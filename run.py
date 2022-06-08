import argparse
from asyncore import write
import configparser
import itertools
import json
import logging
import os
from collections import defaultdict
import random
from typing import Dict
import optuna
from pytorch_lightning.trainer.trainer import Trainer
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from pytorch_lightning.utilities.seed import seed_everything
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data_modules.datamodules import load_data_module
from models.model import PlOTEERE
import shutil


def run(defaults: Dict, random_state):
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config

    print("Hyperparams: {}".format(defaults))
    defaults.update(dict(config.items(job)))
    for key in defaults:
        if defaults[key] in ['True', 'False']:
            defaults[key] = True if defaults[key]=='True' else False
        if defaults[key] == 'None':
            defaults[key] = None
    if job == 'HiEve':
        defaults['loss_weights'] = [6833.0/369, 6833.0/348, 6833.0/162, 6833.0/5954]
    elif job == 'ESL':
        defaults['loss_weights'] = [5.0/6, 1.0/6]
    elif job == 'subevent_mulerx':
        defaults['loss_weights'] = [4.0, 4.0, 1.0]
        defaults['tokenizer'] = f'/vinai/hieumdt/pretrained_models/tokenizers/{args.model}'
        defaults['encoder_name_or_path'] = f'/vinai/hieumdt/pretrained_models/models/{args.model}'
        defaults['data_dir'] = f'datasets/mulerx/subevent-{args.lang}-20'
    
    # parse remaining arguments and divide them into three categories
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    second_parser.set_defaults(**defaults)

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    # print(second_parser.parse_args_into_dataclasses(remaining_args))
    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(remaining_args)
    data_args.datasets = job

    if data_args.train_data == None:
        data_args.train_data = data_args.data_dir
    if data_args.dev_data == None:
        data_args.dev_data = data_args.data_dir
    if data_args.test_data == None:
        data_args.test_data = data_args.data_dir

    if args.tuning:
        training_args.output_dir = './tuning_experiments'
    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass

    f1s = []
    ps = []
    rs = []
    val_f1s = []
    val_ps = []
    val_rs = []
    for i in range(data_args.n_fold):
        print(f"TRAINING AND TESTING IN FOLD {i}: ")
        train_fold_dir = f'{data_args.train_data}/{i}' if data_args.n_fold != 1 else data_args.train_data
        test_fold_dir = f'{data_args.test_data}/{i}' if data_args.n_fold != 1 else data_args.test_data
        dev_fold_dir = f'{data_args.dev_data}/{i}' if data_args.n_fold != 1 else data_args.dev_data
        dm = load_data_module(module_name = 'EERE',
                            data_args=data_args,
                            train_fold_dir=train_fold_dir,
                            dev_fold_dir=dev_fold_dir,
                            test_fold_dir=test_fold_dir)
        
        number_step_in_epoch = len(dm.train_dataloader())/training_args.gradient_accumulation_steps
        # construct name for the output directory
        output_dir = os.path.join(
            training_args.output_dir,
            f'{args.job}'
            f'-{model_args.encoder_name_or_path.split("/")[-1]}'
            f'-{defaults["data_dir"].split(r"/")[-1]}'
            f'-random_state{random_state}'
            f'-{model_args.residual_type}'
            f'-lr{training_args.lr}'
            f'-e_lr{training_args.encoder_lr}'
            f'-eps{training_args.num_epoches}'
            f'-regu_weight{training_args.regular_loss_weight}'
            f'-OT_weight{training_args.OT_loss_weight}'
            f'-gcn_num_layers{model_args.gcn_num_layers}'
            f'-fn_actv{model_args.fn_actv}'
            f'-rnn_hidden{model_args.hidden_size}')
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        if data_args.n_fold != 1:
            output_dir = os.path.join(output_dir,f'fold{i}')
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
        
        checkpoint_callback = ModelCheckpoint(
                                    dirpath=output_dir,
                                    save_top_k=1,
                                    monitor='f1_dev',
                                    mode='max',
                                    save_weights_only=True,
                                    filename='{epoch}-{f1_dev:.2f}', # this cannot contain slashes 
                                    )
        lr_logger = LearningRateMonitor(logging_interval='step')
        model = PlOTEERE(model_args=model_args,
                        training_args=training_args,
                        datasets=job,
                        # scratch_tokenizer=data_args.scratch_tokenizer_name_or_path,
                        num_training_step=int(number_step_in_epoch * training_args.num_epoches)
                        )
        
        trainer = Trainer(
            # logger=tb_logger,
            min_epochs=training_args.num_epoches,
            max_epochs=training_args.num_epoches, 
            gpus=[args.gpu], 
            accumulate_grad_batches=training_args.gradient_accumulation_steps,
            num_sanity_val_steps=0, 
            val_check_interval=1.0, # use float to check every n epochs 
            callbacks = [lr_logger, checkpoint_callback],
        )

        print("Training....")
        dm.setup('fit')
        trainer.fit(model, dm)

        best_model = PlOTEERE.load_from_checkpoint(checkpoint_callback.best_model_path)
        print("Testing .....")
        dm.setup('test')
        trainer.test(best_model, dm)
        # print(best_model.model_results)
        p, r, f1 = best_model.model_results
        val_p, val_r, val_f1 = best_model.best_vals
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        val_f1s.append(val_f1)
        val_ps.append(val_p)
        val_rs.append(val_r)
        print(f"RESULT IN FOLD {i}: ")
        print(f"F1: {f1}")
        print(f"P: {p}")
        print(f"R: {r}")

        # shutil.rmtree(f'{output_dir}')
    
        # with open(output_dir+f'-{f1}', 'w', encoding='utf-8') as f:
        #     f.write(f"F1: {f1} \n")
        #     f.write(f"P: {p} \n")
        #     f.write(f"R: {r} \n")

    f1 = sum(f1s)/len(f1s)
    p = sum(ps)/len(ps)
    r = sum(rs)/len(rs)
    val_f1 = sum(val_f1s)/len(val_f1s)
    val_p = sum(val_ps)/len(val_ps)
    val_r = sum(val_rs)/len(val_rs)
    print(f"F1: {f1} - P: {p} - R: {r}")
    
    return p, f1, r, val_p, val_r, val_f1


def objective(trial: optuna.Trial):
    defaults = {
        'lr': trial.suggest_categorical('lr', [5e-4]),
        'OT_max_iter': trial.suggest_categorical('OT_max_iter', [50]),
        'encoder_lr': trial.suggest_categorical('encoder_lr', [3e-6]),
        'batch_size': trial.suggest_categorical('batch_size', [8]),
        'warmup_ratio': 0.1,
        'num_epoches': trial.suggest_categorical('num_epoches', [30]), # 
        # 'use_pretrained_wemb': trial.suggest_categorical('wemb', [True, False]),
        'regular_loss_weight': trial.suggest_categorical('regular_loss_weight', [0.1]),
        'OT_loss_weight': trial.suggest_categorical('OT_loss_weight', [0.1]),
        'distance_emb_size': trial.suggest_categorical('distance_emb_size', [0]),
        # 'gcn_outp_size': trial.suggest_categorical('gcn_outp_size', [256, 512]),
        'seed': 7890,
        'gcn_num_layers': trial.suggest_categorical('gcn_num_layers', [3]),
        'hidden_size': trial.suggest_categorical('hidden_size', [768]),
        'rnn_num_layers': trial.suggest_categorical('rnn_num_layers', [1]),
        'fn_actv': trial.suggest_categorical('fn_actv', ['leaky_relu']), # 'relu', 'tanh', 'hardtanh', 'silu'
        'residual_type': trial.suggest_categorical('residual_type', ['addtive'])
    }

    random_state = defaults['seed']
    print(f"Random_state: {random_state}")

    seed_everything(random_state, workers=True)

    dataset = args.job
    
    p, f1, r, val_p, val_r, val_f1 = run(defaults=defaults, random_state=random_state)

    record_file_name = 'result.txt'
    if args.tuning:
        record_file_name = f'result_{args.job}_{args.lang}_{defaults["tokenizer"].split(r"/")[-1]}.txt'

    with open(record_file_name, 'a', encoding='utf-8') as f:
        f.write(f"{'--'*10} \n")
        f.write(f"Dataset: {dataset} \n")
        f.write(f"Random_state: {random_state}\n")
        f.write(f"Hyperparams: \n {defaults}\n")
        f.write(f"F1: {f1} - {val_f1} \n")
        f.write(f"P: {p} - {val_p} \n")
        f.write(f"R: {r} - {val_r} \n")

    return f1


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', help='configuration file')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use')
    parser.add_argument('-t', '--tuning', action='store_true', default=False, help='tune hyperparameters')
    parser.add_argument('-m', '--model', type=str, default='mBERT-base', help='Encoder model')
    parser.add_argument('-la', '--lang', type=str, default='en', help='Language')

    args, remaining_args = parser.parse_known_args()
    
    if args.tuning:
        print("tuning ......")
        # sampler = optuna.samplers.TPESampler(seed=1741)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        trial = study.best_trial
        print('Accuracy: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))


