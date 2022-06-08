import argparse
from asyncore import write
import configparser
import itertools
import json
import logging
from opcode import opname
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
from data_modules.datatsets import load_dataset
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

        ranges = [(0, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, 40), (40, 9999)]
        test_loaders = {}
        val_loaders = {}
        for r in ranges:
            test_data = load_dataset(
                name=data_args.datasets,
                tokenizer=data_args.tokenizer,
                encoder=data_args.encoder,
                data_dir=test_fold_dir,
                max_input_length=data_args.max_seq_length,
                seed=1741,
                split = 'test', 
                range_dist=r)
            test_data_loader = DataLoader(
                dataset=test_data,
                batch_size=data_args.batch_size,
                shuffle=False,
                collate_fn=test_data.my_collate,
            )
            test_loaders[f"{r[0]} - {r[1]}"] = test_data_loader

            val_data = load_dataset(
                name=data_args.datasets,
                tokenizer=data_args.tokenizer,
                encoder=data_args.encoder,
                data_dir=dev_fold_dir,
                max_input_length=data_args.max_seq_length,
                seed=1741,
                split = 'val', 
                range_dist=r)
            val_data_loader = DataLoader(
                dataset=val_data,
                batch_size=data_args.batch_size,
                shuffle=True,
                collate_fn=test_data.my_collate,
            )
            val_loaders[f"{r[0]} - {r[1]}"] = val_data_loader
            
        model = PlOTEERE(model_args=model_args,
                        training_args=training_args,
                        datasets=job,
                        # scratch_tokenizer=data_args.scratch_tokenizer_name_or_path,
                        num_training_step=0
                        )
        
        trainer = Trainer(
            # logger=tb_logger,
            min_epochs=training_args.num_epoches,
            max_epochs=training_args.num_epoches, 
            gpus=[args.gpu], 
            accumulate_grad_batches=training_args.gradient_accumulation_steps,
            num_sanity_val_steps=0, 
            val_check_interval=1.0, # use float to check every n epochs 
        )

        best_model = PlOTEERE.load_from_checkpoint('tuning_experiments/HiEve-roberta-large--random_state7890-addtive-lr0.0005-e_lr3e-06-eps30-regu_weight0.1-OT_weight0.1-gcn_num_layers3-fn_actvleaky_relu-rnn_hidden768/epoch=18-f1_dev=0.49.ckpt')
        record_file_name = "exp_distant.txt"
        with open(record_file_name, 'a', encoding='UTF-8') as f:
            f.write(f"{'--'*100} \n")
            f.write(f"Dataset: {dataset} \n")
            f.write(f"Random_state: {random_state}\n")
            f.write(f"Hyperparams: \n {defaults}\n")
        for key in test_loaders.keys():
            print("Experiment on {}")
            trainer.test(best_model, val_loaders[key])
            val_p, val_r, val_f1 = best_model.model_results
            best_model.model_results = [0, 0, 0]

            trainer.test(best_model, test_loaders[key])
            p, r, f1 = best_model.model_results
            with open(record_file_name, 'a', encoding='UTF-8') as f:
                f.write(f"---------Distance range: {key} ----------\n")
                f.write(f"Dev_result: p: {val_p} - r: {val_r} - f1: {val_f1} \n")
                f.write(f"Test_result: p: {p} - r: {r} - f1: {f1} \n")
    
    return [0]*6


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
    
    run(defaults=defaults, random_state=random_state)
    return 0


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


    defaults = {
        'lr': 5e-4,
        'OT_max_iter': 50,
        'encoder_lr': 3e-6,
        'batch_size': 8,
        'warmup_ratio': 0.1,
        'num_epoches': 30, # 
        # 'use_pretrained_wemb': trial.suggest_categorical('wemb', [True, False]),
        'regular_loss_weight': 0.1,
        'OT_loss_weight': 0.1,
        'distance_emb_size': 0,
        # 'gcn_outp_size': trial.suggest_categorical('gcn_outp_size', [256, 512]),
        'seed': 7890,
        'gcn_num_layers': 3,
        'hidden_size': 768,
        'rnn_num_layers': 1,
        'fn_actv': 'leaky_relu', # 'relu', 'tanh', 'hardtanh', 'silu'
        'residual_type': 'addtive'
    }

    random_state = defaults['seed']
    print(f"Random_state: {random_state}")
    seed_everything(random_state, workers=True)
    dataset = args.job
    record_file_name = 'exp_triggers_dist.txt'
    
    if args.tuning:
        print("tuning ......")
        # sampler = optuna.samplers.TPESampler(seed=1741)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        trial = study.best_trial
        print('Accuracy: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))


