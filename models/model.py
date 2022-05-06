from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from transformers import AdamW, get_linear_schedule_with_warmup
from arguments import ModelArguments, TrainingArguments
from models.OTEERE import OTEERE
from utils.utils import compute_f1


class PlOTEERE(pl.LightningModule):
    def __init__(self, 
                model_args: ModelArguments, 
                training_args: TrainingArguments,
                datasets: str,
                # scratch_tokenizer: str,
                
                num_training_step: int
                ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if training_args.encoder_lr == 0:
            self.tune_encoder = False
        else:
            self.tune_encoder = True

        self.model = OTEERE(encoder_model=model_args.encoder_name_or_path,
                            max_seq_len=training_args.max_seq_len,
                            distance_emb_size=model_args.distance_emb_size,
                            hidden_size=model_args.hidden_size,
                            gcn_num_layers=model_args.gcn_num_layers,
                            num_labels=training_args.num_labels,
                            loss_weights=training_args.loss_weights,
                            rnn_num_layers=model_args.rnn_num_layers,
                            dropout=model_args.dropout,
                            OT_eps=model_args.OT_eps,
                            OT_max_iter=model_args.OT_max_iter,
                            OT_reduction=model_args.OT_reduction,
                            fn_actv=model_args.fn_actv,
                            regular_loss_weight=training_args.regular_loss_weight,
                            OT_loss_weight=training_args.OT_loss_weight,
                            tune_encoder=self.tune_encoder,
                            residual_type=model_args.residual_type)
        self.model_results = []
        self.best_vals = [0, 0, 0]
    
    def training_step(self, batch, batch_idx):
        logits, loss, pred_loss, regu_loss, cost, labels = self.model(*batch)
        self.log_dict({'train_loss': loss, 'pred_loss': pred_loss, 'regu_loss': regu_loss, 'OT_loss': cost}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        logits, loss, pred_loss, regu_loss, cost, labels = self.model(*batch)
        # print(logits.size())
        pred_labels = torch.max(logits, dim=1).indices.cpu().numpy()
        labels = labels.cpu().numpy()
        return pred_labels, labels
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        labels = []
        predicts = []
        for output in outputs:
            # print(f"Output: {output}")
            labels.extend(output[1])
            predicts.extend(output[0])
        # print(labels)
        # print(predicts)
        p, r, f1 = compute_f1(dataset=self.hparams.datasets, 
                            num_label=self.hparams.training_args.num_labels, 
                            gold=labels, 
                            pred=predicts,
                            report=True)
        self.log_dict({'f1_dev': f1, 'p_dev': p, 'r_dev': r}, prog_bar=True)
        if f1 >= self.best_vals[-1]:
            print((p, r, f1))
            self.best_vals = [p, r, f1]
        return f1
    
    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        logits, loss, pred_loss, regu_loss, cost, labels = self.model(*batch)
        pred_labels = torch.max(logits, dim=1).indices.cpu().numpy()
        labels = labels.cpu().numpy()
        return pred_labels, labels
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        labels = []
        predicts = []
        for output in outputs:
            # print(f"Output: {output}")
            labels.extend(output[1])
            predicts.extend(output[0])
        p, r, f1 = compute_f1(dataset=self.hparams.datasets, 
                            num_label=self.hparams.training_args.num_labels, 
                            gold=labels, 
                            pred=predicts,
                            report=True)
        self.model_results = (p, r, f1)
        return p, r, f1
    
    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        num_batches = self.hparams.num_training_step / self.trainer.accumulate_grad_batches
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_pretrain_parameters = [
            {
                    "params": [p for n, p in self.model.named_parameters() if 'encoder.' not in n],
                    "weight_decay": 0.0,
                    "lr": self.hparams.training_args.lr
                }
        ]
        if self.tune_encoder == True:
            optimizer_grouped_pretrain_parameters.extend([
                {
                    "params": [p for n, p in self.model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.,
                    "lr": self.hparams.training_args.encoder_lr
                },
                {
                    "params": [p for n, p in self.model.encoder.named_parameters() if  any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": self.hparams.training_args.encoder_lr
                },])
        
        optimizer = AdamW(optimizer_grouped_pretrain_parameters, betas=[0.9, 0.999], eps=1e-8)
        num_warmup_steps = 0.1 * num_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step'
            }
        }

