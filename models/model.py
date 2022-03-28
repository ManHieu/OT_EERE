from typing import Any, Optional
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
                # num_training_step: int
                ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = OTEERE(encoder_model=model_args.encoder_name_or_path,
                            max_seq_len=training_args.max_seq_len,
                            distance_emb_size=model_args.distance_emb_size,
                            # gcn_outp_size=model_args.gcn_outp_size,
                            gcn_num_layers=model_args.gcn_num_layers,
                            num_labels=training_args.num_labels,
                            loss_weights=training_args.loss_weights,
                            rnn_hidden_size=model_args.rnn_hidden_size,
                            rnn_num_layers=model_args.rnn_num_layers,
                            dropout=model_args.dropout,
                            OT_eps=model_args.OT_eps,
                            OT_max_iter=model_args.OT_max_iter,
                            OT_reduction=model_args.OT_reduction,
                            fn_actv=model_args.fn_actv,
                            regular_loss_weight=training_args.regular_loss_weight,)
        self.model_results = []
    
    def training_step(self, batch, batch_idx):
        logits, loss = self.model(*batch)
        self.log_dict({'train_loss': loss,}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        logits, loss = self.model(*batch)
        # print(logits.size())
        pred_labels = torch.max(logits, dim=1).indices.cpu().numpy()
        labels = batch[4].cpu().numpy()
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
                            pred=predicts)
        self.log_dict({'f1_dev': f1, 'p_dev': p, 'r_dev': r}, prog_bar=True)
        return f1
    
    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        logits, loss = self.model(*batch)
        pred_labels = torch.max(logits, dim=1).indices.cpu().numpy()
        labels = batch[4].cpu().numpy()
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
                            pred=predicts)
        self.model_results = (p, r, f1)
        return p, r, f1
    
    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        # num_batches = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_pretrain_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if 'encoder.' not in n],
                "weight_decay": 0.0,
                "lr": self.hparams.training_args.lr
            }]
        optimizer = AdamW(optimizer_grouped_pretrain_parameters, eps=self.hparams.training_args.adam_epsilon)
        # num_warmup_steps = self.hparams.training_args.warmup_ratio * self.hparams.num_training_step
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.hparams.num_training_step
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     'interval': 'step'
            # }
        }

