from os import name
from typing import Dict
from transformers import T5Tokenizer
from torch.utils.data import DataLoader 
import pytorch_lightning as pl
from arguments import DataTrainingArguments
from transformers import AutoTokenizer, AutoModel
import copy
from data_modules.datatsets import load_dataset

DATA_MODULES: Dict[str, pl.LightningDataModule] = {}


def register_data_module(data_module_class: pl.LightningDataModule):
    DATA_MODULES[data_module_class.name] = data_module_class
    return data_module_class


def load_data_module(module_name, data_args: DataTrainingArguments, fold_dir: str) -> pl.LightningDataModule:
    """
    Load a registered data module.
    """
    return DATA_MODULES[module_name](data_args=data_args, fold_dir=fold_dir)


@register_data_module
class EEREDataModule(pl.LightningDataModule):
    """
    Dataset processing for Event Event Relation Extraction.
    """
    name = 'EERE'

    def __init__(self, data_args: DataTrainingArguments, fold_dir: str, seed=0):
        super().__init__()
        self.save_hyperparameters()
        self.data_name = data_args.datasets
        self.tokenizer = data_args.tokenizer
        self.encoder = data_args.encoder
        self.max_seq_len = data_args.max_seq_length
        self.batch_size = data_args.batch_size
        self.data_dir = fold_dir
    
    def train_dataloader(self):
        dataset = load_dataset(
                name=self.data_name,
                tokenizer=self.tokenizer,
                encoder=self.encoder,
                data_dir=self.data_dir,
                max_input_length=self.max_seq_len,
                seed=self.hparams.seed,
                split = 'train')
        dataloader = DataLoader(
            dataset= dataset,
            batch_size= self.batch_size,
            shuffle=True,
            collate_fn=dataset.my_collate,
        )
        return dataloader
    
    def val_dataloader(self):
        dataset = load_dataset(
                name=self.data_name,
                tokenizer=self.tokenizer,
                encoder=self.encoder,
                data_dir=self.data_dir,
                max_input_length=self.max_seq_len,
                seed=self.hparams.seed,
                split = 'val')
        dataloader = DataLoader(
            dataset= dataset,
            batch_size= self.batch_size,
            shuffle=True,
            collate_fn=dataset.my_collate,
        )
        return dataloader
    
    def test_dataloader(self):
        dataset = load_dataset(
                name=self.data_name,
                tokenizer=self.tokenizer,
                encoder=self.encoder,
                data_dir=self.data_dir,
                max_input_length=self.max_seq_len,
                seed=self.hparams.seed,
                split = 'test')
        dataloader = DataLoader(
            dataset= dataset,
            batch_size= self.batch_size,
            shuffle=True,
            collate_fn=dataset.my_collate,
        )
        return dataloader
