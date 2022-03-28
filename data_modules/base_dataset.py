from abc import ABC, abstractmethod
import logging
import math
import os
import pickle
from typing import List
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from data_modules.input_example import InputExample, InputFeatures
from data_modules.utils.tools import mapping_subtok_id, padding


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None         # name of the dataset

    def __init__(
        self,
        tokenizer: str,
        encoder_model: str,
        data_dir: str,
        max_input_length: int,
        seed: int = None,
        split = 'train',
        ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.encoder = AutoModel.from_pretrained(encoder_model, output_hidden_states=True)
        self.encoder.eval()
        self.max_input_length = max_input_length
        self.data_path = data_dir

        self.split = split
        self.examples: List[InputExample] = self.load_data(split=split)
        for example in self.examples:
            example.dataset = self
        
        self.features: List[InputFeatures] = self.compute_features()
        
        self.size = len(self.examples)
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index):
        return self.features[index]
    
    @abstractmethod
    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data(self, split: str) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def _warn_max_sequence_length(self, max_sequence_length: int, input_features: List[InputFeatures]):
        max_length_needed = max(len(x.input_ids) for x in input_features)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f'Max sequence length is {max_sequence_length} but the longest is {max_length_needed} long'
            )
    
    def get_doc_emb(self, input_ids, input_attention_mask):
        num_paras = math.ceil(len(input_ids) / 500.0)
        input_present = []
        for i in range(num_paras):
            start = 500 * i
            end = 500 * (i + 1)
            para_ids = torch.tensor(input_ids[start: end]).unsqueeze(0)
            para_attention_mask = torch.tensor(input_attention_mask[start: end]).unsqueeze(0)
            para_emb = self.encoder(para_ids, para_attention_mask).last_hidden_state
            input_present.append(para_emb)
        input_present = torch.cat(input_present, dim=-1).squeeze(0)
        # print(f"input_present:{input_present.size()}")
        return input_present

    def compute_features(self):
        try:
            os.mkdir(self.data_path + f'featured/{self.split}')
        except FileExistsError:
            pass
        featured_path = self.data_path + f'featured/{self.split}.pkl'
        if os.path.exists(featured_path):
            with open(featured_path, 'rb') as f:
                features = pickle.load(f)  
        else:
            features = []
            for example in tqdm(self.examples, desc="Load features"):    
                input_seq = " ".join(example.tokens)
                input_encoded = self.tokenizer(input_seq)
                input_ids = input_encoded['input_ids']
                input_attention_mask = input_encoded['attention_mask']
                # input_presentation = self.get_doc_emb(input_ids, input_attention_mask)

                subwords_no_space = []
                for index, i in enumerate(input_ids):
                    r_token = self.tokenizer.decode([i])
                    if r_token != ' ':
                        if r_token[0] == ' ':
                            subwords_no_space.append(r_token[1:])
                        else:
                            subwords_no_space.append(r_token)
                
                mapping = mapping_subtok_id(subwords_no_space[1:-1], example.tokens) # w/o <s>, </s> with RoBERTa

                dep_tree = nx.DiGraph()
                for head, tail in zip(example.heads, list(range(len(example.tokens)))):
                    if head != tail:
                        dep_tree.add_edge(head, tail)
                
                adj = torch.eye(len(example.tokens) + 1) # include ROOT and self loop
                for i in range(len(example.tokens)):
                    if dep_tree.has_edge(-1, i):
                        adj[-1, i] = 1
                    for j in range(len(example.tokens)):
                        if dep_tree.has_edge(i, j):
                            adj[i, j] = 1

                for relation in example.relations:
                    label = relation.type.short
                    
                    e1_tok_ids = relation.head.id
                    e2_tok_ids = relation.tail.id
                    trigger_poss = [e1_tok_ids, e2_tok_ids]
                    
                    scores = []
                    for i in range(len(example.tokens)):
                        head_dis = 0 if e1_tok_ids[0] <= i <= e1_tok_ids[-1] else min(abs(i - e1_tok_ids[0]), abs(i - e1_tok_ids[-1]))
                        tail_dis = 0 if e2_tok_ids[0] <= i <= e2_tok_ids[-1] else min(abs(i - e2_tok_ids[0]), abs(i - e2_tok_ids[-1]))
                        scores.append((head_dis, tail_dis))
                    
                    feature = InputFeatures(
                        input_ids=input_ids,
                        input_attention_mask=input_attention_mask,
                        mapping=mapping,
                        label=label,
                        triggers_poss=trigger_poss,
                        dep_path=example.dep_path,
                        adjacent_maxtrix=adj,
                        scores=scores,
                        # input_presentation=input_presentation
                    )
                    features.append(feature)
            with open(featured_path, 'wb') as f:
                pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
            
        # print(f"feature: {features[0]}")
        self._warn_max_sequence_length(self.max_input_length, features)
        print("Loaded {} data points in {} set".format(len(features), self.split))

        return features


    def my_collate(self, batch: List[InputFeatures]):
        max_seq_len = max([len(ex.input_ids) for ex in batch])
        # hidden_size = batch[0].input_presentation.size(-1)
        max_ns = max([len(ex.dep_path) for ex in batch]) # include ROOT

        input_ids = []
        input_attention_mask = []
        # input_ctx_emb = torch.zeros(len(batch), max_seq_len, hidden_size)
        dep_path = []
        labels = []
        adj = torch.zeros((len(batch), max_ns, max_ns), dtype=torch.float)
        masks = torch.zeros((len(batch), max_ns), dtype=torch.float)
        head_dists = []
        tail_dists = []
        for i, ex in enumerate(batch):
            input_ids.append(padding(ex.input_ids, max_sent_len=max_seq_len, pad_tok=self.tokenizer.pad_token_id))
            input_attention_mask.append(padding(ex.input_attention_mask, max_sent_len=max_seq_len, pad_tok=0))
            # input_ctx_emb[i, 0:ex.input_presentation.size(0), :] = ex.input_presentation
            dep_path.append(padding(ex.dep_path, max_sent_len=max_ns, pad_tok=0))
            masks[i, 0:ex.adjacent_maxtrix.size(0)] = masks[i, 0:ex.adjacent_maxtrix.size(0)] + 1
            adj[i, 0:ex.adjacent_maxtrix.size(0), 0:ex.adjacent_maxtrix.size(1)] = ex.adjacent_maxtrix
            labels.append(ex.label)
            head_dists.append(padding([sc[0] for sc in ex.scores], max_sent_len=max_ns, pad_tok=max_ns))
            tail_dists.append(padding([sc[1] for sc in ex.scores], max_sent_len=max_ns, pad_tok=max_ns))
        
        input_ids = torch.tensor(input_ids)
        input_attention_mask = torch.tensor(input_attention_mask)
        dep_path = torch.tensor(dep_path)
        labels = torch.tensor(labels)
        head_dists = torch.tensor(head_dists, dtype=torch.int)
        tail_dists = torch.tensor(tail_dists, dtype=torch.int)

        return (input_ids, input_attention_mask, masks, labels,
                dep_path, adj, head_dists, tail_dists, 
                [ex.mapping for ex in batch], [ex.triggers_poss for ex in batch])



