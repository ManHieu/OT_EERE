from collections import OrderedDict
import math
import pdb
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import AutoModel
# from data_modules.utils.scratch_tokenizer import ScratchTokenizer
from models.GCN import GCN
from models.sinkhorn import SinkhornDistance


class OTEERE(nn.Module):
    def __init__(self,
                encoder_model: str,
                max_seq_len: int,
                distance_emb_size: int,
                hidden_size: int,
                gcn_num_layers: int,
                num_labels: int,
                loss_weights: List[float],
                rnn_num_layers: int = 1,
                dropout: float = 0.5,
                OT_eps: float = 0.1,
                OT_max_iter: int = 100,
                OT_reduction: str = 'mean',
                fn_actv: str = 'relu',
                regular_loss_weight: float = 0.1,
                OT_loss_weight: float = 0.1,
                tune_encoder: bool = False,
                residual_type: str = 'concat',
                ) -> None:
        super().__init__()

        self.drop_out = nn.Dropout(dropout)

        # Encoding layers
        print(f"Load pretrain model from: {encoder_model}")
        self.encoder = AutoModel.from_pretrained(encoder_model, output_hidden_states=True)
        self.tune_encoder = tune_encoder
        self.word_embedding_size = 0
        self.use_wemb = False
        self.in_size = 768 + distance_emb_size * 2 + self.word_embedding_size if 'base' in encoder_model else 1024 + distance_emb_size * 2 + self.word_embedding_size
        self.rnn_hidden_size = hidden_size
        if rnn_num_layers > 1:
            self.rnn = nn.LSTM(self.in_size, self.rnn_hidden_size, rnn_num_layers, 
                                batch_first=True, dropout=dropout, bidirectional=True)
        else:
            self.rnn = nn.LSTM(self.in_size, self.rnn_hidden_size, rnn_num_layers, 
                                batch_first=True, bidirectional=True)

        # OT
        self.sinkhorn = SinkhornDistance(eps=OT_eps, max_iter=OT_max_iter, reduction=OT_reduction)

        # GCN layers
        self.q = nn.Parameter(torch.randn(2))
        self.q.requires_grad = True
        gcn_input_size = 2 * self.rnn_hidden_size
        gcn_outp_size = hidden_size
        self.gcn = GCN(gcn_input_size, gcn_outp_size, gcn_num_layers, hidden_size, rnn_num_layers, dropout)
        self.full_doc_gcn = GCN(gcn_input_size, gcn_outp_size, gcn_num_layers, hidden_size, rnn_num_layers, dropout)

        # Classifier layers
        self.residual_type = residual_type
        if fn_actv == 'relu':
            self.fn_actv = nn.ReLU()
        elif fn_actv == 'leaky_relu':
            self.fn_actv = nn.LeakyReLU(0.2, True)
        elif fn_actv == 'tanh':
            self.fn_actv = nn.Tanh()
        elif fn_actv == 'relu6':
            self.fn_actv = nn.ReLU6()
        elif fn_actv == 'silu':
            self.fn_actv = nn.SiLU()
        elif fn_actv == 'hardtanh':
            self.fn_actv = nn.Hardtanh()

        self.doc_attn = nn.Sequential(OrderedDict([('dropout',self.drop_out), 
                                                    ('fc1', nn.Linear(gcn_outp_size, int(gcn_outp_size/2))), 
                                                    ('dropout', self.drop_out), 
                                                    ('fn_actv', self.fn_actv), 
                                                    ('fc2', nn.Linear(int(gcn_outp_size/2), 1))]
                                                    ))
        
        self.fc = nn.Linear(gcn_input_size, hidden_size)

        if self.residual_type == 'concat':
            classifier_in_size = hidden_size * 3          
        fc1 = nn.Linear(classifier_in_size, int(classifier_in_size/2))
        fc2 = nn.Linear(int(classifier_in_size/2), int(classifier_in_size/4))
        fc3 = nn.Linear(int(classifier_in_size/4), num_labels)
        self.classifier = nn.Sequential(OrderedDict([('dropout',self.drop_out), 
                                                    ('fc1', fc1), 
                                                    ('dropout', self.drop_out), 
                                                    ('fn_actv', self.fn_actv), 
                                                    ('fc2',fc2),
                                                    ('dropout', self.drop_out), 
                                                    ('fn_actv', self.fn_actv), 
                                                    ('fc3',fc3),]))
        
        # Loss fuction
        self.weights = torch.tensor(loss_weights)
        self.loss_pred = nn.CrossEntropyLoss(weight=self.weights)
        self.loss_regu = nn.MSELoss()
        self.regular_loss_weight = regular_loss_weight
        self.OT_loss_weight = OT_loss_weight
    
    def _init_word_embedding(self):
        vocab_size = len(self.vocab.items())
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=100)
        self.word_embedding_size = 50

        glove = pd.read_csv('/vinai/hieumdt/glove/glove.6B.50d.txt', sep=" ", quoting=3, header=None, index_col=0)
        glove_embedding = {key: val.values for key, val in glove.T.items()}

        embedding_matrix=np.zeros((vocab_size,self.word_embedding_size))
        for i, w in self.vocab.items():
            if w in glove_embedding.keys():
                embedding_matrix[i] = glove_embedding[w]
        self.word_embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
        self.word_embedding.weight.requires_grad=True        
    
    def encode_with_rnn(self, inp: torch.Tensor(), ls: List[int]) -> torch.Tensor(): # batch_size x max_seq_len x hidden_dim*2
        packed = pack_padded_sequence(inp, ls, batch_first=True, enforce_sorted=False)
        rnn_encode, _ = self.rnn(packed)
        outp, _ = pad_packed_sequence(rnn_encode, batch_first=True)
        return outp

    def forward(self, 
                input_ids: torch.Tensor, 
                input_attention_mask: torch.Tensor, 
                masks: torch.Tensor, 
                adjs: torch.Tensor, 
                mapping: List[Dict[int, List[int]]], 
                trigger_poss: List[Tuple[List[int], List[int]]],
                input_token_ids: torch.Tensor,
                host_sentences_masks: torch.Tensor,
                dep_path: torch.Tensor,
                scores: torch.Tensor,
                labels: torch.Tensor, 
                ):
        
        bs = input_ids.size(0)
        # Embedding
        # Compute transformer embedding
        if self.tune_encoder:
            # _context_emb = self.encoder(input_ids, input_attention_mask).last_hidden_state # (bs, max_seq_len, encoder_hidden_size)
            _context_emb = []
            num_para = math.ceil(input_ids.size(1) / 512.0)
            # print(f"input_ids: {input_ids.size()}")
            for i in range(num_para):
                start = i * 512
                end = (i + 1) * 512
                para_ids = input_ids[:, start: end]
                para_input_attn_mask = input_attention_mask[:, start:end]
                # print(f"para_ids: {para_ids.size()}")
                # print(f"para_attn_mask: {para_input_attn_mask.size()}")
                para_ctx = self.encoder(para_ids, para_input_attn_mask).last_hidden_state
                # print(f"para_ctx: {para_ctx.size()}")
                _context_emb.append(para_ctx)
            _context_emb = torch.cat(_context_emb, dim=1) # (bs, max_seq_len, encoder_hidden_size)

        else:
            with torch.no_grad():
                # _context_emb = self.encoder(input_ids, input_attention_mask).last_hidden_state # (bs, max_seq_len, encoder_hidden_size)
                _context_emb = []
                num_para = math.ceil(input_ids.size(1) / 512.0)
                # print(f"input_ids: {input_ids.size()}")
                for i in range(num_para):
                    start = i * 512
                    end = (i + 1) * 512
                    para_ids = input_ids[:, start: end]
                    para_input_attn_mask = input_attention_mask[:, start:end]
                    # print(f"para_ids: {para_ids.size()}")
                    # print(f"para_attn_mask: {para_input_attn_mask.size()}")
                    para_ctx = self.encoder(para_ids, para_input_attn_mask).last_hidden_state
                    # print(f"para_ctx: {para_ctx.size()}")
                    _context_emb.append(para_ctx)
                _context_emb = torch.cat(_context_emb, dim=1) # (bs, max_seq_len, encoder_hidden_size)

        # Compute word embedding
        if self.use_wemb:
            word_emb = self.word_embedding(input_token_ids) # (bs)

        context_emb = []
        max_ns = masks.size(1)
        for i, map in enumerate(mapping):
            emb = []
            ns = int(torch.sum(masks[i]))
            n_sub_tok = int(torch.sum(input_attention_mask[i]))
            map[ns-1] = list(range(n_sub_tok-1)) # ROOT mapping
            for tok_id in range(ns):
                token_mapping = map.get(tok_id)
                if token_mapping != None:
                    try:
                        tok_presentation = _context_emb[i, token_mapping, :]
                    except:
                        # pdb.set_trace()
                        pass
                    tok_presentation = torch.max(tok_presentation, dim=0)[0] # (encoder_hidden_size)
                else:
                    tok_presentation = torch.zeros_like(_context_emb[0, 0])
                # print(f"tok_presentation: {tok_presentation.size()}")
                if self.use_wemb:
                    if tok_id != ns - 1:
                        tok_presentation = torch.cat([tok_presentation, word_emb[i, tok_id, :]], dim=-1)
                    else:
                        tok_presentation = torch.cat([tok_presentation, torch.max(word_emb[i, 0:ns, :], dim=0)[0]], dim=-1)
                emb.append(tok_presentation)
            # padding
            if max_ns > ns:
                emb = emb + [torch.zeros(tok_presentation.size()).cuda()] * (max_ns-ns)
            emb = torch.stack(emb, dim=0) # (max_ns, encoder_hidden_size)            
            context_emb.append(emb)

        context_emb = torch.stack(context_emb, dim=0) # (bs, max_ns, hiden_size)
        emb = context_emb

        # Encoding with RNN
        ls = [torch.sum(masks[i]).item() for i in range(bs)]
        gcn_input = self.encode_with_rnn(emb, ls) # (bs, max_ns, gcn_input_size))

        # Regularizarion and classification
        _labels = []
        
        # full_doc_presentations = []
        # full_doc_heads = []
        # full_doc_tails = []
        
        # heads = []
        # tails = []
        # doc_presentations = []

        input_doc_presentations = []
        input_heads = []
        input_tails = []

        _gcn_input = self.fc(gcn_input)
        # full_doc_attn = self.doc_attn(full_doc_gcn_oupt)
        input_doc_attn = self.doc_attn(_gcn_input)
        # doc_attn = self.doc_attn(gcn_outp)

        for i in range(bs):
            pairs = trigger_poss[i]
            label = labels[i]
            for pair, lb in zip(pairs, label):
                _labels.append(lb)
                # full_doc_presentation = torch.sum(full_doc_gcn_oupt[i, :, :] * full_doc_attn[i], dim=0)
                # full_doc_presentations.append(full_doc_presentation)

                input_doc_presentation = torch.sum(_gcn_input[i, :, :] * input_doc_attn[i], dim=0)
                input_doc_presentations.append(input_doc_presentation)
                
                # doc_presentation = torch.sum(gcn_outp[i, :, :] * doc_attn[i], dim=0)
                # doc_presentations.append(doc_presentation)

                head_ids, tail_ids = pair
                
                # full_doc_head = torch.max(full_doc_gcn_oupt[i, head_ids[0]: head_ids[-1] + 1, :], dim=0)[0]
                # full_doc_tail = torch.max(full_doc_gcn_oupt[i, tail_ids[0]: tail_ids[-1] + 1, :], dim=0)[0]
                # full_doc_heads.append(full_doc_head)
                # full_doc_tails.append(full_doc_tail)

                input_head = torch.max(_gcn_input[i, head_ids[0]: head_ids[-1] + 1, :], dim=0)[0]
                input_tail = torch.max(_gcn_input[i, tail_ids[0]: tail_ids[-1] + 1, :], dim=0)[0]
                input_heads.append(input_head)
                input_tails.append(input_tail)

                # head = torch.max(gcn_outp[i, head_ids[0]: head_ids[-1] + 1, :], dim=0)[0]
                # tail = torch.max(gcn_outp[i, tail_ids[0]: tail_ids[-1] + 1, :], dim=0)[0]
                # heads.append(head)
                # tails.append(tail)
        
        _labels = torch.tensor(_labels).cuda()
        # heads = torch.stack(heads, dim= 0)
        # tails = torch.stack(tails, dim=0)
        # doc_presentations = torch.stack(doc_presentations, dim=0)

        input_heads = torch.stack(input_heads, dim= 0)
        input_tails = torch.stack(input_tails, dim=0)
        input_doc_presentations = torch.stack(input_doc_presentations, dim=0)

        # full_doc_heads = torch.stack(full_doc_heads, dim=0)
        # full_doc_tails = torch.stack(full_doc_tails, dim=0)
        # full_doc_presentations = torch.stack(full_doc_presentations, dim=0)
        
        # Classification
        if self.residual_type == 'concat':
            presentations = torch.cat([input_heads,input_tails,input_doc_presentations], dim=1)
        logits = self.classifier(presentations)

        # Compute loss
        pred_loss = self.loss_pred(logits, _labels)
        return logits, pred_loss, pred_loss, 0, 0, _labels

        

