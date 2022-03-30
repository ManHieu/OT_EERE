from collections import OrderedDict
import math
from this import d
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from models.GCN import GCN
from models.sinkhorn import SinkhornDistance


class OTEERE(nn.Module):
    def __init__(self,
                encoder_model: str,
                max_seq_len: int,
                distance_emb_size: int,
                # gcn_outp_size: int,
                gcn_num_layers: int,
                num_labels: int,
                loss_weights: List[float],
                rnn_hidden_size: int,
                rnn_num_layers: int = 1,
                dropout: float = 0.5,
                OT_eps: float = 0.1,
                OT_max_iter: int = 100,
                OT_reduction: str = 'mean',
                fn_actv: str = 'relu',
                regular_loss_weight: float = 0.1,
                ) -> None:
        super().__init__()

        self.drop_out = nn.Dropout(dropout)

        # Encoding layers
        self.encoder = AutoModel.from_pretrained(encoder_model, output_hidden_states=True)
        # self.distance_emb = nn.Embedding(500, distance_emb_size)
        self.in_size = 768 + distance_emb_size * 2 if 'base' in encoder_model else 1024 + distance_emb_size * 2
        if rnn_num_layers > 1:
            self.rnn = nn.LSTM(self.in_size, int(self.in_size/2), rnn_num_layers, 
                                batch_first=True, dropout=dropout, bidirectional=True)
        else:
            self.rnn = nn.LSTM(self.in_size, int(self.in_size/2), rnn_num_layers, 
                                batch_first=True, bidirectional=True)

        # OT
        self.sinkhorn = SinkhornDistance(eps=OT_eps, max_iter=OT_max_iter, reduction=OT_reduction)

        # GCN layers
        gcn_input_size = 2 * int(self.in_size/2)
        gcn_outp_size = gcn_input_size
        self.gcn = GCN(gcn_input_size, gcn_input_size, gcn_num_layers, rnn_hidden_size, rnn_num_layers, dropout)

        # Classifier layers
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
        fc1 = nn.Linear(gcn_outp_size*3, int(gcn_outp_size*1.5))
        fc2 = nn.Linear(int(gcn_outp_size*1.5), num_labels)
        self.classifier = nn.Sequential(OrderedDict([('dropout1',self.drop_out), 
                                                    ('fc1', fc1), 
                                                    ('dropout2', self.drop_out), 
                                                    ('relu', self.fn_actv), 
                                                    ('fc2',fc2) ]))
        
        # Loss fuction
        self.weights = torch.tensor(loss_weights)
        self.loss_pred = nn.CrossEntropyLoss(weight=self.weights)
        self.loss_regu = nn.MSELoss()
        self.regular_loss_weight = regular_loss_weight
    
    def encode_with_rnn(self, inp: torch.Tensor(), ls: List[int]) -> torch.Tensor(): # batch_size x max_seq_len x hidden_dim*2
        packed = pack_padded_sequence(inp, ls, batch_first=True, enforce_sorted=False)
        rnn_encode, _ = self.rnn(packed)
        outp, _ = pad_packed_sequence(rnn_encode, batch_first=True)
        return outp

    def forward(self, 
                input_ids: torch.Tensor, 
                input_attention_mask: torch.Tensor, 
                # ctx_emb: torch.Tensor,
                masks: torch.Tensor, 
                labels: torch.Tensor, 
                dep_paths: torch.Tensor, 
                adjs: torch.Tensor, 
                head_dists: torch.Tensor, 
                tail_dists: torch.Tensor,
                mapping: List[Dict[int, List[int]]], 
                trigger_poss: List[List[List[int]]],
                ):
        
        bs = input_ids.size(0)
        # Embedding
        # Compute transformer embedding
        _context_emb = self.encoder(input_ids, input_attention_mask).last_hidden_state
        with torch.no_grad():
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

        # _context_emb = ctx_emb
        # print(f"_contex_emb: {_context_emb.size()}")
        context_emb = []
        max_ns = masks.size(1)
        for i, map in enumerate(mapping):
            emb = []
            ns = int(torch.sum(masks[i]))
            map[ns-1] = list(range(ns-1)) # ROOT mapping
            for tok_id in range(ns):
                token_mapping = map[tok_id]
                tok_presentation = _context_emb[i, token_mapping[0]: token_mapping[-1]+1, :]
                tok_presentation = torch.max(tok_presentation, dim=0)[0] # (encoder_hidden_size)
                # print(f"tok_presentation: {tok_presentation.size()}")
                emb.append(tok_presentation)
            # padding
            if max_ns > ns:
                emb = emb + [torch.zeros(tok_presentation.size()).cuda()] * (max_ns-ns)

            emb = torch.stack(emb, dim=0) # (max_ns, encoder_hidden_size)
            # print(f"emb: {emb.size()}")
            
            context_emb.append(emb)

        context_emb = torch.stack(context_emb, dim=0) # (bs, max_ns, hiden_size)
        # Compute distance embedding
        # print(head_dists.size())
        # print(self.distance_emb)
        # head_dists_emb = self.distance_emb(head_dists)
        # tail_dists_emb = self.distance_emb(tail_dists)

        emb = context_emb
        # emb = torch.cat(emb, dim=2)

        # Encoding with RNN
        # print(masks)
        ls = [torch.sum(masks[i]).item() for i in range(bs)]
        gcn_input = self.encode_with_rnn(emb, ls) # (bs, max_ns, gcn_input_size)
        # print(f"gcn_input: {gcn_input}")

        # OT
        on_dps = [] # on dependency path token presentation
        off_dps = [] # off denpendency path token presentation
        on_dp_maginals = [] # on dependency path maginal distribution
        off_dp_maginals = [] # off dependency path maginal distribution
        for i in range(bs):
            dep_path = dep_paths[i]
        
            on_dp_score = - torch.min(torch.stack([head_dists[i], tail_dists[i]], dim=0), dim=0)[0] * dep_path
            # print(f"on_dp_score: {on_dp_score.size()}")
            on_dp_score[dep_path==0] = -10000.0
            on_dp_maginal = F.softmax(on_dp_score.float(), dim=0)
            off_dp_score = - torch.min(torch.stack([head_dists[i], tail_dists[i]], dim=0), dim=0)[0] * (1 - dep_path) * masks[i]
            off_dp_score[((1 - dep_path) * masks[i])==0] = -10000.0
            off_dp_maginal = F.softmax(off_dp_score.float(), dim=0) # (max_ns)
            # null_prob = torch.mean(off_dp_maginal, dim=0).unsqueeze(0)
            on_dp_maginal = torch.cat([torch.Tensor([0.5]).cuda(), 0.5 * on_dp_maginal], dim=0) # (max_ns + 1)
            # print(torch.sum(on_dp_maginal))
            # print(f"on_dp_maginal: {on_dp_maginal.size()}")
            # print(f"off_dp_maginal: {off_dp_maginal.size()}")
            
            on_dp = gcn_input[i] * dep_path.unsqueeze(1).expand((gcn_input.size(1), gcn_input.size(2)))
            off_dp = gcn_input[i] * ((1 - dep_path) * masks[i]).unsqueeze(1).expand((gcn_input.size(1), gcn_input.size(2))) # (max_ns, gcn_input_size)
            null_presentation = torch.mean(off_dp, dim=0).unsqueeze(0)
            on_dp = torch.cat([null_presentation, on_dp], dim=0) # (max_ns+1, gcn_input_size)
            # print(f'on_dp: {on_dp.size()}')
            # print(f"off_dp: {off_dp.size()}")
            
            on_dps.append(on_dp)
            off_dps.append(off_dp)
            on_dp_maginals.append(on_dp_maginal)
            off_dp_maginals.append(off_dp_maginal)
        
        on_dps = torch.stack(on_dps, dim=0)
        # print(f"on_dp: {torch.sum(on_dps)}")
        off_dps = torch.stack(off_dps, dim=0)
        # print(f"off_dp: {torch.sum(off_dps)}")
        on_dp_maginals = torch.stack(on_dp_maginals, dim=0)
        off_dp_maginals = torch.stack(off_dp_maginals, dim=0)
        # print(f"off_maginal: {torch.sum(off_dp_maginals)}")
        # print(f"on_maginal: {torch.sum(on_dp_maginals)}")

        cost, pi, C = self.sinkhorn(off_dps, on_dps, off_dp_maginals, on_dp_maginals, cuda=True)
        # print(f"pi: {pi.size()}")
        pi = pi.cuda()
        max_ns = pi.size(1)
        # print(f"pi: {pi[0]}")
        OT_adj = (pi[:, :, 1:])* (1.0 / torch.stack([torch.max(pi, dim=2)[0]] * max_ns, dim=-1)) # + 1 to avoid devide 0
        # print(f"OT_adj: {OT_adj[0]}")
        OT_adj[OT_adj < 1] = 0
        # print(f"OT_adj: {OT_adj[0]}")

        on_dp_masks = torch.stack([dep_paths] * max_ns, dim=2)
        off_dp_masks = torch.stack([(1- dep_paths) * masks] * max_ns, dim=2)
        # print(f"on_dp_masks: {on_dp_masks[0]}")
        # print(f"off_dp_masks: {off_dp_masks[0]}")
        OT_adj = OT_adj * on_dp_masks.transpose(1, 2) * off_dp_masks
        # print(f"OT_adj: {OT_adj[0]}")
        # undirecting
        # branchs = []
        # dep_path_list = []
        OT_adj = OT_adj + OT_adj.transpose(1, 2)
        # for i in range(OT_adj[0].size(1)):
        #     if dep_paths[0, i] == 1:
        #         dep_path_list.append(i)
        #     for j in range(OT_adj[0].size(1)):
        #         if OT_adj[0, i, j] == 1:
        #             branchs.append((i, j))
        # print(f"edges: {branchs}")
        # print(f"dep_path_list: {dep_path_list}")
        
        dep_path_adjs = adjs * on_dp_masks * on_dp_masks.transpose(1,2)
        pruned_adjs = dep_path_adjs + OT_adj
        pruned_adjs[pruned_adjs > 0] = 1

        # GCN
        gcn_outp = self.gcn(gcn_input, pruned_adjs, ls) # (bs x ns x gcn_hidden_size)
        
        # Classification
        heads = []
        tails = []
        doc_presentations = []
        for i, poss in enumerate(trigger_poss):
            head = torch.max(gcn_outp[i, poss[0][0]: poss[0][-1] + 1, :] + gcn_input[i, poss[0][0]: poss[0][-1] + 1, :], dim=0)[0] # residual connection
            tail = torch.max(gcn_outp[i, poss[1][0]: poss[1][-1] + 1, :] + gcn_input[i, poss[1][0]: poss[1][-1] + 1, :], dim=0)[0]
            doc_presentation = torch.max(gcn_outp[i, :, :] + gcn_input[i, :, :], dim=0)[0]
            heads.append(head)
            tails.append(tail)
            doc_presentations.append(doc_presentation)
        heads = torch.stack(heads, dim= 0)
        tails = torch.stack(tails, dim=0)
        doc_presentations = torch.stack(doc_presentations, dim=0)
        presentations = torch.cat([heads, tails, doc_presentations], dim=1)
        # print(f"presentations: {presentations.size()}")
        logits = self.classifier(presentations)

        # Regularizarion
        full_doc_gcn_oupt = self.gcn(gcn_input, adjs, ls)
        full_doc_presentations = []
        for i, poss in enumerate(trigger_poss):
            full_doc_presentation = torch.max(full_doc_gcn_oupt[i, :, :] + gcn_input[i, :, :], dim=0)[0]
            full_doc_presentations.append(full_doc_presentation)
        full_doc_presentations = torch.stack(full_doc_presentations, dim=0)

        # Compute loss
        loss = self.loss_pred(logits, labels) + self.regular_loss_weight * self.loss_regu(doc_presentations, full_doc_presentations)
        return logits, loss

        

