import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, List


class GCN(nn.Module):
    def __init__(self, 
                input_dim: int, 
                output_dim: int, 
                num_layer: int, 
                rnn_hidden_dim: int=0, 
                num_rnn_layer: int=1, 
                dropout: int=0.5) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

        self.use_rnn = False
        if rnn_hidden_dim > 0:
            self.use_rnn = True
            self.rnn_hidden_dim = rnn_hidden_dim
            self.num_rnn_layer = num_rnn_layer
            input_size = self.input_dim
            self.rnn = nn.LSTM(input_size, self.rnn_hidden_dim, num_layers=self.num_rnn_layer,
                                batch_first=True, dropout=0.5, bidirectional=True)
            self.input_dim = self.rnn_hidden_dim * 2
        
        self.num_layer = num_layer
        self.W = nn.ModuleList()
        for i in range(self.num_layer):
            input_dim = self.input_dim if i == 0 else self.output_dim
            self.W.append(nn.Linear(input_dim, self.output_dim))
    
    def encode_with_rnn(self, inp: torch.Tensor(), ls: List[int]) -> torch.Tensor(): # batch_size x max_seq_len x hidden_dim*2
        packed = pack_padded_sequence(inp, ls, batch_first=True, enforce_sorted=False)
        rnn_encode, _ = self.rnn(packed)
        outp, _ = pad_packed_sequence(rnn_encode, batch_first=True)
        return outp

    def forward(self, input, adj, ls) -> torch.Tensor():
        """
        input: (bs, max_seq_len, input_size)
        adj: (bs, max_seq_len, max_seq_len)
        ls: List[int] - list of actual seq len in batch
        """
        denom = adj.sum(2).unsqueeze(2) + 1
        if self.use_rnn:
            gcn_input = self.encode_with_rnn(input, ls)
        else:
            gcn_input = input
        
        for l in range(self.num_layer):
            Ax = adj.bmm(gcn_input)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_input)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_input = self.dropout(gAxW) if l < self.num_layer - 1 else gAxW
        
        return gcn_input  # (batch_size, max_seq_len, gcn_out_dim)