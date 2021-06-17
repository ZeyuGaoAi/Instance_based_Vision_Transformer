import torch
import torch.nn as nn
import numpy as np

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]
        position_embeddings = self.pe(position_ids)
        return x + position_embeddings


        
class LearnedPositionalEncoding2(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding2, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )
    def forward(self, x, position_ids):
        # if position_ids is None:
        #     position_ids = types

        position_ids = position_ids.to(torch.int64).cuda()

        a = nn.Parameter(torch.zeros(position_ids.shape[0], 1)).cuda()

        position_ids = torch.cat((a, position_ids), dim=1).to(torch.int64)

        position_embeddings = self.pe(position_ids)
        
        return x + position_embeddings
    
class LearnedPositionalEncoding3(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding3, self).__init__()
        self.pex = nn.Embedding(max_position_embeddings, int(embedding_dim/2))
        self.pey = nn.Embedding(max_position_embeddings, int(embedding_dim/2))
        self.seq_length = seq_length
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )
    def forward(self, x, px,py):
        # if position_ids is None:
        #     position_ids = types
        
        
        #px = np.ceil(torch.true_divide(px, 10).cpu())
        px = torch.true_divide(px, 20).ceil()
        px = px.to(torch.int64).cuda()
        py = torch.true_divide(py, 20).ceil()
        #py = np.ceil(torch.true_divide(py, 10).cpu())
        py = px.to(torch.int64).cuda()
        a = nn.Parameter(torch.zeros(px.shape[0], 1)).cuda()
        px = torch.cat((a, px), dim=1).to(torch.int64)
        b = nn.Parameter(torch.zeros(py.shape[0], 1)).cuda()
        py = torch.cat((a, py), dim=1).to(torch.int64)
        px_embeddings = self.pex(px)
        py_embeddings = self.pey(py)
        position_embeddings = torch.cat((px_embeddings, py_embeddings), dim=2)
        return x + position_embeddings