import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.k = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dk = attention_dim ** 0.5

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        value = self.v(embedded)
        query = self.q(embedded)
        key = self.k(embedded)
        query_key = torch.matmul(query, torch.transpose(key, 1, 2))
        query_key = torch.tril(query_key) # lower triangle
        query_key = query_key.masked_fill(query_key==0 , float("-inf"))
        query_key = self.softmax(query_key/self.dk)
        value_qk = torch.matmul(query_key, value)
        return value_qk
