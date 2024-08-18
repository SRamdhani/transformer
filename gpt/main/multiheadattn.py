from .selfattn import SingleHeadAttention
from torchtyping import TensorType
import torch.nn as nn
import torch

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.single_heads = nn.ModuleList()

        for i in range(num_heads):
            self.single_heads.append(SingleHeadAttention(embedding_dim, int(attention_dim / num_heads)))

    def forward(self, embedded: TensorType[float]) -> torch.Tensor:
        attn_res = []

        for s in self.singleheads:
            attn_res.append(s(embedded))

        attn_res = torch.cat(attn_res, dim=2)
        return attn_res

