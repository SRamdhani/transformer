from .block import TransformerBlock
from torchtyping import TensorType
import torch.nn as nn
import torch


class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.rng = torch.arange(context_length)
        self.con_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(context_length, model_dim)
        self.transformer = nn.Sequential()

        for i in range(num_blocks):
            self.transformer.append(TransformerBlock(model_dim, num_heads))

        self.layer_norm = nn.LayerNorm(model_dim)
        self.final_layer = nn.Linear(model_dim, vocab_size)

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        torch.manual_seed(0)

        # Positional Embedding
        pos_emb = self.pos_emb(self.rng)

        # Context Embedding
        con_emb = self.con_emb(context)

        # Pos + Context
        pos_context = pos_emb + con_emb

        # Transformer
        transformer = self.transformer(pos_context)

        # Final Layer Norm
        transformer_norm = self.layer_norm(transformer)

        # Final Linear Softmax
        # final = nn.Softmax(dim=2)(self.final_layer(transformer_norm))

        final = self.final_layer(transformer_norm)
        return final
