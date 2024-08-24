from .multiheadattn import MultiHeadedSelfAttention
from .vanilla import VanillaNeuralNetwork
from torchtyping import TensorType
import torch.nn as nn
import torch


class TransformerBlock(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.multi_headed = MultiHeadedSelfAttention(embedding_dim=self.model_dim,
                                                     attention_dim=self.model_dim,
                                                     num_heads=self.num_heads)

        self.first_layerNorm = nn.LayerNorm(model_dim)
        self.second_layerNorm = nn.LayerNorm(model_dim)
        self.vanilla = VanillaNeuralNetwork(model_dim=model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        multi_head_norm = self.first_layerNorm(embedded)
        multi_head_output = self.multi_headed(multi_head_norm) + embedded
        ffw_norm = self.second_layerNorm(multi_head_output)
        ffw_output = self.vanilla(ffw_norm) + multi_head_output
        return ffw_output
