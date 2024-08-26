from transformers import AutoTokenizer
from data.main.data import GPTDATA
from gpt.model.gpt import GPT
import torch.nn as nn
import torch
import os


class GEN:

    @staticmethod
    def pnucleus(tensor: torch.tensor, p: float = 0.95):
        values, idx = tensor.sort()
        psample = idx[values.cumsum(0) > p]
        return psample[torch.randint(len(psample), (1,))]

    @staticmethod
    def generate(prompt: str,
                 length: int,
                 seq_len: int,
                 p: float,
                 gptmodel: GPT,
                 gptdata: GPTDATA,
                 tokenizer: AutoTokenizer):

        tokenized = tokenizer(prompt)['input_ids']

        tokenized.pop(0)
        tokenized.pop(-1)

        for _ in range(length):
            excerpt = tokenized[-seq_len:]
            excerpt_pad, attn = gptdata.padding_and_attn(excerpt, seq_len)
            excerpt_pad = torch.tensor([excerpt_pad])
            attn = torch.tensor(attn)

            output = gptmodel(excerpt_pad).squeeze()
            sampled = GEN.pnucleus(nn.Softmax(dim=0)(output[attn == 1][-1]), p=p)
            tokenized.append(sampled.item())

        return tokenized, tokenizer.decode(tokenized)