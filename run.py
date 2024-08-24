from torch.utils.data.dataloader import DataLoader
from data.main.data import GPTDATA
from gpt.model.gpt import GPT
import torch

gptdata = GPTDATA()
gptdata.curate(max_length = 512)

VOCAB_SIZE = gptdata.vocab_size
SEQ_LEN = gptdata.seq_len
TOKENIZER = gptdata.tokenizer_gpt

gptmodel = GPT(vocab_size=VOCAB_SIZE,
               context_length=SEQ_LEN,
               model_dim=128,
               num_blocks=4,
               num_heads=4)

data = gptdata.hf_dataset

dataloader = DataLoader(
    data,
    batch_size=30
)

ds = next(iter(dataloader))
input_ids = torch.stack(ds['input_ids']).T
label = torch.stack(ds['label']).T

test = gptmodel(input_ids)
print(test.shape)
print(label.shape)
