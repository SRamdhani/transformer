from torch.utils.data.dataloader import DataLoader
from gpt.train.training import TRAIN
from data.main.data import GPTDATA
from gpt.model.gpt import GPT
import torch

gptdata = GPTDATA()
gptdata.curate(max_length=512)

VOCAB_SIZE = gptdata.vocab_size
SEQ_LEN = gptdata.seq_len
TOKENIZER = gptdata.tokenizer_gpt
BATCH_SIZE = 30

gptmodel = GPT(vocab_size=VOCAB_SIZE,
               context_length=SEQ_LEN,
               model_dim=128,
               num_blocks=4,
               num_heads=4)

dataloader = DataLoader(
    gptdata.hf_dataset,
    batch_size=BATCH_SIZE
)

TRAIN.run(gptmodel=gptmodel,
          dataloader=dataloader,
          batch_size=BATCH_SIZE,
          epochs=200)