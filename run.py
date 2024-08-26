from torch.utils.data.dataloader import DataLoader
from gpt.train.training import TRAIN
from data.main.data import GPTDATA
from gpt.gen.generate import GEN
from gpt.model.gpt import GPT
from gpt import MODEL_DIR
import pprint
import torch
import os

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

# TRAIN.run(gptmodel=gptmodel,
#           dataloader=dataloader,
#           batch_size=BATCH_SIZE,
#           epochs=200)

model_path = os.path.join(MODEL_DIR, 'model_weights.pth')
if os.path.exists(model_path):
    print('loading pre-existing weights...')
    gptmodel.load_state_dict(torch.load(model_path))

prompt = """wales want rugby league training wales could follow england s lead by training with a rugby league club.
england have already had a three-day session with leeds rhinos  and wales are thought to be interested in a
similar clinic with rivals st helens. saints coach ian millward has given his approval  but if it does happen it is
unlikely to be this season. saints have a week s training in portugal next week  while wales will play england in the
opening six nations match on 5 february. we have had an approach"""

_, generated = GEN.generate(prompt,
                            length=200,
                            seq_len=SEQ_LEN,
                            p=0.92,
                            gptmodel=gptmodel,
                            gptdata=gptdata,
                            tokenizer=TOKENIZER)

pprint.pp(prompt)
print()
pprint.pp(generated)