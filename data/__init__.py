from datasets import load_dataset, concatenate_datasets

DATA = load_dataset("SetFit/bbc-news")
DATASET = concatenate_datasets([DATA['train'], DATA['test']])