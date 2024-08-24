from transformers import AutoTokenizer
from datasets import Dataset
from .. import DATASET


class GPTDATA:

    def __init__(self, tokenizer_name: str = "distilbert-base-uncased",
                 seq_len: int = 512) -> None:

        self.tokenizer_gpt = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer_gpt.vocab_size
        self.seq_len = seq_len
        self.hf_dataset = None

    def sliding_window_inputs(self, text: str, max_length: int = None) -> list:
        if not max_length:
            max_len = self.seq_len
        else:
            max_len = max_length

        tokenized = self.tokenizer_gpt(text)
        input_ids = tokenized['input_ids']
        # attn_mask = tokenized['attention_mask']
        input_ids.pop(0)   # 101
        input_ids.pop(-1)  # 102

        if len(input_ids) <= max_len:
            return [ [input_ids[:-1], input_ids[1:]] ]
        else:
            data = []

            for i in range(len(input_ids) - max_len):

                data.append(
                    [input_ids[i:(i+max_len)], input_ids[(i+1):(i+1+max_len)]]
                )

            return data

    @staticmethod
    def padding_and_attn(arr_tup: list, max_length: int):
        arr = arr_tup

        ones = len(arr)
        zeros = max_length - len(arr)

        if ones < max_length:
            attn = [1 for _ in range(ones)] + [0 for _ in range(zeros)]
            arr_pad = arr + [0 for _ in range(zeros)]
            return arr_pad, attn
        else:
            return arr, [1 for _ in range(max_length)]


    def curate(self, max_length: int = None) -> None:

        if not max_length:
            max_len = self.seq_len
        else:
            max_len = max_length

        gpt_template = {
            'label': [],
            'input_ids': [],
            'attention_mask': []
        }

        for i in range(len(DATASET)):
            temp = self.sliding_window_inputs(DATASET[i]['text'])

            for t in temp:
                arr1, attn = self.padding_and_attn(t[0], max_len)
                arr2, attn = self.padding_and_attn(t[1], max_len)
                gpt_template['label'].append(arr2)
                gpt_template['attention_mask'].append(attn)
                gpt_template['input_ids'].append(arr1)

        self.hf_dataset = Dataset.from_dict(gpt_template)