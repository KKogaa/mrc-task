import torch
from torch.utils.data import Dataset, DataLoader, random_split


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        # text to get encoded
        text = data_row.text

        # labels used for the text
        labels = data_row.correct

        # encoder
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=labels,
        )
