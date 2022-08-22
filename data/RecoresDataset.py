import torch
from torch.utils.data import Dataset, DataLoader, random_split

class RecoresDataset(Dataset):
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
        context = data_row.context
        #create 5 contexts
        context = [[] * 5]
        #create 5 question option pairs
        question_option = [
            f"{question} {option}" for option in options
        ]

        encoding = self.tokenizer(
            context,
            question_option,
            add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # labels used for the text
        label = data_row.correct
        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            label=label,
        )


    @staticmethod
    def preprocess(tokenizer, max_seq_len, examples):
        # choices_features = []
        # label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        context = [[article] * 4 for article in examples["context"]]
        question_option = [
            [f"{question} {option}" for option in options]
            for options, question in zip(examples["answers"], examples["question"])
        ]

        context = sum(context, [])
        question_option = sum(question_option, [])

        encoding = tokenizer(
            context,
            question_option,
            add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # shape before (4000, 512)
        encoding["input_ids"] = encoding["input_ids"].view(-1, 4, max_seq_len)
        encoding["attention_mask"] = encoding["attention_mask"].view(-1, 4, max_seq_len)
        # shape after (1000, 4, 512)

        encoding["input_ids"] = encoding["input_ids"].view(
            encoding["input_ids"].shape[0], -1
        )
        encoding["attention_mask"] = encoding["attention_mask"].view(
            encoding["attention_mask"].shape[0], -1
        )
        # shape after (1000, 2048)

        # labels = [label_map.get(answer, -1) for answer in examples["answer"]]
        labels = [answer for answer in examples["correct_answer_id"]]

        return {
            "input_ids": encoding["input_ids"].tolist(),
            "attention_mask": encoding["attention_mask"].tolist(),
            "label": labels,
        }