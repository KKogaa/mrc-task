import torch
import datasets
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast
from functools import partial
import datasets
from transformers import AutoTokenizer
    

class QuailDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        dataset_name,
        task_name,
        batch_size,
        max_seq_len,
        num_workers,
        num_proc,
    ):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.num_proc = num_proc
        # self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.dataset = None

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

    def get_dataset(self):
        return self.dataset

    def print_dataset_sanity(self, split):
        input_ids = self.dataset[split][0]["input_ids"]
        print(input_ids.shape)

        return None

    def setup(self, stage=None):
        self.dataset = datasets.load_dataset(self.dataset_name, self.task_name)

        # preprocess
        preprocessor = partial(self.preprocess, self.tokenizer, self.max_seq_len)

        for split in ["train", "validation", "challenge"]:
            self.dataset[split] = self.dataset[split].map(
                preprocessor,
                num_proc=self.num_proc,
                batched=True,
            )

            self.dataset[split].set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["challenge"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )
