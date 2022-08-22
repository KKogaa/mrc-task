import torch
import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast
from functools import partial
from transformers import AutoTokenizer
from .RecoresDataset import RecoresDataset


class RecoresDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        dataset_name,
        task_name,
        batch_size,
        max_seq_len,
        num_workers,
        num_proc,
        df_train,
        df_val,
        df_test,
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
        self.dataset = {
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_val),
            "test": Dataset.from_pandas(df_test),
        }

    @staticmethod
    def preprocess(tokenizer, max_seq_len, examples):

        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        context = [[article] * 5 for article in examples["text"]]
        question_option = []
        for opta, optb, optc, optd, opte, question in zip(
            examples["A"],
            examples["B"],
            examples["C"],
            examples["D"],
            examples["E"],
            examples["question"],
        ):
            question_option.append(f"{question} {opta}")
            question_option.append(f"{question} {optb}")
            question_option.append(f"{question} {optc}")
            question_option.append(f"{question} {optd}")
            question_option.append(f"{question} {opte}")

        context = sum(context, [])
        # question_option = sum(question_option, [])

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
        encoding["input_ids"] = encoding["input_ids"].view(-1, 5, max_seq_len)
        encoding["attention_mask"] = encoding["attention_mask"].view(-1, 5, max_seq_len)
        # shape after (1000, 4, 512)

        encoding["input_ids"] = encoding["input_ids"].view(
            encoding["input_ids"].shape[0], -1
        )
        encoding["attention_mask"] = encoding["attention_mask"].view(
            encoding["attention_mask"].shape[0], -1
        )
        # shape after (1000, 2048)

        labels = [label_map.get(answer, -1) for answer in examples["answer"]]
        # labels = [answer for answer in examples["correct_answer_id"]]

        return {
            "input_ids": encoding["input_ids"].tolist(),
            "attention_mask": encoding["attention_mask"].tolist(),
            "label": labels,
        }

    def setup(self, stage=None):

        # preprocess
        preprocessor = partial(self.preprocess, self.tokenizer, self.max_seq_len)

        for split in ["train", "validation", "test"]:
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
            self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )
