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
        num_choices,
        version,
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
        self.num_choices = num_choices
        self.version = version

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
            question_option,
            context, 
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

    @staticmethod
    def preprocess_binary(tokenizer, max_seq_len, examples):

        context = [article for article in examples["text"]]
        question_option = [
            f"{question} {option}"
            for option, question in zip(examples["answer"], examples["question"])
        ]

        encoding = tokenizer(
            question_option,
            context,
            add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = [answer for answer in examples["correct"]]

        return {
            "input_ids": encoding["input_ids"].tolist(),
            "attention_mask": encoding["attention_mask"].tolist(),
            "label": labels,
        }

    @staticmethod
    def flatten(examples):
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        contexts = [[article] * 5 for article in examples["text"]]

        options = []
        for opta, optb, optc, optd, opte in zip(
            examples["A"],
            examples["B"],
            examples["C"],
            examples["D"],
            examples["E"],
        ):
            options.append([opta])
            options.append([optb])
            options.append([optc])
            options.append([optd])
            options.append([opte])

        questions = [[question] * 5 for question in examples["question"]]

        corrects = []
        for answer in examples["answer"]:
            ans_idx = label_map.get(answer, -1)

            for idx in range(5):
                if ans_idx == idx:
                    corrects.append([1])
                else:
                    corrects.append([0])

        contexts = sum(contexts, [])
        options = sum(options, [])
        questions = sum(questions, [])
        corrects = sum(corrects, [])

        return {
            "text": contexts,
            "question": questions,
            "answers": options,
            "correct": corrects,
        }

    def setup(self, stage=None):

        if self.version == "flat":
            # flatten_preprocessor = partial(self.flatten)
            binary_preprocessor = partial(
                self.preprocess_binary, self.tokenizer, self.max_seq_len
            )

            for split in ["train", "validation", "test"]:

                # self.dataset[split] = self.dataset[split].map(
                #     flatten_preprocessor,
                #     num_proc=self.num_proc,
                #     remove_columns=["A", "B", "C", "D", "E", "reason", "answer"],
                #     batched=True,
                # )

                self.dataset[split] = self.dataset[split].map(
                    binary_preprocessor,
                    num_proc=self.num_proc,
                    batched=True,
                )

                self.dataset[split].set_format(
                    type="torch", columns=["input_ids", "attention_mask", "label"]
                )

        else:
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
