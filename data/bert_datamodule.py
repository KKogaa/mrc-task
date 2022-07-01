import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from .bert_dataset import BERTDataset


class BERTDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer, batch_size, max_token_len):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = BERTDataset(
            self.train_df, self.tokenizer, self.max_token_len
        )

        self.val_dataset = BERTDataset(self.test_df, self.tokenizer, self.max_token_len)

        self.test_dataset = BERTDataset(
            self.test_df, self.tokenizer, self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=5, num_workers=4, drop_last=True
        )
