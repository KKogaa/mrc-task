
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    BertTokenizerFast as BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel,
)


class TEST(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # self.bert = BertModel.from_pretrained(kwargs["model_name"], return_dict=True)
        self.bert = AutoModel.from_pretrained(kwargs["model_name"], return_dict=True)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1)

        self.learning_rate = kwargs["learning_rate"]
        self.batch_size = kwargs["batch_size"]
        self.num_choices = kwargs["num_choices"]
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

        # self.n_training_steps = kwargs['n_training_steps']
        # self.n_warmup_steps = kwargs['n_warmup_steps']

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        x = bert_output.pooler_output
        logits = self.fc1(x)
        reshaped_logits = logits.view(-1, self.num_choices) 
        loss = 0
        if labels is not None:
            loss = self.criterion(reshaped_logits, labels)
        return loss, reshaped_logits

    def training_step(self, batch, batch_idx):
        #unflatten inputids and attention mask (batch_size, 2048) -> (batch_size * 4, 512)
        input_ids = batch["input_ids"].view(batch["input_ids"].shape[0] * self.num_choices, -1)
        attention_mask = batch["attention_mask"].view(batch["attention_mask"].shape[0] * self.num_choices, -1)
        labels = batch["label"]

        loss, logits = self(input_ids, attention_mask, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )

        probs = F.softmax(logits, dim = 1)
        labels_hat = torch.argmax(probs, dim=1)
        correct_count = torch.sum(labels == labels_hat)        
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(batch["input_ids"].shape[0] * self.num_choices, -1)
        attention_mask = batch["attention_mask"].view(batch["attention_mask"].shape[0] * self.num_choices, -1)
        labels = batch["label"]

        loss, logits = self(input_ids, attention_mask, labels)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )

        probs = F.softmax(logits, dim = 1)
        labels_hat = torch.argmax(probs, dim=1)
        correct_count = torch.sum(labels == labels_hat)        
        
        return {
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch["label"])
        }


    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(batch["input_ids"].shape[0] * self.num_choices, -1)
        attention_mask = batch["attention_mask"].view(batch["attention_mask"].shape[0] * self.num_choices, -1)
        labels = batch["label"]

        loss, logits = self(input_ids, attention_mask, labels)

        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )

        probs = F.softmax(logits, dim = 1)
        labels_hat = torch.argmax(probs, dim=1)
        correct_count = torch.sum(labels == labels_hat)        
        
        return {
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch["label"])
        }

    def validation_epoch_end(self, outputs):
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs) 
        self.log(f"val_accuracy_epoch", val_acc)

    def test_epoch_end(self, outputs):
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs) 
        self.log(f"test_accuracy_epoch", test_acc)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #   optimizer,
        #   num_warmup_steps=self.n_warmup_steps,
        #   num_training_steps=self.n_training_steps
        # )
        # return dict(
        #   optimizer=optimizer,
        #   lr_scheduler=dict(
        #     scheduler=scheduler,
        #     interval='step'
        #   )
        # )
        return optimizer
