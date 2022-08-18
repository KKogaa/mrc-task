
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import (
    BertTokenizerFast as BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)


class TEST(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bert = BertModel.from_pretrained(kwargs["model_name"], return_dict=True)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1)

        self.learning_rate = kwargs["learning_rate"]
        self.batch_size = kwargs["batch_size"]
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

        # self.n_training_steps = kwargs['n_training_steps']
        # self.n_warmup_steps = kwargs['n_warmup_steps']

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        x = bert_output.pooler_output
        logits = self.fc1(x)
        reshaped_logits = logits.view(-1, 4) 
        loss = 0
        if labels is not None:
            loss = self.criterion(reshaped_logits, labels)
        return loss, reshaped_logits

    def training_step(self, batch, batch_idx):
        #unflatten inputids and attention mask (batch_size, 2048) -> (batch_size * 4, 512)
        input_ids = batch["input_ids"].view(batch["input_ids"].shape[0] * 4, -1)
        attention_mask = batch["attention_mask"].view(batch["attention_mask"].shape[0] * 4, -1)
        labels = batch["label"]

        loss, outputs = self(input_ids, attention_mask, labels)

        return loss 
        # #change loss calculation to crossentropy
        # loss, outputs = self(input_ids, attention_mask, labels)
        # self.log(
        #     "train_loss",
        #     loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     batch_size=self.batch_size,
        # )

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(batch["input_ids"].shape[0] * 4, -1)
        attention_mask = batch["attention_mask"].view(batch["attention_mask"].shape[0] * 4, -1)
        labels = batch["label"]

        loss, outputs = self(input_ids, attention_mask, labels)

        return loss 
        # input_ids = batch["input_ids"]
        # attention_mask = batch["attention_mask"]
        # labels = batch["labels"]
        # loss, outputs = self(input_ids, attention_mask, labels)
        # self.log(
        #     "val_loss",
        #     loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     batch_size=self.batch_size,
        # )
        # return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=5,
        )

        prob = torch.sigmoid(outputs)
        output = {"loss": loss, "prob": prob.flatten(), "target": labels.int()}

        return output

    def test_step_end(self, outputs):
        predictions = outputs["prob"].detach().cpu()
        targets = outputs["target"].detach().cpu()

        max_prediction = torch.argmax(predictions.flatten())
        max_label = torch.argmax(targets)

        if torch.equal(max_prediction, max_label):
            self.positives = self.positives + 1
        else:
            self.negatives = self.negatives + 1

    def test_epoch_end(self, outputs):
        accuracy = self.positives / (self.positives + self.negatives)
        self.log(f"test_accuracy_epoch", accuracy)

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