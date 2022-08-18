import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import BertTokenizerFast as BertTokenizer

from data.RaceDataModule import RaceDataModule
from model.Test import TEST

if __name__ == "__main__":

    model_name = "bert-base-uncased"
    dataset_name = "race"
    task_name = "all"

    data_module = RaceDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        task_name=task_name,
        batch_size=16,
        max_seq_len=512,
        num_workers=4,
        num_proc=8,
    )

    # data_module.setup()
    # test = dataset.get_dataset()
    # dataset.print_dataset_sanity("train")

    model = TEST(
        learning_rate=5e-5,
        batch_size=16,
        model_name=model_name,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)

    trainer = pl.Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=3,
        gpus=[2, 3],
        strategy="dp",
    )

    trainer.fit(model, data_module)
    # print(test)

    # train_loader = dataset.train_dataloader()

    # print(train_loader[0])

    # for batch in train_loader:
    #     # print(batch)
    #     print("HERE")
    #     print("HERE")
    #     # print(len(batch['input_ids']))
    #     # print(batch)
    #     print(len(batch["input_ids"]))
    #     for x in batch["input_ids"]:
    #         print(x.shape)
    #     print(len(batch["attention_mask"]))
    #     print("HERE")
    #     print("HERE")
    #     break
    # print(len(batch['input_ids']))

    # for x in batch['input_ids']:
    #     print(len(x))
    # print(batch['label'].shape)

    # break
