import pytorch_lightning as pl
import torch
import numpy as np
import wandb
import pandas as pd
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast as BertTokenizer
from data.RecoresDataModule import RecoresDataModule
from model.Test import TEST


def train(
    project,
    entity,
    name,
    max_token_count,
    model_name,
    dataset_name,
    task_name,
    num_choices,
    config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10},
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config, name=name
    ) as run:

        # Extract the config object associated with the run
        config = run.config

        # Construct our LightningModule with the learning rate from the config object
        model = TEST(
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            model_name=model_name,
            num_choices=num_choices,
        )

        # This logger is used when we call self.log inside the LightningModule
        # name_string = f"{config.batch_size}-{config.learning_rate}"
        logger = pl.loggers.WandbLogger(experiment=run, log_model=True)

        # build data module
        df_train = pd.read_csv("data/train_spanish.csv", sep="\t")
        df_val = pd.read_csv("data/dev_spanish.csv", sep="\t")
        df_test = pd.read_csv("data/test_spanish.csv", sep="\t")
        data_module = RecoresDataModule(
            model_name=model_name,
            dataset_name=dataset_name,
            task_name=task_name,
            batch_size=config.batch_size,
            max_seq_len=max_token_count,
            num_workers=4,
            num_proc=8,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
        )

        # Construct a Trainer object with the W&B logger we created and epoch set by the config object
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, verbose=True, monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=config.epochs,
            gpus=[1, 2, 3],
            strategy="dp",
            logger=logger,
        )

        # Execute training
        trainer.fit(model, data_module)

        # load best model
        trainer.test(model=model, datamodule=data_module, ckpt_path="best")

        torch.cuda.empty_cache()



def test(
    project,
    entity,
    name,
    max_token_count,
    model_name,
    dataset_name,
    task_name,
    num_choices,
    config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10},
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config, name=name
    ) as run:

        # Extract the config object associated with the run
        config = run.config

        # Construct our LightningModule with the learning rate from the config object
        model = TEST.load_from_checkpoint(
            "/home/akenichi/mrc-task/recores_test/qxdilsep/checkpoints/epoch=1-step=522.ckpt",
            learning_rate=1e-5,
            num_choices=num_choices
        )

        # This logger is used when we call self.log inside the LightningModule
        # name_string = f"{config.batch_size}-{config.learning_rate}"
        logger = pl.loggers.WandbLogger(experiment=run, log_model=True)

        # build data module
        df_train = pd.read_csv("data/train_spanish.csv", sep="\t")
        df_val = pd.read_csv("data/dev_spanish.csv", sep="\t")
        df_test = pd.read_csv("data/test_spanish.csv", sep="\t")
        data_module = RecoresDataModule(
            model_name=model_name,
            dataset_name=dataset_name,
            task_name=task_name,
            batch_size=config.batch_size,
            max_seq_len=max_token_count,
            num_workers=4,
            num_proc=8,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
        )

        # Construct a Trainer object with the W&B logger we created and epoch set by the config object
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, verbose=True, monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=config.epochs,
            gpus=[1, 2, 3],
            strategy="dp",
            logger=logger,
        )

        # Execute training

        # load best model
        trainer.test(model=model, datamodule=data_module)

        torch.cuda.empty_cache()



if __name__ == "__main__":

    project = "recores_test"
    entity = None
    name = "test"
    config = {"learning_rate": 1e-5, "batch_size": 4, "epochs": 5}
    test(
        project=project,
        entity=entity,
        name=name,
        max_token_count=512,
        model_name="roberta-base",
        dataset_name=None,
        task_name=None,
        num_choices=5,
        config=config,
    )
