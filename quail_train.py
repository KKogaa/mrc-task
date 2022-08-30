import pytorch_lightning as pl
import torch
import numpy as np
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast as BertTokenizer
from data.QuailDataModule import QuailDataModule
from model.Test import TEST
from model.model import BERT
from model.Duma import DUMA


def train(
    config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10},
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config
    ) as run:

        # Extract the config object associated with the run
        config = run.config

        # Construct our LightningModule with the learning rate from the config object
        model = DUMA(
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            model_name=config.model_name,
            num_choices=config.num_choices,
            num_epochs=config.epochs,
            gradient_accumulation_steps=config.accumulate,
            warmup_proportion=config.warmup_proportion,
        )

        # This logger is used when we call self.log inside the LightningModule
        # name_string = f"{config.batch_size}-{config.learning_rate}"
        logger = pl.loggers.WandbLogger(experiment=run, log_model=True)

        # build data module
        data_module = QuailDataModule(
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            task_name=config.task_name,
            batch_size=config.batch_size,
            max_seq_len=config.max_token_count,
            num_workers=4,
            num_proc=8,
            version="en",
            num_choices=config.num_choices,
        )

        # Construct a Trainer object with the W&B logger we created and epoch set by the config object
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, verbose=True, monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=config.epochs,
            gpus=[0, 1],
            strategy="dp",
            logger=logger,
            accumulate_grad_batches=config.accumulate,
        )

        # Execute training
        trainer.fit(model, data_module)

        torch.cuda.empty_cache()

        best_path = trainer.checkpoint_callback.best_model_path

        data_module = QuailDataModule(
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            task_name=config.task_name,
            batch_size=config.batch_size,
            max_seq_len=config.max_token_count,
            num_workers=4,
            num_proc=8,
            version="en",
            num_choices=config.num_choices,
        )

        model = DUMA.load_from_checkpoint(
            best_path,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            model_name=config.model_name,
            num_choices=config.num_choices,
            num_epochs=config.epochs,
            gradient_accumulation_steps=config.accumulate,
            warmup_proportion=config.warmup_proportion,
        )

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=config.epochs,
            gpus=[0, 1, 2, 3],
            strategy="dp",
            logger=logger,
        )

        # load best model
        trainer.test(model=model, datamodule=data_module)

        torch.cuda.empty_cache()

def test(
    config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10},
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config
    ) as run:

        # Extract the config object associated with the run
        config = run.config
        logger = pl.loggers.WandbLogger(experiment=run, log_model=True)

        data_module = QuailDataModule(
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            task_name=config.task_name,
            batch_size=config.batch_size,
            max_seq_len=config.max_token_count,
            num_workers=4,
            num_proc=8,
            version="en",
            num_choices=config.num_choices,
        )

        model = DUMA.load_from_checkpoint(
            "/home/akenichi/mrc-task/quail_v7/6yh7rhav/checkpoints/epoch=1-step=10246.ckpt",
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            model_name=config.model_name,
            num_choices=config.num_choices,
            num_epochs=config.epochs,
            gradient_accumulation_steps=config.accumulate,
            warmup_proportion=config.warmup_proportion,
        )

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, verbose=True, monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=config.epochs,
            gpus=[0, 1],
            strategy="dp",
            logger=logger,
        )

        trainer.test(model=model, datamodule=data_module)

        torch.cuda.empty_cache()

if __name__ == "__main__":

    project = "quail_v8"
    entity = None
    name = None
    sweep_config = {
        "method": "grid",  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        "metric": {  # This is the metric we are interested in minimizing or maximizing
            "name": "test_accuracy_epoch",
            "goal": "maximize",
        },
        # Paramters and parameter values we are sweeping across
        "parameters": {
            "learning_rate": {"values": [1e-5]},
            "batch_size": {"values": [2, 4]},
            "epochs": {"values": [20]},
            "max_token_count": {"values": [512]},
            "model_name": {"values": ["bert-base-multilingual-cased"]},
            "dataset_name": {"values": ["quail"]},
            "task_name": {"values": [None]},
            "num_choices": {"values": [4]},
            "version": {"values": [None]},
            "accumulate": {"values": [2, 4]},
            "warmup_proportion": {"values": [0.1]},
        },
    }
    sweep_id = wandb.sweep(
        sweep_config,
        project=project,
        entity=entity,
    )
    wandb.agent(sweep_id, function=train, count=4)