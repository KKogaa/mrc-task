import pytorch_lightning as pl
import wandb
import torch
import numpy as np


def train(
    project, entity, config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10}
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config
    ) as run:

        # Extract the config object associated with the run
        config = run.config

        # Construct our LightningModule with the learning rate from the config object
        # model = BERT(learning_rate=config.learning_rate)

        # This logger is used when we call self.log inside the LightningModule
        # name_string = f'{config.batch_size}-{config.learning_rate}'
        # logger = pl.loggers.WandbLogger(experiment=run, log_model=True)

        # build data module
        # data_module = BERTDataModule(
        #   df_train,
        #   df_val,
        #   df_test,
        #   tokenizer,
        #   batch_size=config.batch_size,
        #   max_token_len=MAX_TOKEN_COUNT
        # )

        # Construct a Trainer object with the W&B logger we created and epoch set by the config object
        # checkpoint_callback = ModelCheckpoint(
        #   save_top_k=1,
        #   verbose=True,
        #   monitor="val_loss",
        #   mode="min"
        # )
        # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

        # trainer = pl.Trainer(
        #   callbacks=[early_stopping_callback, checkpoint_callback],
        #   max_epochs=config.epochs,
        #   gpus=[2,3],
        #   strategy='dp',
        #   logger=logger
        # )

        # Execute training
        # trainer.fit(model, data_module)

        # load best model
        # trainer.test(model=model, datamodule=data_module, ckpt_path="best")

        # '/drive/MyDrive/qa_datasets_spanish/spanish_mcqa_v1/4ford7jt/checkpoints/epoch=7-step=2615.ckpt'
        # model = BERT.load_from_checkpoint(
        #   '/drive/MyDrive/qa_datasets_spanish/spanish_mcqa_v1/4ford7jt/checkpoints/epoch=7-step=2615.ckpt',
        #   learning_rate = 1e-5,
        # )

        # torch.cuda.empty_cache()


if __name__ == "__main__":
    print(np.__version__)

    # get data
    project = "mrc_test"
    entity = None
    config = {"learning_rate": 1e-5, "batch_size": 16, "epochs": 10}
    train(project=project, entity=entity, config=config)

    # define model

    # train model
