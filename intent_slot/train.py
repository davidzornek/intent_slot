import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from configuration import ClassifierConfig, load_yaml_config
from data import SequenceClassifierDataProcessor, SequenceClassifierDataModule
from model import SequenceClassifierModel


def train(config: ClassifierConfig):
    data_proc = SequenceClassifierDataProcessor(csv_path=config.data, base_model=config.base_model, label_columns = ["label"], text_column="text", max_len=config.max_len)
    data_proc.process_data()

    data_module = SequenceClassifierDataModule(data_proc, batch_size=config.batch_size)
    data_module.setup()

    model = SequenceClassifierModel(
        config.base_model,
        data_proc.label_list,
        config.learning_rate,
    )

    wandb.login()
    run = wandb.init(
        project="test",
        config={
            #### update with our stuff
            "learning_rate": config.learning_rate,
            "epochs": config.max_epochs,
        }
    )
    logger = WandbLogger(log_model="all")


    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        monitor="val_loss",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    wandb.finish()
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    config = ClassifierConfig(**load_yaml_config(args.config_path))
    model = train(config)
