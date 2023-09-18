from collections import Counter
from datasets.dataset_dict import DatasetDict
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.data.processors.utils import DataProcessor
from transformers import AutoTokenizer
import pytorch_lightning as pl


class SequenceClassifierDataset(torch.utils.data.Dataset):
    """Stores train/val/test data in a format easily consumable by
    a Trainer"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class SequenceClassifierDataProcessor(DataProcessor):
    """Preprocesses data for training."""

    def __init__(
        self,
        csv_path: Optional[str] = None,
        split_data: Optional[DatasetDict] = None,
        base_model: str = None,
        max_len: int = 512,
        seed: int = 1337,
        label_columns: List[str] = ["area", "sub area"],
        text_column: str = "description",
        label_level: int = 1,
    ):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.max_len = max_len
        self.seed = seed
        self.label_columns = label_columns
        self.text_column = text_column
        self.label_level = label_level
        self.label_list = None
        self.label2idx = None

        if csv_path and split_data:
            raise ValueError(
                "only supply csv data or a DataDict of train/test/val splits, not both."
            )
        elif csv_path:
            self.data = import_label_csv(
                csv_path=csv_path, label_columns=self.label_columns
            )
            required_columns = self.label_columns + [self.text_column]
            required_columns = [x.lower().replace(" ", "_") for x in required_columns]
            if not all([x in self.data.columns for x in required_columns]):
                raise ValueError(
                    "Either a label column or the text column is not present in the csv."
                )

            self.data["y"] = self.data.label.apply(
                lambda x: ".".join(
                    [x for i, x in enumerate(x.split(".")) if i <= self.label_level]
                )
            )

            self.label_list = list(self.data.y.unique())
            self.label2idx = {label: i for i, label in enumerate(self.label_list)}

            train, test, val = np.split(
                self.data.sample(frac=1, random_state=self.seed),
                [int(0.6 * len(self.data)), int(0.8 * len(self.data))],
            )
            self.train_data = train
            self.test_data = test
            self.val_data = val
        elif split_data:
            self.data = split_data
            try:
                self.train_data = self.data["train"]
            except KeyError:
                self.train_data = None

            try:
                self.val_data = self.data["val"]
            except KeyError:
                self.val_data = None

            try:
                self.test_data = self.data["test"]
            except KeyError:
                self.test_data = None

    def process_data(self):
        """Performas any preprocessing and tokenization necessary"""
        if self.train_data is not None:
            self.processed_train_data = self._create_encodings(self.train_data)
        if self.val_data is not None:
            self.processed_val_data = self._create_encodings(self.val_data)
        if self.test_data is not None:
            self.processed_test_data = self._create_encodings(self.test_data)

    def _create_encodings(self, data):
        encodings = self.tokenizer(
            data[self.text_column].tolist(), padding="max_length", max_length=self.max_len, truncation=True,
        )
        labels = [self.label2idx[x] for x in data.y.tolist()]
        return SequenceClassifierDataset(encodings, labels)

    def summarize_labels(self, label_level=0, percentages=False):
        temp = self.data.label.apply(
            lambda x: ".".join(
                [x for i, x in enumerate(x.split(".")) if i <= label_level]
            )
        ).tolist()

        label_dist = dict(Counter(temp).most_common())
        if percentages:
            return {k: np.round(v / len(temp), 3) for k, v in label_dist.items()}
        else:
            return label_dist


def import_label_csv(
    csv_path: str,
    label_columns: List[str] = ["area", "sub area"],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="|", encoding="ISO-8859-1", lineterminator="\n")
    df.columns = [x.lower().replace(" ", "_") for x in df.columns]
    df.columns = [x if x != "name" else "new_name" for x in df.columns]  # temporary
    label_columns = [x.lower().replace(" ", "_") for x in label_columns]

    for col in label_columns:
        df[f"formatted_{col}"] = df[col].apply(lambda x: x.lower().replace(" ", "_"))
    df["label"] = df.apply(
        lambda x: ".".join([x[col].lower().replace(" ", "_") for col in label_columns]),
        axis=1,
    )

    if save_path is not None:
        df.to_csv(save_path)

    return df


class SequenceClassifierDataModule(pl.LightningDataModule):
    """Loads data batches to Trainer"""

    def __init__(
        self, data_processor: SequenceClassifierDataProcessor, batch_size: int
    ):
        super().__init__()
        self.data_processor = data_processor
        self.batch_size = batch_size

    def setup(self):
        self.train_data = self.data_processor.processed_train_data
        self.val_data = self.data_processor.processed_val_data
        self.test_data = self.data_processor.processed_test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=8)
