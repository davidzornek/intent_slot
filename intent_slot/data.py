from datasets import Dataset
from datasets.dataset_dict import DatasetDict

from torch.utils.data import DataLoader
from transformers.data.processors.utils import DataProcessor
from transformers import AutoTokenizer
import pytorch_lightning as pl


class IntentSlotDataProcessor(DataProcessor):
    def __init__(self, data: DatasetDict, base_model: str, max_len: int = 512):
        self.data = data
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.max_len = max_len
        try:
            self.train_data = self.data["train"]
        except KeyError:
            self.train_data = None

        try:
            self.val_data = self.data["validation"]
        except KeyError:
            self.val_data = None

        try:
            self.test_data = self.data["test"]
        except KeyError:
            self.test_data = None

    def process_data(self):
        """Performas any preprocessing and tokenization necessary"""
        if self.train_data:
            self.processed_train_data = self._tokenize_dataset(self.train_data)
        if self.val_data:
            self.processed_val_data = self._tokenize_dataset(self.val_data)
        if self.test_data:
            self.processed_test_data = self._tokenize_dataset(self.test_data)

    def _tokenize_dataset(self, dataset):
        """tokenizes data and aligns labels to account for word splitting and
        special tokens"""
        tokenized_inputs = self.tokenizer(dataset["tokens"], padding="max_length", is_split_into_words=True)

        labels = []
        for i, label in enumerate(dataset[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:  # Set the special tokens to -100.
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        return Dataset.from_dict(tokenized_inputs)


class IntentSlotDataModule(pl.LightningDataModule):
    def __init__(self, data_processor: IntentSlotDataProcessor, batch_size: int = 32):
        super().__init__()
        self.data_processor = data_processor
        self.batch_size = batch_size

    def setup(self):
        self.train_data = self.data_processor.processed_train_data
        self.val_data = self.data_processor.processed_val_data
        self.test_data = self.data_processor.processed_test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
