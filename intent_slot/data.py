from datasets.dataset_dict import DatasetDict

from transformers.data.processors.utils import DataProcessor
from transformers import AutoTokenizer


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
        tokenized_inputs = self.tokenizer(dataset["tokens"], is_split_into_words=True)

        labels = []
        for i, label in enumerate(dataset[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None: # Set the special tokens to -100.
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        return tokenized_inputs
