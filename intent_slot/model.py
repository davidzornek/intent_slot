import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SequenceClassifierModel(pl.LightningModule):
    def __init__(self, base_model, num_labels, learning_rate=2e-5):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.label_list = [f'label_{x}' for x in range(num_labels)]
        self.label2idx = {label: i for i, label in enumerate(self.label_list)}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=self.num_labels
        ) # .embeddings

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = torch.tensor([batch["input_ids"]])
        attention_mask = torch.tensor([batch["attention_mask"]])
        labels = torch.tensor([batch["labels"]])
        logits = self.forward(input_ids, attention_mask)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = self.forward(input_ids, attention_mask)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def tokenize_inputs(self, text_list):
        return self.tokenizer.batch_encode_plus(
            text_list, padding="max_length", truncation=True, return_tensors="pt"
        )

    def predict(self, text_list):
        inputs = self.tokenize_inputs(text_list)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        logits = self.transformer.forward(input_ids, attention_mask).logits
        logit_list = logits.tolist()

        probabilities = nn.functional.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        return {
            "logits": logit_list,
            "prediction": [self.idx2label[x] for x in predicted_labels.tolist()],
            "scores": [
                {self.idx2label[i]: p for i, p in enumerate(item_probs.tolist())}
                for item_probs in probabilities
            ]
        }
