import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModelForTokenClassification, AutoTokenizer


class IntentSlotModel(pl.LightningModule):
    def __init__(self, base_model, num_labels, learning_rate=2e-5):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.base_model, num_labels=self.num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self.forward(input_ids, attention_mask)
        loss_fn = nn.CrossEntropyLoss()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
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
        logits = self.forward(input_ids, attention_mask)
        probabilities = nn.functional.softmax(logits, dim=2)
        predicted_labels = torch.argmax(probabilities, dim=2)
        return predicted_labels
