import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import os
from pipeline.preprocess import load_data, preprocess_data
from monitoring.staleness_monitor import check_staleness
from monitoring.email_alert import send_email_alert

# Load and preprocess data
data = load_data('data/imbi_dataset.csv')
train_data, test_data = train_test_split(data, test_size=0.2)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Preprocess data
train_encodings = preprocess_data(train_data, tokenizer)
test_encodings = preprocess_data(test_data, tokenizer)

# Create Dataset objects
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_data['label'].tolist())
test_dataset = Dataset(test_encodings, test_data['label'].tolist())

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Check model staleness
if check_staleness(test_dataset):
    send_email_alert()  # Alert if the model needs retraining
