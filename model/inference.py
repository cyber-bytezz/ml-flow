import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model():
    model = BertForSequenceClassification.from_pretrained('results/checkpoint-<latest_check_point>')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def predict(text):
    model, tokenizer = load_model()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax(-1).item()
    return predicted_class
