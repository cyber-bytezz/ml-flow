import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data, tokenizer):
    return tokenizer(data['text'].tolist(), padding=True, truncation=True, return_tensors="pt")
