import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)

def load_model():
    model = BertForSequenceClassification.from_pretrained('results/checkpoint-latest')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def predict(text):
    model, tokenizer = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax(-1).item()
    logging.info(f"Text: {text}, Predicted class: {predicted_class}")
    return predicted_class
