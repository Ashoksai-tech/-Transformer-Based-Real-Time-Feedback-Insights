from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import mlflow
import uvicorn
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.getenv("MODEL_PATH", default="C:/models/transformer_model/")
tokenizer = None
model = None
model_name = "distilbert-base-uncased-finetuned-sentiment"

logging.basicConfig(level=logging.INFO)

def load_model():
    global tokenizer, model
    try:
        logging.info(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Directory does not exist: {model_path}")
        
        files = os.listdir(model_path)
        required_files = ['config.json', 'pytorch_model.bin']
        for file in required_files:
            if file not in files:
                raise FileNotFoundError(f"Required file {file} not found in {model_path}")
        
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        logging.info("Model and tokenizer loaded successfully")
        
        try:
            latest_model = mlflow.pytorch.load_model(f"models:/{model_name}/latest")
            model = latest_model
            logging.info(f"Successfully loaded the latest {model_name} from MLflow")
        except Exception as mlflow_error:
            logging.error(f"Error loading {model_name} from MLflow: {str(mlflow_error)}")
            logging.info("Using the locally loaded model")
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.info("Falling back to pre-trained model")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

def predict(text):
    if tokenizer is None or model is None:
        load_model()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

def batch_predict(texts):
    return [predict(text) for text in texts]

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API"}

@app.post("/predict")
async def predict_sentiment(input_data: TextInput):
    prediction = predict(input_data.text)
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"text": input_data.text, "prediction": sentiment}

@app.post("/batch_predict")
async def batch_sentiment(input_data: BatchInput):
    predictions = batch_predict(input_data.texts)
    return {"texts": input_data.texts, "predictions": predictions}

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
