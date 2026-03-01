# Retrieve the required libraries 
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import joblib
from io import StringIO

# Load pre-trained model and transformers
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
cat_encoder = joblib.load("category_feature_encoder.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI(title="DataFlow AI")

# Utility function for preprocessing
def preprocess(df: pd.DataFrame):
    # Data Cleaning
    df.fillna(method='ffill', inplace=True)
    
    # Feature Engineering
    if 'date' in df.columns:
        df['day'] = pd.to_datetime(df['date']).dt.day
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['year'] = pd.to_datetime(df['date']).dt.year

    # Scaling
    num_features = ['numeric_feature1', 'numeric_feature2', 'day', 'month']
    df[num_features] = scaler.transform(df[num_features])

    # Encoding
    cat_features = ['category_feature']
    for col in cat_features:
        df[col] = cat_encoder.transform(df[col])

    # Text features
    text_features = ['text_feature']
    for col in text_features:
        X_text = tfidf_vectorizer.transform(df[col]).toarray()
        tfidf_df = pd.DataFrame(X_text, columns=[f"{col}_tfidf_{i}" for i in range(X_text.shape[1])])
        df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)

    return df

# API endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read() #Reading the file 
    df = pd.read_csv(StringIO(content.decode('utf-8')))
    
    processed_df = preprocess(df)
    predictions = model.predict(processed_df)
    
    return {"predictions": predictions.tolist()}
