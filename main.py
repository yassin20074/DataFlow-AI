#Retrieve the required libraries 
from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import StringIO
import joblib
from preprocessing import preprocess
from monitoring import log_request, log_prediction, detect_drift, send_email_alert

# Load model and transformers
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
cat_encoder = joblib.load("models/category_feature_encoder.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Reference data for drift detection
reference_data = pd.read_csv("data/sample_data.csv")

app = FastAPI(title="DataFlow AI")

#construction endpoint 
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()# Read the file 
    df = pd.read_csv(StringIO(content.decode('utf-8')))

    # Log request
    log_request(file.filename, len(df))

    # Preprocess
    processed_df = preprocess(df, scaler, cat_encoder, tfidf_vectorizer)

    # Predict
    predictions = model.predict(processed_df)
    log_prediction(predictions.tolist())

    # Drift detection
    drift_detected = detect_drift(reference_data, df)
    if drift_detected:
        logging.warning("Data drift detected!")
        # Send email alert
        send_email_alert(
            subject=" Data Drift Detected in DataFlow AI",
            body=f"Drift detected in uploaded file: {file.filename}",
            to_email="Enter your email address here"
        )

    return {
        "predictions": predictions.tolist(),
        "drift_detected": drift_detected
    }
