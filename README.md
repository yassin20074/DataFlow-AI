# 🚀 DataFlow AI

**DataFlow AI** is a Production-Ready Machine Learning Pipeline that handles:

- Data Preprocessing  
- Feature Engineering  
- Scaling & Encoding  
- Model Training & Evaluation  
- Real-Time Prediction via API  
- Logging, Monitoring & Drift Detection with Email Alerts  

It supports numeric, categorical, and textual data, and is ready for production deployment.

---

# 🌟 Key Features

- **End-to-End Pipeline:** Data Cleaning → Feature Engineering → Scaling → Encoding → Model Training → Prediction  
- **Supports Multiple Data Types:** Numeric, Categorical, Text  
- **Production-Ready API:** FastAPI endpoint for frontend integration  
- **Scalable:** Can handle large datasets and model upgrades  
- **Evaluation Metrics:** MSE, R² Score, etc.  
- **Logging & Monitoring:** Logs requests, predictions, and errors  
- **Drift Detection:** Detects data drift using Evidently AI  
- **Email Alerts:** Sends notifications automatically if drift is detected  

# 🧠 Pipeline Overview
Data Cleaning → handle missing values & outliers
Feature Engineering → create new useful columns
Scaling → normalize numeric features
Encoding → categorical → LabelEncoder, text → TF-IDF / embeddings
Train/Test Split → prepare data for model
Model Training → RandomForest / other ML models
Prediction → new data passes through same pipeline
Evaluation → MSE, R², etc.
Logging & Monitoring → logs all requests, predictions, errors
Drift Detection → detects data drift & triggers email alerts automatically

# 🔧 Notes for Production
Save scalers, encoders, and model using joblib for consistent preprocessing
Use Evidently AI dashboard to monitor drift visually
Canary deploy new models to a small % of users before full rollout
Keep training pipeline separate from online inference for stability
Log all API requests, responses, and errors for debugging & auditing

# made by: yassin sanad 
