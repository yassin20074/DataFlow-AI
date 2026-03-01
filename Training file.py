#Retrieve the required libraries 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Data 
df = pd.read_csv("data/sample_data.csv")  # 
print("Original Data:\n", df.head())

#  Data Cleaning 
# Filling in the missing values 
df.fillna(method='ffill', inplace=True)

# Feature Engineering 
if 'date' in df.columns:
    df['day'] = pd.to_datetime(df['date']).dt.day
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['year'] = pd.to_datetime(df['date']).dt.year
  
# Scaling 
num_features = ['numeric_feature1', 'numeric_feature2', 'day', 'month']  # numerical Data 
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

#Encoding 
# Categorical Data 
cat_features = ['category_feature']
for col in cat_features:
    le = LabelEncoder() 
    df[col] = le.fit_transform(df[col])

# Text data 
text_features = ['text_feature']
tfidf = TfidfVectorizer(max_features=50)
for col in text_features:
    X_text = tfidf.fit_transform(df[col]).toarray()
    tfidf_df = pd.DataFrame(X_text, columns=[f"{col}_tfidf_{i}" for i in range(X_text.shape[1])])
    df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)

#  Train Test Split 
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training 
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#  Prediction
y_pred = model.predict(X_test)

#  Evaluation 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

#  Save Scalers & Encoders 
import joblib
joblib.dump(scaler, "scaler.pkl")
for col in cat_features:
    joblib.dump(le, f"{col}_encoder.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Pipeline Complete. Ready for Production!")
