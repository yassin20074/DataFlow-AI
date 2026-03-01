# Retrieve the required libraries 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
#a preprocessing function 
def preprocess(df, scaler, cat_encoder, tfidf_vectorizer):
    # Data Cleaning
    df.fillna(method='ffill', inplace=True)
    
    # Feature Engineering
    if 'date' in df.columns:
        df['day'] = pd.to_datetime(df['date']).dt.day
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['year'] = pd.to_datetime(df['date']).dt.year

    # Scaling
    num_features = ['numeric_feature1', 'numeric_feature2', 'day', 'month'] #numerical data
    df[num_features] = scaler.transform(df[num_features])

    # Categorical Encoding
    cat_features = ['category_feature']
    for col in cat_features:
        df[col] = cat_encoder.transform(df[col])

    # Text Features
    text_features = ['text_feature']
    for col in text_features:
        X_text = tfidf_vectorizer.transform(df[col]).toarray()
        tfidf_df = pd.DataFrame(X_text, columns=[f"{col}_tfidf_{i}" for i in range(X_text.shape[1])])
        df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)

    return df #return data
