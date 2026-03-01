# Retrieve the required libraries 
import joblib
from sklearn.ensemble import RandomForestRegressor

#Training function 
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/random_forest_model.pkl")# Save the model 
    return model
