# Retrieve the required libraries 
import logging
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

# Logging config
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#Save orders 
def log_request(filename: str, num_rows: int):
    logging.info(f"File received: {filename} | Rows: {num_rows}")
    
#Save the predictions 
def log_prediction(predictions: list):
    logging.info(f"Predictions: {predictions[:10]} ...")  # first 10 predictions

#Function to determine deviation 
def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    profile = Profile(sections=[DataDriftProfileSection()])
    profile.calculate(reference_df, current_df)
   # If the value is greater than 0.1, then there is a deviation. 
    drift_detected = any([col['drift_score'] > 0.1 for col in profile.json()['data_drift']['metrics_by_columns'].values()])
    return drift_detected

#Email sending function if detected Email
def send_email_alert(subject, body, to_email):
    sender_email = "Enter your email address "
    sender_password = "enter your password "  # Use App Password for Gmail

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = to_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    
#If there is an email address, please send it. 
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()
        logging.info(f"Drift alert email sent to {to_email}")
    except Exception as e:
        logging.error(f"Failed to send email alert: {e}")
