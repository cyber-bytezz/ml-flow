import smtplib
from email.mime.text import MIMEText

def send_email_alert():
    sender_email = "your_email@example.com"
    receiver_email = "recipient_email@example.com"
    subject = "Model Staleness Alert"
    body = "The model has been detected as stale and requires retraining."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, 'your_password') 
        server.send_message(msg)
