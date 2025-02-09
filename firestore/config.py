import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env.local
env_path = Path('../URBANWORKS_V2/.env.local')
load_dotenv(dotenv_path=env_path)

def get_firebase_config():
    """Get Firebase configuration from environment variables."""
    return {
        "apiKey": os.getenv("NEXT_PUBLIC_FIREBASE_API_KEY"),
        "authDomain": os.getenv("NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN"),
        "projectId": os.getenv("NEXT_PUBLIC_FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("NEXT_PUBLIC_FIREBASE_APP_ID"),
        "measurementId": os.getenv("NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID")
    }

def create_firebase_credentials():
    """Create Firebase credentials file from environment variables."""
    config = get_firebase_config()
    
    # You'll need to get these from Firebase Console > Project Settings > Service Accounts > Generate New Private Key
    creds = {
        "type": "service_account",
        "project_id": config["projectId"],
        "private_key_id": "",  # You'll need to add this manually
        "private_key": "",     # You'll need to add this manually
        "client_email": f"firebase-adminsdk@{config['projectId']}.iam.gserviceaccount.com",
        "client_id": "",       # You'll need to add this manually
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk@{config['projectId']}.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }
    
    # Write credentials to file
    with open('firebase-credentials.json', 'w') as f:
        json.dump(creds, f, indent=2)

if __name__ == "__main__":
    create_firebase_credentials() 