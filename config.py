import os
import json
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, storage

def is_replit():
    """Check if we're running on Replit"""
    return os.getenv('REPL_ID') is not None

def get_env_var(var_name: str, default: str = None) -> str:
    """Get environment variable from either Replit secrets or local .env file"""
    if is_replit():
        # In Replit, get from secrets
        return os.getenv(var_name, default)
    else:
        # Locally, try to load from .env.local
        from dotenv import load_dotenv
        env_path = Path('../URBANWORKS_V2/.env.local')
        load_dotenv(dotenv_path=env_path)
        return os.getenv(var_name, default)

def get_firebase_config():
    """Get Firebase configuration"""
    return {
        "apiKey": get_env_var("NEXT_PUBLIC_FIREBASE_API_KEY"),
        "authDomain": get_env_var("NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN"),
        "projectId": get_env_var("NEXT_PUBLIC_FIREBASE_PROJECT_ID"),
        "storageBucket": get_env_var("NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": get_env_var("NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID"),
        "appId": get_env_var("NEXT_PUBLIC_FIREBASE_APP_ID"),
        "measurementId": get_env_var("NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID")
    }

def get_api_keys():
    """Get all API keys"""
    return {
        "ANTHROPIC_API_KEY": get_env_var("ANTHROPIC_API_KEY"),
        "TAVILY_API_KEY": get_env_var("TAVILY_API_KEY"),
        "REPLICATE_API_TOKEN": get_env_var("REPLICATE_API_TOKEN"),
        "LLAMA_CLOUD_API_KEY": get_env_var("LLAMA_CLOUD_API_KEY")
    }

def get_firebase_credentials():
    """Get Firebase credentials from environment or file"""
    if is_replit():
        # In Replit, construct from environment variables
        creds_json = get_env_var("FIREBASE_CREDENTIALS")
        if creds_json:
            return json.loads(creds_json)
        else:
            raise ValueError("FIREBASE_CREDENTIALS not found in Replit secrets")
    else:
        # Locally, read from file
        current_dir = Path(__file__).parent.absolute()
        cred_path = current_dir / "firebase-credentials.json"
        if cred_path.exists():
            with open(cred_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Firebase credentials file not found at {cred_path}")

def initialize_environment():
    """Initialize all necessary environment variables"""
    if is_replit():
        print("Running on Replit - using secrets for configuration")
    else:
        print("Running locally - using .env.local for configuration")

    # Test all required environment variables
    required_vars = [
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "REPLICATE_API_TOKEN",
        "NEXT_PUBLIC_FIREBASE_API_KEY",
        "NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN",
        "NEXT_PUBLIC_FIREBASE_PROJECT_ID",
        "NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET",
        "NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID",
        "NEXT_PUBLIC_FIREBASE_APP_ID"
    ]

    missing_vars = []
    for var in required_vars:
        if not get_env_var(var):
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Test Firebase credentials
    try:
        get_firebase_credentials()
    except Exception as e:
        raise ValueError(f"Failed to load Firebase credentials: {str(e)}")

    print("Environment initialized successfully")

def initialize_firebase():
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(get_firebase_credentials())
            firebase_config = get_firebase_config()
            bucket_name = firebase_config.get("storageBucket")
            
            if not bucket_name:
                raise ValueError("Storage bucket name not found in configuration")
                
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
            
            # Verify bucket exists
            bucket = storage.bucket()
            if not bucket.exists():
                print(f"Warning: Bucket '{bucket_name}' does not exist")
                print("Please create the bucket in the Firebase Console")
                
        except Exception as e:
            print(f"Firebase initialization error: {str(e)}")
            raise

if __name__ == "__main__":
    initialize_environment() 