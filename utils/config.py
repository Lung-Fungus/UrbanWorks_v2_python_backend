import os
import json
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, storage, auth
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

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
        
        # Try these paths in order
        paths_to_try = [
            Path('./.env.local'),                # In current directory (backend root)
            Path('../.env.local'),               # In parent directory
            Path('./URBANWORKS_V2/.env.local'),  # From project root to URBANWORKS_V2
            Path('../URBANWORKS_V2/.env.local'), # From backend dir to URBANWORKS_V2
            # Absolute path as fallback
            Path('C:/Users/danie/OneDrive/Desktop/UrbanWorks_Frontend/URBANWORKS_V2/.env.local')
        ]
        
        # Try each path
        for path in paths_to_try:
            if path.exists():
                load_dotenv(dotenv_path=path)
                break
        
        # Return the value
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
    # First try to get from environment variable
    creds_json = get_env_var("FIREBASE_CREDENTIALS")
    if creds_json:
        try:
            return json.loads(creds_json)
        except json.JSONDecodeError:
            # If it's not valid JSON, assume it's a filepath
            if os.path.exists(creds_json):
                with open(creds_json, 'r') as f:
                    return json.load(f)
            # Otherwise, fall through to other methods
    
    if is_replit():
        # In Replit, we should already have returned above, but just in case
        raise ValueError("FIREBASE_CREDENTIALS not found in Replit secrets")
    else:
        # Try to use environment variables to construct credentials
        # This requires having the individual Firebase credential fields in .env.local
        firebase_config = {}
        try:
            # Check for essential Firebase credential fields
            project_id = get_env_var("FIREBASE_PROJECT_ID") or get_env_var("NEXT_PUBLIC_FIREBASE_PROJECT_ID")
            private_key = get_env_var("FIREBASE_PRIVATE_KEY")
            client_email = get_env_var("FIREBASE_CLIENT_EMAIL")
            
            if project_id and private_key and client_email:
                # Build credentials object from environment variables
                # Replace escaped newlines with actual newlines if present
                if "\\n" in private_key:
                    private_key = private_key.replace("\\n", "\n")
                
                firebase_config = {
                    "type": "service_account",
                    "project_id": project_id,
                    "private_key": private_key,
                    "client_email": client_email,
                    "client_id": get_env_var("FIREBASE_CLIENT_ID", ""),
                    "auth_uri": get_env_var("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                    "token_uri": get_env_var("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
                    "auth_provider_x509_cert_url": get_env_var("FIREBASE_AUTH_PROVIDER_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
                    "client_x509_cert_url": get_env_var("FIREBASE_CLIENT_CERT_URL", "")
                }
                return firebase_config
        except Exception as e:
            logger.warning(f"Failed to construct Firebase credentials from environment variables: {str(e)}")
        
        # Finally, try to read from file
        current_dir = Path(__file__).parent.absolute()
        # Look in the parent directory (backend) instead of utils
        backend_dir = current_dir.parent
        cred_path = backend_dir / "firebase-credentials.json"
        if cred_path.exists():
            with open(cred_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(
                "Firebase credentials not found. Please either:\n"
                "1. Add a firebase-credentials.json file to the backend directory, or\n"
                "2. Add FIREBASE_CREDENTIALS as JSON in your .env.local file, or\n"
                "3. Add individual Firebase credential environment variables: FIREBASE_PROJECT_ID, FIREBASE_PRIVATE_KEY, FIREBASE_CLIENT_EMAIL"
            )

def initialize_environment():
    """Initialize all necessary environment variables"""
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

    logger.debug("Environment initialized successfully")

def initialize_firebase():
    """Initialize Firebase if it hasn't been already."""
    print("DEBUG: initialize_firebase() function called from utils/config.py")
    if not firebase_admin._apps:
        try:
            # Get the Firebase project ID from environment
            project_id = get_env_var('NEXT_PUBLIC_FIREBASE_PROJECT_ID')
            
            # Use the correct storage bucket path
            storage_bucket = "urbanworks-v2.firebasestorage.app"
            print(f"DEBUG: About to initialize Firebase with bucket: {storage_bucket}")
            
            # Initialize with the storage bucket and service account credentials
            cred = credentials.Certificate(get_firebase_credentials())
            firebase_admin.initialize_app(cred, {
                'projectId': project_id,
                'storageBucket': storage_bucket
            })
            print(f"DEBUG: Firebase initialized with bucket: {storage_bucket}")
            print("DEBUG: Bucket creation code has been removed, this should not attempt to create a bucket")
            
            # Removed automatic bucket creation to prevent errors
            # If you need to create a bucket, please do so manually in the Firebase Console
            
        except Exception as e:
            print(f"DEBUG: Error in initialize_firebase: {e}")
            logger.error(f"Error initializing Firebase: {e}")
            # Continue without Firebase to allow local development
            pass

def verify_firebase_token(token: str) -> dict:
    """
    Verify Firebase ID token and return user info
    
    Args:
        token (str): Firebase ID token
        
    Returns:
        dict: User info from decoded token
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication credentials: {str(e)}"
        )

if __name__ == "__main__":
    initialize_environment() 