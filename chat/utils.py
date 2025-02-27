"""
Utility functions for the chat backend.
"""

import logging
import pytz
from datetime import datetime
import firebase_admin
from firebase_admin import firestore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Add console handler
    ]
)
logger = logging.getLogger(__name__)

# Add a startup log message to verify logging is working
logger.info("\n=== CHAT BACKEND STARTED ===")
logger.info("Logging configured with console output")

# Define Central Time Zone
central_tz = pytz.timezone('US/Central')

# Initialize Firebase
def get_db():
    """Get Firestore database client."""
    if not firebase_admin._apps:
        from config import initialize_firebase
        initialize_firebase()
    
    return firestore.client()

def get_current_time():
    """Get current date and time in Central Time formatted as a string."""
    return datetime.now(central_tz).strftime("%B %d, %Y %I:%M %p")

def get_current_datetime():
    """Get current datetime object in Central Time."""
    return datetime.now(central_tz) 