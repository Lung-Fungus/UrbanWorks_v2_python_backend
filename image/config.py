"""
Configuration settings for the image module.
"""
import os
import pytz
import logging
import replicate
import anthropic
from firebase_admin import storage
from utils.config import get_api_keys, get_firebase_config

# API base URL
API_BASE_URL = os.getenv('API_BASE_URL', 'https://urbanworks-v2-pythonbackend.replit.app')

# Define Central Time Zone
CENTRAL_TZ = pytz.timezone('US/Central')

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Get API keys
api_keys = get_api_keys()

# Initialize clients
replicate_client = replicate.Client(api_token=api_keys["REPLICATE_API_TOKEN"])
claude_client = anthropic.Anthropic(api_key=api_keys["ANTHROPIC_API_KEY"])

# Use the correct storage bucket path
storage_bucket = "urbanworks-v2.firebasestorage.app"
bucket = storage.bucket(storage_bucket)
# Only log at debug level to avoid duplicate messages
logger.debug(f"Using Firebase Storage bucket: {storage_bucket}")

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.gif', '.webp')

# Default folder
DEFAULT_FOLDER = "Uncategorized" 