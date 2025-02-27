import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import our config for Firebase initialization
from config import initialize_firebase

# First, initialize Firebase properly without Storage
initialize_firebase()

# Now import all the modules that might need Firebase
from new_chat import app as chat_app, initialize_app
from proposal.proposal_generator import app as proposal_app
from social_media.social_media_backend import app as social_app
from image.image_generation import app as image_app
from firestore_upload import app as upload_app
from auth_middleware import firebase_auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize chat app
initialize_app()

# Create main FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add auth dependency to all routes except health checks
@app.middleware("http")
async def add_auth_dependency(request, call_next):
    path = request.url.path
    if not path.endswith("/health") and path != "/":
        # Add auth dependency to request
        request.state.dependencies = [Depends(firebase_auth)]
    return await call_next(request)

# Mount the chat app
# This means chat_app routes are accessible at:
# - /chat/ for the POST endpoint
# - /chat/test for the test endpoints
# - /chat/health for the health check
# - /chat/graph for the graph visualization
app.mount("/chat", chat_app)

# Mount all the other sub-applications
app.mount("/proposal", proposal_app)
app.mount("/social", social_app)
app.mount("/image", image_app)
app.mount("/upload", upload_app)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "UrbanWorks Backend API is running",
        "endpoints": {
            "chat": "/chat",
            "proposal": "/proposal",
            "social": "/social",
            "image": "/image",
            "upload": "/upload"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 