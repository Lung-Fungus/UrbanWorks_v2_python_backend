import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
# Import the router from the new chat package instead of the whole app
from chat import router as chat_router
from proposal_generator import app as proposal_app
from social_media_backend import app as social_app
from image_generation import app as image_app
from firestore_upload import app as upload_app
from auth_middleware import firebase_auth

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

# Include the router for chat instead of mounting the app
app.include_router(chat_router, prefix="/chat")

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