import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chat_backend import app as chat_app
from proposal_generator import app as proposal_app
from social_media_backend import app as social_app
from image_generation import app as image_app
from firestore_upload import app as upload_app

# Create main FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all the sub-applications
app.mount("/chat", chat_app)
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