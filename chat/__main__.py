"""
Entry point for standalone execution of the chat backend.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

# Create FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(router)

if __name__ == "__main__":
    print("Starting chat backend in standalone mode...")
    uvicorn.run(app, host="0.0.0.0", port=8081) 