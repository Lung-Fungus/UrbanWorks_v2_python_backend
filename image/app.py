"""
Main FastAPI application for the image module.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://urbanworks-v2.web.app",
        "https://urbanworks-v2.firebaseapp.com",
        "https://urbanworks-v2-pythonbackend.replit.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Import routes
from .routes.generation import router as generation_router
from .routes.management import router as management_router

# Include routers
app.include_router(generation_router)
app.include_router(management_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "image"} 