"""
FastAPI application for the chat backend.

This module defines the FastAPI application and endpoints for the chat backend.
"""

import logging
from datetime import datetime
from typing import Dict

import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

from utils.auth_middleware import firebase_auth
from utils.config import get_firebase_credentials

from new_chat.models import ChatRequest, ChatResponse
from new_chat.agent import (
    clarke_agent, 
    validate_and_parse_result, 
    ClarkeDependencies,
    central_tz
)

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = None

def initialize_app():
    """Initialize the application dependencies."""
    global db
    
    try:
        # Since main.py should have already initialized Firebase, we just need to get the client
        db = firestore.client()
        print("Successfully initialized Firestore client in new_chat app")
    except Exception as e:
        # If an error occurs, try initializing Firebase ourselves
        if not firebase_admin._apps:
            try:
                print("Firebase not initialized, initializing now in new_chat app...")
                cred = credentials.Certificate(get_firebase_credentials())
                # Initialize Firebase with storageBucket set to None to prevent automatic bucket creation
                firebase_admin.initialize_app(cred, {
                    'storageBucket': None  # Disable automatic Storage bucket initialization
                })
                db = firestore.client()
                print("Successfully initialized Firebase in new_chat app")
            except Exception as e:
                print(f"Error initializing Firebase in new_chat app: {e}")
                # Continue without Firestore - this should be handled in the routes
                db = None
        else:
            # Firebase is initialized but Firestore client failed
            print(f"Error getting Firestore client: {e}")
            db = None
    
    return app

# ========== API Endpoints ==========
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/graph")
async def graph_visualization():
    """
    Returns a JSON description of the agent's tools instead of a graph visualization
    since we're not using LangGraph anymore.
    """
    try:
        tools = [
            {"name": "web_search", 
             "description": "Search the web for current information"},
            {"name": "extract_url", 
             "description": "Extract and analyze content from a specific URL"},
            {"name": "get_conversation_history", 
             "description": "Retrieves conversation history"}
        ]
        return {"agent": "Clarke", "tools": tools}
    except Exception as e:
        logger.error("Error generating tools description", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating tools description")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, user_data: Dict = Depends(firebase_auth)):
    """Main chat endpoint that processes user messages and returns AI responses."""
    try:
        # Log incoming request
        logger.info("=== NEW CHAT REQUEST ===")
        #logger.info(f"Message: {request.message}")
        #logger.info(f"Conversation ID: {request.conversation_id}")
        #logger.info(f"User Display Name: {request.user_display_name}")
        #logger.info(f"Files attached: {len(request.files) if request.files else 0}")
        #logger.info(f"Authenticated User ID: {user_data.get('uid')}")

        # Get conversation and verify ownership
        conversation_ref = db.collection('conversations').document(request.conversation_id)
        conversation_doc = conversation_ref.get()

        if not conversation_doc.exists:
            logger.error(f"Conversation not found: {request.conversation_id}")
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation_data = conversation_doc.to_dict()
        logger.info(f"Conversation exists, owner: {conversation_data.get('userId') or conversation_data.get('userid')}")

        # Verify user owns this conversation
        conversation_owner = conversation_data.get('userId') or conversation_data.get('userid')
        if conversation_owner != user_data['uid']:
            logger.error(f"Authorization error: User {user_data['uid']} attempted to access conversation owned by {conversation_owner}")
            raise HTTPException(status_code=403, detail="Not authorized to access this conversation")

        # Initialize dependencies
        deps = ClarkeDependencies(request.conversation_id, db)
        deps.user_display_name = request.user_display_name
        deps.files = request.files
        
        # Fetch available collections from Firestore
        try:
            # Get collections specifically from the 'collections' collection
            # These are the ones that appear in DatabaseDisplay component
            collections_ref = db.collection('collections')
            collections_docs = collections_ref.stream()
            available_collections = []
            
            for doc in collections_docs:
                collection_data = doc.to_dict()
                if 'name' in collection_data:
                    available_collections.append(collection_data['name'])
            
            logger.info(f"Available database collections: {available_collections}")
            
            # Store collections in dependencies for agent context
            deps.available_collections = available_collections
        except Exception as collections_error:
            logger.error(f"Error fetching database collections: {str(collections_error)}")
            # Continue without collections data

        # Log before running the agent
        logger.info("Initializing AI agent with dependencies")
        
        try:
            # Run the AI agent
            result = await clarke_agent.run(
                request.message,
                deps=deps
            )
            logger.info("AI agent run completed successfully")
        except Exception as agent_error:
            logger.error(f"Error running AI agent: {str(agent_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"AI processing error: {str(agent_error)}")

        try:
            # Validate and parse the result
            logger.info("Validating AI agent result")
            validated_result = await validate_and_parse_result(result.data)
            logger.info("Result validation successful")
        except Exception as validation_error:
            logger.error(f"Result validation error: {str(validation_error)}", exc_info=True)
            if hasattr(result, 'data'):
                logger.error(f"Raw result data: {result.data}")
            raise HTTPException(status_code=422, detail=f"Invalid result format: {str(validation_error)}")

        # Save to Firestore
        messages_ref = conversation_ref.collection('messages')
        
        # DON'T save messages here - the frontend handles this with proper formatting
        # We only want to update the conversation metadata
        
        # messages_ref.add({
        #     "role": "user",
        #     "content": request.message,
        #     "timestamp": datetime.now(central_tz)
        # })
        
        # messages_ref.add({
        #     "role": "assistant",
        #     "content": validated_result.content,
        #     "analysis": validated_result.analysis,
        #     "file_content": validated_result.file_content,
        #     "timestamp": datetime.now(central_tz)
        # })

        # Update conversation metadata
        conversation_ref.update({
            "lastMessageTimestamp": datetime.now(central_tz),
            "messageCount": firestore.Increment(2)
        })

        logger.info("=== CHAT REQUEST COMPLETED ===")
        logger.info(f"Final response content:\n{validated_result.content}")
        logger.info(f"Final analysis content:\n{validated_result.analysis}")
        if validated_result.file_content:
            logger.info(f"Final file content:\n{validated_result.file_content}")

        return {
            "message": {
                "role": "assistant",
                "content": validated_result.content
            },
            "analysis": validated_result.analysis,
            "file_content": validated_result.file_content
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint that doesn't require authentication."""
    return {
        "status": "ok",
        "message": "Chat API is running properly",
        "time": datetime.now(central_tz).isoformat(),
        "db_connection": db is not None
    }

@app.post("/test")
async def test_post_endpoint(request: ChatRequest):
    """Test endpoint for validating request format without authentication."""
    return {
        "status": "ok",
        "received": {
            "message": request.message,
            "conversation_id": request.conversation_id,
            "user_display_name": request.user_display_name,
            "files_count": len(request.files) if request.files else 0
        }
    } 