"""
FastAPI routes for the chat backend.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from firebase_admin import firestore
from langchain_core.messages import HumanMessage, AIMessage
from datetime import timedelta
from .models import ChatRequest
from .agent import create_chat_graph
from .utils import logger, get_db, get_current_datetime
from auth_middleware import firebase_auth

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@router.get("/graph")
async def graph_visualization():
    """
    Returns a PNG image visualization of the current chat graph.
    """
    try:
        # Compile the current chat graph
        chat_graph = create_chat_graph()
        # Generate PNG bytes from the graph visualization
        img_bytes = chat_graph.get_graph().draw_mermaid_png()
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        logger.error("Error generating graph visualization", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating graph visualization")

@router.post("/chat")
async def chat(request: ChatRequest, user_data: dict = Depends(firebase_auth)):
    """
    Endpoint to handle chat requests.
    """
    try:
        # Log incoming request
        logger.info("=== NEW CHAT REQUEST ===")
        logger.info(f"Message: {request.message}")

        # Get firestore database client
        db = get_db()
        
        # Get conversation and verify ownership
        conversation_ref = db.collection('conversations').document(request.conversation_id)
        conversation_doc = conversation_ref.get()

        if not conversation_doc.exists:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation_data = conversation_doc.to_dict()

        # Verify user owns this conversation
        conversation_owner = conversation_data.get('userId') or conversation_data.get('userid')
        if conversation_owner != user_data['uid']:
            raise HTTPException(status_code=403, detail="Not authorized to access this conversation")

        # Get conversation history from Firestore
        messages_ref = conversation_ref.collection('messages')
        messages_query = messages_ref.order_by('timestamp').stream()

        # Convert Firestore messages to LangChain messages
        messages = []
        for msg in messages_query:
            msg_data = msg.to_dict()
            if msg_data['role'] == 'user':
                messages.append(HumanMessage(content=msg_data['content']))
            else:
                messages.append(AIMessage(content=msg_data['content']))

        # Add the new user message
        messages.append(HumanMessage(content=request.message))

        # Initialize state
        current_datetime = get_current_datetime()
        initial_state = {
            "messages": messages,
            "current_date": current_datetime.strftime("%B %d, %Y %I:%M %p"),
            "user_display_name": request.user_display_name,
            "files": request.files,
            "response": None,
            "tool_response": None,
            "tool_input": None,
            "tool_name": None
        }

        # Create and run the graph
        graph = create_chat_graph()
        try:
            final_state = graph.invoke(initial_state)
        except Exception as e:
            if str(e) == "'__end__'":
                # This is expected when the graph ends normally
                final_state = initial_state
            else:
                raise

        # Get the response
        response = final_state.get("response")
        if not response:
            raise ValueError("No response generated")

        # Save to Firestore
        messages_ref.add({
            "role": "user",
            "content": request.message,
            "timestamp": current_datetime,
            **({"displayContent": request.display_content} if request.display_content else {}),
            **({"parsedFileContent": request.parsed_file_content} if request.parsed_file_content else {})
        })

        # Add 1 second to assistant timestamp to ensure proper ordering
        # Use timedelta to preserve timezone
        assistant_timestamp = current_datetime + timedelta(seconds=1)
        messages_ref.add({
            "role": "assistant",
            "content": response["content"],
            "analysis": response["analysis"],
            "timestamp": assistant_timestamp
        })

        conversation_ref.update({
            "lastMessageTimestamp": assistant_timestamp,
            "messageCount": firestore.Increment(2),
            "lastMessage": request.display_content or request.message
        })

        logger.info("=== CHAT REQUEST COMPLETED ===")
        logger.info(f"Final response content:\n{response['content']}")
        logger.info(f"Final analysis content:\n{response['analysis']}")

        return {
            "message": {
                "role": "assistant",
                "content": response["content"]
            },
            "analysis": response["analysis"]
        }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 