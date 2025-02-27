"""
Pydantic models for the chat application.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    conversation_id: str
    user_display_name: str
    files: Optional[List[Dict[str, str]]] = None  # List of parsed file contents

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    message: Dict[str, str]
    analysis: str

class ChatAnalysisResponse(BaseModel):
    """Structured response from Clarke AI agent."""
    analysis: str = Field(..., description="Private analysis of the user's message and context")
    content: str = Field(..., description="Response to be shown to the user") 