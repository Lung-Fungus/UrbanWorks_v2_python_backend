"""
Data models for the chat functionality.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, TypedDict, Annotated
from langchain_core.messages import BaseMessage

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    conversation_id: str
    user_display_name: str
    files: Optional[List[Dict[str, str]]] = None  # List of parsed file contents
    display_content: Optional[str] = None  # Text shown in UI, may differ from full message content
    parsed_file_content: Optional[str] = None  # Content extracted from files, stored separately

class State(TypedDict):
    """State for the chat agent."""
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    current_date: Annotated[str, "The current date and time"]
    user_display_name: Annotated[str, "The user's display name"]
    files: Annotated[List[Dict[str, str]], "List of parsed file contents"]
    response: Annotated[Optional[Dict[str, str]], "The structured response with analysis and content"]
    tool_response: Annotated[Optional[str], "The response from any tool calls"]
    tool_input: Annotated[Optional[str], "The input to any tool calls"]
    tool_name: Annotated[Optional[str], "The name of the tool being called"] 