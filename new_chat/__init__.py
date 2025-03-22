"""
Chat application using Pydantic AI.

This package provides a chat backend service powered by Claude 3.7 Sonnet
with tools for web search and URL content extraction.
"""

from new_chat.app import app, initialize_app

__all__ = ['app', 'initialize_app'] 