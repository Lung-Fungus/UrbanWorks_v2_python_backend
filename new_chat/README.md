# Chat Backend with Pydantic AI

This package provides a modern chat backend implementation using [Pydantic AI](https://ai.pydantic.dev/) and Claude 3.7 Sonnet.

## Features

- Conversational AI powered by Claude 3.7 Sonnet
- Web search capability via Tavily API
- URL content extraction
- Conversation history management
- Firebase integration for data persistence

## Architecture

The package is organized as follows:

- `__init__.py` - Package initialization and exports
- `models.py` - Pydantic models for request/response handling
- `agent.py` - AI agent definition and tools
- `app.py` - FastAPI application and endpoints

## Usage

The chat backend is mounted as a sub-application in the main FastAPI app. It provides the following endpoints:

- `POST /chat` - Main chat endpoint for processing user messages
- `GET /health` - Health check endpoint
- `GET /graph` - Tool information endpoint

## Tools

The chat backend provides the following tools:

1. `web_search` - Search the web for current information using Tavily
2. `extract_url` - Extract and analyze content from a specific URL
3. `get_conversation_history` - Retrieve conversation history from the database

## Response Format

The AI agent provides structured responses with two components:
- `analysis` - Private analysis for internal use
- `content` - Response content to be shown to the user 