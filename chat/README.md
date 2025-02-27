# Chat Backend Module

This module contains the refactored chat functionality for the UrbanWorks application.

## Structure

The module is organized as follows:

- `__init__.py` - Package initialization, exports the router
- `__main__.py` - Entry point for standalone execution
- `agent.py` - LangGraph agent and graph implementation
- `llm.py` - Custom LLM implementation (CustomChatAnthropic)
- `models.py` - Pydantic models and data structures
- `routes.py` - FastAPI routes and endpoint handlers
- `tools.py` - Tool implementations (web search, URL extraction)
- `utils.py` - Utility functions and logging setup

## Usage

### As Part of the Main Application

The chat module is automatically included in the main application through the `main.py` file:

```python
from chat import router as chat_router
app.include_router(chat_router, prefix="/chat")
```

### Standalone Mode

The chat module can also be run in standalone mode for testing:

```bash
python -m backend.chat
```

This will start the chat backend on port 8081.

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /graph` - Graph visualization endpoint (returns PNG)
- `POST /chat` - Main chat endpoint

## Dependencies

- FastAPI - Web framework
- LangChain - LLM integration
- LangGraph - Agent orchestration
- Anthropic - Claude API
- Tavily - Web search and URL extraction 