"""
Clarke AI agent and tools.

This module defines the Clarke AI agent, its dependencies, and tools
for web search and URL content extraction.
"""

import logging
import re
import requests
from typing import Any, List, Dict, Optional
from datetime import datetime

import pytz
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.settings import ModelSettings

from new_chat.models import ChatAnalysisResponse
from config import get_api_keys
from prompts import get_clarke_system_prompt

# Get API keys
api_keys = get_api_keys()
ANTHROPIC_API_KEY = api_keys["ANTHROPIC_API_KEY"]
TAVILY_API_KEY = api_keys["TAVILY_API_KEY"]

# Configure logging
logger = logging.getLogger(__name__)

# Define Central Time Zone
central_tz = pytz.timezone('US/Central')

class ClarkeDependencies:
    """Dependency container for Clarke AI agent."""
    def __init__(self, conversation_id: Optional[str] = None, db=None):
        self.conversation_id = conversation_id
        self.db = db
        self.current_date = datetime.now(central_tz).strftime("%B %d, %Y %I:%M %p")
        self.user_display_name = None
        self.files = None
        self.tool_response = None

# Initialize the Anthropic model
model = AnthropicModel('claude-3-7-sonnet-20250219', api_key=ANTHROPIC_API_KEY)

# Create the Clarke AI agent
clarke_agent = Agent(
    model,
    deps_type=ClarkeDependencies,
    result_type=ChatAnalysisResponse,
    model_settings=ModelSettings(
        max_tokens=20000,
    ),
    system_prompt=get_clarke_system_prompt(datetime.now(central_tz))
)

async def validate_and_parse_result(result: Any) -> ChatAnalysisResponse:
    """Validates and parses the AI response format."""
    try:
        if isinstance(result, ChatAnalysisResponse):
            return result

        content = str(result)

        analysis_match = re.search(r'<analysis>(.*?)</analysis>', content, re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)

        if not analysis_match or not response_match:
            # Default to full content if no specific tags
            if not analysis_match and not response_match:
                return ChatAnalysisResponse(
                    analysis="",
                    content=content
                )
            elif not analysis_match:
                return ChatAnalysisResponse(
                    analysis="",
                    content=response_match.group(1).strip()
                )
            else:
                return ChatAnalysisResponse(
                    analysis=analysis_match.group(1).strip(),
                    content=content.replace(f"<analysis>{analysis_match.group(1)}</analysis>", "").strip()
                )

        analysis = analysis_match.group(1).strip()
        main_response = response_match.group(1).strip()

        return ChatAnalysisResponse(
            analysis=analysis,
            content=main_response
        )

    except Exception as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to validate response: {str(e)}")

@clarke_agent.tool
async def web_search(ctx: RunContext[ClarkeDependencies], query: str) -> str:
    """
    Perform a web search using Tavily API and return formatted results.
    """
    logger.info(f"\n=== WEB SEARCH TOOL EXECUTION ===")
    logger.info(f"Search query: {query}")
    try:
        logger.info("Making API call to Tavily...")
        
        search_url = "https://api.tavily.com/search"
        payload = {
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "max_results": 10
        }
        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(search_url, json=payload, headers=headers)
        response.raise_for_status()
        search_results = response.json()

        logger.info(f"Search successful:")
        logger.info(f"- Number of results: {len(search_results.get('results', []))}")
        logger.info(f"- Has summary: {'Yes' if search_results.get('answer') else 'No'}")

        if search_results.get("answer"):
            logger.info(f"Search summary: {search_results['answer'][:200]}...")

        formatted_results = "Search Results:\n\n"
        if search_results.get("answer"):
            formatted_results += f"Summary: {search_results['answer']}\n\n"

        formatted_results += "Sources:\n"
        for idx, result in enumerate(search_results.get("results", [])):
            logger.info(f"\nProcessing result {idx + 1}:")
            logger.info(f"- Title: {result['title']}")
            logger.info(f"- URL: {result['url']}")
            logger.info(f"- Content length: {len(result['content'])} characters")

            formatted_results += f"- {result['title']}\n"
            formatted_results += f"  URL: {result['url']}\n"
            formatted_results += f"  Content: {result['content']}\n\n"

        logger.info("\n=== WEB SEARCH COMPLETED ===")
        logger.info(f"Total formatted results length: {len(formatted_results)} characters")
        
        # Store the tool response for context
        ctx.deps.tool_response = formatted_results
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error performing web search: {str(e)}", exc_info=True)
        return f"Error performing web search: {str(e)}"

@clarke_agent.tool
async def extract_url(ctx: RunContext[ClarkeDependencies], url: str) -> str:
    """
    Extract content from a given URL using Tavily's extraction API.
    """
    logger.info(f"\n=== URL CONTENT EXTRACTION TOOL EXECUTION ===")
    logger.info(f"URL: {url}")

    try:
        extract_url = "https://api.tavily.com/extract"
        payload = {
            "urls": url,
            "include_images": False,
            "extract_depth": "advanced",
        }
        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json"
        }

        logger.info("Making API call to Tavily extraction endpoint...")
        logger.info(f"Request payload: {payload}")

        response = requests.post(extract_url, json=payload, headers=headers)
        logger.info(f"Response status code: {response.status_code}")

        response.raise_for_status()
        content = response.json()
        logger.info("Successfully parsed JSON response")

        # Format the extracted content
        formatted_content = ""

        if content.get("results"):
            # Get the first result's raw_content
            result = content["results"][0]
            if "raw_content" in result:
                formatted_content = result["raw_content"].strip()
            else:
                formatted_content = "No raw content found in the extraction results"
        else:
            formatted_content = "No results found in the extraction response"

        if content.get("failed_results"):
            failed_reasons = [f"- {failed.get('error', 'Unknown error')}" for failed in content["failed_results"]]
            if failed_reasons:
                formatted_content += "\n\nExtraction Warnings:\n" + "\n".join(failed_reasons)

        logger.info("\n=== URL EXTRACTION COMPLETED ===")
        logger.info(f"Total formatted content length: {len(formatted_content)} characters")
        
        # Store the tool response for context
        ctx.deps.tool_response = formatted_content
        
        return formatted_content

    except Exception as e:
        error_msg = f"Error extracting URL content: {str(e)}"
        logger.error(error_msg)
        return error_msg

@clarke_agent.tool
async def get_conversation_history(ctx: RunContext[ClarkeDependencies]) -> List[Dict]:
    """Retrieves conversation history from Firestore."""
    try:
        if not ctx.deps.conversation_id or not ctx.deps.db:
            return []

        messages_ref = ctx.deps.db.collection("conversations").document(
            ctx.deps.conversation_id).collection("messages")
        docs = messages_ref.order_by("timestamp").stream()

        return [
            {
                "role": doc.get("role"),
                "content": doc.get("content"),
                "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None
            } for doc in docs
        ]
    except Exception as e:
        logger.error(f"Error fetching conversation history: {str(e)}")
        return []

# Override the run method to inject user display name into the message context
original_run = clarke_agent.run

async def run_with_user_context(message: str, deps: ClarkeDependencies, **kwargs):
    """
    Wrapper for the agent run method that injects the user display name into the context.
    """
    logger.info(f"Running agent with user display name: {deps.user_display_name}")
    
    # Prepare context with user display name
    user_context = f"""
<user_displayname>{deps.user_display_name}</user_displayname>
<current_date>{deps.current_date}</current_date>
<user_message>{message}</user_message>
"""
    
    # Add the user context to the message
    context_message = f"{user_context}\n{message}"
    
    # Call the original run method with the enhanced message
    return await original_run(context_message, deps=deps, **kwargs)

# Replace the original run method with our wrapper
clarke_agent.run = run_with_user_context 