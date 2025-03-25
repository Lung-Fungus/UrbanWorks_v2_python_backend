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
from firebase_admin import firestore

from new_chat.models import ChatAnalysisResponse
from utils.config import get_api_keys
from utils.prompts import get_clarke_system_prompt

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
        self.available_collections = []

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
        file_content_match = re.search(r'<file_content>(.*?)</file_content>', content, re.DOTALL)

        # Extract file content if available
        file_content = file_content_match.group(1).strip() if file_content_match else None

        if not analysis_match or not response_match:
            # Default to full content if no specific tags
            if not analysis_match and not response_match:
                return ChatAnalysisResponse(
                    analysis="",
                    content=content,
                    file_content=file_content
                )
            elif not analysis_match:
                return ChatAnalysisResponse(
                    analysis="",
                    content=response_match.group(1).strip(),
                    file_content=file_content
                )
            else:
                return ChatAnalysisResponse(
                    analysis=analysis_match.group(1).strip(),
                    content=content.replace(f"<analysis>{analysis_match.group(1)}</analysis>", "").strip(),
                    file_content=file_content
                )

        analysis = analysis_match.group(1).strip()
        main_response = response_match.group(1).strip()

        return ChatAnalysisResponse(
            analysis=analysis,
            content=main_response,
            file_content=file_content
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
    """Gets the conversation history."""
    try:
        # Get the conversation ID from dependencies
        conversation_id = ctx.deps.conversation_id
        if not conversation_id:
            logger.warning("No conversation ID provided, returning empty history")
            return []

        # Check if db is available in dependencies
        if not ctx.deps.db:
            logger.error("Firestore DB not available in dependencies")
            return []

        # Get messages collection for this conversation
        messages_ref = ctx.deps.db.collection('conversations').document(conversation_id).collection('messages')

        # Order by timestamp and get all messages
        messages_query = messages_ref.order_by('timestamp')
        messages_docs = messages_query.stream()

        # Format messages for context
        conversation_history = []
        for doc in messages_docs:
            message_data = doc.to_dict()
            conversation_history.append({
                "role": message_data.get("role", "unknown"),
                "content": message_data.get("content", ""),
                "timestamp": message_data.get("timestamp", None)
            })

        logger.info(f"Retrieved {len(conversation_history)} messages from conversation history")
        return conversation_history

    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
        return []

@clarke_agent.tool
async def get_urbanworks_collection_data(ctx: RunContext[ClarkeDependencies], collection_name: str) -> Dict:
    """
    Retrieves all data from an UrbanWorks Firestore database collection.
    Returns all documents without any limits.

    Args:
        collection_name (str): The name of the collection to retrieve data from (e.g., "UrbanWorks Projects")

    Returns:
        Dict: A dictionary containing collection data and metadata
    """
    try:
        logger.info(f"Retrieving data from UrbanWorks collection: {collection_name}")
        try:
            db = firestore.client()
        except Exception as e:
            logger.error(f"Error getting Firestore client: {str(e)}")
            return {
                "status": "error",
                "message": "Unable to connect to the database",
                "documents": []
            }

        # Get collection data - retrieve all documents (no limit)
        collection_ref = db.collection(collection_name)
        docs = collection_ref.stream()

        # Extract document data
        documents = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id  # Add document ID
            documents.append(doc_data)

        logger.info(f"Retrieved {len(documents)} documents from collection: {collection_name}")

        # Calculate field statistics to help the AI understand the data
        field_stats = {}
        if documents:
            # Get all unique fields
            all_fields = set()
            for doc in documents:
                all_fields.update(doc.keys())

            # Calculate stats for each field
            for field in all_fields:
                field_values = [doc.get(field) for doc in documents if field in doc]
                value_types = set(type(val).__name__ for val in field_values if val is not None)

                field_stats[field] = {
                    "count": len(field_values),
                    "types": list(value_types),
                    "sample_values": field_values[:3] if field_values else []
                }

        result = {
            "collection_name": collection_name,
            "document_count": len(documents),
            "documents": documents,
            "field_stats": field_stats,
            "retrieved_at": datetime.now(central_tz).strftime("%Y-%m-%d %H:%M:%S"),
            "note": "This data is from the UrbanWorks internal database."
        }

        return result
    except Exception as e:
        error_message = f"Error retrieving collection data: {str(e)}"
        logger.error(error_message)
        return {
            "error": error_message,
            "collection_name": collection_name,
            "documents": [],
            "document_count": 0
        }

@clarke_agent.tool
async def get_available_urbanworks_collections(ctx: RunContext[ClarkeDependencies]) -> Dict:
    """
    Retrieves a list of all available UrbanWorks database collections shown in the DatabaseDisplay component.
    These are the collections that users can interact with through the database interface.

    Returns:
        Dict: A dictionary containing the list of collection names
    """
    try:
        logger.info("Retrieving list of available UrbanWorks collections")
        try:
            db = firestore.client()
        except Exception as e:
            logger.error(f"Error getting Firestore client: {str(e)}")
            return {
                "status": "error",
                "message": "Unable to connect to the database",
                "collections": []
            }

        # Get collections list from the collections collection
        collections_ref = db.collection('collections')
        collections_docs = collections_ref.stream()

        # Extract collection names
        collections = []
        for doc in collections_docs:
            collection_data = doc.to_dict()
            if 'name' in collection_data:
                collections.append(collection_data['name'])

        result = {
            "collections": collections,
            "count": len(collections),
            "retrieved_at": datetime.now(central_tz).strftime("%Y-%m-%d %H:%M:%S")
        }

        return result
    except Exception as e:
        error_message = f"Error retrieving collections list: {str(e)}"
        logger.error(error_message)
        return {
            "error": error_message,
            "collections": [],
            "count": 0
        }

# Override the run method to inject user display name into the message context
original_run = clarke_agent.run

async def run_with_user_context(message: str, deps: ClarkeDependencies, **kwargs):
    """
    Wrapper for the agent run method that injects the user display name into the context.
    """
    logger.info(f"Running agent with user display name: {deps.user_display_name}")

    # Prepare file content if files are present
    file_content = ""
    if deps.files and len(deps.files) > 0:
        file_content = "Parsed File Content:\n\n"
        for idx, file_info in enumerate(deps.files):
            file_content += f"File {idx + 1}: {file_info.get('filename', 'Unknown')}\n"
            file_content += f"Content: {file_info.get('content', 'No content')}\n\n"
        logger.info(f"Prepared file content for {len(deps.files)} files")

    # Prepare available collections
    collections_info = ""
    if deps.available_collections and len(deps.available_collections) > 0:
        collections_info = "<available_collections>\n"
        for collection in deps.available_collections:
            collections_info += f"- {collection}\n"
        collections_info += "</available_collections>\n"
        logger.info(f"Including {len(deps.available_collections)} collections in context")

    # Prepare context with user display name and file content
    user_context = f"""
<user_displayname>{deps.user_display_name}</user_displayname>
<current_date>{deps.current_date}</current_date>
{collections_info}<user_message>{message}</user_message>
"""

    # Add the user context to the message
    context_message = f"{user_context}\n{message}"

    # Call the original run method with the enhanced message
    result = await original_run(context_message, deps=deps, **kwargs)

    # Add file content to the result if present
    if isinstance(result.data, ChatAnalysisResponse) and file_content:
        result.data.file_content = file_content

    return result

# Replace the original run method with our wrapper
clarke_agent.run = run_with_user_context 