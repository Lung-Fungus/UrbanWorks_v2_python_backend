"""
Tool implementations for the chat backend.
"""

import requests
from langchain.tools import Tool
from config import get_api_keys
from .utils import logger

# Get API keys
api_keys = get_api_keys()
TAVILY_API_KEY = api_keys["TAVILY_API_KEY"]

def perform_web_search(query: str) -> str:
    """
    Perform a web search using Tavily API and return formatted results
    """
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
    logger.info(f"\n=== WEB SEARCH TOOL EXECUTION ===")
    logger.info(f"Search query: {query}")
    try:
        logger.info("Making API call to Tavily...")
        search_results = tavily_client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            max_results=10
        )

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
        return formatted_results
    except Exception as e:
        logger.error(f"Error performing web search: {str(e)}", exc_info=True)
        return f"Error performing web search: {str(e)}"

def extract_url_content(url: str) -> str:
    """
    Extract content from a given URL using Tavily's extraction API.

    Args:
        url (str): The URL to extract content from

    Returns:
        str: Formatted content from the URL
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
        return formatted_content

    except Exception as e:
        error_msg = f"Error extracting URL content: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Define tools
def get_tools():
    """Get all available tools for the chat agent."""
    return [
        Tool(
            name="web_search",
            description="Search the web for current information. Use this for any queries about regulations, codes, or up-to-date information. Input should be your search query.",
            func=perform_web_search,
        ),
        Tool(
            name="extract_url",
            description="Extract and analyze content from a specific URL. Input should be the complete URL including http:// or https://",
            func=extract_url_content,
        )
    ] 