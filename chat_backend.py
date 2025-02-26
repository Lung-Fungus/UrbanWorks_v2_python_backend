from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, TypedDict, Annotated, Any
from datetime import datetime
import firebase_admin
from firebase_admin import firestore
import logging
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import anthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from tavily import TavilyClient
import re
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain.tools import Tool
from fastapi.responses import Response
from config import get_api_keys, initialize_environment
import json
import uuid
import requests
from auth_middleware import firebase_auth
import pytz  # Add pytz for timezone handling
from prompts import get_clarke_system_prompt  # Import the system prompt

# Initialize environment
initialize_environment()

# Get API keys
api_keys = get_api_keys()
ANTHROPIC_API_KEY = api_keys["ANTHROPIC_API_KEY"]
TAVILY_API_KEY = api_keys["TAVILY_API_KEY"]

# Initialize clients
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Custom ChatAnthropic implementation
class CustomChatAnthropic(BaseChatModel):
    client: Optional[anthropic.Client] = None
    model_name: str = "claude-3-7-sonnet-20250219"
    temperature: float = 1.0
    max_tokens: int = 64000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.client:
            self.client = anthropic.Client(api_key=ANTHROPIC_API_KEY)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = []
        for message in messages:
            if isinstance(message, HumanMessage):
                message_dicts.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                message_dicts.append({"role": "assistant", "content": message.content})

        # Get the system message from kwargs without a default
        system = kwargs.get('system')
        if system is None:
            raise ValueError("System message must be provided")

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=message_dicts,
            system=system
        )

        message = AIMessage(content=response.content[0].text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "anthropic-chat"

    def bind_tools(self, tools: List[Tool]) -> 'CustomChatAnthropic':
        """Bind tools to the model."""
        tools_str = "\n".join(
            f"{i+1}. {tool.name}: {tool.description}" 
            for i, tool in enumerate(tools)
        )

        def _new_system_message(system: str) -> str:
            return f"{system}\n\nYou have access to the following tools:\n{tools_str}\n\nTo use a tool, output a message in this format:\n<tool_calls>\n<tool>tool_name</tool>\n<input>tool input</input>\n</tool_calls>"

        def _new_generate(*args, **kwargs):
            if "system" in kwargs:
                kwargs["system"] = _new_system_message(kwargs["system"])
            return self._generate(*args, **kwargs)

        new_model = CustomChatAnthropic(
            client=self.client,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        new_model._generate = _new_generate
        return new_model

# Initialize Firebase Admin if not already initialized
if not firebase_admin._apps:
    from config import initialize_firebase
    initialize_firebase()

db = firestore.client()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    user_display_name: str
    files: Optional[List[Dict[str, str]]] = None  # List of parsed file contents

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Add console handler
        #logging.FileHandler('chat_backend.log', encoding='utf-8', mode='a')  # Only file handler with UTF-8
    ]
)
logger = logging.getLogger(__name__)

# Add a startup log message to verify logging is working
logger.info("\n=== CHAT BACKEND STARTED ===")
logger.info("Logging configured with console output")

# Define Central Time Zone
central_tz = pytz.timezone('US/Central')

# Get Clarke's system prompt with the current date in Central Time
CLARKE_SYSTEM_MESSAGE = get_clarke_system_prompt(datetime.now(central_tz))

def perform_web_search(query: str) -> str:
    """
    Perform a web search using Tavily API and return formatted results
    """
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

# Define tools first
tools = [
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

# Initialize LangChain components
llm = CustomChatAnthropic()
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools=tools)

def create_agent_node():
    def agent_node(state: State):
        logger.info("\n=== AGENT NODE PROCESSING ===")
        messages = state["messages"]
        user_display_name = state["user_display_name"]
        current_date = state["current_date"]
        files = state.get("files", [])
        tool_response = state.get("tool_response")
        previous_tool_input = state.get("tool_input")

        try:
            # Get the last message
            last_message = messages[-1] if messages else None
            if not last_message:
                logger.warning("No messages found in state")
                return state

            # Format conversation history
            conversation_history = "\\n\\n".join(
                [
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                    for msg in messages[:-1]
                ]
            )

            # Format context for the LLM
            context = f"""<conversation_history>{conversation_history if conversation_history else 'No previous messages'}</conversation_history>

<user_message>
{last_message.content}
</user_message>

<user_displayname>
{user_display_name}
</user_displayname>

<current_date>
{current_date}
</current_date>

<previous_tool_input>{previous_tool_input if previous_tool_input else ''}</previous_tool_input>
"""
            context += f"""
{f'<tool_response>{json.dumps(tool_response)}</tool_response>' if tool_response else ''}
<files>{json.dumps(files) if files else '[]'}</files>
"""

            # Get response from Clarke with context using llm_with_tools
            response = llm_with_tools.invoke([HumanMessage(content=context)], system=CLARKE_SYSTEM_MESSAGE, tool_response=tool_response)
            content = response.content.strip()

            # Extract tool calls from the content
            tool_calls_match = re.search(r'<tool_calls>([\s\S]*?)</tool_calls>', content)
            extracted_tool_calls = []
            if tool_calls_match:
                tool_calls_text = tool_calls_match.group(1).strip()
                tool_name_match = re.search(r'<tool>(.*?)</tool>', tool_calls_text)
                tool_input_match = re.search(r'<input>(.*?)</input>', tool_calls_text)

                if tool_name_match and tool_input_match:
                    extracted_tool = {
                        "name": tool_name_match.group(1).strip(),
                        "args": {"query": tool_input_match.group(1).strip()},
                        "id": str(uuid.uuid4())
                    }
                    extracted_tool_calls.append(extracted_tool)
                    # Remove the tool_calls block from the content
                    content = re.sub(r'<tool_calls>[\s\S]*?</tool_calls>', '', content).strip()

            # Parse the analysis section
            analysis_match = re.search(r'<analysis>([\s\S]*?)</analysis>', content)
            analysis = analysis_match.group(1).strip() if analysis_match else ""

            response_match = re.search(r'<response>([\s\S]*)', content)
            if response_match:
                main_response = response_match.group(1).strip()
                main_response = re.sub(r'</response>.*$', '', main_response, flags=re.DOTALL).strip()
            else:
                main_response = content

            # Create a new AI message and attach tool_calls if found
            new_message = AIMessage(content=main_response)
            if extracted_tool_calls:
                setattr(new_message, "tool_calls", extracted_tool_calls)
                logger.info(f"Extracted tool_calls: {extracted_tool_calls}")

            # Update state
            state["response"] = {
                "content": main_response,
                "analysis": analysis
            }
            state["messages"] = messages + [new_message]
            return state

        except Exception as e:
            logger.error(f"Error in agent node: {str(e)}", exc_info=True)
            raise

    return agent_node

def should_continue(state: State) -> str:
    """Determine if we should continue running tools or end."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        return "agent"

    # Check for tool calls in the message
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    elif isinstance(last_message.content, str) and "<tool_calls>" in last_message.content:
        return "tools"

    # If we have a response but no tool calls, we're done
    if state.get("response"):
        return "end"

    return "agent"

def create_chat_graph():
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("agent", create_agent_node())
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
            "agent": "agent"
        }
    )
    workflow.add_edge("tools", "agent")

    workflow.set_entry_point("agent")
    return workflow.compile()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/graph")
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

@app.post("/chat")
async def chat(request: ChatRequest, user_data: dict = Depends(firebase_auth)):
    try:
        # Log incoming request
        logger.info("=== NEW CHAT REQUEST ===")
        logger.info(f"Message: {request.message}")

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
        initial_state = {
            "messages": messages,
            "current_date": datetime.now(central_tz).strftime("%B %d, %Y %I:%M %p"),
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
            "timestamp": datetime.now(central_tz)
        })

        messages_ref.add({
            "role": "assistant",
            "content": response["content"],
            "analysis": response["analysis"],
            "timestamp": datetime.now(central_tz)
        })

        conversation_ref.update({
            "lastMessageTimestamp": datetime.now(central_tz),
            "messageCount": firestore.Increment(2)
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

if __name__ == "__main__":
    print("Please run main.py to start the server")