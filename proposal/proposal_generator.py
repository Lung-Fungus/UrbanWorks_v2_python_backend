from fastapi import FastAPI, HTTPException, Depends
from utils.auth_middleware import firebase_auth
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, TypedDict, Annotated, Any
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.tools import Tool
import anthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from tavily import TavilyClient
from utils.config import get_api_keys, get_firebase_credentials, initialize_environment
import pytz  # Add pytz for timezone handling
from utils.prompts import get_clarke_system_prompt  # Import the system prompt

# Initialize environment
initialize_environment()

# Get API keys
api_keys = get_api_keys()
ANTHROPIC_API_KEY = api_keys["ANTHROPIC_API_KEY"]
TAVILY_API_KEY = api_keys["TAVILY_API_KEY"]

# Initialize clients
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Custom ChatAnthropic implementation to avoid pydantic v2 issues
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

        # Get the system message from kwargs
        system = kwargs.get('system', "You are Clarke, an expert proposal writer for UrbanWorks architecture, specializing in creating compelling architectural, design, and construction, and business proposals.")

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

# Initialize Firebase Admin if not already initialized
try:
    # Since main.py should have already initialized Firebase, we just need to get the client
    db = firestore.client()
except Exception as e:
    # If an error occurs, try initializing Firebase ourselves
    if not firebase_admin._apps:
        try:
            # Initialize Firebase without storage bucket to prevent automatic bucket creation
            cred = credentials.Certificate(get_firebase_credentials())
            firebase_admin.initialize_app(cred, {
                'storageBucket': None  # Disable automatic Storage bucket initialization
            })
            db = firestore.client()
        except Exception as e:
            print(f"Error initializing Firebase in proposal generator: {e}")
            # Create a placeholder db that will be replaced when Firebase is available
            db = None

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProposalRequest(BaseModel):
    user_name: str
    additional_instructions: Optional[str] = None
    proposal_name: str
    client_name: str
    project_details: str
    budget: Optional[str] = None
    timeline: Optional[str] = None
    ai_instructions: Optional[str] = None
    sections: List[Dict[str, Any]] = []  # List of section objects with name, description, order, and template

class State(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    template_content: Annotated[str, "The content of the selected template"]
    proposal_data: Annotated[Dict, "The proposal request data"]
    current_section: Annotated[Optional[str], "The current section being generated"]
    generated_sections: Annotated[Dict[str, str], "The generated sections of the proposal"]
    sections: Annotated[List[str], "The list of sections to generate"]
    is_chat: Annotated[bool, "Whether this is a chat interaction"]
    chat_message: Annotated[Optional[str], "The chat message if this is a chat interaction"]
    agent_action: Annotated[Optional[Any], "The action to be executed by a tool"]
    tool_result: Annotated[Optional[Any], "The result from executing a tool"]

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Add console handler
    ]
)
logger = logging.getLogger(__name__)

# Define Central Time Zone
central_tz = pytz.timezone('US/Central')

# Define tools
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
if not LLAMA_CLOUD_API_KEY:
    raise ValueError("LLAMA_CLOUD_API_KEY environment variable is not set in .env.local")

async def get_template_content(template_id: str) -> str:
    """Get the content of a proposal template from Firestore."""
    try:
        template_doc = db.collection('proposal_templates').document(template_id).get()
        if not template_doc.exists:
            raise ValueError("Template not found")
        template_data = template_doc.to_dict()

        # Get the parsed content
        content = template_data.get('content', '')
        if not content:
            raise ValueError("Template has no content")

        logger.info(f"Retrieved template: {template_data.get('name', 'Unknown')}")
        logger.info(f"Content length: {len(content)}")
        return content

    except Exception as e:
        logger.error(f"Error retrieving template: {str(e)}")
        raise ValueError(f"Error retrieving template: {str(e)}")

def get_current_date() -> str:
    """Get the current date in a formatted string."""
    return datetime.now(central_tz).strftime("%B %d, %Y")

def perform_web_search(query: str) -> str:
    """
    Perform a web search using Tavily API and return formatted results
    """
    try:
        search_results = tavily_client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            max_results=10
        )

        # Format the results
        formatted_results = "Search Results:\n\n"
        if search_results.get("answer"):
            formatted_results += f"Summary: {search_results['answer']}\n\n"

        formatted_results += "Sources:\n"
        for result in search_results.get("results", []):
            formatted_results += f"- {result['title']}\n"
            formatted_results += f"  URL: {result['url']}\n"
            formatted_results += f"  Content: {result['content']}\n\n"

        logger.info(f"Web search performed for query: {query}")
        return formatted_results
    except Exception as e:
        logger.error(f"Error performing web search: {str(e)}")
        return f"Error performing web search: {str(e)}"

tools = [
    Tool(
        name="get_template",
        description="Get the content of a proposal template",
        func=get_template_content,
    ),
    Tool(
        name="get_date",
        description="Get the current date",
        func=get_current_date,
    ),
    Tool(
        name="web_search",
        description="Search the web for current information. Use this for any queries about regulations, codes, or up-to-date information. Input should be your search query.",
        func=perform_web_search,
    )
]

# Initialize LangChain components
llm = CustomChatAnthropic()
tool_executor = tool_executor(tools)

# Define agent nodes
def create_agent_node():
    def agent_node(state: State):
        messages = state["messages"]
        proposal_data = state["proposal_data"]
        current_section = state.get("current_section")
        is_chat = state.get("is_chat", False)
        chat_message = state.get("chat_message")

        # Create base system message
        base_system = """You are Clarke, an expert proposal writer for UrbanWorks architecture, specializing in creating compelling architectural, design, and construction, and business proposals.

You have access to these tools:
1. web_search: Search the web for current information about regulations, codes, or any up-to-date information you need
2. get_template: Get proposal template content
3. get_date: Get the current date

When users ask about regulations, codes, or current information, use the web_search tool to ensure accuracy.""".strip()

        # Add user's AI instructions if they exist
        if proposal_data.get("ai_instructions"):
            system_message = f"{base_system}\n\nAdditional Instructions: {proposal_data['ai_instructions']}\n\nNote: Follow the additional user instructions while maintaining core functionality and user experience."
        else:
            system_message = base_system

        if is_chat and chat_message:
            logger.info("Processing chat message")
            # For chat, we just add the human message and let the LLM respond
            human_message = HumanMessage(content=chat_message)
            messages.append(human_message)
            
            response = llm._generate(
                messages=messages,
                system=system_message
            )
            return {"messages": messages + [response.generations[0].message]}
        
        # If we're generating a section
        elif current_section:
            logger.info(f"Generating section: {current_section}")
            
            # Get the current section info
            section_info = None
            for section in proposal_data.get("sections", []):
                if section["name"] == current_section:
                    section_info = section
                    break
            
            if not section_info:
                logger.error(f"Section info not found for {current_section}")
                return {"messages": messages + [AIMessage(content=f"Error: Section info not found for {current_section}")]}
            
            # Construct a prompt for generating this section
            section_prompt = f"""
            I need you to write the {current_section} section for a proposal.
            
            Project Name: {proposal_data.get('proposal_name')}
            Client: {proposal_data.get('client_name')}
            Project Details: {proposal_data.get('project_details')}
            """
            
            if proposal_data.get('budget'):
                section_prompt += f"\nBudget: {proposal_data.get('budget')}"
            
            if proposal_data.get('timeline'):
                section_prompt += f"\nTimeline: {proposal_data.get('timeline')}"
            
            section_prompt += f"\n\nSection Description: {section_info.get('description', 'No description provided.')}"
            
            # If there's a template ID, we should retrieve it
            template_id = section_info.get('template')
            if template_id:
                section_prompt += "\n\nI'll try to get the template for this section to help guide the writing."
            
            human_message = HumanMessage(content=section_prompt)
            messages.append(human_message)
            
            response = llm._generate(
                messages=messages,
                system=system_message
            )
            
            return {"messages": messages + [response.generations[0].message]}
        
        # Initial prompt
        else:
            logger.info("Processing initial prompt")
            sections = ", ".join([s['name'] for s in proposal_data.get("sections", [])])
            
            # Construct the initial prompt
            initial_prompt = f"""
            I need you to help me write a proposal for the following project:
            
            Project Name: {proposal_data.get('proposal_name')}
            Client: {proposal_data.get('client_name')}
            Project Details: {proposal_data.get('project_details')}
            """
            
            if proposal_data.get('budget'):
                initial_prompt += f"\nBudget: {proposal_data.get('budget')}"
            
            if proposal_data.get('timeline'):
                initial_prompt += f"\nTimeline: {proposal_data.get('timeline')}"
            
            if sections:
                initial_prompt += f"\n\nThe proposal will include these sections: {sections}"
            
            if proposal_data.get('additional_instructions'):
                initial_prompt += f"\n\nAdditional Instructions: {proposal_data.get('additional_instructions')}"
            
            human_message = HumanMessage(content=initial_prompt)
            messages.append(human_message)
            
            response = llm._generate(
                messages=messages,
                system=system_message
            )
            
            return {"messages": messages + [response.generations[0].message]}
        
    return agent_node

def tools_node(state: State):
    """Tool executor node that executes tools."""
    agent_action = state["agent_action"]
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input
    
    # Execute the tool
    observation = tool_executor.execute(tool_name, tool_input)
    
    # Update the state with the result
    return {"tool_result": observation}

# Create the graph
def create_proposal_graph():
    workflow = StateGraph(State)
    workflow.add_node("agent", create_agent_node())
    workflow.add_node("tools", tools_node)

    def should_continue(state: State) -> str:
        if state.get("is_chat", False):
            logger.info("Chat interaction complete")
            return END

        current_section = state.get("current_section")
        sections = state.get("sections", [])

        logger.info(f"Checking if should continue. Current section: {current_section}")
        logger.info(f"Current sections in state: {list(state.get('generated_sections', {}).keys())}")
        logger.info(f"All sections: {sections}")

        if not sections:
            logger.info("No sections list, continuing to agent")
            return "agent"

        if not current_section:
            logger.info("No current section, continuing to agent")
            return "agent"

        try:
            current_idx = sections.index(current_section)
            # If we've generated all sections, end the graph
            if current_idx >= len(sections) - 1:
                logger.info("Reached last section, ending graph")
                return END

            # Otherwise, continue to the next section
            logger.info(f"Continuing to next section after {current_section}")
            return "agent"
        except ValueError:
            logger.error(f"Current section '{current_section}' not found in sections list")
            raise ValueError(f"Current section '{current_section}' not found in sections list: {sections}")

    # Define the agent's conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "agent": "agent",
            END: END
        }
    )

    # Add edge to tools node if needed
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "agent")

    workflow.set_entry_point("agent")
    return workflow.compile()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/generate-proposal")
async def generate_proposal(request: ProposalRequest, user_data: dict = Depends(firebase_auth)):
    try:
        # Initialize state
        initial_state = {
            "messages": [],
            "template_content": "",  # No primary template
            "proposal_data": request.dict(),
            "current_section": None,
            "generated_sections": {},
            "sections": [],
            "is_chat": False,
            "chat_message": None
        }

        # Create and run the graph
        graph = create_proposal_graph()
        final_state = graph.invoke(initial_state)

        # Combine sections into final proposal
        sections = [
            content 
            for section, content in final_state["generated_sections"].items()
        ]
        final_proposal = "\n\n".join(sections)

        # Create a new proposal document in Firestore
        proposal_ref = db.collection('proposals').document()
        proposal_ref.set({
            "userid": user_data['uid'],  # Add user ID
            "proposal_data": request.dict(),
            "generated_sections": final_state["generated_sections"],
            "sections": final_state["sections"],
            "title": f"Proposal: {request.proposal_name}",
            "summary": f"Proposal for {request.client_name}",
            "messageCount": 2,
            "lastMessageTimestamp": datetime.now(central_tz),
            "created_at": datetime.now(central_tz)
        })

        # Add initial messages
        messages_ref = proposal_ref.collection('messages')
        messages_ref.add({
            "role": "user",
            "content": f"Generate a proposal for {request.proposal_name}",
            "timestamp": datetime.now(central_tz)
        })
        messages_ref.add({
            "role": "assistant",
            "content": final_proposal,
            "timestamp": datetime.now(central_tz)
        })

        return {
            "proposal": final_proposal,
            "conversation_id": proposal_ref.id
        }
    except Exception as e:
        logger.error(f"Error in generate-proposal endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str
    conversation_id: str

@app.post("/chat")
async def chat(request: ChatRequest, user_data: dict = Depends(firebase_auth)):
    try:
        # Get the proposal document first to get the context
        proposal_ref = db.collection('proposals').document(request.conversation_id)
        proposal_doc = proposal_ref.get()

        if not proposal_doc.exists:
            raise HTTPException(status_code=404, detail="Proposal not found")

        proposal_data = proposal_doc.to_dict()

        # Verify user owns this conversation
        if proposal_data.get('userid') != user_data['uid']:
            raise HTTPException(status_code=403, detail="Not authorized to access this conversation")

        # Get the conversation messages from Firestore
        messages_ref = proposal_ref.collection('messages')
        messages_query = messages_ref.order_by('timestamp').stream()

        # Convert Firestore messages to LangChain messages
        messages = []
        for msg in messages_query:
            msg_data = msg.to_dict()
            if msg_data['role'] == 'user':
                messages.append(HumanMessage(content=msg_data['content']))
            else:
                messages.append(AIMessage(content=msg_data['content']))

        # Get the template content if it exists
        template_content = ""
        if 'template_id' in proposal_data:
            try:
                template_content = await get_template_content(proposal_data['template_id'])
            except Exception as e:
                logger.warning(f"Could not retrieve template content: {str(e)}")

        # Initialize state for chat with proposal context
        initial_state = {
            "messages": messages,
            "template_content": template_content,
            "proposal_data": proposal_data.get('proposal_data', {}),
            "current_section": None,
            "generated_sections": proposal_data.get('generated_sections', {}),
            "sections": proposal_data.get('sections', []),
            "is_chat": True,
            "chat_message": request.message
        }

        # Create and run the graph
        graph = create_proposal_graph()
        final_state = graph.invoke(initial_state)

        # Get the last assistant message
        last_message = final_state["messages"][-1]

        # Save messages to Firestore
        messages_ref = proposal_ref.collection('messages')
        messages_ref.add({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now(central_tz)
        })
        messages_ref.add({
            "role": "assistant",
            "content": last_message.content,
            "timestamp": datetime.now(central_tz)
        })

        # Update conversation metadata
        proposal_ref.update({
            "lastMessageTimestamp": datetime.now(central_tz),
            "messageCount": firestore.Increment(2)
        })

        return {"response": last_message.content}

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Please run main.py to start the server") 