from fastapi import FastAPI, HTTPException, Depends
from auth_middleware import firebase_auth
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, TypedDict, Annotated, Any
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.tools import Tool
import anthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from tavily import TavilyClient
from config import get_api_keys, get_firebase_credentials, initialize_environment

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
    model_name: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.8
    max_tokens: int = 8192

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
if not firebase_admin._apps:
    cred = credentials.Certificate(get_firebase_credentials())
    firebase_admin.initialize_app(cred)

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        #logging.FileHandler('proposal_generator.log')
    ]
)
logger = logging.getLogger(__name__)

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
    return datetime.now().strftime("%B %d, %Y")

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
tool_node = ToolNode(tools=tools)

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
            try:
                messages.append(HumanMessage(content=chat_message))
                response = llm.invoke(messages, system=system_message)
                messages.append(AIMessage(content=response.content.strip()))
                state["messages"] = messages
                return state
            except Exception as e:
                logger.error(f"Error processing chat message: {str(e)}")
                raise

        # If not chat, proceed with proposal generation
        logger.info(f"Processing section: {current_section if current_section else 'Initial'}")

        markdown_formatting_guide = """
        Format your response using proper markdown with consistent spacing:

        1. For paragraphs (primary format):
           - Write in clear, well-structured paragraphs
           - Each paragraph should be 3-5 sentences
           - Use strong narrative flow between paragraphs
           - Convert all list-like information into proper paragraphs
           - DO NOT use bullet points or lists unless explicitly requested
           - Focus on descriptive, flowing text instead of enumeration

        2. For main sections (# headings):
           - Add 2 blank lines before each main section
           - Use a single # for main section headers
           - Add 1 blank line after the heading
           - Always follow headers with a full paragraph, not a list

        3. For subsections (### headings):
           - Add 2 blank lines before subsection headers
           - Use ### for subsection headers
           - Add 1 blank line after the heading
           - Always follow with a complete paragraph, never a list
           - Use transitional phrases between subsections

        4. For lists (ONLY when explicitly requested):
           - Convert all lists into paragraphs by default
           - Only use lists if the prompt specifically asks for them
           - If lists are required, keep them to 3-5 items maximum
           - Follow every list with an explanatory paragraph

        5. For tables (use only when data comparison is needed):
           - Add 2 blank lines before and 1 blank line after tables
           - Use standard markdown table format
           - Include a header row with column names
           - Use | to separate columns and - for header separator
           - Align columns using : in the separator row
           - Follow tables with explanatory paragraphs

        Example table format:

        | Column 1 | Column 2 | Column 3 |
        |:---------|:--------:|----------:|
        | Left     | Center   | Right    |
        | align    | align    | align    |

        IMPORTANT: Write in full, flowing paragraphs. Do not use bullet points or lists unless specifically asked to do so.
        """

        if not current_section:
            # Use user-defined sections instead of generating them
            sections = [section["name"] for section in sorted(proposal_data["sections"], key=lambda x: x["order"])]
            logger.info(f"Using user-defined sections: {sections}")
            
            if not sections:
                logger.error("No sections provided in the request")
                raise ValueError("No sections provided in the request")
            
            state["sections"] = sections
            first_section = sections[0]
            state["current_section"] = first_section
            
            # Get section data including template
            current_section_data = next(
                (s for s in proposal_data["sections"] if s["name"] == first_section),
                None
            )
            
            if not current_section_data or not current_section_data.get("template_content"):
                logger.error(f"No template content found for section: {first_section}")
                raise ValueError(f"No template content found for section: {first_section}")
                
            section_description = current_section_data.get("description", "")
            section_template = current_section_data["template_content"]
            
            # Generate first section
            prompt = f"""
            {markdown_formatting_guide}

            Generate the content for the {first_section} section of the proposal.
            {f'Section Description: {section_description}' if section_description else ''}
            Do not include the section name in your response.

            Project Information:
            - Proposal Name: {proposal_data['proposal_name']}
            - Client: {proposal_data['client_name']}
            - Project Details: {proposal_data['project_details']}
            - Budget: {proposal_data.get('budget', 'Not specified')}
            - Timeline: {proposal_data.get('timeline', 'Not specified')}

            Template Reference:
            {section_template}

            Additional Instructions:
            {proposal_data.get('additional_instructions', 'None provided')}
            {proposal_data.get('ai_instructions', '')}

            IMPORTANT GUIDELINES:
            1. Match the approximate length of the template provided (about {len(section_template.split())} words).
            2. Do not make assumptions or add details that aren't explicitly provided.
            3. If any information is unclear or missing, note it with [Additional information needed: specify what information].
            4. Focus on factual information from the provided details only.
            5. Follow any additional instructions provided above.
            6. Maintain the style and structure of the template while adapting the content.

            Format your response using proper markdown following the guide above.
            """
        else:
            sections = state.get("sections", [])
            if not sections:
                raise ValueError("No sections found in state")
                
            current_idx = sections.index(current_section)
            
            if current_idx < len(sections) - 1:
                next_section = sections[current_idx + 1]
                logger.info(f"Moving to next section: {next_section} (from {current_section})")
                state["current_section"] = next_section

                # Build context from previously generated sections
                previous_sections = []
                for section in sections[:current_idx + 1]:
                    content = state["generated_sections"].get(section, "").replace("# " + section + "\n\n", "")
                    previous_sections.append(f"{section}:\n{content}")
                previous_context = "\n\n".join(previous_sections)

                # Get section data including template
                next_section_data = next(
                    (s for s in proposal_data["sections"] if s["name"] == next_section),
                    None
                )
                
                if not next_section_data or not next_section_data.get("template_content"):
                    logger.error(f"No template content found for section: {next_section}")
                    raise ValueError(f"No template content found for section: {next_section}")
                    
                section_description = next_section_data.get("description", "")
                section_template = next_section_data["template_content"]

                prompt = f"""
                {markdown_formatting_guide}

                Here are the sections generated so far:

                {previous_context}

                Generate the content for the {next_section} section of the proposal.
                {f'Section Description: {section_description}' if section_description else ''}
                Make sure this section flows naturally from and complements the previous sections.
                Do not include the section name in your response.

                Project Information:
                - Proposal Name: {proposal_data['proposal_name']}
                - Client: {proposal_data['client_name']}
                - Project Details: {proposal_data['project_details']}
                - Budget: {proposal_data.get('budget', 'Not specified')}
                - Timeline: {proposal_data.get('timeline', 'Not specified')}

                Template Reference:
                {section_template}

                Additional Instructions:
                {proposal_data.get('additional_instructions', 'None provided')}
                {proposal_data.get('ai_instructions', '')}

                IMPORTANT GUIDELINES:
                1. Match the approximate length of the template provided (about {len(section_template.split())} words).
                2. Do not make assumptions or add details that aren't explicitly provided.
                3. If any information is unclear or missing, note it with [Additional information needed: specify what information].
                4. Focus on factual information from the provided details only.
                5. Follow any additional instructions provided above.
                6. Maintain the style and structure of the template while adapting the content.

                Format your response using proper markdown following the guide above.
                """

        # Get response from LLM with system message
        logger.info(f"Generating content for section: {state['current_section']}")
        logger.info("Sending prompt to LLM:")
        logger.info(f"System: {system_message}")
        logger.info(f"Prompt: {prompt}")
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)], system=system_message)
            logger.info(f"LLM Response: {response.content}")
            messages.append(HumanMessage(content=prompt))
            messages.append(response)
            
            # Store the generated section with its heading
            section_content = f"# {state['current_section']}\n\n{response.content.strip()}"
            state["generated_sections"][state["current_section"]] = section_content
            logger.info(f"Successfully generated content for {state['current_section']}")
            logger.info(f"Current sections completed: {list(state['generated_sections'].keys())}")

            return state
        except Exception as e:
            logger.error(f"Error generating content for {state['current_section']}: {str(e)}")
            raise

    return agent_node

# Create the graph
def create_proposal_graph():
    workflow = StateGraph(State)
    workflow.add_node("agent", create_agent_node())
    
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
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "agent": "agent",
            END: END
        }
    )
    
    workflow.set_entry_point("agent")
    return workflow.compile()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/generate-proposal")
async def generate_proposal(request: ProposalRequest):
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
            "proposal_data": request.dict(),
            "generated_sections": final_state["generated_sections"],
            "sections": final_state["sections"],
            "title": f"Proposal: {request.proposal_name}",
            "summary": f"Proposal for {request.client_name}",
            "messageCount": 2,
            "lastMessageTimestamp": datetime.now(),
            "created_at": datetime.now()
        })
        
        # Add initial messages
        messages_ref = proposal_ref.collection('messages')
        messages_ref.add({
            "role": "user",
            "content": f"Generate a proposal for {request.proposal_name}",
            "timestamp": datetime.now()
        })
        messages_ref.add({
            "role": "assistant",
            "content": final_proposal,
            "timestamp": datetime.now()
        })
        
        return {
            "proposal": final_proposal,
            "conversation_id": proposal_ref.id
        }
        
    except Exception as e:
        logger.error(f"Error in generate_proposal: {str(e)}", exc_info=True)
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
            "timestamp": datetime.now()
        })
        messages_ref.add({
            "role": "assistant",
            "content": last_message.content,
            "timestamp": datetime.now()
        })
        
        # Update conversation metadata
        proposal_ref.update({
            "lastMessageTimestamp": datetime.now(),
            "messageCount": firestore.Increment(2)
        })
        
        return {"response": last_message.content}
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Please run main.py to start the server") 