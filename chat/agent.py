"""
Agent and graph implementation for the chat backend.
"""

import re
import json
import uuid
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from .models import State
from .llm import CustomChatAnthropic
from .tools import get_tools
from .utils import logger
from prompts import get_clarke_system_prompt
from datetime import datetime
import pytz

# Define Central Time Zone
central_tz = pytz.timezone('US/Central')

# Get Clarke's system prompt with the current date in Central Time
CLARKE_SYSTEM_MESSAGE = get_clarke_system_prompt(datetime.now(central_tz))

# Initialize LangChain components
llm = CustomChatAnthropic()
llm_with_tools = llm.bind_tools(get_tools())
tool_node = ToolNode(tools=get_tools())

def create_agent_node():
    """Create the agent node for the graph."""
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
    """Create the chat graph."""
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