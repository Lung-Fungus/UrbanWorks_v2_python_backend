"""
Custom LLM implementation for the chat backend.
"""

from typing import Optional, List, Dict, Any
import anthropic
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.tools import Tool
from config import get_api_keys

# Get API keys
api_keys = get_api_keys()
ANTHROPIC_API_KEY = api_keys["ANTHROPIC_API_KEY"]

class CustomChatAnthropic(BaseChatModel):
    """Custom ChatAnthropic implementation for better tool integration."""
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