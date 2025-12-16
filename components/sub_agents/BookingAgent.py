from components.utils import AgentFactory, ToolRegistry
from components.common import ModelSettings
from components import MechaniGoContext
from typing import Optional
from config import settings

INSTRUCTIONS = """
You are {name}, a bookings and payment agent for MechaniGo.ph.\n
Your goal is to collect user details, then confirm and save.\n

### Required details (ask together in one concise request):
- name
- email
- address/location
- contact number
- car/make/year/model
- service type
- schedule date and time
- preferred payment type

# Flow

1) First response: ask for ALL required details in one short message (one paragraph or a tight bullet list).
2) When the user provides the required details, call `extract_user_info` to parse/merge into the working record.
3) If any required fields are still missing, ask again but include ONLY the missing fields; keep it concise.
4) When all the required fields are extracted, immediately call `save_user_info` with the current record, summarize the infor, acknowledge success, and end the conversation.

# Tools

- `extract_user_info` to parse user-provided details.
- `save_user_info` to save the confirmed record.

# Current User Memory:
{user_memory}
"""

class BookingAgent(AgentFactory):
    def __init__(
        self,
        api_key: str,
        name: Optional[str] = "booking_agent",
        model: Optional[str] = None,
        max_tokens: Optional[int] = settings.OPENAI_MAX_TOKENS,
        temperature: Optional[float] = settings.SUB_AGENT_TEMPERATURE,
        context: Optional[MechaniGoContext] = None
    ):
        """
        :param api_key: OpenAI API key for the agent.
        :type api_key: str
        :param name: Display name used in prompts; defaults to "mechanic_agent".
        :type name: Optional[str]
        :param model: Model used for the LLM; defaults to `None`.
        :type model: Optional[str]
        :param max_tokens: Max tokens for response; defaults to `500` (set in `settings.MAX_TOKENS`).
        :type max_tokens: Optional[int]
        :param temperature: Sampling temperature; defaults to `0.1` (set in `settings.SUB_AGENT_TEMPERATURE`).
        :type temperature: Optional[float]
        :param context: Prebuild context; created if not provided.
        :type context: Optional[MechaniGoContext]
        """
        super().__init__(api_key=api_key)
        self.name = name
        self.model = model
        self.instructions = INSTRUCTIONS
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.orchestrator_tool = None # agent instance
        self.context = context

    @property
    def as_tool(self):
        if self.orchestrator_tool is None:
            self.orchestrator_tool = self.build().as_tool(
                tool_name=self.get_name(),
                tool_description=self.get_handoff_description()
            )
        return self.orchestrator_tool

    def dynamic_instructions(self):
        user_dump = {}
        if self.context and getattr(self.context, "user_ctx", None):
            user = self.context.user_ctx.user_memory
            if user:
                try:
                    user_dump = user.model_dump()
                except Exception:
                    user_dump = {}
        return self.instructions.format(
            name=self.get_name(),
            user_memory=user_dump
        )

    def get_model(self) -> str:
        return self.model

    def get_name(self) -> str:
        return self.name

    def get_handoff_description(self) -> str:
        return "Handles user info extraction and booking services."

    def get_instructions(self):
        return self.dynamic_instructions()

    def get_tools(self):
        return [
            ToolRegistry.get_tool("extract.user_info"),
            ToolRegistry.get_tool("booking.save_user_info")
        ]

    def get_input_guardrails(self):
        return [] # No guardrails for now

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )