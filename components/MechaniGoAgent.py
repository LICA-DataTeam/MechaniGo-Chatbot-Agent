from components.common import (
    ModelSettings, Runner,
    TResponseInputItem
)

from components.utils import (
    mechanigo_guardrail,
    SessionHandler,
    ToolRegistry,
    AgentFactory
)
from components.schemas import User
from config import settings

from typing import Optional, List
from pydantic import BaseModel

INSTRUCTIONS = """
You are {name}, the manager agent for MechaniGo PH. 

Your job is to: (1) use tools for FAQs, and (2) delegate to sub-agents when needed

Capabilities:
- Use `faq_tool` only when the user asks for factual MechaniGo info that you cannot answer from context. For generic booking intent or conversational back-and-forth, stay in chat mode.
    - When the user asks in Tagalog, translate it to English first before using `faq_tool` to answer their question.
- Use `mechanic_agent` whenever the user has an inquiry related to automotives.
    - **ALWAYS** forward the exact text from `mechanic_agent` with no paraphrasing, no extra bullets, and no added content.
- Use `booking_agent` when the user wants to book an appointment (PMS, Secondhand Car Inspection, Parts Replacement, Car Diagnosis).

Rules:
- Don’t invent diagnoses or company policies.
- Ask clear follow-up questions when information is missing.
- Always choose the appropriate tool/agent; don’t answer mechanic questions yourself.
- Provide friendly, concise replies in a customer-service tone.
- Do not call `user_extract_info` on intent-only messages; wait for details first.

Workflow:
1) Detect intent → 2) Choose tool/agent → 3) Get results

Communication style:
- Be warm, respectful, and casual
- Use simple Taglish, use 'po' and 'opo' time to time.
- Do not send long paragraphs, prefer short bullet-style sentences.
- After `faq_tool` and `user_extract_info`, answer using exactly 3 bullet lines.
- When using `mechanic_agent` and `booking_agent`, relay its output verbatim.
"""

class UserInfoContext(BaseModel):
    user_memory: User
    model_config = {"arbitrary_types_allowed": True}

class MechaniGoContext(BaseModel):
    user_ctx: UserInfoContext
    model_config = {"arbitrary_types_allowed": True}

class OutputModelSettings(BaseModel):
    max_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int

class ChatbotResponse(BaseModel):
    response: str
    model: str
    model_settings: OutputModelSettings
    usage: Usage
    history_items: List[TResponseInputItem]


class MechaniGoAgent(AgentFactory):
    """
    Manager agent for MechaniGo PH bot that wires default instructions, tools, and guardrails.
    """
    def __init__(
        self,
        api_key: str,
        name: Optional[str] = "MechaniGo Bot",
        model: Optional[str] = settings.OPENAI_MODEL,
        description: Optional[str] = None,
        max_tokens: Optional[int] = settings.OPENAI_MAX_TOKENS,
        temperature: Optional[float] = settings.MAIN_AGENT_TEMPERATURE,
        session: Optional[SessionHandler] = None,
        user_id: Optional[str] = None,
        context: Optional[MechaniGoContext] = None
    ):
        """
        Creates a new MechaniGo Agent.
        
        :param self: Instance.
        :param api_key: OpenAI API key for the agent.
        :type api_key: str
        :param name: Display name used in prompts; defaults to "MechaniGo Bot".
        :type name: Optional[str]
        :param model: Model used for the LLM; defaults to "gpt-4.1" (set in `settings.OPENAI_MODEL`).
        :type model: Optional[str]
        :param description: Short handoff description shown when delegating to this agent.
        :type description: Optional[str]
        :param max_tokens: Max tokens for response; defaults to `500` (set in `settings.MAX_TOKENS`).
        :type max_tokens: Optional[int]
        :param temperature: Sampling temperature; defaults to `0.2` (set in `settings.MAIN_AGENT_TEMPERATURE`).
        :type temperature: Optional[float]
        :param session: Session store for chat history.
        :type session: Optional[SessionHandler]
        :param user_id: Unique user identifier for context and memory.
        :type user_id: Optional[str]
        :param context: Prebuild context; created if not provided.
        :type context: Optional[MechaniGoContext]
        """
        super().__init__(api_key=api_key)
        self.name = name
        self.model = model
        self.description = description
        self.instructions = INSTRUCTIONS
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = session
        self.user_id = user_id
        self.agent = None # agent instance
        self._context = context or None

    @property
    def context(self):
        if self._context is None:
            self._context = MechaniGoContext(
                user_ctx=UserInfoContext(user_memory=User(uid=self.user_id))
            )
        return self._context

    def get_model(self) -> str:
        return self.model

    def get_name(self) -> str:
        return self.name

    def get_handoff_description(self) -> str:
        return self.description

    def get_instructions(self) -> str:
        return self.instructions.format(name=self.get_name())
    
    def get_tools(self):
        return [
            ToolRegistry.get_tool("booking_agent"),
            ToolRegistry.get_tool("mechanic_agent"),
            ToolRegistry.get_tool("knowledge.faq_tool")
        ]

    def get_input_guardrails(self):
        return [mechanigo_guardrail]

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

    def builder(self):
        self.agent = super().build()
        return self.agent

    async def inquire(self, inquiry: str) -> ChatbotResponse:
        """
        Run the manager agent against a user inquiry.
        
        :param inquiry: Raw user message to process.
        :type inquiry: str
        :return: Final agent output plus model info, token usage, and the collected history items.
        :rtype: ChatbotResponse


        Notes
        -----
        Builds the agent if needed, executes via Runner with session/context, stores new history,
        and surfaces token usage from the first raw response.
        """
        response = await Runner.run(
            starting_agent=self.build(),
            input=inquiry,
            context=self.context,
            session=self.session
        )

        new_history_items = response.raw_responses[0].to_input_items()
        await self.session.collect_items(new_history_items)
        return ChatbotResponse(
            response=response.final_output,
            model=self.get_model(),
            model_settings=OutputModelSettings(
                max_tokens=self.agent.model_settings.max_tokens
            ),
            usage=Usage(
                input_tokens=response.raw_responses[0].usage.input_tokens,
                output_tokens=response.raw_responses[0].usage.output_tokens,
                total_tokens=response.raw_responses[0].usage.total_tokens,
            ),
            history_items=new_history_items
        )