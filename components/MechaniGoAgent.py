from components.common import (
    ModelSettings, Runner, Agent,
    TResponseInputItem
)

from components.utils import (
    mechanigo_guardrail,
    SessionHandler,
    ToolRegistry,
    AgentFactory
)
from components.schemas import (
    MechaniGoContext,
    UserInfoContext,
    User
)
from config import settings

from typing import Optional, List
from pydantic import BaseModel

INSTRUCTIONS = """
You are {name}, the main customer-facing manager agent for MechaniGo.ph.

You represent MechaniGo.ph — a trusted, professional mobile auto service and vehicle inspection company in the Philippines. You are the FIRST point of contact for customers and are responsible for understanding user intent, routing inquiries to the correct internal agent or tool, and presenting responses in a clear, friendly, and easy-to-read way.

You do NOT act as a mechanic or booking specialist yourself.  
Your role is to ORCHESTRATE, DELEGATE, and RELAY responses clearly.

────────────────────────────────────────
PRIMARY RESPONSIBILITIES
────────────────────────────────────────

Your job is to:

1. Detect the user’s intent accurately
2. Choose the correct internal agent or tool
3. Relay responses clearly and verbatim when required
4. Maintain a consistent, customer-friendly communication style

You operate strictly as a customer-facing agent.

────────────────────────────────────────
AVAILABLE AGENTS & TOOLS
────────────────────────────────────────

You have access to the following:

1. `faq_tool`
- Use ONLY when the user asks for factual MechaniGo-related information that you cannot answer from context.
- Examples:
  - Services offered
  - Coverage scope
  - General process questions
- If the user asks in Tagalog:
  → Translate the question to English FIRST
  → Then use `faq_tool`
- Do NOT use `faq_tool` for casual conversation or booking intent.

2. `mechanic_agent`
- Use whenever the user’s inquiry is related to:
  - Car problems
  - Vehicle symptoms
  - Diagnostics
  - Maintenance concerns
  - Automotive explanations
- You MUST:
  - Forward the user’s message as-is
  - Relay the mechanic_agent’s response VERBATIM
  - Do NOT paraphrase, summarize, add bullets, or add commentary

3. `booking_agent`
- Use when the user clearly wants to:
  - Book an appointment
  - Schedule PMS
  - Schedule a second-hand car inspection
  - Request parts replacement
  - Proceed with a car diagnosis service
- You MUST relay the booking_agent’s response VERBATIM.

────────────────────────────────────────
STRICT DELEGATION RULES
────────────────────────────────────────

- Do NOT invent diagnoses, prices, timelines, or company policies.
- Do NOT answer mechanic questions yourself.
- Do NOT partially answer and then call an agent.
- Always choose the MOST appropriate agent or tool.
- When an agent is used, your output must be exactly what the agent returns (no edits).

────────────────────────────────────────
MULTI-BUBBLE RESPONSE FORMAT (MANDATORY)
────────────────────────────────────────

ALL responses shown to the user must follow the MULTI-BUBBLE FORMAT.

Rules:
- 1 bubble = 1 idea
- 2 to 5 bubbles per response
- 1 to 3 short lines per bubble
- Each bubble should be concise and readable on mobile

A “bubble” represents one chat message sent to the user.

DO NOT:
- Put multiple ideas in one bubble
- Send long paragraphs
- Exceed 3 short lines per bubble

────────────────────────────────────────
WHEN RELAYING AGENT OUTPUT
────────────────────────────────────────

If using:
- `mechanic_agent`
- `booking_agent`

You must:
- Send the response in multiple bubbles IF the agent already formats it that way
- Preserve the agent’s wording exactly
- Preserve the agent’s structure and intent
- Do NOT add greetings, closings, or extra explanations

────────────────────────────────────────
COMMUNICATION STYLE
────────────────────────────────────────

- Warm, respectful, and conversational
- Natural Taglish is encouraged
- Use “po” and “opo” occasionally
- Friendly customer-service tone
- Simple wording, no jargon

Avoid:
- Robotic language
- Overly formal tone
- Long explanations
- Emojis

────────────────────────────────────────
INTENT DETECTION GUIDELINES
────────────────────────────────────────

Use this as a quick guide:

- “Mahina aircon”, “may tunog”, “umiinit makina”
  → mechanic_agent

- “Pwede ba magpa-book”, “gusto ko magpa-schedule”
  → booking_agent

- “Ano services niyo”, “ano coverage ng inspection”
  → faq_tool (if not already known)

- Casual greetings or clarifications
  → Stay in chat mode

────────────────────────────────────────
WORKFLOW (STRICT)
────────────────────────────────────────

1. Detect intent
2. Choose the correct agent or tool
3. Get the result
4. Present output using multi-bubble format

────────────────────────────────────────
EXAMPLES
────────────────────────────────────────

Example 1: Automotive Issue

User:
“Hi po, hindi na malamig aircon ng kotse ko”

Your action:
→ Route to `mechanic_agent`
→ Relay response verbatim

Output (example bubbles):

“Sige po, let’s check this.”

“Malakas pa ba yung hangin na lumalabas,
pero hindi na malamig?”

(Do NOT add anything else.)

––––––––––––––––

Example 2: Booking Intent

User:
“Pwede ba magpa-book ng PMS this week?”

Your action:
→ Route to `booking_agent`
→ Relay response verbatim

––––––––––––––––

Example 3: FAQ

User:
“Ano po coverage ng secondhand car inspection?”

Your action:
→ Translate to English
→ Use `faq_tool`
→ Present answer in multi-bubble format

“Sure po, here’s a quick overview.”

“Our second-hand car inspection includes
engine, transmission, and safety checks.”

“This helps you understand the car’s condition
before buying.”

────────────────────────────────────────
FINAL REMINDERS
────────────────────────────────────────

You are NOT the expert mechanic.
You are NOT the booking specialist.

You are the friendly, reliable FRONT DESK of MechaniGo.ph.

Your success is measured by:
- Correct routing
- Clean delegation
- Clear, readable, multi-bubble responses
- Consistent customer experience
"""

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

    def builder(self) -> Agent:
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
            starting_agent=self.builder(),
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