# Current tasks:
# MechanicAgent
# - web search: car-diagnosis and troubleshooting
#     - https://www.carparts.com/blog/auto-repair/
# - file search: car-diagnosis (cms blog post - vector store)
# FAQAgent
# - file search: vector store (faqs.json)
from agents import Agent, Runner, RunContextWrapper, SQLiteSession
from config import DEFAULT_AGENT_HANDOFF_DESCRIPTION
from components.agent_tools import (
    MechanicAgent, MechanicAgentContext,
    FAQAgent
)
from components.utils import create_agent
from schemas import UserCarDetails
from pydantic import BaseModel
from typing import Optional
import openai
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MechaniGoContext(BaseModel):
    mechanic_ctx: MechanicAgentContext
    model_config = {
        "arbitrary_types_allowed": True
    }

class MechaniGoAgent:
    """Serves as the general agent and customer facing agent."""
    def __init__(
        self,
        api_key: str = None,
        name: str = "MechaniGo Bot",
        model: str = "gpt-4.1",
        context: Optional[MechaniGoContext] = None,
        input_guardrail: Optional[list] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or openai.api_key or None
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided.")
        
        self.name = name
        self.handoff_description = DEFAULT_AGENT_HANDOFF_DESCRIPTION
        self.session = SQLiteSession(session_id=str(uuid.uuid4()), db_path="conversations.db")
        self.model = model
        self.input_guardrail = input_guardrail

        if not context:
            context = MechaniGoContext(
                mechanic_ctx=MechanicAgentContext(
                    car_memory=UserCarDetails()
                )
            )

        self.context = context
        self.mechanic_agent = MechanicAgent(api_key=self.api_key)
        self.faq_agent = FAQAgent(api_key=self.api_key)

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.handoff_description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tool_names=("mechanic_agent", "faq_agent"),
            input_guardrails=self.input_guardrail,
            guardrail_names=("input_generic",),
        )

    async def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[MechaniGoContext],
        agent: Agent
    ):
        self.logger.info("========== orchestrator_agent called! ==========")
        prompt = (
            f"You are {agent.name}, the main orchestrator agent and a helpful assistant for MechaniGo.ph.\n"
            "FAQ HANDLING:\n"
            " - If the user asks a general MechaniGo question (e.g., location, hours, pricing, services) — especially at the very start — immediately use faq_agent to answer.\n"
            " - After responding, return to the service flow to answer more inquiries.\n\n"
            "MechanicAgent HANDLING:\n"
            "- If the user has a car related issue or question, always call mechanic_agent.\n"
            "- After responding, optionally ask their car detail.\n\n"
            " - It has its own internal lookup tool that can answer any car related inquiries.\n"
            " - It can search the web and use a file-based vector store to answer car-related questions, including topics like diagnosis and maintenance.\n"
            " - ALWAYS use the output of mechanic_agent when answering car related inquiries.\n"
            " - If mechanic_agent does not return any relevant information, use your own knowledge base/training data as a LAST RESORT.\n"
            "TOOLS AND WHEN TO USE THEM:\n"
            "- mechanic_agent:\n"
            " - When the user seeks assistance/questions for any car related issues, diagnostic, troubleshooting, etc.(e.g., 'My car's engine is smoking, can you assist me?')\n"
            " - Whenever the user updates car details (even in free text), parse them and call mechanic_agent.\n"
            " - It can parse a free-form car string into make/model/year.\n"
            " - After a successful extraction of car information, summarize the saved fields.\n"
            " - **Do not reset the topic** or ask what they want to ask again if an issue was already provided earlier.\n"
            "   - Example:\n"
            "       - User: 'My aircon is getting warmer.'\n"
            "       - You: 'Can I get your car details?'\n"
            "       - User: 'Ford Everest 2015.'\n"
            "       - After mechanic_agent returns car info, respond like: 'Got it—Ford Everest 2015. Since you mentioned your aircon is getting warmer, here’s what we can check…'\n"
            "- faq_agent:\n"
            " - Use to answer official FAQs. Quote the official answer.\n"
            " - After answering, continue the flow.\n\n"
            "MEMORY AND COMPLETENESS:\n"
            " - Check what's already in memory and avoid re-asking.\n"
            " - Always retain and reference the customer’s last described problem or issue (e.g., 'engine light on', 'aircon not cooling', 'strange noise').\n"
            " - Check what's already in memory and avoid re-asking questions unnecessarily.\n"
            " - Maintain continuity between tool calls. The customer should feel like the conversation flows naturally without restarting.\n\n"
            "SCOPE:\n"
            "Currently, you only handle two agents: mechanic_agent and faq_agent. You only need to answer a customer's general inquiries about MechaniGo (FAQs) and car-related questions (e.g., PMS, diagnosis and troubleshooting).\n"
            "If they ask about booking related questions (i.e., they want to book an appointment for PMS or secondhand car-buying), let them know you cannot assist them with that yet. You can only handle car-diagnosis and MechaniGo FAQs.\n"
            "COMMUNICATION STYLE:\n"
            "- Always introduce yourself to customers cheerfully and politely.\n"
            "- Be friendly, concise, and proactive.\n"
            "- The customer may speak in English, Filipino, or a mix of both. Expect typos and slang.\n"
            "- Use a mix of casual and friendly Tagalog and English as appropriate in a cheerful and polite conversational tone, occasionally using 'po' to show respect, regardless of the customer's language.\n"
            "- Summarize updates after each tool call so the user knows what's saved.\n\n"
            "- If the user asks FAQs at any point → use faq_agent, then resume this flow.\n"
            "- Only call a sub-agent if it will capture missing information or update fields the user explicitly changed.\n"
            "- If a tool returns no_change, do not call it again this turn.\n"
        )
        return prompt

    async def inquire(self, inquiry: str):
        response = await Runner.run(
            starting_agent=self.agent,
            input=inquiry,
            context=self.context,
            session=self.session
        )
        return response.final_output