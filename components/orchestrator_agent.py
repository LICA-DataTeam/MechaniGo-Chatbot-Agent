from components.agent_tools import (
    UserInfoAgent, UserInfoAgentContext,
    MechanicAgent, MechanicAgentContext,
    FAQAgent
)
from components.utils import create_agent, BigQueryClient
from config import DEFAULT_AGENT_HANDOFF_DESCRIPTION
from agents import Agent, Runner, RunContextWrapper
from schemas import User, UserCarDetails
from pydantic import BaseModel
from typing import Optional
import openai
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TABLE_NAME = "chatbot_users_table"

class MechaniGoContext(BaseModel):
    user_ctx: UserInfoAgentContext
    mechanic_ctx: MechanicAgentContext

    model_config = {
        "arbitrary_types_allowed": True
    }

class MechaniGoAgent:
    """Serves as the general agent and customer facing agent."""
    def __init__(
        self,
        api_key: str = None,
        bq_client: BigQueryClient = None,
        name: str = "MechaniGo Bot",
        model: str = "gpt-4o-mini",
        context: Optional[MechaniGoContext] = None
    ):
        self.api_key = api_key or openai.api_key or None
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via environment variables.")

        self.bq_client = bq_client
        self.name = name
        self.handoff_description = DEFAULT_AGENT_HANDOFF_DESCRIPTION
        self.model = model

        if not context:
            context = MechaniGoContext(
                user_ctx=UserInfoAgentContext(
                    user_memory=User(uid=str(uuid.uuid4())),
                    bq_client=self.bq_client,
                    table_name=TABLE_NAME
                ),
                mechanic_ctx=MechanicAgentContext(
                    car_memory=UserCarDetails()
                )
            )

        self.context = context
        user_info_agent = UserInfoAgent(
            api_key=self.api_key,
            bq_client=self.bq_client,
            table_name=TABLE_NAME,
            model=self.model
        )

        mechanic_agent = MechanicAgent(
            api_key=self.api_key
        )

        faq_agent = FAQAgent(
            api_key=self.api_key
        )

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.handoff_description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tools=[user_info_agent.as_tool, mechanic_agent.as_tool, faq_agent.as_tool]
        )
        self.logger = logging.getLogger(__name__)
    
    async def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[MechaniGoContext],
        agent: Agent
    ):
        user_name = ctx.context.user_ctx.user_memory.name or "Unknown User"

        car = ctx.context.user_ctx.user_memory.car
        car_summary = f"{car or ''}"

        self.logger.info("========== DETAILS ==========")
        self.logger.info(f"User name: {user_name}")
        self.logger.info(f"Car summary: {car_summary}")
        prompt = (
            f"You are {agent.name}, the main orchestrator agent and a helpful assistant for MechaniGo.ph, a business that offers home maintenance (PMS) and car-buying assistance.\n\n"
            "You are the customer-facing agent which handles responding to customer inquiries.\n\n"
            f"The current user is {user_name}.\n\n"
            f"Their car is {car_summary}."
            "Your job is to use the tools given to you to accomplish your tasks:\n\n"
            "- Use user_info_agent for user info related tasks\n\n"
            "- Use mechanic_agent for car-related inquiries\n\n"
            "- Do not attempt to solve the tasks directly; always use the tools to accomplish the tasks.\n\n"
            "- Provide a clear and concise response back to the customer.\n\n"
        )
        return prompt

    async def inquire(self, inquiry: str):
        response = await Runner.run(
            starting_agent=self.agent,
            input=inquiry,
            context=self.context
        )
        return response.final_output