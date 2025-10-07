from components.agent_tools import UserInfoAgent, UserInfoAgentContext
from components.agent_tools import MechanicAgent, MechanicAgentContext
from components.utils import create_agent, BigQueryClient
from config import (
    DEFAULT_AGENT_HANDOFF_DESCRIPTION,
    DEFAULT_AGENT_INSTRUCTIONS
)
from schemas import User, UserCarDetails
from pydantic import BaseModel
from typing import Optional
from agents import Runner
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
        self.instructions = DEFAULT_AGENT_INSTRUCTIONS
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

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.handoff_description,
            instructions=self.instructions,
            model=self.model,
            tools=[user_info_agent.as_tool, mechanic_agent.as_tool]
        )
        self.logger = logging.getLogger(__name__)

    async def inquire(self, inquiry: str):
        response = await Runner.run(
            starting_agent=self.agent,
            input=inquiry,
            context=self.context
        )
        return response.final_output