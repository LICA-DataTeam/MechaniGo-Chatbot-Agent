from components.utils import create_agent, BigQueryClient
from components.agent_tools import UserInfoAgent
from config import (
    DEFAULT_AGENT_HANDOFF_DESCRIPTION,
    DEFAULT_AGENT_INSTRUCTIONS
)
from agents import Agent, Runner
import openai
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RunnerWrapper:
    @staticmethod
    async def run(agent: Agent, inquiry: str):
        return await Runner.run(agent, inquiry)

class MechaniGoAgent:
    """Serves as the general agent and customer facing agent."""
    def __init__(
        self,
        api_key: str = None,
        bq_client: BigQueryClient = None,
        name: str = "MechaniGo Bot",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key or openai.api_key or None
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via environment variables.")

        self.bq_client = bq_client
        self.name = name
        self.handoff_description = DEFAULT_AGENT_HANDOFF_DESCRIPTION
        self.instructions = DEFAULT_AGENT_INSTRUCTIONS
        self.model = model

        user_info_agent = UserInfoAgent(api_key=self.api_key, bq_client=self.bq_client, model=self.model)

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.handoff_description,
            instructions=self.instructions,
            model=self.model,
            tools=[user_info_agent.as_tool]
        )
        self.runner = RunnerWrapper
        self.logger = logging.getLogger(__name__)

    async def inquire(self, inquiry: str):
        response = await self.runner.run(self.agent, inquiry)
        return response.final_output