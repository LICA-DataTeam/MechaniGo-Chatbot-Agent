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

def create_agent(
    api_key: str,
    name: str,
    handoff_description: str,
    instructions: str,
    model: str,
    tools: list = None
):
    openai.api_key = api_key
    return Agent(
        name=name,
        handoff_description=handoff_description,
        instructions=instructions,
        model=model,
        tools=tools or []
    )


class RunnerWrapper:
    @staticmethod
    async def run(agent: Agent, inquiry: str):
        return await Runner.run(agent, inquiry)


class MechaniGoAgent:
    def __init__(
        self,
        api_key: str = None,
        name: str = "MechaniGo Assistant",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key or openai.api_key or None
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via environment variables.")

        self.name = name
        self.handoff_description = DEFAULT_AGENT_HANDOFF_DESCRIPTION
        self.instructions = DEFAULT_AGENT_INSTRUCTIONS
        self.model = model
        self.agent = create_agent(
            self.api_key,
            self.name,
            self.handoff_description,
            self.instructions,
            self.model
        )
        self.runner = RunnerWrapper
        self.logger = logging.getLogger(__name__)

    # tools
    
    async def inquire(self, inquiry: str):
        response = await self.runner.run(self.agent, inquiry)
        return response.final_output