from agents import Agent, Runner
from dotenv import load_dotenv
import openai
import logging
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_agent(
    api_key: str,
    name: str,
    handoff_description: str,
    instructions: str,
    model: str
):
    openai.api_key = api_key
    return Agent(
        name=name,
        handoff_description=handoff_description,
        instructions=instructions,
        model=model
    )


class RunnerWrapper:
    @staticmethod
    async def run(agent: Agent, inquiry: str):
        return await Runner.run(agent, inquiry)


class MechaniGoAgent:
    def __init__(
        self,
        api_key: str = None,
        name: str = None,
        handoff_description: str = None,
        instructions: str = None,
        model: str = "gpt-4o-mini"
    ):
        self.api_key = OPENAI_API_KEY if api_key is None else api_key
        self.name = "MechaniGo Assistant" if name is None else name
        self.model = model
        self.handoff_description = handoff_description
        self.instructions = instructions
        self.agent = create_agent(
            self.api_key,
            self.name,
            self.handoff_description,
            self.instructions,
            self.model
        )
        self.runner = RunnerWrapper
        self.logger = logging.getLogger(__name__)

    async def inquire(self, inquiry: str):
        response = await self.runner.run(self.agent, inquiry)
        return response.final_output