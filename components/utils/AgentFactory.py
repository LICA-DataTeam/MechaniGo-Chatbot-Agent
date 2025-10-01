from agents import Agent
import openai

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