from typing import Optional, Any
from agents import Agent
import openai

def create_agent(
    api_key: str,
    name: str,
    handoff_description: str,
    instructions: str,
    output_type: Optional[Any] = None,
    model: str = None,
    tools: list = None,
    input_guardrails: list = None
):
    openai.api_key = api_key
    return Agent(
        name=name,
        handoff_description=handoff_description,
        instructions=instructions,
        output_type=output_type,
        model=model,
        tools=tools or [],
        input_guardrails=input_guardrails or []
    )