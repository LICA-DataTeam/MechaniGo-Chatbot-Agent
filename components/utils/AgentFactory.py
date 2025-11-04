from __future__ import annotations

from typing import Any, Iterable, Optional

from agents import Agent
from components.utils.registry import guardrail_registry, tool_registry
import openai


def _resolve_tools(
    tools: Optional[Iterable[Any]] = None,
    tool_names: Optional[Iterable[str]] = None,
) -> list[Any]:
    resolved: list[Any] = []
    if tools:
        resolved.extend(tools)
    if tool_names:
        resolved.extend(tool_registry.get_many(tool_names))
    return resolved


def _resolve_guardrails(
    guardrails: Optional[Iterable[Any]] = None,
    guardrail_names: Optional[Iterable[str]] = None,
) -> list[Any]:
    resolved: list[Any] = []
    if guardrails:
        resolved.extend(guardrails)
    if guardrail_names:
        resolved.extend(guardrail_registry.get_many(guardrail_names))
    return resolved


def create_agent(
    api_key: str,
    name: str,
    handoff_description: str,
    instructions: Any,
    output_type: Optional[Any] = None,
    model: Optional[str] = None,
    tools: Optional[Iterable[Any]] = None,
    tool_names: Optional[Iterable[str]] = None,
    input_guardrails: Optional[Iterable[Any]] = None,
    guardrail_names: Optional[Iterable[str]] = None,
) -> Agent:
    openai.api_key = api_key
    resolved_tools = _resolve_tools(tools, tool_names)
    resolved_guardrails = _resolve_guardrails(input_guardrails, guardrail_names)

    return Agent(
        name=name,
        handoff_description=handoff_description,
        instructions=instructions,
        output_type=output_type,
        model=model,
        tools=resolved_tools,
        input_guardrails=resolved_guardrails,
    )
