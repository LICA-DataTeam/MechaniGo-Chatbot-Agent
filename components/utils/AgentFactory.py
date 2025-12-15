from components.common import Agent, ModelSettings, openai
from typing import Optional, List, Literal, Iterable, Any
from abc import ABC, abstractmethod

def build_agent(
    api_key: str,
    name: str,
    handoff_description: str,
    instructions: str,
    output_type: Optional[Any] = None,
    model: Optional[str] = None,
    tools: Optional[Iterable[Any]] = None,
    model_settings: ModelSettings = None,
    tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] = "run_llm_again",
    input_guardrails: Optional[List[Any]] = None
) -> Agent:
    """
    Build and configure an Agent with the given OpenAI settings, tools, and behaviors.
    
    :param api_key: OpenAI API key used for requests.
    :type api_key: str
    :param name: Display name for the Agent.
    :type name: str
    :param handoff_description: Short description shown when control is handed to this Agent.
    :type handoff_description: str
    :param instructions: System prompt or core instructions for the Agent.
    :type instructions: str
    :param output_type: Expected output schema or parser; defaults to None.
    :type output_type: Optional[Any]
    :param model: LLM Model used; falls back to `ModelSettings`.
    :type model: Optional[str]
    :param tools: Iterable of tool definitions available to the Agent.
    :type tools: Optional[Iterable[Any]]
    :param model_settings: Model configuration; created with defaults if omitted.
    :type model_settings: ModelSettings
    :param tool_use_behavior: Strategy for handling tool calls; defaults to `run_llm_again`.
    :type tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"]
    :param input_guardrails: Input validation/filtering guards; defaults to None.
    :type input_guardrails: Optional[List[Any]]
    :return: Configured Agent.
    :rtype: Agent[Any]
    """
    openai.api_key = api_key
    if model_settings is None:
        model_settings = ModelSettings()
    return Agent(
        name=name,
        handoff_description=handoff_description,
        instructions=instructions,
        output_type=output_type,
        model=model,
        tools=tools,
        tool_use_behavior=tool_use_behavior,
        input_guardrails=input_guardrails,
        model_settings=model_settings
    )


class AgentFactory(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def get_model(self) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_handoff_description(self) -> str:
        pass

    @abstractmethod
    def get_instructions(self) -> str:
        pass

    @abstractmethod
    def get_tools(self) -> Optional[Iterable[Any]]:
        pass

    @abstractmethod
    def get_input_guardrails(self) -> Optional[List[Any]]:
        return []

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings()

    def get_output_type(self) -> Optional[Any]:
        return None
    
    def get_tool_use_behavior(self) -> Literal["run_llm_again", "stop_on_first_tool"]:
        return "run_llm_again"

    def build(self) -> Agent:
        """
        Agent builder method.
        
        :return: Configured agent.
        :rtype: Agent[Any]
        """
        return build_agent(
            api_key=self.api_key,
            name=self.get_name(),
            handoff_description=self.get_handoff_description(),
            instructions=self.get_instructions(),
            output_type=self.get_output_type(),
            model=self.get_model(),
            tools=self.get_tools(),
            tool_use_behavior=self.get_tool_use_behavior(),
            model_settings=self.get_model_settings(),
            input_guardrails=self.get_input_guardrails()
        )