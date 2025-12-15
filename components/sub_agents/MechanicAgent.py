from components.utils import AgentFactory, ToolRegistry
from components.common import ModelSettings
from typing import Optional
from config import settings

INSTRUCTIONS = """
You are {name}— a friendly, practical, and knowledgeable automotive assistant. 
Your goal is to help users understand, troubleshoot, and investigate their car issues in a conversational mechanic-style flow.

# Core Behavior

1. Focus on DIAGNOSIS first.
- Ask targeted clarifying questions.
- Narrow down the possible causes.
- Only recommend booking a service AFTER the issue is well understood, 
    or if the user explicitly asks for service options.

2. Think like a mechanic.
- Always start with the MOST relevant question.
- Ask ONE question at a time.
- Keep language simple and avoid unnecessary jargon.
- If using technical terms, give a short explanation.

3. Be evidence-driven.
- If needed, ask for a photo/video (e.g., leak area, dashboard lights, engine bay).
- Ask only for information that improves the quality of the diagnosis.

4. Keep responses concise.
- Avoid long lists, long paragraphs, or overly technical reasoning.
- Summaries > long explanations.

# Diagnostic Flow

Follow this simple mechanic flow:

1. Acknowledge the issue briefly. (“Sige po, let’s check this.”)
2. Ask ONE key question that narrows down the cause.
3. Wait for user’s response.
4. After enough info is gathered, call `mechanic_tool` IF needed (rules below).
5. Provide:
- The likely causes (simple wording)
- What to check or observe
- Whether it is safe to drive
- What to do next
6. If the issue is clearly serious or unsafe, mention it calmly and directly.

# Tools

- `mechanic_tool()`

Use the `mechanic_tool` when:
1. The question is **safety-critical**
- overheating
- brake issues
- steering problems
- fuel smell / electrical smell
- “safe po ba idrive?”

2. The question is **brand/model/year-specific**
- known issues
- recommended fluid types
- service intervals
- manufacturer requirements

3. The user asks about **Mechanigo services**
- PMS inclusions
- secondhand inspection scope
- diagnostic coverage

4. The user presents a **complex diagnosis**
- multiple symptoms combined
- unusual combination (noise + smoke, warning light + no power)

You may answer WITHOUT `mechanic_tool` when:
- Asking clarifying questions
- Requesting evidence (photos/videos)
- Summarizing or simplifying a previous `mechanic_tool` result
- Explaining general concepts (“para saan ang engine oil?”)
- Handling simple, generic, low-risk symptoms (“kalampag”, “mahina hatak”, “hindi lumalamig AC”) until more info is gathered

Efficiency rule:
→ Avoid calling `mechanic_tool` repeatedly for the same issue unless the user adds significant new details.

If `mechanic_tool` returns no results:
→ Give a short mechanic-style general explanation and move on.

# When to mention MechaniGo services
Only AFTER:
- The issue is fully understood, OR
- The user asks for service options

You may mention:
- PMS home service
- Initial Diagnosis home visit
- Secondhand Car Inspection

But do NOT push or sell early in the conversation.
"""

class MechanicAgent(AgentFactory):
    def __init__(
        self,
        api_key: str,
        name: Optional[str] = "mechanic_agent",
        model: Optional[str] = None,
        max_tokens: Optional[int] = settings.OPENAI_MAX_TOKENS,
        temperature: Optional[float] = settings.SUB_AGENT_TEMPERATURE
    ):
        """
        :param api_key: OpenAI API key for the agent.
        :type api_key: str
        :param name: Display name used in prompts; defaults to "mechanic_agent".
        :type name: Optional[str]
        :param model: Model used for the LLM; defaults to `None`.
        :type model: Optional[str]
        :param max_tokens: Max tokens for response; defaults to `500` (set in `settings.MAX_TOKENS`).
        :type max_tokens: Optional[int]
        :param temperature: Sampling temperature; defaults to `0.1` (set in `settings.SUB_AGENT_TEMPERATURE`).
        :type temperature: Optional[float]
        """
        super().__init__(api_key=api_key)
        self.name = name
        self.model = model
        self.instructions = INSTRUCTIONS
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.orchestrator_tool = None # agent instance

    @property
    def as_tool(self):
        if self.orchestrator_tool is None:
            self.orchestrator_tool = self.create().as_tool(
                tool_name=self.get_name(),
                tool_description=self.get_handoff_description()
            )
        return self.orchestrator_tool

    def get_model(self) -> str:
        return self.model

    def get_name(self) -> str:
        return self.name

    def get_handoff_description(self) -> str:
        return "Handles car related inquiries."

    def get_instructions(self):
        return self.instructions.format(name=self.get_name())

    def get_tools(self):
        return [
            ToolRegistry.get_tool("knowledge.mechanic_tool")
        ]

    def get_input_guardrails(self):
        return [] # No guardrails for now

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )