from components.utils import create_agent
from agents import function_tool
from schemas import User

class UserInfoAgent:
    """Handles all user info related processing."""
    def __init__(
        self,
        api_key: str,
        name: str = "user_info_agent",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key
        self.name = name
        self.model = model
        self.description = "Handles processing of user information."

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions="You extract the necessary user information from the conversation.",
            model=self.model,
            tools=[self.extract_user_info]
        )

    @property
    def as_tool(self):
        return self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )

    @function_tool
    def extract_user_info(user: User):
        return {
            "status": "success",
            "user": user.model_dump()
        }