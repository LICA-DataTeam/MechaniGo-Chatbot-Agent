from components.utils import create_agent, BigQueryClient
from agents import function_tool
from schemas import User
import traceback
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class UserInfoAgent:
    """Handles all user info related processing."""
    def __init__(
        self,
        api_key: str,
        bq_client: BigQueryClient = None,
        table_name: str = "mgo_chatbot_users",
        name: str = "user_info_agent",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key
        self.bq_client = bq_client
        self.table_name = table_name
        self.name = name
        self.model = model
        self.description = "Handles processing of user information."
        self.logger = logging.getLogger(__name__)

        extract_tool = self._create_extract_user_tool()

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=(
                "You are a user information extraction and storage agent.\n\n"
                "ABSOLUTE REQUIREMENT:\n"
                "When you receive ANY user information (even partial), you MUST IMMEDIATELY call "
                "the extract_user_info function to save it. Do NOT wait for complete information. "
                "Do NOT ask for more details before saving. Save first, then ask for missing details.\n\n"
                "WORKFLOW (MANDATORY):\n"
                "1. Extract whatever user information is available from the message\n"
                "2. IMMEDIATELY call extract_user_info() with the extracted data (even if partial)\n"
                "3. AFTER saving, you may ask for missing information\n\n"
                "EXAMPLES:\n"
                "❌ WRONG: 'Thank you. Please provide schedule date...'\n"
                "✅ CORRECT: Call extract_user_info(User(name='Walter', address='308...')) FIRST, "
                "then say 'Information saved. Please provide schedule date...'\n\n"
                "REMEMBER: You MUST call extract_user_info for EVERY message that contains user data. "
                "Partial data is acceptable - save what you have!\n\n"
                "Show the error logs if any, ex: if you can't save the information due to a technical issue.\n\n"
                "IMPORTANT: Do not attempt to guess or extract the uid, the uid is always generated internally by the system."
            ),
            model=self.model,
            tools=[extract_tool]
        )

    @property
    def as_tool(self):
        return self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )

    def _create_extract_user_tool(self):
        @function_tool
        def extract_user_info(user: User):
            return self._save_user(user)

        return extract_user_info

    def _save_user(self, user: User):
        try:
            if self.bq_client:
                if not user.uid:
                    user.uid = str(uuid.uuid4())
                self.logger.info(f"Inserting user {user.name} to {self.table_name}...")
                self.bq_client.insert_user(self.table_name, user)
                return {
                    "status": "success",
                    "message": f"User information for {user.name} extracted and saved to BigQuery.",
                    "user": user
                }
            else:
                self.logger.error("BigQuery client not configured!")
                return {
                    "status": "success",
                    "message": "User information extracted.",
                    "user": user
                }
        except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Failed to save user information: {str(e)}",
                "user": user
            }