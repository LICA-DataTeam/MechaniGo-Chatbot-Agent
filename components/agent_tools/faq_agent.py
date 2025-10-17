from components.utils import create_agent
from agents import function_tool
from openai import OpenAI
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class FAQAgent:
    """Handles all FAQs related inquiries."""
    def __init__(
        self,
        api_key: str,
        name: str = "faq_agent",
        model: str = "gpt-4o-mini"
    ):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.name = name
        self.model = model
        self.description = "Handles FAQs for MechaniGo.ph."

        ask = self._create_ask()
        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=(
                f"You are {self.name}, responsible for answering faqs for MechaniGo.ph\n\n"
                "You have access to a file search tool connected to the FAQ knowledge base.\n\n"
                "Always use `extract_faq_from_json` to answer FAQs.\n\n"
                "IMPORTRANT RULES:\n\n"
                "- Always quote the official answer exactly as written in the FAQ file.\n\n"
                "- DO NOT summarize the official answer from the FAQs file.\n\n"
                "- If you cannot find a relevant FAQ, suggest contacting support (in the user's language if you wish).\n\n"
            ),
            model=self.model,
            tools=[ask]
        )

    @property
    def as_tool(self):
        return self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )

    def _create_ask(self):
        @function_tool
        def ask(question: str):
            return self._ask(question)
        return ask

    def _ask(self, question: str):
        self.logger.info("========== _ask() Called! ==========")

        try:
            client = OpenAI()
            with open("vector_store_id.json") as f:
                vector_store_id = json.load(f)["id"]

            response = client.responses.create(
                model=self.model,
                input=question,
                tools=[
                    {"type": "file_search", "vector_store_ids": [vector_store_id]}
                ],
                max_tool_calls=5
            )

            self.logger.info(f"'_ask()' output: {response.model_dump()}")
            text = response.output_text or ""
            if "issue retrieving" in text.lower():
                self.logger.warning("FAQ fallback triggered; check vector store.")
            return response.output_text
        except Exception as e:
            self.logging.error(f"Exception occurred while answering FAQs: {e}")