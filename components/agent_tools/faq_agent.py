from __future__ import annotations

import logging
import os

from agents import function_tool
from components.utils import create_agent, register_tool
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LOGGER = logging.getLogger(__name__)
FAQ_VECTOR_STORE_ID = os.getenv("FAQ_VECTOR_STORE_ID")


class FAQAgent:
    """Handles all FAQs related inquiries."""

    def __init__(
        self,
        api_key: str,
        name: str = "faq_agent",
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.name = name
        self.model = model
        self.description = "Handles FAQs for MechaniGo.ph."
        self.logger = LOGGER.getChild(self.name)
        self.vector_store_id = FAQ_VECTOR_STORE_ID
        self.openai_client = OpenAI(api_key=self.api_key)

        self.logger.setLevel(logging.INFO)
        self._ask_tool = self._create_ask_tool()

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=(
                f"You are {self.name}, responsible for answering FAQs for MechaniGo.ph.\n\n"
                "You have access to a file search tool connected to the FAQ knowledge base.\n\n"
                "Always use `ask` to answer FAQs.\n\n"
                "IMPORTANT RULES:\n"
                "- If a relevant FAQ is found, return ONLY the exact answer text verbatim. Do not paraphrase.\n"
                "- Do NOT summarize the official answer from the FAQs file.\n"
                "- If you cannot find a relevant FAQ, suggest contacting support (in the user's language if you wish).\n"
            ),
            model=self.model,
            tools=[self._ask_tool],
        )

        self._orchestrator_tool = self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description,
        )

        register_tool(
            name="faq_tool",
            target=self._ask_tool,
            description="Searches the FAQ knowledge base for official answers.",
            scopes=("default", "faq_suite"),
            overwrite=True,
        )

        register_tool(
            name="faq_agent",
            target=self._orchestrator_tool,
            description="FAQ agent orchestrator hook.",
            scopes=("default",),
            overwrite=True,
        )

    @property
    def as_tool(self):
        return self._orchestrator_tool

    def _create_ask_tool(self):
        @function_tool
        def ask(question: str):
            return self._ask(question)

        return ask

    def _ask(self, question: str):
        """
        Sends a question to the OpenAI model using a vector store for retrieval.

        :param question: Input for the model.
        :type question: str

        :returns:
            Model response text on success. Returns error on failure.
        :rtype: Union[str, dict]
        """
        self.logger.info("========== faq_agent._ask() called ==========")
        if not self.vector_store_id:
            self.logger.info("No vector store ID found.")
            return {
                "status": "error",
                "message": "No vector store ID found.",
            }

        try:
            response = self.openai_client.responses.create(
                model=self.model,
                input=question,
                tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id]}],
                max_tool_calls=5,
                temperature=0,
            )
            return response.output_text
        except Exception as exc:  # pragma: no cover - network errors
            self.logger.error("Exception occurred while answering FAQs: %s", exc)
            return {
                "status": "error",
                "message": "Error retrieving FAQ answer.",
            }
