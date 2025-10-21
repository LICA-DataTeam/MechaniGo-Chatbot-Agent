from components.utils import create_agent
from urllib.parse import urlparse
from agents import function_tool
from dotenv import load_dotenv
from openai import OpenAI
import logging
import os

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

        load_dotenv()
        self.vector_store_id = os.getenv("VECTOR_STORE_ID")

        ask = self._create_ask()
        web_search = self._create_web_search()
        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=(
                f"You are {self.name}, responsible for answering faqs for MechaniGo.ph\n\n"
                "You have access to a file search tool connected to the FAQ knowledge base.\n\n"
                "Always use `ask` to answer FAQs.\n\n"
                "If the FAQ search tool fails or is unavailable, call `web_searc` and cite the source.\n\n"
                "IMPORTRANT RULES:\n\n"
                "- If a relevant FAQ is found, return ONLY the exact answer text verbatim. Do not paraphrase.\n\n"
                "- DO NOT summarize the official answer from the FAQs file.\n\n"
                "- If you cannot find a relevant FAQ, suggest contacting support (in the user's language if you wish).\n\n"
            ),
            model=self.model,
            tools=[ask, web_search]
        )

    @property
    def as_tool(self):
        return self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )
    
    @staticmethod
    def domain_extract(response: dict, target_domain: str):
        used_domains = set()
        
        for i in response.get("output", []):
            if i.get("type") == "message":
                for content_block in i.get("content", []):
                    annotations = content_block.get("annotations", [])
                    for annotation in annotations:
                        if annotation.get("type") == "url_citation":
                            url = annotation.get("url")
                            if url:
                                parsed = urlparse(url)
                                domain = parsed.netloc.replace("www.", "")
                                used_domains.add(domain)

        return target_domain in used_domains, used_domains

    def _create_ask(self):
        @function_tool
        def ask(question: str):
            return self._ask(question)
        return ask

    def _create_web_search(self):
        @function_tool
        def web_search(question: str):
            return self._web_search(question)
        return web_search

    def _ask(self, question: str):
        self.logger.info("========== _ask() Called! ==========")
        if not self.vector_store_id:
            self.logger.info("No vector store ID found; using web_search fallback...")
            return self._web_search(question)

        try:
            client = OpenAI()
            response = client.responses.create(
                model=self.model,
                input=question,
                tools=[
                    {"type": "file_search", "vector_store_ids": [self.vector_store_id]}
                ],
                max_tool_calls=5
            )

            text = response.output_text or ""
            if not text or "issue retrieving" in text.lower():
                self.logger.warning("FAQ fallback triggered; check vector store.")
                return self._web_search(question)
            return text
        except Exception as e:
            self.logger.error(f"Exception occurred while answering FAQs: {e}")
            return {
                "status": "error",
                "message": "Error retrieving FAQ answer."
            }

    def _web_search(self, question: str):
        self.logger.info("========== _web_search() Called! ==========")

        try:
            client = OpenAI()
            input = [
                {"role": "system", "content": "You are an assistant that answers based on the MechaniGo PH website. Always cite your answers."},
                {"role": "user", "content": question}
            ]
            response = client.responses.create(
                model="gpt-5",
                input=input,
                tools=[{"type": "web_search", "filters": {"allowed_domains": ["mechanigo.ph"]}}],
                tool_choice="auto",
                include=["web_search_call.action.sources"]
            )
            
            _, domains = FAQAgent.domain_extract(response.model_dump(), "mechanigo.ph")
            self.logger.info(f"Domains cited: {domains}")
            return response.output_text
        except Exception as e:
            self.logger.error(f"Exception occurred while web searching: {e}")
            return {
                "status": "error",
                "message": "Error retrieving FAQ answer."
            }