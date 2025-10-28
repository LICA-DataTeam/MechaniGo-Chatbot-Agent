from agents import Agent, RunContextWrapper, function_tool
from components.utils import create_agent
from schemas import UserCarDetails
from urllib.parse import urlparse
from typing import Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MechanicAgentContext(BaseModel):
    car_memory: UserCarDetails
    model_config = {
        "arbitrary_types_allowed": True
    }

class PMSAgent:
    """Expert at handling inquiries about PMS related questions."""
    def __init__(
        self,
        api_key: str,
        name: str = "pms_agent",
        model: str = "gpt-4.1"
    ):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.name = name
        self.model = model
        self.description = "An expert at Preventive and Periodic Maintenance-Oil Change Services."

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=self._dynamic_instructions,
            model=self.model
        )

    @property
    def as_tool(self):
        return self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )

    def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[Any],
        agent: Agent
    ):
        mechanic_ctx = getattr(ctx.context, "mechanic_ctx", None)
        car = getattr(mechanic_ctx or ctx.context, "car_memory", None) or UserCarDetails()
        car_details = f"{car.make or 'Unknown'} {car.model or ''} {car.year or ''}".strip()
        self.logger.info("========== pms_agent instructions ==========")
        print(f"car: {car_details}")
        instructions = (
            f"You are {agent.name}, an expert in Preventive and Periodic Maintenance Service (PMS).\n\n"
            f"The user owns a {car_details}.\n"
            f"Your role is to provide authoritative, accurate, and practical information about this car's preventive maintenance.\n\n"
            "Your expertise includes:\n"
            "- Explaining what PMS is and why it is important for car longevity and safety.\n"
            "- Recommending service intervals (e.g., every 5,000 or 10,000 km, or every 6 months) depending on the manufacturer and driving habits.\n"
            "- Detailing what tasks are done at each PMS stage: oil and filter changes, fluid checks, brake inspection, tire rotation, etc.\n"
            "- Advising the user on what to expect during a PMS visit — cost, duration, and checklist.\n"
            "- Suggesting appropriate fluids, filters, spark plugs, or other consumables based on the car's make and year.\n"
            "- Warning about consequences of missed or delayed PMS (e.g., voided warranty, reduced engine performance, or safety risks).\n"
            "- Comparing dealership vs third-party PMS options when asked.\n\n"
            "When answering:\n"
            "- Always ground your explanations in preventive maintenance context.\n"
            "- If the user asks a troubleshooting question (symptom, noise, malfunction), clarify that PMS is for *scheduled maintenance*, not repair diagnosis.\n"
            "- If needed, suggest the user ask the main MechanicAgent for troubleshooting help.\n"
            "- Keep answers concise, practical, and formatted for clarity (lists, steps, or tables when helpful).\n"
        )
        return instructions

class MechanicAgent:
    """Handles all car related inquiries."""
    def __init__(
        self,
        api_key: str,
        name: str = "mechanic_agent",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key
        self.name = name
        self.model = model
        self.description = "Handles car related inquiries."
        self.openai_client = OpenAI(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

        extract_car_info = self._create_extract_car_info()
        lookup = self._create_lookup()

        load_dotenv()
        self.vector_store_id = os.getenv("MECHANIC_VECTOR_STORE_ID")

        pms_agent = PMSAgent(
            api_key=self.api_key
        )

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tools=[extract_car_info, lookup, pms_agent.as_tool]
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
                                domain = parsed.netloc.replace("www", "")
                                used_domains.add(domain)
        return target_domain in used_domains, used_domains

    def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[Any],
        agent: Agent
    ):
        car = ctx.context.mechanic_ctx.car_memory
        car_details = f"{car.make or 'Unknown'} {car.model or ''} {car.year or ''}".strip()

        missing_fields = []
        if not (car.make or "").strip():
            missing_fields.append("make")
        if not (car.model or "").strip():
            missing_fields.append("model")
        if not car.year:
            missing_fields.append("year")

        prompt = (
            f"You are {agent.name}, a car mechanic sub-agent.\n\n"
            f"User's car deatils in memory: {car_details}\n\n"
            "CRITICAL RULES:\n"
            "1. If this turn contains ANY car information and memory is missing make or model, call extract_car_info first before replying or using other tools.\n"
            "2. When calling extract_car_info, pass every car field mentioned in the current turn (make, model, year, fuel type, transmission).\n"
            "3. After extract_car_info returns success, acknowledge what changed and ask the user to confirm.\n"
            "4. Use lookup for troubleshooting or general maintenance questions, citing sources.\n"
            "5. Use pms_agent strictly for Preventive/Periodic Maintenance Service (PMS) questions (intervals, checklists, consumables, etc.).\n\n"
            "TOOLS AND WHEN TO USE THEM:\n"
            "- extract_car_info: whenever the user states or updates car details.\n"
            "- lookup: diagnosis, troubleshooting, or non-PMS maintenance questions.\n"
            "- pms_agent: PMS intervals, schedules, tasks, or consumables.\n"
        )

        if missing_fields:
            prompt += (
                "\nNEXT-ACTION:\n"
                f"- Car memory still missing: {', '.join(missing_fields)}.\n"
                "- IMMEDIATELY call extract_car_info with any car info from the current turn.\n"
                "- Do not call pms_agent or lookup, and do not answer, until extract_car_info returns success.\n"
            )
        else:
            prompt += (
                "\nNEXT-ACTION:\n"
                "- Car details complete. Continue assisting.\n"
                "- For PMS questions, prefer pms_agent and summarize the answer for the user.\n"
                "- For troubleshooting or general maintenance questions, call lookup and cite your sources.\n"
            )
        return prompt

    def _create_extract_car_info(self):
        @function_tool
        def extract_car_info(
            ctx: RunContextWrapper[MechanicAgentContext],
            make: Optional[str] = None,
            model: Optional[str] = None,
            year: Optional[str] = None,
            fuel_type: Optional[str] = None,
            transmission: Optional[str] = None
        ):
            return self._extract_car_info(
                ctx, make, model, year, fuel_type, transmission
            )
        return extract_car_info

    def _create_lookup(self):
        @function_tool
        def lookup(question: str):
            return self._lookup(question)
        return lookup

    def _extract_car_info(
        self,
        ctx: RunContextWrapper[Any], # MechaniGoContext
        make: Optional[str] = None,
        model: Optional[str] = None,
        year: Optional[str] = None,
        fuel_type: Optional[str] = None,
        transmission: Optional[str] = None,
    ):
        self.logger.info("========== Extracting Car Info ==========")
        self.logger.info(f"Received: make={make}, model={model}, year={year}")

        car = ctx.context.mechanic_ctx.car_memory
        
        def norm_str(x): return (x or "").strip()
        def norm_int_str(x):
            if x is None:
                return None
            try:
                return int(str(x).strip())
            except (ValueError, TypeError):
                return None

        incoming = {
            "make": norm_str(make) or None,
            "model": norm_str(model) or None,
            "year": norm_int_str(year) or None,
            "fuel_type": norm_str(fuel_type) or None,
            "transmission": norm_str(transmission) or None
        }

        current = {
            "make": norm_str(car.make) or None,
            "model": norm_str(car.model) or None,
            "year": car.year if isinstance(car.year, int) else norm_int_str(car.year),
            "fuel_type": norm_str(car.fuel_type) or None,
            "transmission": norm_str(car.transmission) or None
        }

        changed_fields = {}
        if incoming["make"] is not None and incoming["make"] != current["make"]:
            car.make = incoming["make"]; changed_fields["make"] = car.make
        if incoming["model"] is not None and incoming["model"] != current["model"]:
            car.model = incoming["model"]; changed_fields["model"] = car.model
        if incoming["year"] is not None and incoming["year"] != current["year"]:
            car.year = incoming["year"]; changed_fields["year"] = car.year
        if incoming["fuel_type"] is not None and incoming["fuel_type"] != current["fuel_type"]:
            car.fuel_type = incoming["fuel_type"]; changed_fields["fuel_type"] = car.fuel_type
        if incoming["transmission"] is not None and incoming["transmission"] != current["transmission"]:
            car.transmission = incoming["transmission"]; changed_fields["transmission"] = car.transmission

        if not changed_fields:
            self.logger.info("========== _extract_car_info() ==========")
            self.logger.info(f"Current car_memory: {car.model_dump()}")
            return {
                "status": "no_change",
                "message": "Car details unchanged.",
                "car_details": car.model_dump()
            }
        self.logger.info(f"Updated user car details: {car}")

        return {
            "status": "success",
            "changed_fields": changed_fields,
            "car_details": car.model_dump(),
            "message": f"Updated car details: {changed_fields}. "
                        f"Please confirm if these are correct."
        }

    def _lookup(self, question: str):
        # MechanicAgent's tool for car-diagnosis and troubleshooting.
        # Prefers vector store (file_search) ONLY IF it is configured and indexed.
        # To configure vector store -> components/utils/file_search_utils.py
        # Fallback: web_search
        self.logger.info("========== _lookup() called ==========")
        domain = "mechanigo.ph"

        try:
            if getattr(self, "vector_store_id", None):
                try:
                    self.logger.info("Using mechanic knowledge base vector store via file_search...")
                    response = self.openai_client.responses.create(
                        model="gpt-4.1",
                        input=question,
                        tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id]}],
                        max_tool_calls=3,
                        temperature=0
                    )
                    return {
                        "status": "success",
                        "source": "vector_store",
                        "answer": (response.output_text or "").strip()
                    }
                except Exception as e:
                    self.logger.error(f"Vector store lookup failed: {e}. Using web_search...")
            try:
                input = [
                    {"role": "system", "content": "You are an assistant that answers car diagnosis and troubleshooting. Always cite your answers."},
                    {"role": "user", "content": question}
                ]
                response = self.openai_client.responses.create(
                    model="gpt-5",
                    input=input,
                    tools=[{"type": "web_search", "filters": {"allowed_domains": [domain]}}],
                    tool_choice="auto",
                    include=["web_search_call.action.sources"]
                )
                _, domains = MechanicAgent.domain_extract(response.model_dump(), domain)
                return {
                    "status": "success",
                    "source": "web_search",
                    "answer": (response.output_text or "").strip(),
                    "citations": domains
                }
            except Exception as e:
                return {"status": "error", "message": "web_search failed."}
        except Exception as e:
            self.logger.error(f"Exception occurred while web searching: {e}")
            return {
                "status": "error",
                "message": "Exception occurred. Error retrieving FAQ answer."
            }