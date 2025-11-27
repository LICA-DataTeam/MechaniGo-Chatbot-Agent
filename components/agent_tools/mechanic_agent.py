from __future__ import annotations

import logging
import os
from typing import Any, Optional

from agents import Agent, RunContextWrapper, function_tool
from components.utils import create_agent, register_tool
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from schemas import UserCarDetails

from agents.model_settings import ModelSettings

load_dotenv()

LOGGER = logging.getLogger(__name__)
VECTOR_STORE_ID = os.getenv("MECHANIC_VECTOR_STORE_ID")


class MechanicAgentContext(BaseModel):
    car_memory: UserCarDetails
    model_config = {"arbitrary_types_allowed": True}


class MechanicAgent:
    """Handles all car related inquiries."""

    def __init__(
        self,
        api_key: str,
        name: str = "mechanic_agent",
        model: str = "gpt-4.1",
    ):
        self.api_key = api_key
        self.name = name
        self.model = model
        self.description = "Handles car related inquiries."
        self.logger = LOGGER.getChild(self.name)
        self.openai_client = OpenAI(api_key=self.api_key)
        self.vector_store_id = VECTOR_STORE_ID

        self.logger.setLevel(logging.INFO)
        self._extract_car_info_tool = self._create_extract_car_info_tool()
        self._lookup_tool = self._create_lookup_tool()
        model_settings = ModelSettings(max_tokens=500)

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tools=[self._extract_car_info_tool, self._lookup_tool],
            model_settings=model_settings
        )

        self._orchestrator_tool = self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description,
        )

        register_tool(
            name="mechanic_extract_car_info",
            target=self._extract_car_info_tool,
            description="Parses and stores user car details in conversation context.",
            scopes=("mechanic_suite", "default"),
            overwrite=True,
        )

        register_tool(
            name="mechanic_lookup",
            target=self._lookup_tool,
            description="Mechanic knowledge lookup with vector store / web fallback.",
            scopes=("mechanic_suite", "default"),
            overwrite=True,
        )

        register_tool(
            name="mechanic_agent",
            target=self._orchestrator_tool,
            description="Mechanic agent orchestrator hook.",
            scopes=("default",),
            overwrite=True,
        )

    @property
    def as_tool(self):
        return self._orchestrator_tool

    def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[Any],
        agent: Agent,
    ) -> str:
        car = ctx.context.mechanic_ctx.car_memory
        car_details = f"{car.make or 'Unknown'} {car.model or ''} {car.year or ''}".strip()

        missing_fields = []
        if not (car.make or "").strip():
            missing_fields.append("make")
        if not (car.model or "").strip():
            missing_fields.append("model")
        if not car.year:
            missing_fields.append("year")

        prompt = """
        You are Mechanigo’s Mechanic Agent — a friendly, practical, and knowledgeable automotive assistant. 
        Your goal is to help users understand, troubleshoot, and investigate their car issues in a conversational mechanic-style flow.

        =====================================================
        CORE BEHAVIOR
        =====================================================

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

        =====================================================
        CAR INFO RULES
        =====================================================

        Use the stored car details only when relevant. 
        If an issue is general (battery weak, AC not cold, flat tire, car won’t start), 
        you can proceed without knowing make/model/year.

        Ask for car details only if:
        - They matter for accuracy, OR
        - The user asks for model-specific guidance.

        If the user provides any car details:
        → Immediately call the extract_car_info tool.

        Do NOT ask for all details upfront. Only ask what’s needed.

        =====================================================
        DIAGNOSTIC FLOW
        =====================================================

        Follow this simple mechanic flow:

        1. Acknowledge the issue briefly. (“Sige po, let’s check this.”)
        2. Ask ONE key question that narrows down the cause.
        3. Wait for user’s response.
        4. After enough info is gathered, call lookup IF needed (rules below).
        5. Provide:
        - The likely causes (simple wording)
        - What to check or observe
        - Whether it is safe to drive
        - What to do next
        6. If the issue is clearly serious or unsafe, mention it calmly and directly.

        =====================================================
        TOOL USAGE
        =====================================================

        =========================
        1. extract_car_info
        =========================
        Call this tool ONLY when:
        - The user provides new or updated car info (make/model/year/fuel type/transmission).

        Do NOT call this tool:
        - For filler messages (“ok po”, “sige”)
        - When the car info is irrelevant to the issue
        - When you already have the necessary fields

        =========================
        2. lookup (Hybrid RAG Rules)
        =========================

        Use the lookup tool when:
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

        You may answer WITHOUT lookup when:
        - Asking clarifying questions
        - Requesting evidence (photos/videos)
        - Summarizing or simplifying a previous lookup result
        - Explaining general concepts (“para saan ang engine oil?”)
        - Handling simple, generic, low-risk symptoms (“kalampag”, “mahina hatak”, “hindi lumalamig AC”) until more info is gathered

        Efficiency rule:
        → Avoid calling lookup repeatedly for the same issue unless the user adds significant new details.

        If lookup returns no results:
        → Give a short mechanic-style general explanation and move on.

        =====================================================
        TONE & STYLE
        =====================================================

        - Friendly Filipino-English (“Taglish mechanic style”)
        - Clear, direct, non-technical unless needed
        - Avoid overwhelming the customer
        - Break down steps simply
        - Never sound like reading a script

        =====================================================
        WHEN TO MENTION MECHANIGO SERVICES
        =====================================================

        Only AFTER:
        - The issue is fully understood, OR
        - The user asks for service options

        You may mention:
        - PMS home service
        - Initial Diagnosis home visit
        - Secondhand Car Inspection

        But do NOT push or sell early in the conversation.
        """
        self.logger.info("========== mechanic_agent called! ==========")
        return prompt

    def _create_extract_car_info_tool(self):
        @function_tool
        def extract_car_info(
            ctx: RunContextWrapper[MechanicAgentContext],
            make: Optional[str] = None,
            model: Optional[str] = None,
            year: Optional[str] = None,
            fuel_type: Optional[str] = None,
            transmission: Optional[str] = None,
        ):
            return self._extract_car_info(
                ctx,
                make,
                model,
                year,
                fuel_type,
                transmission,
            )

        return extract_car_info

    def _create_lookup_tool(self):
        @function_tool
        def lookup(question: str):
            return self._lookup(question)

        return lookup

    def _extract_car_info(
        self,
        ctx: RunContextWrapper[Any],
        make: Optional[str] = None,
        model: Optional[str] = None,
        year: Optional[str] = None,
        fuel_type: Optional[str] = None,
        transmission: Optional[str] = None,
    ):
        """
        Extracts and updates the user's car information based on the context.

        :param ctx: The context used in the conversation.
        :type ctx: RunContextWrapper[Any]
        :param make: Car make. Defaults to ``None``.
        :type make: Optional[str]
        :param model: Car model. Defaults to ``None``.
        :type model: Optional[str]
        :param year: Car manufacturing year. Defaults to ``None``.
        :type year: Optional[str]
        :param fuel_type: Type of fuel. Defaults to ``None``.
        :type fuel_type: Optional[str]
        :param transmission: Transmission type. Defaults to ``None``.
        :type transmission: Optional[str]

        :returns: A dictionary containing the status of the update, changed fields, current car details,
              and a message describing the result.
              - If no changes: `{"status": "no_change", "message": "...", "car_details": {...}}`
              - If updates were made: `{"status": "success", "changed_fields": {...}, "car_details": {...}, "message": "..."}`
        :rtype: dict
        """
        self.logger.info("========== Extracting Car Info ==========")
        self.logger.info("Received: make=%s, model=%s, year=%s", make, model, year)

        car = ctx.context.mechanic_ctx.car_memory

        def norm_str(value: Optional[str]) -> Optional[str]:
            return (value or "").strip() or None

        def norm_int_str(value: Optional[str]) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(str(value).strip())
            except (ValueError, TypeError):
                return None

        incoming = {
            "make": norm_str(make),
            "model": norm_str(model),
            "year": norm_int_str(year),
            "fuel_type": norm_str(fuel_type),
            "transmission": norm_str(transmission),
        }

        current = {
            "make": norm_str(car.make),
            "model": norm_str(car.model),
            "year": car.year if isinstance(car.year, int) else norm_int_str(car.year),
            "fuel_type": norm_str(car.fuel_type),
            "transmission": norm_str(car.transmission),
        }

        changed_fields = {}
        if incoming["make"] is not None and incoming["make"] != current["make"]:
            car.make = incoming["make"]
            changed_fields["make"] = car.make
        if incoming["model"] is not None and incoming["model"] != current["model"]:
            car.model = incoming["model"]
            changed_fields["model"] = car.model
        if incoming["year"] is not None and incoming["year"] != current["year"]:
            car.year = incoming["year"]
            changed_fields["year"] = car.year
        if incoming["fuel_type"] is not None and incoming["fuel_type"] != current["fuel_type"]:
            car.fuel_type = incoming["fuel_type"]
            changed_fields["fuel_type"] = car.fuel_type
        if incoming["transmission"] is not None and incoming["transmission"] != current["transmission"]:
            car.transmission = incoming["transmission"]
            changed_fields["transmission"] = car.transmission

        if not changed_fields:
            self.logger.info("========== _extract_car_info() ==========")
            self.logger.info("Current car_memory: %s", car.model_dump())
            return {
                "status": "no_change",
                "message": "Car details unchanged.",
                "car_details": car.model_dump(),
            }

        self.logger.info("Updated user car details: %s", car)
        return {
            "status": "success",
            "changed_fields": changed_fields,
            "car_details": car.model_dump(),
            "message": f"Updated car details: {changed_fields}. Please confirm if these are correct.",
        }

    def _lookup(self, question: str):
        """
        The main tool of `MechanicAgent` for answering car-related inquiries such as car diagnosis, troubleshooting, maintenance, etc.

        If `MechanicAgent` has a `vector_store_id` initialized, first tries a vector store for answer retrieval. Otherwise,
        web search is used as a fallback.

        :param question: Question/inquiry.
        :type question: str

        :returns: A dictionary containing the status of the update.
        :rtype: dict
        """
        self.logger.info("========== _lookup() called ==========")
        domain = ["carparts.com", "mechanigo.ph"]

        try:
            if self.vector_store_id:
                try:
                    self.logger.info("Using mechanic knowledge base vector store via file_search...")
                    prompt = [
                        {"role": "user", "content": question}
                    ]
                    response = self.openai_client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt,
                        tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id]}],
                        max_tool_calls=1,
                        temperature=0,
                        max_output_tokens=300
                    )
                    answer = (response.output_text or "").strip()
                    if answer and answer != "__NO_RESULTS__":
                        self.logger.info("vector_store found answer!")
                        return {
                            "status": "success",
                            "source": "vector_store",
                            "answer": answer
                        }
                    else:
                        self.logger.info("Vector store returned no text, falling back to web_search...")
                except Exception as exc:
                    self.logger.error("Vector store lookup failed: %s. Using web_search...", exc)

            try:
                self.logger.info("Attempting web_search...")
                prompt = [
                    {"role": "user", "content": question},
                ]
                response = self.openai_client.responses.create(
                    model="gpt-4o-mini",
                    input=prompt,
                    tools=[{"type": "web_search", "filters": {"allowed_domains": domain}}],
                    tool_choice="required",
                    include=["web_search_call.action.sources"],
                    max_tool_calls=1,
                    temperature=0,
                    max_output_tokens=300
                )
                payload = response.model_dump()
                answer = (response.output_text or "").strip()
                if answer and payload is not None:
                    self.logger.info("web_search answer found!")
                    ans = {
                        "status": "success",
                        "source": "web_search",
                        "answer": answer,
                        "citations": domain
                    }
                    return ans
                else:
                    self.logger.info("web_search failed to retrieve any relevant information.")
                    return {
                        "status": "fail",
                        "source": "web_search"
                    }
            except Exception:
                return {"status": "error", "message": "web_search failed."}
        except Exception as exc:
            self.logger.error("Exception occurred while web searching: %s", exc)
            return {
                "status": "error",
                "message": "Exception occurred. Error retrieving mechanic answer.",
            }
