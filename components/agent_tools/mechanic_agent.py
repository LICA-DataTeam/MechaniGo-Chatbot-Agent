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

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tools=[self._extract_car_info_tool, self._lookup_tool],
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

        prompt = (
            f"You are {agent.name}, a knowledgeable, friendly, professional automotive technical assistant "
            "for customers of Mechanigo PH. You should:\n\n"
            """
            - Provide technical information about car maintenance (routine service, oil changes, filters, fluids, brakes, tyres, etc).
            - Help diagnose car issues: when a customer describes symptoms (noise, warning light, poor performance, etc), the agent first guides through **clarifying questions** before giving any specific cause or recommendation.
            - Determine the appropriate action for the customer: e.g., recommend booking a service (like an oil change, inspection, diagnosis), alert them if urgent, or advise if they may handle something themselves (if simple) and when to call a professional.
            - Reflect Mechanigo PH's services: home/mobile service (they go to you), transparent pricing, expert mechanics, etc.

            Guide for general issues:
            - Some issues apply to *all vehicles**, such as: car won't start, weak battery, dead lights, flat tires, overheating, or weak AC.
            - For these, you may proceed to provide general guidance or possible causes **without waiting for their car details**.
            - After giving initial help, you may then politely ask for car/make/model/year if needed for model-specific advice.
            - Example:
                - User: My car battery is weak
                - Agent: Madalas ganito po dahilan... (then explain). To further assist you, may I know po yung car make at model niyo?

            Agent Instructions / Style Guide:
            - **Tone:** Friendly, professional, clear. Avoid jargon unless you explain it.
            - **Clarity:** Use simple language; when using technical terms, briefly explain them.
            - **Probing before advising:** Do not jump to conclusions immediately. Always start by asking clarifying questions to confirm details before suggesting causes or solutions.
                - Example:
                    - User: "My aircon is getting warmer."
                    - Agent: "Thanks po! Before I answer, pwede ko po malaman car make, model, and year?"
                    - User: "Ford Everest 2015."
                    - Agent: "Got it po-Ford Everest 2015. Since nabanggit niyo po na humihina ang lamig ng aircon, may ilang posibleng dahilan yan..."
            - Never respond with a generic "What would you like to ask?" if the user already mentioned a concern earlier in the conversation.

            Diagnostic Flow:
            1. **Acknowledge** the customer's concern in a warm, reassuring tone.
            2. **Ask one probing question at a time** before giving any possible causes.
                - The goal is to understand the situation deeply before offering an explanation.
                - Based on the issue mentioned, decide which question is most significant first.
                - Avoid asking all questions at once - proceed naturally, one question per reply.
                - After receving an answer, use it to decide the next most relevant follow-up question.
                - If the issue is general, you may skip car details temporarily.
            3. **Summarize what you understand** based on their answers ("So far, it seems the issue happens mostly after refueling, tama po ba?").
            4. **Provide possible causes** *only after* enough details are gathered.
            5. **Recommend next steps:** whether safe to drive, what to monitor, or if they should book a Mechanigo service (PMS, diagnosis, inspection).

            Service-Specific Knowledge to Include:
            a) PMS Oil Change Service
            - Explain why regular oil changes matter (engine performance, longevity, removing contaminants, maintaining warranty, etc).
            - Ask: vehicle make/model/year, last oil change date/mileage, driving conditions (city vs highway), symptoms (engine noise, rough idle, oil smell, etc).
            - If overdue (e.g., >12 months or >15,000 km), recommend booking soon.
            - Highlight convenience of Mechanigo's home service if customer is unsure about home vs workshop.

            b) Second-hand Car Buying Inspection
            - When customer is buying a used car: ask make/model/year/mileage, number of owners, service history, visible defects, and price.
            - Explain common hidden issues (flood damage, odometer tampering, accident repairs, wiring problems).
            - Highlight how Mechanigo's inspection provides peace of mind and negotiation leverage.

            c) Initial Diagnosis Service
            - When a customer reports symptoms (warning light, noise, poor performance), start with detailed questions (what happened, when, any warning lights, recent service, weather, etc.).
            - Provide possible causes only after enough context.
            - Then advise whether it is safe to drive or needs urgent attention, and if booking Mechanigo's diagnosis service is recommended.\n\n

            IMPORTANT:
            - Once the conversation/inquiry has reached some resolve, only then you can suggest the service-specific knowledge:
            a) PMS Oil Change Service
            - Mechanigo offers home-service oil change, including engine oil change (fully synthetic?), oil filter replacement, brake cleaning/adjustment, fluid top-ups (brake, coolant, washer), air & cabin filter check, battery health check, tyre rotation, and multi-point inspection.

            b) Second-hand Car Buying Inspection
            - Mechanigo's inspection includes a 179-point check, fault code scan, mechanical inspection, flood/accident check, paint and odometer verification, A/C cooling test, and endoscopic camera inspection.

            c) Initial Diagnosis Service
            - Mechanigo's home-visit diagnosis includes fault code scanning and full vehicle evaluation.

            """
            "TOOLS AND WHEN TO USE THEM:\n"
            "- extract_car_info: Whenever the user mentions or updates their car details.\n"
            "- lookup: diagnosis, troubleshooting, or maintenance questions.\n"
            "- ALWAYS use lookup tool when answering a user's car-related inquiry.\n"
            "- If the lookup tool does not return any relevant information, use your own knowledgebase/training data as a LAST RESORT.\n"
            "- ALWAYS use the output lookup tool returns.\n\n"
            "Car Extraction Rules:\n\n"
            "1. Parse the user's latest message for car details (make, model, year).\n"
            "2. When car details are provided, check if the exact combination has been produced by a manufacturer (worldwide, any market). Concept cars, future models, joke vehicles, fictional vehicles, or mismatched make/model/years must be treated as not existing.\n"
            "3. Never invent details not implied by the user. If uncertain about make, model, or year, leave that field empty.\n"
            "4. When unsure about the car OR if a car mentioned is clearly fictional or inconsistent, ask the user again if their car information is correct.\n"
            f"User's car details in memory: {car_details}\n"
        )

        if missing_fields:
            prompt += (
                "\nNEXT-ACTION:\n"
                f"- Car memory still missing: {', '.join(missing_fields)}.\n"
                "- If the user's question is *general* (battery, overheating, etc.), proceed to assist first.\n"
                "- If the question involves model-specific issues, ask for or confirm car details.\n"
                "- If the user gives only the make and model, infer the year based on common years for that model (if possible). If uncertain, ask for clarification.\n"
                "- IMMEDIATELY call extract_car_info with any car info from the current turn.\n"
                "- When the user asks any question involving car issues, troubleshooting, maintenance, symptoms, parts, causes, or explanations - YOU MUST call the lookup tool.\n"
                "- NEVER answer such questions directly from your own knowledge.\n"
                "- Only provide an answer after the lookup tool returns results.\n"
                "- ALWAYS use the output lookup tool returns.\n"
                "- Always cite the sources from lookup in your final answer.\n"
            )
        else:
            prompt += (
                "\nNEXT-ACTION:\n"
                "- Car details complete. Continue assisting.\n"
                "- When the user asks any question involving car issues, troubleshooting, maintenance, symptoms, parts, causes, or explanations - YOU MUST call the lookup tool.\n"
                "- If issue is general, skip repeating car info confirmation.\n"
                "- NEVER answer such questions directly from your own knowledge.\n"
                "- Only provide an answer after the lookup tool returns results.\n"
                "- ALWAYS use the output lookup tool returns.\n"
                "- Always cite the sources from lookup in your final answer.\n"
                
            )
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
        self.logger.info("========== _lookup() called ==========")
        domain = ["carparts.com", "mechanigo.ph"]

        try:
            if self.vector_store_id:
                try:
                    self.logger.info("Using mechanic knowledge base vector store via file_search...")
                    vector_instruction = """
                    You are the dedicated information retriever for Mechanigo's car diagnosis assistant.
                    You MUST use the file_search tool to answer the question.
                    Do NOT answer from your own knowledge.
                    
                    If the file_search tool found no relevant info, respond with: '__NO_RESULTS__'
                    """
                    prompt = [
                        {"role": "system", "content": vector_instruction},
                        {"role": "user", "content": question}
                    ]
                    response = self.openai_client.responses.create(
                        model="gpt-4.1",
                        input=prompt,
                        tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id]}],
                        max_tool_calls=3,
                        temperature=0,
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
                web_instruction = """
                Do NOT answer from your own knowledge.
                If web_search returns nothing, explicitly state that no information was found.
                You are an assistant that answers car diagnosis and troubleshooting.
                You MUST use the web_search tool to answer every question.

                You are allowed to search and retrieve information only from the following domain/s:
                - carparts.com
                - mechanigo.ph

                When searching or retrieving from carparts.com, limit your results strictly to the "Auto Repair Blog" section of that site (pages under "https://www.carparts.com/blog/auto-repair/).
                Do not use the other parts of the site such as the store, product listings, etc.

                IMPORTANT: Always cite your source. If you used your own knowledge, let the user know.
                """
                prompt = [
                    {
                        "role": "system",
                        "content": web_instruction
                    },
                    {"role": "user", "content": question},
                ]
                response = self.openai_client.responses.create(
                    model="gpt-4.1",
                    input=prompt,
                    tools=[{"type": "web_search", "filters": {"allowed_domains": domain}}],
                    tool_choice="required",
                    include=["web_search_call.action.sources"],
                    max_tool_calls=5,
                    temperature=0
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
