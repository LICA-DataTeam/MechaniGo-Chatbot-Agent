# Current tasks:
# UserInfoAgent
# - extract user info
# - get user info (from context)
# BookingAgent
# - extract payment type
# MechanicAgent
# - web search: car-diagnosis and troubleshooting
#     - https://www.carparts.com/blog/auto-repair/
# - file search: car-diagnosis (cms blog post - vector store)
# FAQAgent
# - file search: vector store (faqs.json)
from agents import Agent, Runner, RunContextWrapper, SQLiteSession, function_tool
from config import DEFAULT_AGENT_HANDOFF_DESCRIPTION
from components.agent_tools import (
    UserInfoAgentContext,
    MechanicAgentContext,
    BookingAgentContext
)
from components.utils import create_agent, register_tool, BigQueryClient
from agents.model_settings import ModelSettings
from schemas import User, UserCarDetails
from google.cloud import bigquery
from typing import Optional, Any
from pydantic import BaseModel
import openai
import logging
import uuid
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

VECTOR_STORE_ID = os.getenv("MECHANIC_VECTOR_STORE_ID")
FAQ_VECTOR_STORE_ID = os.getenv("FAQ_VECTOR_STORE_ID")

class MechaniGoContext(BaseModel):
    user_ctx: UserInfoAgentContext
    mechanic_ctx: MechanicAgentContext
    booking_ctx: BookingAgentContext
    model_config = {
        "arbitrary_types_allowed": True
    }

class MechaniGoAgent:
    """Serves as the general agent and customer facing agent."""
    def __init__(
        self,
        api_key: str = None,
        bq_client: BigQueryClient = None,
        table_name: str = "chatbot_users_test",
        name: str = "MechaniGo Bot",
        model: str = "gpt-4o-mini",
        context: Optional[MechaniGoContext] = None,
        session: SQLiteSession = None,
        input_guardrail: Optional[list] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or openai.api_key or None
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided.")
        
        self.bq_client = bq_client
        self.table_name = table_name
        self.name = name
        self.handoff_description = DEFAULT_AGENT_HANDOFF_DESCRIPTION

        if session is None:
            raise ValueError("Session must be provided.")
        self.session = session

        self.model = model
        self.input_guardrail = input_guardrail

        if not self.bq_client:
            self.logger.info("BigQuery client not initialized! Initializing one...")
            from config import DATASET_NAME
            self.bq_client = BigQueryClient('google_creds.json', DATASET_NAME)
            self.logger.info("Done.")

        if self.bq_client:
            self._ensure_users_table()

        if not context:
            context = MechaniGoContext(
                user_ctx=UserInfoAgentContext(
                    user_memory=User(uid=str(uuid.uuid4()))
                ),
                mechanic_ctx=MechanicAgentContext(
                    car_memory=UserCarDetails()
                ),
                booking_ctx=BookingAgentContext()
            )

        self.openai_client = openai.OpenAI(api_key=self.api_key)
        self.vector_store_id = VECTOR_STORE_ID
        self.vector_store_id = FAQ_VECTOR_STORE_ID

        self.context = context
        model_settings = ModelSettings(max_tokens=1000)

        # tools
        extract_user_info = self._create_ctx_extract_user_tool()
        get_user_info = self._create_ctx_get_user_tool()
        get_car_info = self._create_extract_car_info_tool()
        lookup_tool = self._create_lookup_tool()
        faq_tool = self._create_ask_tool()

        # register tools
        register_tool(
            name="user_extract_info",
            target=extract_user_info,
            description="Parses user information in conversation context.",
            scopes=("user_suite", "default"),
            overwrite=True,
        )

        register_tool(
            name="user_get_info",
            target=get_user_info,
            description="Retrieves user info from memory in conversation context.",
            scopes=("user_suite", "default"),
            overwrite=True,
        )

        register_tool(
            name="extract_car_info",
            target=get_car_info,
            description="Parses and stores user car details in conversation context.",
            scopes=("mechanic_suite", "default"),
            overwrite=True,
        )

        register_tool(
            name="lookup",
            target=lookup_tool,
            description="Mechanic knowledge lookup with vector store/web fallback.",
            scopes=("mechanic_suite", "default"),
            overwrite=True,
        )

        register_tool(
            name="faq_tool",
            target=faq_tool,
            description="Searches the FAQ knowledge base for official answers.",
            scopes=("default", "faq_suite"),
            overwrite=True,
        )

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.handoff_description,
            instructions=self._dynamic_instructions,
            model=self.model,
            # tool_names=(),
            tool_names=("user_extract_info", "user_get_info", "extract_car_info", "lookup", "faq_tool"),
            input_guardrails=self.input_guardrail,
            guardrail_names=("input_generic",),
            model_settings=model_settings,
            # tool_use_behavior="stop_on_first_tool"
        )


    async def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[MechaniGoContext],
        agent: Agent
    ):
        self.logger.info("========== orchestrator_agent called! ==========")

        # raw values
        user_name = ctx.context.user_ctx.user_memory.name
        user_sched_date = ctx.context.user_ctx.user_memory.schedule_date
        user_sched_time = ctx.context.user_ctx.user_memory.schedule_time
        user_payment = ctx.context.user_ctx.user_memory.payment
        user_service_type = ctx.context.user_ctx.user_memory.service_type
        car = self.sync_user_car(ctx)

        user_email = ctx.context.user_ctx.user_memory.email
        user_contact = ctx.context.user_ctx.user_memory.contact_num
        user_address = ctx.context.user_ctx.user_memory.address

        # Check completeness before setting display values
        self.logger.info("========== VERIFYING USER INFORMATION ==========")
        has_user_info = user_name is not None and bool(user_name.strip())
        has_email = user_email is not None and bool(user_email.strip())
        has_user_contact = user_contact is not None and bool(user_contact.strip())
        has_service = user_service_type is not None and bool(user_service_type.strip())
        has_address = user_address is not None and bool(user_address.strip())
        has_schedule = (
            user_sched_date is not None and bool(user_sched_date.strip()) and
            user_sched_time is not None and bool(user_sched_time.strip())
        )
        has_payment = user_payment is not None and bool(user_payment.strip())
        has_car = car is not None and bool(car.strip())

        display_name = user_name if has_user_info else "Unknown user"
        display_email = user_email if has_email else "Unknown email"
        display_contact = user_contact if has_user_contact else "No contact"
        display_service_type = user_service_type if has_service else "No service type"
        display_sched_date = user_sched_date if has_schedule else "Unknown date"
        display_sched_time = user_sched_time if has_schedule else "Unknown time"
        display_address = user_address if has_address else "No address"
        display_payment = user_payment if has_payment else "No payment"
        display_car = car if has_car else "No car specified"

        self.logger.info("========== DETAILS ==========")
        self.logger.info(f"Complete: user={display_name}, email={display_email}, contact_num={display_contact}, service={display_service_type}, car={display_car}, schedule={display_sched_date} @{display_sched_time}, payment={display_payment}, address={display_address}")
        prompt = (
            f"You are {agent.name}, the main orchestrator and customer-facing bot of MechaniGo.ph.\n"
            "You always reply in a friendly, helpful Taglish tone and use 'po' where appropriate to show respect.\n"
            "Keep replies concise but clear — usually 2–5 short sentences, plus a follow-up question if needed.\n\n"
            "==============================\n"
            "MAIN ROLE\n"
            "==============================\n"
            "- Ikaw ang kausap ng customer. You understand their concern, reply in Taglish, and only call tools when needed.\n"
            "- Use the information already saved (name, email, contact, address, car details, etc.) and avoid re-asking the same thing.\n"
            "- Aim for low token usage and low latency: short answers, minimal tool calls, and no unnecessary repetition.\n\n"
            "==============================\n"
            "COMMUNICATION STYLE\n"
            "==============================\n"
            "- Be warm, respectful, at medyo casual: e.g., 'Sige po, tutulungan ko kayo diyan.'\n"
            "- Use simple Taglish, explain terms briefly if technical.\n"
            "- Don’t send long paragraphs. Prefer short bullet-style sentences when explaining steps.\n"
            "- Always keep track of the last issue the customer mentioned; don’t act like you forgot.\n\n"
            "==============================\n"
            "TOOLS USE CASES\n"
            "==============================\n"
            "1) user_extract_info\n"
            "- Call when the user **provides or updates** their details: name, email, contact number, address, car details, service type, payment type, and/or schedule (date and time).\n"
            "- Once details are saved, reuse them; do not re-ask unless the user corrects something.\n\n"
            "2) user_get_info\n"
            "- Call user_get_info when asked to recall details.\n"
            "3) extract_car_info\n"
            "- Call extract_car_info when the user provides new or updated car info (make/model/year/fuel type/transmission).\n"
            "4) lookup_tool\n"
            "- Use when the user asks about:\n"
            "  - Car symptoms or problems (ingay, usok, ilaw sa dashboard, mahina hatak, hindi lumalamig ang aircon, etc.).\n"
            "  - Car maintenance, PMS, parts, or secondhand car inspection questions.\n"
            "- Let lookup_tool handle the **technical explanation and diagnosis flow**.\n"
            "- After lookup_tool returns, give a short Taglish summary for the user and continue the conversation.\n"
            "5) faq_tool\n"
            "- Use when the user asks general MechaniGo questions:\n"
            "  - 'Ano po services niyo?', 'Saan kayo nagse-service?', 'Magkano usually PMS?', 'Available kayo weekends?'\n"
            "- Let faq_tool provide factual info (based on official content), then you reply concisely in Taglish.\n\n"
            "==============================\n"
            "FLOW & DECISION RULES\n"
            "==============================\n"
            "- Avoid calling multiple tools in the same turn unless absolutely necessary.\n"
            "- Do not call a tool if it would obviously return the same state (e.g., user repeats info you already saved).\n"
            "- If the user is just clarifying or saying 'thank you', you usually do **not** need to call any sub-agent.\n\n"
            "Booking-related guidance:\n"
            "- If the user says they want to book or schedule, guide them step-by-step:\n"
            "  1) Confirm what service they need.\n"
            "  2) Confirm or ask for car details if relevant.\n"
            "  3) Ask for location if missing.\n"
            "  4) Ask for schedule (date and time) if missing.\n"
            "  5) Ask for preferred payment method if missing.\n"
            "- Each time the user provides new info, call the appropriate tool **once**, then summarize briefly.\n\n"
            "Mechanic-related guidance:\n"
            "- If the main concern is the car issue, prioritize mechanic_agent first before pushing for booking.\n"
            "- Help the user understand the problem in simple terms, then **optionally** offer booking once they seem ready.\n\n"
            "==============================\n"
            "QUALITY & EFFICIENCY\n"
            "==============================\n"
            "- Target: helpful but short responses. Avoid long stories.\n"
            "- Never ignore existing memory (user info, car, schedule). Use it to sound consistent and avoid re-asking.\n"
            "- Only use tools when they clearly add value (save new info, diagnose, answer FAQs, or structure a booking).\n"
            "CURRENT STATE SNAPSHOT:\n"
            f"- User: {user_name}\n"
            f"- Email: {user_email}\n"
            f"- Contact: {user_contact}\n"
            f"- Service: {user_service_type}\n"
            f"- Car: {car}\n"
            f"- Location: {user_address}\n"
            f"- Schedule: {display_sched_date} @{display_sched_time}\n"
            f"- Payment: {user_payment}\n"
        )

        missing = []
        if not has_user_info:
            missing.append("name")
        if not has_email:
            missing.append("email")
        if not has_service:
            missing.append("service type")
        if not has_car:
            missing.append("car details")
        if not has_user_contact:
            missing.append("contact number")
        if not has_address:
            missing.append("service location")
        if not has_schedule:
            missing.append("schedule (date/time)")
        if not has_payment:
            missing.append("payment method")

        if not missing:
            prompt += (
            "STATUS: All required information is complete — Booking is ready!\n\n"
            "- Present a concise final confirmation of service need, car, location, date, time, and payment.\n"
            "- Thank the user and avoid calling any sub‑agents unless they request changes.\n"
            )
        else:
            prompt += "STATUS: Incomplete - still missing: " + ", ".join(missing) + ".\n"

        return prompt

    @staticmethod
    def _complete_user_data(user: User) -> bool:
        if not user or not user.uid:
            return False
        def filled(value: Optional[str]) -> bool:
            return bool(value and value.strip())
        return all([
            filled(user.name),
            filled(user.email),
            filled(user.contact_num),
            filled(user.service_type),
            filled(user.address),
            filled(user.schedule_date),
            filled(user.schedule_time),
            filled(user.payment),
            filled(user.car),
        ])

    async def inquire(self, inquiry: str):
        prev_email = (self.context.user_ctx.user_memory.email or "").strip()
        response = await Runner.run(
            starting_agent=self.agent,
            input=inquiry,
            context=self.context,
            session=self.session
        )
        
        new_email = (self.context.user_ctx.user_memory.email or "").strip()
        if new_email and new_email != prev_email:
            try:
                linked = self.link_session_by_email(new_email)
                if linked:
                    self.logger.info(f"Session linked via email: {new_email}.")
                else:
                    self.logger.info(f"No existing user found for email: {new_email}.")
            except Exception as e:
                self.logger.error(f"Error linking by email: {e}")

        if self._complete_user_data(self.context.user_ctx.user_memory):
            self.save_user_state()
        else:
            self.logger.info("User profile still incomplete, skipping BigQuery save.")
        return {
            "text": response.final_output,
            "usage": {
                "input_tokens": response.raw_responses[0].usage.input_tokens,
                "output_tokens": response.raw_responses[0].usage.output_tokens,
                "total_tokens": response.raw_responses[0].usage.total_tokens
            }
        }
    
    def save_user_state(self):
        try:
            user = self.context.user_ctx.user_memory
            if not user.uid:
                self.logger.warning("No user UID - skipping save.")
                return
            self.logger.info(f"Saving user to BigQuery: name={user.name}, uid={user.uid}")
            self.bq_client.upsert_user(
                table_name=self.table_name,
                user=user
            )
        except Exception as e:
            self.logger.info(f"Error saving user information: {e}")

    def sync_user_car(self, ctx: RunContextWrapper[MechaniGoContext]):
        car = ctx.context.mechanic_ctx.car_memory
        user = ctx.context.user_ctx.user_memory
        self.logger.info(f"Current car_memory: {car}")

        def mechanics_to_string():
            parts = [car.make, car.model]
            if car.year:
                parts.append(str(car.year))
            return " ".join(filter(None, parts)).strip()

        formatted = mechanics_to_string()
        if formatted:
            if user.car != formatted:
                user.car = formatted
            return formatted
        return user.car or ""

    def link_session_by_email(self, email: str) -> bool:
        try:
            if not self.bq_client:
                self.logger.warning("No BigQuery client set.")
                self.context.user_ctx.user_memory.email = email
                return False
            
            normalized = (email or "").strip()
            if not normalized:
                self.logger.info("Empty email provided; skipping link...")
                return False

            existing = self.bq_client.get_user_by_email(self.table_name, email=normalized)
            if not existing:
                self.logger.info(f"No existing user for email: {normalized}")
                self.context.user_ctx.user_memory.email = normalized
                return False

            current = self.context.user_ctx.user_memory
            if current and current.uid and existing.uid and current.uid != existing.uid:
                self.logger.info(
                    f"Switching session uid from {current.uid} to {existing.uid} (contact match)."
                )
            self.context.user_ctx.user_memory = existing
            self._hydrate_mechanic_car_from_user(existing)
            self.logger.info(f"Linked session to uid={existing.uid} for email={normalized}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to link by email: {e}")
            return False

    def _hydrate_mechanic_car_from_user(self, user: User):
        car_str = (user.car or "").strip()
        if not car_str:
            return

        parts = car_str.split()
        year = parts[-1] if parts and parts[-1].isdigit() else None
        make_model = parts[:-1] if year else parts

        mechanic_car = self.context.mechanic_ctx.car_memory
        if not mechanic_car:
            mechanic_car = self.context.mechanic_ctx.car_memory
        if year:
            mechanic_car.year = int(year)
        if make_model:
            mechanic_car.make = make_model[0]
            mechanic_car.model = " ".join(make_model[1:]) or None

    def _ensure_users_table(self):
        self.logger.info("Ensuring dataset and table in BigQuery...")
        schema = [
            bigquery.SchemaField("uid", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("email", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("address", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("service_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("contact_num", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("schedule_date", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("schedule_time", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("payment", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("car", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("raw_json", "STRING", mode="NULLABLE"),
        ]
        self.bq_client.ensure_dataset()
        self.bq_client.ensure_table(self.table_name, schema)
        self.logger.info("Done!")
    
    # For tests
    def _delete_table(self):
        full_id = f"{self.bq_client.client.project}.{self.bq_client.dataset_id}.{self.table_name}"
        self.logger.info(f"Deleting table {full_id}...")
        self.bq_client.client.delete_table(full_id)
        self.logger.info("Done dropping table!")
    
    # TOOLS
    # ========================
    # USER INFO TOOLS
    # ========================
    def _create_ctx_extract_user_tool(self):
        @function_tool
        def ctx_extract_user_info(
            ctx: RunContextWrapper[UserInfoAgentContext],
            name: Optional[str] = None,
            address: Optional[str] = None,
            email: Optional[str] = None,
            contact_num: Optional[str] = None,
            service_type: Optional[str] = None,
            schedule_date: Optional[str] = None,
            schedule_time: Optional[str] = None,
            payment: Optional[str] = None,
            car: Optional[str] = None
        ):
            return self._ctx_extract_user_info(ctx, name, address, email, contact_num, service_type, schedule_date, schedule_time, payment, car)
        return ctx_extract_user_info

    def _create_ctx_get_user_tool(self):
        @function_tool
        def ctx_get_user_info(ctx: RunContextWrapper[UserInfoAgentContext]):
            return self._ctx_get_user_info(ctx)
        return ctx_get_user_info

    def _ctx_extract_user_info(
        self,
        ctx: RunContextWrapper[Any],
        name: Optional[str] = None,
        address: Optional[str] = None,
        email: Optional[str] = None,
        contact_num: Optional[str] = None,
        service_type: Optional[str] = None,
        schedule_date: Optional[str] = None,
        schedule_time: Optional[str] = None,
        payment: Optional[str] = None,
        car: Optional[str] = None
    ):
        user = ctx.context.user_ctx.user_memory

        def norm(x): return (x or "").strip()

        incoming = {
            "name": norm(name) or None,
            "email": norm(email) or None,
            "address": norm(address) or None,
            "contact_num": norm(contact_num) or None,
            "service_type": norm(service_type) or None,
            "schedule_date": norm(schedule_date) or None,
            "schedule_time": norm(schedule_time) or None,
            "payment": norm(payment) or None,
            "car": norm(car) or None
        }
        current = {
            "name": norm(user.name) or None,
            "email": norm(user.email) or None,
            "address": norm(user.address) or None,
            "contact_num": norm(user.contact_num) or None,
            "service_type": norm(user.service_type) or None,
            "schedule_date": norm(user.schedule_date) or None,
            "schedule_time": norm(user.schedule_time) or None,
            "payment": norm(user.payment) or None,
            "car": norm(user.car) or None,
        }

        changed_fields = {}
        for field, new_val in incoming.items():
            if new_val is not None and new_val != current[field]:
                setattr(user, field, new_val)
                changed_fields[field] = new_val

        if not changed_fields:
            self.logger.info("========== _ctx_extract_user_info() ==========")
            self.logger.info(f"User unchanged: {user}")
            return {"status": "no_change", "message": "No updates needed.", "user": user}

        self.logger.info(f"Updated user memory: {user}")
        return {
            "status": "updated",
            "changed_fields": changed_fields,
            "user": user
        }

    def _ctx_get_user_info(self, ctx: RunContextWrapper[Any]):
        self.logger.info("========== _ctx_extract_user_info() ==========")
        user = ctx.context.user_ctx.user_memory
        if user and any([user.name, user.email, user.address, user.car, user.uid]):
            return {"status": "success", "user": user.model_dump()}
        return {"status": "not_found", "message": "No user data yet."}

    # TOOLS
    # ========================
    # MECHANIC TOOLS
    # ========================
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
                transmission
            )
        return extract_car_info

    def _extract_car_info(
        self,
        ctx: RunContextWrapper[Any],
        make: Optional[str] = None,
        model: Optional[str] = None,
        year: Optional[str] = None,
        fuel_type: Optional[str] = None,
        transmission: Optional[str] = None
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

    def _create_lookup_tool(self):
        @function_tool
        def lookup(question: str):
            return self._lookup(question)
        return lookup

    def _lookup(self, question: str):
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

    # ========================
    # FAQ TOOLS
    # ========================
    def _create_ask_tool(self):
        @function_tool
        def ask(question: str):
            return self._ask(question)

        return ask

    def _ask(self, question: str):
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