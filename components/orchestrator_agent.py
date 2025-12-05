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
from agents import Agent, Runner, RunContextWrapper, SQLiteSession
from config import DEFAULT_AGENT_HANDOFF_DESCRIPTION
from components.agent_tools import (
    UserInfoAgent, UserInfoAgentContext,
    MechanicAgent, MechanicAgentContext,
    BookingAgent, BookingAgentContext,
    FAQAgent
)
from components.utils import create_agent, BigQueryClient
from agents.model_settings import ModelSettings
from schemas import User, UserCarDetails
from google.cloud import bigquery
from pydantic import BaseModel
from typing import Optional
import openai
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
        model: str = "gpt-4.1",
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

        self.context = context
        self.user_info_agent = UserInfoAgent(api_key=self.api_key)
        self.mechanic_agent = MechanicAgent(api_key=self.api_key)
        self.faq_agent = FAQAgent(api_key=self.api_key)
        self.booking_agent = BookingAgent(api_key=self.api_key)
        model_settings = ModelSettings(max_tokens=1000)

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.handoff_description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tool_names=("mechanic_agent", "faq_agent", "user_info_agent", "booking_agent"),
            input_guardrails=self.input_guardrail,
            guardrail_names=("input_generic",),
            model_settings=model_settings
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
            "- Ikaw ang unang kausap ng customer. You understand their concern, reply in Taglish, and only call sub-agents when needed.\n"
            "- Use the information already saved (name, email, contact, address, car details, schedule, etc.) and avoid re-asking the same thing.\n"
            "- Aim for low token usage and low latency: short answers, minimal tool calls, and no unnecessary repetition.\n\n"
            "When the user sends a message, first decide:\n"
            "- Are they asking about their **car issue or car service**? (MechanicAgent)\n"
            "- Are they asking about **MechaniGo in general**? (FAQAgent)\n"
            "- Are they trying to **book or change an appointment**? (BookingAgent + UserInfoAgent)\n"
            "- Are they just giving or updating their **personal details**? (UserInfoAgent)\n\n"
            "==============================\n"
            "COMMUNICATION STYLE\n"
            "==============================\n"
            "- Be warm, respectful, at medyo casual: e.g., 'Sige po, tutulungan ko kayo diyan.'\n"
            "- Use simple Taglish, explain terms briefly if technical.\n"
            "- Don’t send long paragraphs. Prefer short bullet-style sentences when explaining steps.\n"
            "- Always keep track of the last issue the customer mentioned; don’t act like you forgot.\n\n"
            "==============================\n"
            "SUB-AGENT USE CASES\n"
            "==============================\n"
            "1) user_info_agent\n"
            "- Use when the user **provides or updates** their details: name, email, contact number, address and/or car details.\n"
            "- Do NOT ask for these details unless they are needed for the current goal (e.g., booking) and still missing.\n"
            "- Once details are saved, reuse them; do not re-ask unless the user corrects something.\n\n"
            "2) mechanic_agent\n"
            "- Use when the user asks about:\n"
            "  - Car symptoms or problems (ingay, usok, ilaw sa dashboard, mahina hatak, hindi lumalamig ang aircon, etc.).\n"
            "  - Car maintenance, PMS, parts, or secondhand car inspection questions.\n"
            "- Let mechanic_agent handle the **technical explanation and diagnosis flow**.\n"
            "- After mechanic_agent returns, give a short Taglish summary for the user and continue the conversation.\n\n"
            "- Whenever the user updates car details (even in free text), parse them and call mechanic_agent.\n"
            "3) booking_agent\n"
            "- Use when the user clearly wants to **book, confirm, or change** an appointment.\n"
            "- booking_agent is for extracting/saving:\n"
            "  - service type (PMS, secondhand inspection, diagnosis, parts replacement)\n"
            "  - schedule date and time\n"
            "  - payment method (cash, gcash, credit)\n"
            "- Only ask for these if they are still missing or the user is changing them.\n\n"
            "4) faq_agent\n"
            "- Use when the user asks general MechaniGo questions:\n"
            "  - 'Ano po services niyo?', 'Saan kayo nagse-service?', 'Magkano usually PMS?', 'Available kayo weekends?'\n"
            "- Let faq_agent provide factual info (based on official content), then you reply concisely in Taglish.\n\n"
            "==============================\n"
            "FLOW & DECISION RULES\n"
            "==============================\n"
            "- For each message, choose the **single most relevant** sub-agent to call, or answer directly if no tool is needed.\n"
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
            "- Each time the user provides new info, call the appropriate agent (user_info_agent or booking_agent) **once**, then summarize briefly.\n\n"
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

        # marked "dirty" whenever context changes
        self.session.context_dirty = True
        
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
            "model": self.agent.model,
            "model_settings": {
                "max_tokens": self.agent.model_settings.max_tokens
            },
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