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
        self.session = SQLiteSession(session_id=str(uuid.uuid4()), db_path="conversations.db")
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

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.handoff_description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tool_names=("mechanic_agent", "faq_agent", "user_info_agent", "booking_agent"),
            input_guardrails=self.input_guardrail,
            guardrail_names=("input_generic",),
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
        car = self.sync_user_car(ctx)

        user_email = ctx.context.user_ctx.user_memory.email
        user_contact = ctx.context.user_ctx.user_memory.contact_num
        user_address = ctx.context.user_ctx.user_memory.address

        # Check completeness before setting display values
        self.logger.info("========== VERIFYING USER INFORMATION ==========")
        has_user_info = user_name is not None and bool(user_name.strip())
        has_email = user_email is not None and bool(user_email.strip())
        has_user_contact = user_contact is not None and bool(user_contact.strip())
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
        display_sched_date = user_sched_date if has_schedule else "Unknown date"
        display_sched_time = user_sched_time if has_schedule else "Unknown time"
        display_address = user_address if has_address else "No address"
        display_payment = user_payment if has_payment else "No payment"
        display_car = car if has_car else "No car specified"

        self.logger.info("========== DETAILS ==========")
        self.logger.info(f"Complete: user={display_name}, email={display_email}, contact_num={display_contact}, car={display_car}, schedule={display_sched_date} @{display_sched_time}, payment={display_payment}, address={display_address}")
        prompt = (
            f"You are {agent.name}, the main orchestrator agent and a helpful assistant for MechaniGo.ph.\n"
            "FAQ HANDLING:\n"
            " - If the user asks a general MechaniGo question (e.g., location, hours, pricing, services) — especially at the very start — immediately use faq_agent to answer.\n"
            " - After responding, return to the service flow to answer more inquiries.\n\n"
            "You lead the customer through a 3-step service flow and only call sub-agents when **NEEDED**.\n"
            "BUSINESS FLOW (follow strictly): \n\n"
            "1) Get an estimate/quote\n\n"
            " - Understand what the car needs (diagnosis, maintenance, or car-buying help).\n"
            " - If the user's name email, or contact number is missing, politely ask for them and, once provided,\n"
            " call user_info_agent.ctx_extract_user_info(name=..., email=..., contact_num=...). Do not re-ask if saved.\n"
            " - Ensure car details are known. If missing or ambiguous, call mechanic_agent to parse/collect car details.\n"
            " - Provide a transparent, ballpark estimate and clarify it is subject to confirmation on site.\n"
            " - If the user asks general questions, you may use faq_agent to answer, then return to the main flow.\n\n"
            "2) Book an Appointment\n"
            " - Ask for service location (home or office). Save it with user_info_agent if given.\n"
            " - Ask for preferred date and time; when provided, call booking_agent.ctx_extract_sched to save schedule.\n"
            " - Ask for preferred payment type (GCash, cash, credit); when provided, call booking_agent.ctx_extract_payment_type.\n"
            " - Never re‑ask for details already in memory.\n\n"
            "3) Expert Service at Your Door\n"
            " - Confirm that a mobile mechanic will come equipped, perform the job efficiently, explain work, and take care of the car.\n"
            " - Provide a clear confirmation summary (service need, car, location, date, time, payment).\n"
            " - If the user requests changes, use the appropriate sub‑agent to update, then re‑confirm.\n\n"
            "MechanicAgent HANDLING:\n"
            "- If the user has a car related issue or question, always call mechanic_agent.\n"
            "- After responding, optionally ask their car detail.\n\n"
            " - It has its own internal lookup tool that can answer any car related inquiries.\n"
            " - It can search the web and use a file-based vector store to answer car-related questions, including topics like diagnosis and maintenance.\n"
            " - ALWAYS use the output of mechanic_agent when answering car related inquiries.\n"
            " - If mechanic_agent does not return any relevant information, use your own knowledge base/training data as a LAST RESORT.\n"
            "TOOLS AND WHEN TO USE THEM:\n"
            "- user_info_agent:\n"
            " - When the user provides their details (e.g., name, email, contact address), always call user_info_agent.\n"
            "- booking_agent:\n"
            " - Use when the user provides their schedule date/time.\n"
            " - Use when the user provides payment preference.\n"
            "- mechanic_agent:\n"
            " - When the user seeks assistance/questions for any car related issues, diagnostic, troubleshooting, etc.(e.g., 'My car's engine is smoking, can you assist me?')\n"
            " - Whenever the user updates car details (even in free text), parse them and call mechanic_agent.\n"
            " - It can parse a free-form car string into make/model/year.\n"
            " - After a successful extraction of car information, summarize the saved fields.\n"
            " - **Do not reset the topic** or ask what they want to ask again if an issue was already provided earlier.\n"
            "   - Example:\n"
            "       - User: 'My aircon is getting warmer.'\n"
            "       - You: 'Can I get your car details?'\n"
            "       - User: 'Ford Everest 2015.'\n"
            "       - After mechanic_agent returns car info, respond like: 'Got it—Ford Everest 2015. Since you mentioned your aircon is getting warmer, here’s what we can check…'\n"
            "- faq_agent:\n"
            " - Use to answer official FAQs. Quote the official answer.\n"
            " - After answering, continue the flow.\n\n"
            "MEMORY AND COMPLETENESS:\n"
            " - Check what's already in memory and avoid re-asking.\n"
            " - Always retain and reference the customer’s last described problem or issue (e.g., 'engine light on', 'aircon not cooling', 'strange noise').\n"
            " - Check what's already in memory and avoid re-asking questions unnecessarily.\n"
            " - Maintain continuity between tool calls. The customer should feel like the conversation flows naturally without restarting.\n\n"
            " - Drive toward completeness: once service need + car + location + schedule + payment are known, the booking is ready.\n\n"
            "SCOPE:\n"
            "Currently, you only handle the following agents: user_info_agent, mechanic_agent and faq_agent.\n"
            "You need to answer a customer's general inquiries about MechaniGo (FAQs) and car-related questions (e.g., PMS, diagnosis and troubleshooting).\n"
            "If they ask about booking related questions (i.e., they want to book an appointment for PMS or Secondhand car-buying), ask for their information first (name, email, contact, address, etc.)\n"
            "COMMUNICATION STYLE:\n"
            "- Always introduce yourself to customers cheerfully and politely.\n"
            "- Be friendly, concise, and proactive.\n"
            "- The customer may speak in English, Filipino, or a mix of both. Expect typos and slang.\n"
            "- Use a mix of casual and friendly Tagalog and English as appropriate in a cheerful and polite conversational tone, occasionally using 'po' to show respect, regardless of the customer's language.\n"
            "- Summarize updates after each tool call so the user knows what's saved.\n\n"
            "- If the user asks FAQs at any point → use faq_agent, then resume this flow.\n"
            "- Only call a sub-agent if it will capture missing information or update fields the user explicitly changed.\n"
            "- If a tool returns no_change, do not call it again this turn.\n\n"
            "CURRENT STATE SNAPSHOT:\n"
            f"- User: {user_name}\n"
            f"- Email: {user_email}\n"
            f"- Contact: {user_contact}\n"
            f"- Car: {display_car}\n"
            f"- Location: {user_address}\n"
            f"- Schedule: {display_sched_date} @{display_sched_time}\n"
            f"- Payment: {display_payment}\n"
        )

        missing = []
        if not has_user_info:
            missing.append("name")
        if not has_email:
            missing.append("email")
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
        return response.final_output
    
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