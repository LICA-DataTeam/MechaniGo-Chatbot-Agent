from components.agent_tools import (
    UserInfoAgent, UserInfoAgentContext,
    MechanicAgent, MechanicAgentContext,
    BookingAgent, BookingAgentContext,
    FAQAgent
)
from components.utils import create_agent, BigQueryClient
from config import DEFAULT_AGENT_HANDOFF_DESCRIPTION
from agents import Agent, Runner, RunContextWrapper
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
        model: str = "gpt-4o-mini",
        context: Optional[MechaniGoContext] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or openai.api_key or None
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via environment variables.")

        self.bq_client = bq_client
        self.table_name = table_name
        self.name = name
        self.handoff_description = DEFAULT_AGENT_HANDOFF_DESCRIPTION
        self.model = model

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
        user_info_agent = UserInfoAgent(
            api_key=self.api_key,
            model=self.model
        )

        mechanic_agent = MechanicAgent(
            api_key=self.api_key
        )

        faq_agent = FAQAgent(
            api_key=self.api_key
        )

        booking_agent = BookingAgent(
            api_key=self.api_key
        )

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.handoff_description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tools=[user_info_agent.as_tool, mechanic_agent.as_tool, faq_agent.as_tool, booking_agent.as_tool]
        )
    
    async def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[MechaniGoContext],
        agent: Agent
    ):
        # raw values
        user_name = ctx.context.user_ctx.user_memory.name
        user_sched_date = ctx.context.user_ctx.user_memory.schedule_date
        user_sched_time = ctx.context.user_ctx.user_memory.schedule_time
        user_payment = ctx.context.user_ctx.user_memory.payment
        car = ctx.context.user_ctx.user_memory.car
        user_contact = ctx.context.user_ctx.user_memory.contact_num
        user_address = ctx.context.user_ctx.user_memory.address

        # Check completeness before setting display values
        self.logger.info("========== Verifying User Information ==========")
        has_user_info = user_name is not None and bool(user_name.strip())
        has_user_contact = user_contact is not None and bool(user_contact.strip())
        hass_address = user_address is not None and bool(user_address.strip())
        has_schedule = (
            user_sched_date is not None and bool(user_sched_date.strip()) and
            user_sched_time is not None and bool(user_sched_time.strip())
        )
        has_payment = user_payment is not None and bool(user_payment.strip())
        has_car = car is not None and bool(car.strip())

        display_name = user_name if has_user_info else "Unknown User"
        display_sched_date = user_sched_date if user_sched_date else "Unknown date"
        display_sched_time = user_sched_time if user_sched_time else "Unknown time"
        display_payment = user_payment if has_payment else "No payment"
        display_car = car if has_car else "No car specified"

        self.logger.info("========== DETAILS ==========")
        self.logger.info(f"User name: {display_name}")
        self.logger.info(f"Contact: {user_contact}")
        self.logger.info(f"Car summary: {display_car}")
        self.logger.info(f"Schedule: {display_sched_date} @{display_sched_time}")
        self.logger.info(f"Payment: {display_payment}")
        self.logger.info(f"Complete: user={has_user_info}, contact_num={has_user_contact}, car={has_car}, schedule={has_schedule}, payment={has_payment}")
        prompt = (
            f"You are {agent.name}, the main orchestrator agent and a helpful assistant for MechaniGo.ph, a business that offers home maintenance (PMS) and car-buying assistance.\n\n"
            "You lead the customer through a 3-step service flow and only call sub-agents when needed.\n\n"
            "BUSINESS FLOW (follow strictly in order):\n"
            "1) Get an estimate/quote\n\n"
            " - Understand what the car needs (diagnosis, maintenance, or car-buying help).\n"
            " - If the user's name or contact number is missing, politely ask for them and, once provided,\n"
            " call user_info_agent.ctx_extract_user_info(name=..., contact_num=...). Do not re-ask if saved.\n"
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
            "TOOLS AND WHEN TO USE THEM:\n"
            "- user_info_agent:\n"
            " - Use to extract/update user name, address, contact number.\n"
            " - Use ctx_get_user_info if the user asks you to recall saved details.\n"
            "- mechanic_agent:\n"
            " - Use when car details are missing/ambiguous or the user changes car info.\n"
            " - It can parse a free‑form car string into make/model/year and sync a car string into user memory.\n"
            "- booking_agent:\n"
            " - Use ctx_extract_sched right after the user gives schedule date/time.\n"
            " - Use ctx_extract_payment_type right after the user gives payment preference.\n"
            "- faq_agent:\n"
            " - Use to answer official FAQs. Quote the official answer from the KB.\n"
            " - After answering, continue the flow toward booking completion.\n\n"
            "MEMORY AND COMPLETENESS:\n"
            "- Before asking, check what's already in memory and avoid re‑asking.\n"
            "- Drive toward completeness: once service need + car + location + schedule + payment are known, the booking is ready.\n\n"
            "CURRENT STATE SNAPSHOT:\n"
            f"- User: {display_name}\n"
            f"- Contact: {user_contact or 'Not provided'}\n"
            f"- Car: {display_car}\n"
            f"- Location: {user_address or 'Not provided'}\n"
            f"- Schedule: {display_sched_date} @{display_sched_time}\n"
            f"- Payment: {display_payment}\n\n"
        )

        missing = []
        if not has_user_info:
            missing.append("name")
        if not has_car:
            missing.append("car details")
        if not has_user_contact:
            missing.append("contact number")
        if not hass_address:
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
            prompt += "STATUS: Incomplete — still missing: " + ", ".join(missing) + ".\n\n"
            prompt += (
            "NEXT‑ACTION POLICY:\n"
            "- If missing name or contact → ask for them, then call user_info_agent.ctx_extract_user_info(name=..., contact_num=...).\n"
            "- If missing car details → call mechanic_agent to extract/confirm car details (e.g., make/model/year).\n"
            "- If missing service location → use user_info_agent to save the address.\n"
            "- If missing schedule → after the user provides date/time, call booking_agent.ctx_extract_sched.\n"
            "- If missing payment → after the user provides a method, call booking_agent.ctx_extract_payment_type.\n"
            "- If the user asks FAQs at any point → use faq_agent, then resume this flow.\n\n"
            "COMMUNICATION STYLE:\n"
            "- Be friendly, concise, and proactive.\n"
            "- Briefly explain what you are doing and why, especially after tool calls.\n"
            "- Summarize updates after each tool call so the user knows what's saved.\n"
            )


        self.logger.info("========== Orchestrator Agent Prompt ==========")
        print(prompt)
        return prompt

    async def inquire(self, inquiry: str):
        prev_contact = (self.context.user_ctx.user_memory.contact_num or "").strip()
        response = await Runner.run(
            starting_agent=self.agent,
            input=inquiry,
            context=self.context
        )

        new_contact = (self.context.user_ctx.user_memory.contact_num or "").strip()
        if new_contact and new_contact != prev_contact:
            try:
                linked = self.link_session_by_contact(new_contact)
                if linked:
                    self.logger.info(f"Session linked via contact: {new_contact}.")
                else:
                    self.logger.info(f"No existing user found for contact: {new_contact}.")
            except Exception as e:
                self.logger.error(f"Error linking by contact_num: {e}")
        self.save_user_state()
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
            self.logger.info("User saved to BigQuery successfully!")
        except Exception as e:
            self.logger.error(f"Error saving user information: {e}")

    def link_session_by_contact(self, contact_num: str) -> bool:
        try:
            if not self.bq_client:
                self.logger.warning("No BigQuery client set.")
                self.context.user_ctx.user_memory.contact_num = contact_num
                return False
            
            normalized = (contact_num or "").strip()
            if not normalized:
                self.logger.info("Empty contact provided; skipping link...")
                return False

            existing = self.bq_client.get_user_by_contact_num(self.table_name, contact_num=normalized)
            if not existing:
                self.logger.info(f"No existing user for contact: {normalized}")
                self.context.user_ctx.user_memory.contact_num = normalized
                return False

            current = self.context.user_ctx.user_memory
            if current and current.uid and existing.uid and current.uid != existing.uid:
                self.logger.info(
                    f"Switching session uid from {current.uid} to {existing.uid} (contact match)."
                )
            self.context.user_ctx.user_memory = existing
            self.logger.info(f"Linked session to uid={existing.uid} for contact={normalized}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to link by contact: {e}")
            return False

    def _ensure_users_table(self):
        self.logger.info("Ensuring dataset and table in BigQuery...")
        schema = [
            bigquery.SchemaField("uid", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
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