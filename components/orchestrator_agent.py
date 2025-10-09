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
from pydantic import BaseModel
from typing import Optional
import openai
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TABLE_NAME = "chatbot_users_table"

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
        name: str = "MechaniGo Bot",
        model: str = "gpt-4o-mini",
        context: Optional[MechaniGoContext] = None
    ):
        self.api_key = api_key or openai.api_key or None
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via environment variables.")

        self.bq_client = bq_client
        self.name = name
        self.handoff_description = DEFAULT_AGENT_HANDOFF_DESCRIPTION
        self.model = model

        if not context:
            context = MechaniGoContext(
                user_ctx=UserInfoAgentContext(
                    user_memory=User(uid=str(uuid.uuid4())),
                    bq_client=self.bq_client,
                    table_name=TABLE_NAME
                ),
                mechanic_ctx=MechanicAgentContext(
                    car_memory=UserCarDetails()
                ),
                booking_ctx=BookingAgentContext()
            )

        self.context = context
        user_info_agent = UserInfoAgent(
            api_key=self.api_key,
            bq_client=self.bq_client,
            table_name=TABLE_NAME,
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
        self.logger = logging.getLogger(__name__)
    
    async def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[MechaniGoContext],
        agent: Agent
    ):
        # user_name = ctx.context.user_ctx.user_memory.name or "Unknown User"
        # user_sched_date = ctx.context.user_ctx.user_memory.schedule_date or "Unknown date"
        # user_sched_time = ctx.context.user_ctx.user_memory.schedule_time or "Unknown time"
        # user_payment = ctx.context.user_ctx.user_memory.payment or "No payment"
        # car = ctx.context.user_ctx.user_memory.car

        # raw values
        user_name = ctx.context.user_ctx.user_memory.name
        user_sched_date = ctx.context.user_ctx.user_memory.schedule_date
        user_sched_time = ctx.context.user_ctx.user_memory.schedule_time
        user_payment = ctx.context.user_ctx.user_memory.payment
        car = ctx.context.user_ctx.user_memory.car

        # Check completeness before setting display values
        self.logger.info("========== Verifying User Information ==========")
        has_user_info = user_name is not None and bool(user_name.strip())
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
        self.logger.info(f"Car summary: {display_car}")
        self.logger.info(f"Schedule: {display_sched_date} @{display_sched_time}")
        self.logger.info(f"Payment: {display_payment}")
        self.logger.info(f"Complete: user={has_user_info}, car={has_car}, schedule={has_schedule}, payment={has_payment}")
        prompt = (
            f"You are {agent.name}, the main orchestrator agent and a helpful assistant for MechaniGo.ph, a business that offers home maintenance (PMS) and car-buying assistance.\n\n"
            "You are the customer-facing agent which handles responding to customer inquiries.\n\n"
            "CURRENT STATE:\n\n"
        )

        if has_user_info:
            prompt += f"- User: {display_name}\n"
        else:
            prompt += f"- User: not provided yet.\n"

        if has_car:
            prompt += f"- Car: {display_car}\n"
        else:
            prompt += "- Car: not provided yet.\n"

        if has_schedule:
            prompt += f"- Schedule: {display_sched_date} @{display_sched_time}\n"
        else:
            prompt += "- Schedule: not provided yet.\n"

        if has_payment:
            prompt += f"- Payment: {display_payment}\n\n"
        else:
            prompt += f"- Payment: not provided yet.\n\n"

        if has_user_info and has_schedule and has_payment and has_car:
            prompt += (
                "All information is complete - Booking is ready!\n\n"
                "- Call user_info_agent again to update and save the user information.\n\n"
                "If user confirms, simply acknowledge and thank them. DO NOT call any sub-agents again.\n\n"
                "If user wants to modify something, use the appropriate sub-agent.\n\n"
                "- You can still use faq_agent if the user has questions.\n\n"
                "- Provide clear, friendly responses.\n\n"
            )
        else:
            prompt += (
                "Incomplete - the following information still needs to be collected:\n\n"
            )

            if not has_user_info:
                prompt += "- Use user_info_agent to collect user details (name, address, contact).\n"
            if not has_car:
                prompt += "- Use mechanic_agent to collect car information.\n"
            if not has_schedule or not has_payment:
                prompt += "- Use booking_agent to collect schedule and payment.\n"

            prompt += (
                "\nINSTRUCTIONS:\n"
                "- Use the appropriate sub-agent to collect missing information.\n"
                "- Provide clear and concise responses to the customer.\n"
                "- Do not call sub-agents unnecessarily if the information is already saved.\n"
                "- You can still use faq_agent if the user has questions.\n\n"
            )
        self.logger.info("========== Orchestrator Agent Prompt ==========")
        print(prompt)
        return prompt

    async def inquire(self, inquiry: str):
        response = await Runner.run(
            starting_agent=self.agent,
            input=inquiry,
            context=self.context
        )
        return response.final_output