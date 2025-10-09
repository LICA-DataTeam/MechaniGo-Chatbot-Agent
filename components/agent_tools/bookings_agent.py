from agents import RunContextWrapper, function_tool
from components.utils import create_agent
from typing import Optional, Any
from pydantic import BaseModel
from agents import Agent
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class BookingAgentContext(BaseModel):
    pass

class BookingAgent:
    """Handles booking and payment processing."""
    def __init__(
        self,
        api_key: str,
        name: str = "booking_agent",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key
        self.name = name
        self.model = model
        self.description = "Handles booking and payment processing."
        self.logger = logging.getLogger(__name__)

        extract_schedule = self._create_ctx_extract_sched()
        extract_payment_type = self._create_ctx_extract_payment_type()
        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tools=[extract_schedule, extract_payment_type]
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
        self.logger.info("========== dynamic_instructions ==========")
        user = ctx.context.user_ctx.user_memory
        user_payment_string = getattr(ctx.context.user_ctx.user_memory, "payment", None)
        self.logger.info(f"User.name: {user.name}")
        self.logger.info(f"User.schedule_date: {user.schedule_date}")
        self.logger.info(f"User.schedule_time: {user.schedule_time}")
        self.logger.info(f"User.payment: {user_payment_string}")
        prompt = (
                f"You are {agent.name}, a bookings and payment agent for MechaniGo.ph.\n\n"
                "Your task is to handle scheduling the booking and payment of the users.\n\n"
                # "Make sure all user information is provided before confirming their schedule and payment type.\n\n"
                "Once the user decides what service they want, ask them the schedule and their payment type.\n\n"
                "Call extract_schedule once when the customer provides their schedule.\n\n"
                "Call extract_payment_type once when the customer provides their preferred payment type.\n\n"
                "After both schedule and payment are collected, confirm the details with the user and inform them their booking is complete.\n\n"
            )
        self.logger.info("========== BookingAgent prompt ==========")
        self.logger.info(prompt)
        return prompt

    def _create_ctx_extract_sched(self):
        @function_tool
        def ctx_extract_sched(
            ctx: RunContextWrapper[Any],
            schedule_date: Optional[str] = None,
            schedule_time: Optional[str] = None
        ):
            return self._extract_sched(ctx, schedule_date, schedule_time)
        return ctx_extract_sched

    def _extract_sched(
        self,
        ctx: RunContextWrapper[Any],
        schedule_date: Optional[str] = None,
        schedule_time: Optional[str] = None
    ):
        user = ctx.context.user_ctx.user_memory
        user.schedule_date = schedule_date
        user.schedule_time = schedule_time
        self.logger.info("========== _extract_sched() Called! ==========")
        self.logger.info(f"User name: {user.name}")
        if schedule_time or schedule_date:
            self.logger.info(f"Date: {schedule_date} @{schedule_time}")
        else:
            self.logger.info("No schedule yet!")

        return {
            "status": "success",
            "message": f"Schedule saved: {schedule_date} at {schedule_time}"
        }

    def _create_ctx_extract_payment_type(self):
        @function_tool
        def ctx_extract_payment_type(
            ctx: RunContextWrapper[Any],
            payment: Optional[str] = None
        ):
            return self._extract_payment_type(ctx, payment)
        return ctx_extract_payment_type

    def _extract_payment_type(
        self,
        ctx: RunContextWrapper[Any],
        payment: Optional[str] = None
    ):
        user = ctx.context.user_ctx.user_memory
        user.payment = payment
        self.logger.info("========== _extract_payment_type() Called! ==========")
        self.logger.info(f"User name: {user.name}")
        if payment:
            self.logger.info(f"Payment Type: {payment}")
        else:
            self.logger.info("No payment yet!")

        return {
            "status": "success",
            "message": f"Payment method saved: {payment}"
        }