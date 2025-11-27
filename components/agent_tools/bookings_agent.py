from __future__ import annotations

from components.utils import create_agent, register_tool
from agents import RunContextWrapper, function_tool
from typing import Optional, Any
from pydantic import BaseModel
from agents import Agent
import logging

from agents.model_settings import ModelSettings

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
        extract_service_type = self._create_ctx_extract_service()
        model_settings = ModelSettings(max_tokens=1000)

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tools=[extract_schedule, extract_payment_type, extract_service_type],
            model_settings=model_settings
        )

        self._orchestrator_tool = self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )

        register_tool(
            name="booking_agent",
            target=self._orchestrator_tool,
            description="Booking agent orchestrator hook.",
            scopes=("default",),
            overwrite=True,
        )

        register_tool(
            name="bookings_extract_schedule",
            target=extract_schedule,
            description="Parses and stores user schedule.",
            scopes=("booking_suite", "default"),
            overwrite=True,
        )

        register_tool(
            name="bookings_extract_payment",
            target=extract_payment_type,
            description="Extracts the user payment type.",
            scopes=("booking_suite", "default"),
            overwrite=True,
        )

        register_tool(
            name="bookings_extract_service",
            target=extract_service_type,
            description="Extracts the user service type.",
            scopes=("booking_suite", "default"),
            overwrite=True,
        )

    @property
    def as_tool(self):
        return self._orchestrator_tool

    def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[Any],
        agent: Agent
    ):
        self.logger.info("========== dynamic_instructions ==========")
        user = ctx.context.user_ctx.user_memory
        self.logger.info(f"User.name: {user.name}")
        self.logger.info(f"User.service_type: {user.service_type}")
        self.logger.info(f"User.schedule_date: {user.schedule_date}")
        self.logger.info(f"User.schedule_time: {user.schedule_time}")
        self.logger.info(f"User.payment: {user.payment}")
        prompt = (
                f"You are {agent.name}, a bookings and payment agent for MechaniGo.ph.\n\n"
                "Your task is to handle scheduling the booking and payment of the users.\n\n"
                # "Make sure all user information is provided before confirming their schedule and payment type.\n\n"
                "Call extract_service_type once when the customer provides the type of service they need.\n\n"
                "The only acceptable service type options are pms, secondhand car-buying inspection, parts replacement, and car diagnosis.\n\n"
                "Once the user decides what service they want, ask them the schedule and their payment type.\n\n"
                "Call extract_schedule once when the customer provides their schedule.\n\n"
                "Call extract_payment_type once when the customer provides their preferred payment type.\n\n"
                "If the user gives both date and time, immediately call extract_schedule even if the time isn't perfectly worded.\n\n"
                "When a user clarifies only one part (either date or time), reuse the other value from memory and call extract_schedule with both. Don't ask for the missing part again if it already exists in memory.\n\n"
                "If neither date nor time is stored, ask the user to provide both in the same message.\n\n"
                "Examples: \n\n"
                "User: I want to schedule an appointment for PMS.\n"
                "Call extract_service_type(service_type='pms')\n\n"
                "User: 'My schedule is October 12, 2025 at around 10 am'.\n"
                "Call extract_sched(schedule_date='October 12, 2025', schedule_time='10 am')\n\n"
                "If you're not 100% sure about the date or time correctly, ask the user to restate both.\n\n"
                "When calling extract_schedule, always pass the exact date/time strings the user just confirmed; do not infer or adjust them.\n\n"
                "extract_schedule returns 'status': 'error' if you call it without both date and time (and no stored value yet). If you encounter that, immediately ask the user for both date and time.\n\n"
                "When you already have a saved date or time, you can call extract_schedule with just the part the user is changing; the function reuses the stored value for the other part.\n\n"
                "The only acceptable payment options are GCash, Cash, or Credit. If the user gives anything else or it's unclear, ask them to restate.\n\n"
                "extract_payment_type returns 'status': 'error' when no payment value is provided; When you encounter this, ask the user again.\n\n"
                "After both schedule and payment are collected, confirm the details with the user and inform them their booking is complete.\n\n"
                "Example: \n\n"
                "User: 'Book me on October 30 at 12 PM.'\n"
                "User: 'Make it 1 PM instead'\n"
                "- Call extract_schedule(schedule_date='October 30', schedule_time='1 PM')\n\n"
            )
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
        """
        Validate, update, and report the user’s scheduled date/time in memory.

        :param ctx: Conversation context whose ``user_ctx.user_memory`` holds the schedule fields.
        :type ctx: RunContextWrapper[Any]
        :param schedule_date: New schedule date provided by the user, if any.
        :type schedule_date: Optional[str]
        :param schedule_time: New schedule time provided by the user, if any.
        :type schedule_time: Optional[str]
        :returns: Status payload describing whether the schedule was updated, kept unchanged, or rejected due to missing inputs.
        :rtype: dict
        """
        user = ctx.context.user_ctx.user_memory
        prev_date, prev_time = user.schedule_date, user.schedule_time

        def norm(x): return (x or "").strip()
        new_date_n, new_time_n = norm(schedule_date), norm(schedule_time)
        prev_date_n, prev_time_n = norm(prev_date), norm(prev_time)

        if not prev_date_n and not new_date_n:
            self.logger.info("========== _extract_sched() Missing date ==========")
            return {
                "status": "error",
                "message": "No schedule date provided. Ask the user to restate both date and time."
            }

        if not prev_time_n and not new_time_n:
            self.logger.info("========== _extract_sched() Missing time ==========")
            return {
                "status": "error",
                "message": "No schedule time provided. Ask the user to restate both date and time."
            }

        first_set = (not prev_date_n and new_date_n) or (not prev_time_n and new_time_n)

        if not first_set and new_date_n == prev_date_n and new_time_n == prev_time_n:
            self.logger.info("========== _extract_sched() No changes ==========")
            self.logger.info(f"User name: {user.name}")
            self.logger.info(f"Existing schedule unchanged: {prev_date} @{prev_time}")
            return {
                "status": "no_change",
                "message": "Schedule already set to the same values."
            }
        if schedule_date is not None:
            user.schedule_date = schedule_date
        if schedule_time is not None:
            user.schedule_time = schedule_time

        self.logger.info("========== _extract_sched() Called! ==========")
        self.logger.info(f"User name: {user.name}")
        self.logger.info(f"Date: {user.schedule_date} @{user.schedule_time}")

        return {
            "status": "success",
            "message": f"Schedule saved: {user.schedule_date} at {user.schedule_time}"
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
        """
        Validate, update, and report the user’s payment type in memory.

        :param ctx: Conversation context whose ``user_ctx.user_memory`` holds the schedule fields.
        :type ctx: RunContextWrapper[Any]
        :param payment: User's payment type.
        :type payment: Optional[str]
        :returns: Status payload describing whether the schedule was updated, kept unchanged, or rejected due to missing inputs.
        :rtype: dict
        """
        user = ctx.context.user_ctx.user_memory
        prev_payment_norm = (user.payment or "").strip().lower()
        new_payment_norm = (payment or "").strip().lower()

        if not new_payment_norm:
            self.logger.info("========== _extract_payment_type() ==========")
            self.logger.info(f"User name: {user.name}")
            self.logger.info("No payment value provided.")
            return {
                "status": "error",
                "message": "No payment method received. Please ask the user to specify GCash, Cash, or Credit."
            }

        first_set = (not prev_payment_norm) and bool(new_payment_norm)
        if not first_set and new_payment_norm  == prev_payment_norm:
            self.logger.info("========== _extract_payment_type() No change ==========")
            self.logger.info(f"User name: {user.name}")
            self.logger.info(f"Payment unchanged: {user.payment}")
            return {
                "status": "no_change",
                "message": "Payment method unchanged."
            }

        user.payment = payment
        self.logger.info("========== _extract_payment_type() Called! ==========")
        self.logger.info(f"User name: {user.name}")
        self.logger.info(f"Payment Type: {user.payment}")

        return {
            "status": "success",
            "message": f"Payment method saved: {user.payment}"
        }

    def _create_ctx_extract_service(self):
        @function_tool
        def ctx_extract_service(
            ctx: RunContextWrapper[Any],
            service_type: Optional[str] = None
        ):
            return self._extract_service_type(ctx, service_type)
        return ctx_extract_service

    def _extract_service_type(
        self,
        ctx: RunContextWrapper[Any],
        service_type: Optional[str] = None
    ):
        """
        Validate, update, and report the user’s service type in memory.

        :param ctx: Conversation context whose ``user_ctx.user_memory`` holds the schedule fields.
        :type ctx: RunContextWrapper[Any]
        :param service_type: User's preferred type of service.
        :type service_type: Optional[str]
        :returns: Status payload describing whether the schedule was updated, kept unchanged, or rejected due to missing inputs.
        """
        user = ctx.context.user_ctx.user_memory
        prev_service = (user.service_type or "").strip().lower()
        new_service = (service_type or "").strip().lower()

        if not new_service:
            self.logger.info(f"User: {user.name}")
            return {
                "status": "error",
                "message": "No service type. Please ask the user to specify their service."
            }

        first_set = (not prev_service) and bool(new_service)
        if not first_set and new_service == prev_service:
            self.logger.info("No change for service type.")
            return {
                "status": "no_change",
                "message": "Payment method unchanged"
            }

        user.service_type = service_type
        self.logger.info("========== extract_service_type() called ==========")
        self.logger.info(f"User: {user.name}")
        self.logger.info(f"Service Type: {user.service_type}")

        return {
            "status": "success",
            "message": f"Service type saved: {user.service_type}"
        }