from __future__ import annotations

from components.utils import create_agent, register_tool
from agents import RunContextWrapper, function_tool
from pydantic import BaseModel
from typing import Optional, Any
from schemas import User
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class UserInfoAgentContext(BaseModel):
    user_memory: User
    model_config = {
        "arbitrary_types_allowed": True
    }

class UserInfoAgent:
    """Handles all user info related processing."""
    def __init__(
        self,
        api_key: str,
        name: str = "user_info_agent",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key
        self.name = name
        self.model = model
        self.description = "Handles processing of user information."
        self.logger = logging.getLogger(__name__)

        extract_user_info = self._create_ctx_extract_user_tool()
        get_user_info = self._create_ctx_get_user_tool()

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=(
                "You are a user information memory agent.\n\n"
                "ABSOLUTE REQUIREMENT:\n"
                "Call extract_user_info when the customer provides their details\n\n"
                "You can call get_user_info when asked to recall details\n\n"
                "Do not re-ask for details that are already known\n\n"
                "IMPORTANT: Do not attempt to guess or extract the uid, the uid is always generated internally by the system."
            ),
            output_type=User,
            model=self.model,
            tools=[extract_user_info, get_user_info],
        )

        self._orchestrator_tool = self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )

        register_tool(
            name="user_info_agent",
            target=self._orchestrator_tool,
            description="User information agent orchestrator hook.",
            scopes=("default",),
            overwrite=True,
        )

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

    @property
    def as_tool(self):
        return self._orchestrator_tool

    def _create_ctx_extract_user_tool(self):
        @function_tool
        def ctx_extract_user_info(
            ctx: RunContextWrapper[UserInfoAgentContext],
            name: Optional[str] = None,
            address: Optional[str] = None,
            email: Optional[str] = None,
            contact_num: Optional[str] = None,
            schedule_date: Optional[str] = None,
            schedule_time: Optional[str] = None,
            payment: Optional[str] = None,
            car: Optional[str] = None
        ):
            return self._ctx_extract_user_info(ctx, name, address, email, contact_num, schedule_date, schedule_time, payment, car)
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
        user = ctx.context.user_ctx.user_memory
        if user and any([user.name, user.email, user.address, user.car, user.uid]):
            return {"status": "success", "user": user.model_dump()}
        return {"status": "not_found", "message": "No user data yet."}