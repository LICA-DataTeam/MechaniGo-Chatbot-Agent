from agents import RunContextWrapper, function_tool
from components.utils import create_agent
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

    @property
    def as_tool(self):
        return self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )

    def _create_ctx_extract_user_tool(self):
        @function_tool
        def ctx_extract_user_info(
            ctx: RunContextWrapper[UserInfoAgentContext],
            name: Optional[str] = None,
            address: Optional[str] = None,
            contact_num: Optional[str] = None,
            schedule_date: Optional[str] = None,
            schedule_time: Optional[str] = None,
            payment: Optional[str] = None,
            car: Optional[str] = None
        ):
            return self._ctx_extract_user_info(ctx, name, address, contact_num, schedule_date, schedule_time, payment, car)
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
            "address": norm(address) or None,
            "contact_num": norm(contact_num) or None,
            "schedule_date": norm(schedule_date) or None,
            "schedule_time": norm(schedule_time) or None,
            "payment": norm(payment) or None,
            "car": norm(car) or None
        }
        current = {
            "name": norm(user.name) or None,
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
        if user and any([user.name, user.address, user.car, user.uid]):
            return {"status": "success", "user": user.model_dump()}
        return {"status": "not_found", "message": "No user data yet."}