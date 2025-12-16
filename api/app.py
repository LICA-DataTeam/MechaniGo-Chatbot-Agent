from contextlib import asynccontextmanager
from api import send_msg_router
from fastapi import FastAPI
from config import settings

from components import MechaniGoAgent, MechaniGoContext, UserInfoContext
from components.sub_agents import MechanicAgent, BookingAgent
from components.utils import SessionHandler, ToolRegistry
from components.schemas import User
from dataclasses import dataclass
from typing import Tuple, Dict
import uuid
import os

os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)

@dataclass
class AgentState:
    session: SessionHandler
    context: MechaniGoContext
    mechanic_agent: MechanicAgent
    booking_agent: BookingAgent

_AGENT_STATE: Dict[str, AgentState] = {}

def _initialize_session_context_and_sub_agents(session_id: str, user_id: str) -> Tuple[SessionHandler, MechaniGoContext]:
    """
    Helper method that initializes the session, context, and sub-agents during app startup.
    """
    model = "gpt-4.1-mini" # Sub-agents use mini
    resolved_user_id = user_id or session_id

    session = SessionHandler(session_id=session_id) # session_id == user_id
    ctx = MechaniGoContext(
        user_ctx=UserInfoContext(
            user_memory=User(uid=resolved_user_id)
        )
    )

    mechanic_agent = MechanicAgent(
        api_key=settings.OPENAI_API_KEY,
        model=model
    )

    booking_agent = BookingAgent(
        api_key=settings.OPENAI_API_KEY,
        model=model,
        context=ctx
    )

    return AgentState(
        session=session,
        context=ctx,
        mechanic_agent=mechanic_agent,
        booking_agent=booking_agent
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    def agent_factory(user_id: str | None) -> MechaniGoAgent:
        # Initialize and register the sub-agents on startup
        session_id = user_id or f"anon-{uuid.uuid4()}"
        state = _AGENT_STATE.get(session_id)
        if state is None:
            state = _initialize_session_context_and_sub_agents(
                session_id=session_id,
                user_id=user_id
            )
            _AGENT_STATE[session_id] = state
        
        ToolRegistry.register_tool(
            "mechanic_agent",
            state.mechanic_agent.as_tool,
            category="agent",
            description=state.mechanic_agent.get_handoff_description()
        )

        ToolRegistry.register_tool(
            "booking_agent",
            state.booking_agent.as_tool,
            category="agent",
            description=state.booking_agent.get_handoff_description()
        )
        return MechaniGoAgent(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            session=state.session,
            user_id=user_id,
            context=state.context
        )

    app.state.agent_factory = agent_factory
    yield

app = FastAPI(
    lifespan=lifespan,
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)

app.include_router(send_msg_router, prefix=f"{settings.API_PREFIX}/send", tags=["chatbot"])

@app.get("/")
def index():
    openai_api_key = settings.OPENAI_API_KEY
    model = settings.OPENAI_MODEL
    max_tokens = settings.OPENAI_MAX_TOKENS
    supabase_api_key = settings.SUPABASE_API_KEY
    return {
        "app": settings.APP_NAME,
        "debug": settings.DEBUG,
        "environment": settings.ENV,
        "openai_settings": {
            "api_key": openai_api_key, # delete later
            "chatbot_settings": {
                "model": model,
                "max_tokens": max_tokens
            }
        },
        "supabase_settings": {
            "api_key": supabase_api_key # delete later
        }
    }