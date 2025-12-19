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

REQUIRED_TOOLS = {
    "booking_agent",
    "mechanic_agent",
    "knowledge.faq_tool"
}

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
        model=settings.OPENAI_MODEL
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

def _register_agents(state: AgentState) -> None:
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

def _missing_tools() -> set[str]:
    registered = set(ToolRegistry.list_tools().keys())
    return REQUIRED_TOOLS - registered

# used for dev (health check)
def _warm_tools(app: FastAPI):
    if _missing_tools():
        app.state.agent_factory(user_id="health_check")

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
        _register_agents(state)
        return MechaniGoAgent(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            session=state.session,
            user_id=user_id,
            context=state.context
        )

    app.state.agent_factory = agent_factory
    _warm_tools(app)
    yield

app = FastAPI(
    lifespan=lifespan,
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)

app.include_router(send_msg_router, prefix=f"{settings.API_PREFIX}/send", tags=["chatbot"])

if settings.ENV == "development":
    @app.get("/", tags=["health"])
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

    @app.get("/health", tags=["health"])
    def health_check():
        _warm_tools(app)
        missing = _missing_tools()
        warnings = []
        errors = []

        if missing:
            warnings.append(f"Missing tools: {', '.join(missing)}")

        if not settings.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY missing")

        status = "ok"
        if errors:
            status = "error"
        elif warnings:
            status = "warnings found"

        return {
            "status": status,
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "warnings": warnings,
            "error": errors
        }

@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENV,
        "status": "running"
    }