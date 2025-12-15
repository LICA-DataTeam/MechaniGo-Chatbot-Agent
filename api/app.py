from contextlib import asynccontextmanager
from api import send_msg_router
from fastapi import FastAPI
from config import settings

from components import MechaniGoAgent, MechaniGoContext, UserInfoContext
from components.sub_agents import MechanicAgent, BookingAgent
from components.utils import SessionHandler, ToolRegistry
from components.schemas import User
import uuid
import os

os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)

def initialize_and_register_sub_agents() -> None:
    """
    Method to initialize and register the sub-agents to the main registry during app startup.
    """
    model = "gpt-4.1-mini" # Sub-agents use mini

    mechanic_agent = MechanicAgent(
        api_key=settings.OPENAI_API_KEY,
        model=model
    )

    booking_agent = BookingAgent(
        api_key=settings.OPENAI_API_KEY,
        model=model
    )

    ToolRegistry.register_tool(
        "mechanic_agent",
        mechanic_agent.as_tool,
        category="agent",
        description=mechanic_agent.get_handoff_description()
    )

    ToolRegistry.register_tool(
        "booking_agent",
        booking_agent.as_tool,
        category="agent",
        description=booking_agent.get_handoff_description()
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and register the sub-agents on startup
    initialize_and_register_sub_agents()
    def session_factory(user_id: str) -> SessionHandler:
        return SessionHandler(session_id=user_id)

    def agent_factory(user_id: str | None) -> MechaniGoAgent:
        session_id = user_id or f"anon-{uuid.uuid4()}"
        ctx = MechaniGoContext(
            user_ctx=UserInfoContext(
                user_memory=User(uid=user_id)
            )
        )
        return MechaniGoAgent(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            session=session_factory(session_id),
            user_id=user_id or f"anon-{uuid.uuid4()}",
            context=ctx
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