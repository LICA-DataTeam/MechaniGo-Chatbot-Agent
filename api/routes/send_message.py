from api.common import (
    BackgroundTasks,
    HTTPException,
    JSONResponse,
    APIRouter,
    Request,
    Depends,
    status
)

from components import MechaniGoAgent
from utils import log_execution_time
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

class UserMessagePayload(BaseModel):
    message: str

router = APIRouter()

async def resolve_user_id(request: Request) -> str | None:
    user_id = request.headers.get("X-User-Id") # arbitrary
    return user_id.strip() if user_id else str(uuid4())

def get_agent(request: Request, user_id: str | None = Depends(resolve_user_id)):
    factory = getattr(request.app.state, "agent_factory", None)
    if factory is None:
        raise RuntimeError("Agent not initialized.")
    return factory(user_id)

@router.post("/send-message")
@log_execution_time("POST /api/v1/send/send-message")
async def send(
    bg_tasks: BackgroundTasks,
    payload: UserMessagePayload,
    user_id: Optional[str] = Depends(resolve_user_id),
    agent: MechaniGoAgent = Depends(get_agent)
):
    if not payload.message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message is required.")

    try:
        session_id = getattr(agent.session, "session_id", user_id)
        result = await agent.inquire(inquiry=payload.message)
        bg_tasks.add_task(agent.session.persist_items)
        return JSONResponse(
            content={
                "response": result.response,
                "session_id": session_id,
                "user_id": user_id,
                "model": result.model,
                "usage": result.usage.model_dump()
            }
        )
    except Exception as e:
        return JSONResponse(
            content={
                "message": str(e)
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )