from api.common import (
    InputGuardrailTripwireTriggered,
    SQLiteSession,
    JSONResponse,
    BaseModel,
    APIRouter,
    PH_TZ,
    status,
    Query
)

from api.routes.utils import run as run_chatbot
from datetime import datetime
from uuid import uuid4
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

router = APIRouter()

class UserMessage(BaseModel):
    message: str

# - session ID
# - message
@router.post("/send-message")
async def send(
    payload: UserMessage,
    session_id: str = Query(None, description="Session ID for context")
):
    try:
        if session_id:
            session = SQLiteSession(session_id=session_id, db_path="conversations.db")
        else:
            session = SQLiteSession(session_id=str(uuid4()), db_path="conversations.db")
            session_id = session.session_id
    except Exception as e:
        logging.error(f"Error initializing session: {e}")
        return JSONResponse(
            content={
                "status": "Error initializing session."
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    try:
        response = await run_chatbot(inquiry=payload.message, session=session)
        return JSONResponse(
            content={
                "status": "Success",
                "sender_ts": datetime.now(tz=PH_TZ).isoformat(),
                "data": {
                    "sender_message": payload.message,
                    "session_id": session_id,
                    "response": response
                }
            }
        )
    except Exception as e:
        logging.info(f"Exception: {e}")
        return JSONResponse(
            content={
                "status": "Error"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )