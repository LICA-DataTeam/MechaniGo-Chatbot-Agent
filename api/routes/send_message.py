from api.common import (
    JSONResponse,
    BaseModel,
    APIRouter,
    PH_TZ,
    status,
    Query
)

from api.routes.utils import run as run_chatbot
from components.utils import SessionHandler
from datetime import datetime
import logging

# for testing
from config import DATASET_NAME

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
    sender_ts = datetime.now(tz=PH_TZ)
    try:
        session = SessionHandler(
            session_id=session_id,
            dataset_id=DATASET_NAME,
            table_name="chatbot_api_test"
        )
        session.current_turn_ts = sender_ts
        response = await run_chatbot(inquiry=payload.message, session=session)
        return JSONResponse(
            content={
                "status": "Success",
                "sender_ts": sender_ts.isoformat(),
                "data": {
                    "sender_message": payload.message,
                    "session_id": session.session_id,
                    "response": response
                }
            }
        )
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        return JSONResponse(
            content={
                "status": "Error"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )