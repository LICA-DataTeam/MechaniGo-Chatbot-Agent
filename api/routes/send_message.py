from api.common import (
    JSONResponse,
    BaseModel,
    APIRouter,
    PH_TZ,
    status,
    Query
)
from utils import (
    record_session_tokens,
    record_response_time,
    record_request,
    record_session,
)

from api.routes.utils import run as run_chatbot
from components.utils import SessionHandler
from helpers import save_convo
from datetime import datetime
import logging
import time

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
    start = time.perf_counter()
    try:
        record_request()
        session = SessionHandler(
            session_id=session_id,
            dataset_id=DATASET_NAME,
            table_name="chatbot_api_test"
        )
        record_session(session_id=session.session_id)
        session.current_turn_ts = sender_ts
        response = await run_chatbot(inquiry=payload.message, session=session)
        if "usage" in response:
            record_session_tokens(session.session_id, response["usage"])
        try:
            chat_history = [
                {"role": "user", "message": payload.message, "timestamp": sender_ts},
                {"role": "assistant", "message": response.get("text", ""), "timestamp": sender_ts}
            ]
            save_convo(
                dataset_id=DATASET_NAME,
                table_name="chatbot_chat_history_test_2",
                uid=session.session_id,
                entries=chat_history
            )
        except Exception as e:
            logging.error(f"Exception while saving: {e}")
        record_response_time(start)
        return JSONResponse(
            content={
                "status": "Success",
                "sender_ts": sender_ts.isoformat(),
                "data": {
                    "sender_message": payload.message,
                    "session_id": session.session_id,
                    "response": response.get("text", "")
                }
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Exception occurred: {e}")
        return JSONResponse(
            content={
                "status": "Error"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )