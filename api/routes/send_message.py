from api.common import (
    JSONResponse,
    BaseModel,
    APIRouter,
    PH_TZ,
    status,
    Query,
    Request
)
from utils import (
    record_session_tokens,
    record_response_time,
    record_request,
    record_session,
    get_metrics
)

from api.routes.utils import run as run_chatbot
from components.utils import SessionHandler
from helpers import save_convo
from datetime import datetime
import logging
import time

# for testing
from config import DATASET_NAME

from fastapi import BackgroundTasks
from starlette.concurrency import run_in_threadpool

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
    request: Request,
    session_id: str = Query(None, description="Session ID for context")
):
    metrics_sink = request.app.state.metrics_sink
    sender_ts = datetime.now(tz=PH_TZ)
    start = time.perf_counter()
    try:
        record_request()
        session = SessionHandler(
            session_id=session_id,
            dataset_id=DATASET_NAME,
            table_name="chatbot_api_test",
            bq_client=metrics_sink.client
        )
        record_session(session_id=session.session_id)
        session.current_turn_ts = sender_ts
        response = await run_chatbot(api_key=request.app.state.api_key, inquiry=payload.message, session=session, bq_client=metrics_sink.client)
        model = response.get("model")
        if usage := response.get("usage"):
            record_session_tokens(session.session_id, response["usage"])
            metrics_sink.record_usage(
                session_id=session.session_id,
                usage=usage,
                model=model
            )
        try:
            chat_history = [
                {"role": "user", "message": payload.message, "timestamp": sender_ts},
                {"role": "assistant", "message": response.get("text", ""), "timestamp": sender_ts}
            ]
            save_convo(
                dataset_id=DATASET_NAME,
                table_name="chatbot_chat_history_test_2",
                uid=session.session_id,
                entries=chat_history,
                bq_client=metrics_sink.client
            )
        except Exception as e:
            logging.error(f"Exception while saving: {e}")
        elapsed = record_response_time(start)
        metrics_sink.record_session(
            session_id=session.session_id,
            request_ts=sender_ts,
            response_latency=elapsed,
            status="success",
            extra={
                "request_count": get_metrics().get("request_count")
            }
        )
        return JSONResponse(
            content={
                "status": "Success",
                "sender_ts": sender_ts.strftime("%Y-%m-%d %H:%M:%S"),
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
        metrics_sink.record_session(  # optional failure logging
            session_id=session.session_id if "session" in locals() else session_id or "",
            request_ts=sender_ts,
            response_latency=time.perf_counter() - start,
            status="error",
        )
        return JSONResponse(
            content={
                "status": "Error"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# testing: using bacgkround tasks
@router.post("/send-msg-bgt")
async def send(
    payload: UserMessage,
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str = Query(None, description="Session ID for context")
):
    metrics_sink = request.app.state.metrics_sink
    sender_ts = datetime.now(tz=PH_TZ)
    start = time.perf_counter()

    record_request()
    session = SessionHandler(
        session_id=session_id,
        dataset_id=DATASET_NAME,
        table_name="chatbot_api_test",
        bq_client=metrics_sink.client,
    )
    record_session(session_id=session.session_id)
    session.current_turn_ts = sender_ts

    response = await run_chatbot(
        api_key=request.app.state.api_key,
        inquiry=payload.message,
        session=session,
        bq_client=metrics_sink.client,
    )
    model = response.get("model")
    usage = response.get("usage")

    chat_history = [
        {"role": "user", "message": payload.message, "timestamp": sender_ts},
        {"role": "assistant", "message": response.get("text", ""), "timestamp": sender_ts},
    ]

    async def log_save_convo():
        await run_in_threadpool(
            save_convo,
            DATASET_NAME,
            "chatbot_chat_history_test_2",
            session.session_id,
            chat_history,
            metrics_sink.client,
        )

    async def log_usage():
        if usage:
            record_session_tokens(session.session_id, usage)
            await run_in_threadpool(
                metrics_sink.record_usage,
                session_id=session.session_id,
                usage=usage,
                model=model,
            )

    async def log_session(elapsed: float):
        await run_in_threadpool(
            metrics_sink.record_session,
            session_id=session.session_id,
            request_ts=sender_ts,
            response_latency=elapsed,
            status="success",
            extra={"request_count": get_metrics()["request_count"]},
        )

    elapsed = record_response_time(start)

    background_tasks.add_task(log_save_convo)
    background_tasks.add_task(log_usage)
    background_tasks.add_task(log_session, elapsed)

    return JSONResponse(
        content={
            "status": "Success",
            "sender_ts": sender_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "data": {
                "sender_message": payload.message,
                "session_id": session.session_id,
                "response": response.get("text", ""),
            },
        }
    )