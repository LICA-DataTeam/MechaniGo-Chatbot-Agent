from api.common import (
    JSONResponse,
    APIRouter,
    status
)
from utils import get_metrics
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

router = APIRouter()

@router.get("/session-metrics")
def session_metrics(session_id: str):
    try:
        return JSONResponse(
            content={
                "session_id": session_id,
                "metrics": get_metrics()
            },
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        return JSONResponse(
            content={
                "status": "Error"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )