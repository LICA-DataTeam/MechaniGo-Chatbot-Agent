# Monitoring utilities
# - Token usage
# - Request count
# - Unique sessions created
# - Latency of responses

# Capture metrics after each .inquire() from Runner.run()

# Metric transport -> collector for metrics
# Options: stats service OR using Python by building a MetricsClient module
from contextlib import contextmanager
from typing import Dict
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

metrics_dict = {
    "request_count": 0,
    "unique_sessions": set(),
    "response_time": 0.0
}

session_token_usage: Dict[str, Dict[str, int]] = {}

def record_request():
    metrics_dict["request_count"] += 1

def record_session(session_id: str):
    metrics_dict["unique_sessions"].add(session_id)

def record_response_time(start_time: float):
    metrics_dict["response_time"] = time.perf_counter() - start_time

def record_session_tokens(session_id: str, usage: dict):
    if session_id not in session_token_usage:
        session_token_usage[session_id] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    session_token_usage[session_id]["input_tokens"] += usage.get("input_tokens", 0)
    session_token_usage[session_id]["output_tokens"] += usage.get("output_tokens", 0)
    session_token_usage[session_id]["total_tokens"] += usage.get("total_tokens", 0)

def get_metrics():
    return {
        "request_count": metrics_dict["request_count"],
        "unique_sessions_count": len(metrics_dict["unique_sessions"]),
        "response_time": metrics_dict["response_time"],
        "session_token_usage": session_token_usage
    }

@contextmanager
def track_phase(
    name: str,
    session_id: str | None = None
):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info("phase=%s session=%s duration=%.3fs", name, session_id, elapsed)