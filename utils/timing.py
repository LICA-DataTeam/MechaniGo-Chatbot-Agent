from typing import Any, Callable, TypeVar, Coroutine
from fastapi.responses import JSONResponse
from functools import wraps
import logging
import asyncio
import time
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

F = TypeVar("F", Callable[..., Any], Callable[..., Coroutine[Any, Any, Any]])

def _attached_elapse_to_json(result: JSONResponse, elapsed: float) -> JSONResponse:
    body = result.body or b"{}"
    payload = json.loads(body.decode(result.charset or "utf-8") or "{}")
    payload.setdefault("backend_response_time", elapsed)

    headers = {k: v for k, v in result.headers.items() if k.lower() != "content-length"}

    return JSONResponse(
        content=payload,
        status_code=result.status_code,
        headers=headers,
        background=result.background,
        media_type=result.media_type
    )

def log_execution_time(label: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logging.info(f"%s took %.2f ms", label, elapsed)

                if isinstance(result, JSONResponse):
                    return _attached_elapse_to_json(result, elapsed)
                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = (time.perf_counter() - start) * 1000
                    logging.info("%s took %.2f ms", label, elapsed)
            return sync_wrapper
    return decorator