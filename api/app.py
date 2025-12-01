from api import send_msg_router, fetch_metrics_router
from api.routes.utils import BigQueryMetricsSink
from components.utils import BigQueryClient
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = BigQueryClient(credentials_file="google_creds.json", dataset_id="conversations")
    app.state.metrics_sink = BigQueryMetricsSink(
        client=client,
        dataset_id="conversations",
        session_table="mechanigo_chatbot_metrics_sessions",
        usage_table="mechanigo_chatbot_metrics_usage"
    )
    try:
        yield
    finally:
        pass

app = FastAPI(
    lifespan=lifespan,
    title="MechaniGo Chatbot API"
)

app.include_router(send_msg_router, prefix="/send", tags=["send-message"])
app.include_router(fetch_metrics_router, prefix="/metrics", tags=["session-metrics"])

@app.get("/")
def root():
    return "working"