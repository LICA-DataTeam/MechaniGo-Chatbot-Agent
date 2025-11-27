from api import send_msg_router, fetch_metrics_router
from fastapi import FastAPI

app = FastAPI(
    title="MechaniGo Chatbot API"
)

app.include_router(send_msg_router, prefix="/send", tags=["send-message"])
app.include_router(fetch_metrics_router, prefix="/metrics", tags=["session-metrics"])

@app.get("/")
def root():
    return "working"