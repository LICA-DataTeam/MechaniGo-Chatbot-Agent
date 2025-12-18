"""
User-friendly interface for MechaniGo Bot (uses the API).
"""
import streamlit as st
import requests
import logging
import time
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(
    page_title="MechaniGo Chatbot",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000/mgo-chatbot-api/v1/send/send-message"
MAX_MESSAGES = 15

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []

def send_request(message: str) -> tuple[str, float, float]:
    start = time.perf_counter()
    response = requests.post(
        API_URL,
        json={"message": message},
        headers={"X-User-Id": st.session_state.session_id},
        timeout=30
    )
    response.raise_for_status()
    frontend_elapsed = time.perf_counter() - start
    resp = response.json()
    backend_elapsed = resp.get("backend_response_time", None)
    return resp.get("response"), backend_elapsed, frontend_elapsed

def main():
    st.title("MechaniGo Bot :robot:")
    st.caption("Helpful AI assistant for MechaniGo users.")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    if user_input := st.chat_input("Ask MechaniGo Bot..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        with st.chat_message("ai"):
            reply, backend, frontend = send_request(user_input)
            st.markdown(reply)
            if backend is not None:
                st.caption(
                    f"Backend: {backend:.2f}s | "
                    f"UI total: {frontend:.2f}s | "
                    f"Overhead: {(frontend - backend):.2f}s"
                )
            st.session_state.chat_history.append(("assistant", reply))