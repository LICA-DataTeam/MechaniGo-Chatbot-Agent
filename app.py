from mechanigo_agent import MechaniGoAgent
from agents import set_default_openai_key
import streamlit as st
import asyncio
import os

st.set_page_config(
    page_title="MechaniGo Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

async def async_validate_api_key(api_key: str) -> bool:
    try:
        set_default_openai_key(api_key)
        return True
    except Exception:
        return False

def validate_api_key(api_key: str) -> bool:
    return asyncio.run(async_validate_api_key(api_key))

async def handle_user_input(agent: MechaniGoAgent, message: str):
    return await agent.inquire(message)

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "validated" not in st.session_state:
        st.session_state.validated = False
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "agent" not in st.session_state:
        st.session_state.agent = None

    with st.sidebar:
        user_key = st.text_input(
            "OpenAI API key",
            key="chatgpt_api_key",
            type="password"
        )

        if st.button("Validate"):
            if not user_key:
                st.error("Please enter your OpenAI API key.")
            else:
                with st.spinner("Validating API key..."):
                    is_valid = validate_api_key(user_key)
                    if is_valid:
                        st.session_state.api_key = user_key
                        st.session_state.validated = True

                        os.environ["OPENAI_API_KEY"] = user_key
                        st.session_state.agent = MechaniGoAgent(api_key=user_key)
                        st.session_state.chat_history = []
                        st.success("OpenAI API key validated!")
                    else:
                        st.session_state.validate = False
                        st.error("Invalid OpenAI API key.")

    if st.button("Reset"):
        st.session_state.api_key = None
        st.session_state.validated = False
        st.session_state.agent = None
        st.session_state.chat_history = []
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        st.info("Cleared cache for this session.")

    st.title("MechaniGo.ph Chatbot")
    st.caption("Helpful AI assistant for MechaniGo customers.")

    if not st.session_state.validated or st.session_state.agent is None:
        st.info("Please enter a valid OpenAI API key to start chatting.")
        return

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    if user_input := st.chat_input("Ask MechaniGo bot..."):
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.chat_history.append(("user", user_input))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(handle_user_input(st.session_state.agent, user_input))
                    st.markdown(response)
                    st.session_state.chat_history.append(("assistant", response))
                except Exception as e:
                    st.error(f"Exception occurred: {e}")

main()