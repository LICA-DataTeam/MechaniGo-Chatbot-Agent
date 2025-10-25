from components.agent_tools import MechanicAgentContext
from components import MechaniGoAgent, MechaniGoContext
from schemas import UserCarDetails
import streamlit as st
import asyncio

st.set_page_config(
    page_title="MechaniGo Chatbot",
    layout="wide"
)

async def handle_user_input(agent: MechaniGoAgent, message: str):
    return await agent.inquire(message)

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "agent" not in st.session_state:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except KeyError:
            st.error("No OpenAI API key found in secrets.")
            return

        if "context" not in st.session_state:
            st.session_state.context = MechaniGoContext(
                mechanic_ctx=MechanicAgentContext(
                    car_memory=UserCarDetails()
                )
            )
        st.session_state.agent = MechaniGoAgent(
            api_key=api_key,
            context=st.session_state.context
        )

    if st.button("Reset"):
        st.session_state.agent = None
        st.session_state.chat_history = []
        st.info("Cleared cache for this session.")

    st.title("MechaniGo.ph Chatbot")
    st.caption("Helpful AI assistant for MechaniGo customer service and customers.")

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