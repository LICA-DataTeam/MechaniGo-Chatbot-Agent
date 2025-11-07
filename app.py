from components.agent_tools import UserInfoAgentContext, MechanicAgentContext, BookingAgentContext
from components import MechaniGoAgent, MechaniGoContext
from agents import InputGuardrailTripwireTriggered
from config import TEST_TABLE_NAME, DATASET_NAME # change table name later
from components.utils import BigQueryClient
from schemas import User, UserCarDetails
import streamlit as st
import asyncio
import uuid

st.set_page_config(
    page_title="MechaniGo Chatbot",
    layout="wide"
)

def init_bq_client(credentials_file: str, dataset_id: str):
    try:
        return BigQueryClient(credentials_file=credentials_file, dataset_id=dataset_id)
    except Exception as e:
        st.error(f"Failed to initialize BigQuery: {e}")
        return None

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

        if "bq_client" not in st.session_state:
            st.session_state.bq_client = init_bq_client('google_creds.json', DATASET_NAME)

        if "context" not in st.session_state:
            st.session_state.context = MechaniGoContext(
                user_ctx=UserInfoAgentContext(
                    user_memory=User(uid=str(uuid.uuid4()))
                ),
                mechanic_ctx=MechanicAgentContext(
                    car_memory=UserCarDetails()
                ),
                booking_ctx=BookingAgentContext()
            )
        st.session_state.agent = MechaniGoAgent(
            api_key=api_key,
            bq_client=st.session_state.bq_client,
            table_name=TEST_TABLE_NAME,
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
                except InputGuardrailTripwireTriggered:
                    st.error("Sorry we cannot process your message right now.")
                except Exception as e:
                    st.error(f"Exception occurred: {e}")

main()