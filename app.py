from components.agent_tools import UserInfoAgentContext, MechanicAgentContext, BookingAgentContext
from helpers import ensure_chat_history_table_ready, save_convo
from components.utils import BigQueryClient, SessionHandler
from components import MechaniGoAgent, MechaniGoContext
from agents import InputGuardrailTripwireTriggered
from config import TEST_TABLE_NAME, DATASET_NAME # change table name later
from agents import set_default_openai_key
from schemas import User, UserCarDetails
from datetime import datetime
import streamlit as st
import asyncio
import logging
import pytz
import json
import uuid
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(
    page_title="MechaniGo Chatbot",
    layout="wide"
)

PH_TZ = pytz.timezone("Asia/Manila")

def init_bq_client(credentials_file: str, dataset_id: str):
    try:
        if os.path.exists(credentials_file):
            creds_source = "local"
            client = BigQueryClient(credentials_file=credentials_file, dataset_id=dataset_id)
        else:
            creds_source = "streamlit_secrets"
            google_creds = st.secrets["gcp_service_account"]
            creds_json = json.dumps(dict(google_creds))
            with open("google_creds.json", "w") as f:
                f.write(creds_json)
            client = BigQueryClient(credentials_file="google_creds.json", dataset_id=dataset_id)
        logging.info(f"BigQuery initialized using {creds_source}.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize BigQuery: {e}")
        return

def load_secrets():
    api_key = None
    api_key_source = None

    logging.info("Loading API keys...")
    try:
        secrets_entry = st.secrets["OPENAI_API_KEY"]["OPENAI_API_KEY"]
        if isinstance(secrets_entry, dict):
            api_key = secrets_entry.get("OPENAI_API_KEY")
        else:
            api_key = secrets_entry
        if api_key:
            api_key_source = "Streamlit secrets"
    except Exception as e:
        st.error(f"Could not load OpenAI API key from secrets: {e}")

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            api_key_source = "environment variables"
    
    if not api_key:
        st.error("No OpenAI API key found.")
        return
    
    os.environ["OPENAI_API_KEY"] = api_key
    set_default_openai_key(api_key)
    logging.info(f"OpenAI API key loaded from {api_key_source}.")

    logging.info("Loading vector store IDs...")
    try:
        os.environ["FAQ_VECTOR_STORE_ID"] = st.secrets["FAQ_VECTOR_STORE_ID"]
        os.environ["MECHANIC_VECTOR_STORE_ID"] = st.secrets["MECHANIC_VECTOR_STORE_ID"]
    except Exception:
        st.warning("Vector stores unable to load.")

    return api_key

async def handle_user_input(agent: MechaniGoAgent, message: str):
    return await agent.inquire(message)

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.last_saved = 0

    if "agent" not in st.session_state:
        try:
            api_key = load_secrets()
        except Exception as e:
            logging.error(f"OpenAI API key did not load: {e}")
            st.error(f"Error loading OpenAI API key: {e}")
            st.stop()

        if not api_key:
            st.error("OpenAI API key is missing.")
            st.stop()

        if "bq_client" not in st.session_state:
            st.session_state.bq_client = init_bq_client('google_creds.json', DATASET_NAME)
        
        if not st.session_state.bq_client:
            st.error("BigQuery client is not available.")
            st.stop()

        if not st.session_state.get("chat_table_ready"):
            ensure_chat_history_table_ready(st.session_state.bq_client)
            st.session_state.chat_table_ready = True

        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

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

        if "session" not in st.session_state:
            st.session_state.session = SessionHandler(
                session_id=st.session_state.session_id,
                dataset_id="conversations",
                table_name="chatbot_api_test",
                bq_client=st.session_state.bq_client
            )

        st.session_state.agent = MechaniGoAgent(
            api_key=api_key,
            bq_client=st.session_state.bq_client,
            table_name=TEST_TABLE_NAME,
            context=st.session_state.context,
            session=st.session_state.session
        )

    if st.button("Reset"):
        st.session_state.pop("agent", None)
        st.session_state.pop("context", None)
        st.session_state.last_saved = 0
        st.session_state.chat_history = []
        st.info("Cleared cache for this session.")

    st.title("MechaniGo.ph Chatbot")
    st.caption("Helpful AI assistant for MechaniGo customer service and customers.")

    history_container = st.container()
    with history_container:
        for entry in st.session_state.chat_history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["message"])

    if user_input := st.chat_input("Ask MechaniGo bot..."):
        user_ts = datetime.now(tz=PH_TZ)
        logging.info(f"user_ts: {user_ts}")
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_input,
            "timestamp": user_ts
        })

        with history_container:
            st.chat_message("user").markdown(user_input)
            assistant_placeholder = st.chat_message("assistant")

        try:
            with history_container, st.spinner("Thinking..."):
                response = asyncio.run(handle_user_input(st.session_state.agent, user_input))
                assistant_ts = datetime.now(tz=PH_TZ)
                logging.info(f"assistant_ts: {assistant_ts}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "message": response["text"],
                    "timestamp": assistant_ts
                })

            with history_container:
                assistant_placeholder.markdown(response["text"])
        except InputGuardrailTripwireTriggered:
            st.error("Sorry we cannot process your message right now.")
        except Exception as e:
            st.error(f"Exception occurred: {e}")
    
    unsaved = st.session_state.chat_history[st.session_state.last_saved:]
    if unsaved:
        if not st.session_state.bq_client:
            st.warning("BigQuery client not initalized; Conversation not saved.")
        else:
            try:
                import traceback
                save_convo(
                    bq_client=st.session_state.bq_client,
                    dataset_id=DATASET_NAME,
                    table_name="chatbot_chat_history_test",
                    uid=st.session_state.context.user_ctx.user_memory.uid,
                    entries=unsaved
                )
            except Exception as e:
                traceback.print_exc()
                st.error(f"Failed to save chat history: {e}")
            else:
                st.session_state.last_saved = len(st.session_state.chat_history)

main()