from api.common import (
    UserInfoAgentContext, MechanicAgentContext, BookingAgentContext,
    MechaniGoAgent, MechaniGoContext,
    InputGuardrailTripwireTriggered,
    TEST_TABLE_NAME, DATASET_NAME,
    set_default_openai_key,
    User, UserCarDetails,
    BigQueryClient
)

from agents import SQLiteSession
import streamlit as st
import logging
import uuid
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def _init_bq_client(credentials_file: str, dataset_id: str):
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
        logging.error(f"Failed to initialize BigQuery: {e}")
        return

def _load_secrets():
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
        logging.error(f"Could not load OpenAI API key from secrets: {e}")

    if not api_key:
        logging.error("No OpenAI API key loaded.")
        return

    os.environ["OPENAI_API_KEY"] = api_key
    set_default_openai_key(api_key)
    logging.info(f"OpenAI API key loaded fro {api_key_source}.")
    logging.info("Loading vector store IDs...")
    try:
        os.environ["FAQ_VECTOR_STORE_ID"] = st.secrets["FAQ_VECTOR_STORE_ID"]
        os.environ["MECHANIC_VECTOR_STORE_ID"] = st.secrets["MECHANIC_VECTOR_STORE_ID"]
    except Exception:
        st.warning("Vector stores unable to load.")

    return api_key

async def run(inquiry: str, session: SQLiteSession = None):
    """
    Main entry-point for the calling the chatbot.

    :param inquiry: User inquiry to the LLM.
    :type inquiry: str

    :param session: Session object that maintains state across interactions.
    :type session: SQLiteSession
    """
    api_key = _load_secrets()
    if not api_key:
        logging.error("OpenAI API key is missing.")
        raise RuntimeError("OpenAI API key not configured.")

    bq_client = _init_bq_client("google_creds.json", DATASET_NAME)
    if not bq_client:
        logging.error("BigQuery client is not available.")
        return

    try:
        ctx = MechaniGoContext(
            user_ctx=UserInfoAgentContext(
                user_memory=User(uid=str(uuid.uuid4()))
            ),
            mechanic_ctx=MechanicAgentContext(
                car_memory=UserCarDetails()
            ),
            booking_ctx=BookingAgentContext()
        )

        if not session:
            raise ValueError("Session object missing.")

        agent = MechaniGoAgent(
            api_key=api_key,
            bq_client=bq_client,
            table_name=TEST_TABLE_NAME,
            context=ctx,
            session=session
        )
        return await agent.inquire(inquiry=inquiry)
    except InputGuardrailTripwireTriggered:
        logging.error("Input Guardrail Tripwire triggered!")
        return {"status": "Error", "message": "Sorry we cannot process your message right now."}
    except Exception as e:
        logging.error(f"Exception occurred: {e}")