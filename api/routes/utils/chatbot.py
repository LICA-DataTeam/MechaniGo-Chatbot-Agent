from api.common import (
    UserInfoAgentContext, MechanicAgentContext, BookingAgentContext,
    MechaniGoAgent, MechaniGoContext,
    InputGuardrailTripwireTriggered,
    TEST_TABLE_NAME, DATASET_NAME,
    set_default_openai_key,
    User, UserCarDetails,
    BigQueryClient,
    PH_TZ
)

from google.cloud.bigquery import SchemaField, ScalarQueryParameter
from agents import SQLiteSession
from datetime import datetime
import streamlit as st
import logging
import uuid
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# For persistent context after each turn
CONTEXT_TABLE_NAME = "chatbot_context"
CONTEXT_SCHEMA = [
    SchemaField("session_id", "STRING", mode="REQUIRED"),
    SchemaField("user_ctx", "STRING", mode="NULLABLE"),
    SchemaField("mechanic_ctx", "STRING", mode="NULLABLE"),
    SchemaField("booking_ctx", "STRING", mode="NULLABLE"),
    SchemaField("updated_at", "DATETIME", mode="REQUIRED")
]

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
        logging.info(f"Initializing context table: {CONTEXT_TABLE_NAME}")
        client.ensure_table(table_name=CONTEXT_TABLE_NAME, schema=CONTEXT_SCHEMA)
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

def _hydrate_context(bq: BigQueryClient, session_id: str) -> MechaniGoContext:
    query = """
    SELECT user_ctx, mechanic_ctx, booking_ctx
    FROM `{}.{}.{}`
    WHERE session_id = @session_id
    LIMIT 1
    """.format(bq.client.project, bq.dataset_id, CONTEXT_TABLE_NAME)
    result = bq.query_to_json(query, params=[ScalarQueryParameter("session_id", "STRING", session_id)])
    if not result:
        return MechaniGoContext(
            user_ctx=UserInfoAgentContext(user_memory=User(uid=str(uuid.uuid4()))),
            mechanic_ctx=MechanicAgentContext(car_memory=UserCarDetails()),
            booking_ctx=BookingAgentContext()
        )

    row = result[0]
    user_data = json.loads(row["user_ctx"]) if row.get("user_ctx") else {}
    mech_data = json.loads(row["mechanic_ctx"]) if row.get("mechanic_ctx") else {}
    booking_data = json.loads(row["booking_ctx"]) if row.get("booking_ctx") else {}
    return MechaniGoContext(
        user_ctx=UserInfoAgentContext(user_memory=User(**user_data)),
        mechanic_ctx=MechanicAgentContext(car_memory=UserCarDetails(**mech_data)),
        booking_ctx=BookingAgentContext(**booking_data)
    )

def _persist_context(bq: BigQueryClient, session_id: str, ctx: MechaniGoContext):
    payload = {
        "session_id": session_id,
        "user_ctx": json.dumps(ctx.user_ctx.user_memory.model_dump()),
        "mechanic_ctx": json.dumps(ctx.mechanic_ctx.car_memory.model_dump()),
        "booking_ctx": json.dumps(ctx.booking_ctx.model_dump()),
        "updated_at": datetime.now(tz=PH_TZ).strftime("%Y-%m-%d %H:%M:%S")
    }
    bq.upsert_json(
        rows=[payload],
        table_name=CONTEXT_TABLE_NAME,
        key_col=("session_id",),
        schema=CONTEXT_SCHEMA
    )

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

    if not session:
        raise ValueError("Session object missing.")

    try:
        ctx = _hydrate_context(bq_client, session.session_id)

        if not session:
            raise ValueError("Session object missing.")

        agent = MechaniGoAgent(
            api_key=api_key,
            bq_client=bq_client,
            table_name=TEST_TABLE_NAME,
            context=ctx,
            session=session
        )
        result = await agent.inquire(inquiry=inquiry)
        _persist_context(bq_client, session.session_id, ctx)
        return result
    except InputGuardrailTripwireTriggered:
        logging.error("Input Guardrail Tripwire triggered!")
        return {"status": "Error", "message": "Sorry we cannot process your message right now."}
    except Exception as e:
        logging.error(f"Exception occurred: {e}")