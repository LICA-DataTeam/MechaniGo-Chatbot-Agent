from components.utils import BigQueryClient
import streamlit as st

st.set_page_config(
    page_title="Conversation Viewer",
    page_icon="💬",
    layout="wide"
)

def get_convo_data(session_id: str):
    bq = BigQueryClient("google_creds.json", "conversations")
    query = """
    SELECT
    *
    FROM `{}.{}.chatbot_chat_history_test_2`
    WHERE uid = '{}'
    """.format("mechanigo-liveagent", "conversations", session_id)

    return bq.execute_query(query)

with st.container(border=True):
    st.title("MechaniGo Chatbot Conversation Viewer")

with st.form("convo_viewer"):
    session_id = st.text_input("Session ID:")
    search = st.form_submit_button("Search")

    if search:
        with st.spinner("Searching..."):
            df = get_convo_data(session_id)

        if df is not None and not df.empty:
            st.success(f"Found {len(df)} messages.")
            st.dataframe(df, hide_index=True)
        else:
            st.warning("No data found!")