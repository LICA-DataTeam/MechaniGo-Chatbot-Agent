from typing import Iterable
from google.cloud import bigquery
from components.utils import BigQueryClient

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def ensure_chat_history_table_ready(bq_client: BigQueryClient, table_name: str = "chatbot_chat_history_test"):
    """
    A helper that ensures chat history table is ready.

    :param bq_client: BigQuery client instance.
    :type bq_client: BigQueryClient
    :param table_name: The name of the BigQuery table to save entries. Defaults to `"chatbot_chat_history_test"`.
    :type table_name: str

    :returns: None
    """
    logging.info("Ensuring convo history table...")
    schema = [
        bigquery.SchemaField("uid", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("role", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("message", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("timestamp", "DATETIME", mode="NULLABLE"),
    ]
    bq_client.ensure_dataset()
    bq_client.ensure_table(table_name=table_name, schema=schema)
    logging.info("Done!")

def save_convo(
    dataset_id: str,
    table_name: str,
    uid: str,
    entries: Iterable[dict],
    bq_client: BigQueryClient=None
) -> None:
    """
    Saves a conversation history (iterable) to a specifed BigQuery table.

    :param bq_client: BigQuery client instance.
    :type bq_client: BigQueryClient
    :param dataset_id: The BigQuery dataset ID.
    :type dataset_id: str
    :param table_name: The name of the BigQuery table to save entries to.
    :type table_name: str
    :param uid: Session identifier for the conversation; Will be used to link users and their chat history.
    :type uid: str
    :param entries: Iterable of conversation entries. Each must contain:
        * ``role`` (*str*) – The sender’s role (e.g., "user", "assistant").
        * ``message`` (*str*) – The message content.
        * ``timestamp`` (*datetime* or *str*) – When the message was created.
    :type entries: Iterable[dict]

    :returns: None
    """
    if bq_client is None:
        logging.warning("Missing BigQuery client! Creating new one...")
        bq_client = BigQueryClient("google_creds.json", "conversations")
    if not isinstance(bq_client, BigQueryClient):
        raise TypeError(f"Expected BigQueryClient instance, got {type(bq_client).__name__}")

    table_id = f"{bq_client.client.project}.{dataset_id}.{table_name}"
    payload = [
        {
            "uid": uid,
            "role": entry["role"],
            "message": entry["message"],
            "timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        }
        for entry in entries
    ]

    errors = bq_client.client.insert_rows_json(table_id, payload)
    if errors:
        raise RuntimeError(f"Error inserting to BigQuery: {errors}")