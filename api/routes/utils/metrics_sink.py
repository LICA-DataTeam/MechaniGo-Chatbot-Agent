from typing import Any, Dict, Iterable, Optional
from components.utils import BigQueryClient
from google.cloud import bigquery
from datetime import datetime

class BigQueryMetricsSink:
    def __init__(
        self,
        *,
        client: Optional[BigQueryClient] = None,
        credentials_file: Optional[str] = None,
        dataset_id: str,
        session_table: str,
        usage_table: str
    ):
        if client is None:
            if not credentials_file:
                raise ValueError("Provide an existing BigQueryClient or credentials file.")
            client = BigQueryClient(credentials_file=credentials_file, dataset_id=dataset_id)
        self.client = client
        self.dataset_id = dataset_id
        self.session_table = session_table
        self.usage_table = usage_table

    @staticmethod
    def _session_schema():
        return [
            bigquery.SchemaField("session_id", "STRING"),
            bigquery.SchemaField("request_count", "INT64"),
            bigquery.SchemaField("request_ts", "DATETIME"),
            bigquery.SchemaField("response_latency", "FLOAT64"),
            bigquery.SchemaField("status", "STRING"),
            bigquery.SchemaField("extra_json", "JSON")
        ]

    @staticmethod
    def _usage_schema():
        return [
            bigquery.SchemaField("session_id", "STRING"),
            bigquery.SchemaField("model", "STRING"),
            bigquery.SchemaField("input_tokens", "INT64"),
            bigquery.SchemaField("output_tokens", "INT64"),
            bigquery.SchemaField("total_tokens", "INT64"),
            bigquery.SchemaField("logged_at", "DATETIME")
        ]

    def record_session(
        self,
        *,
        session_id: str,
        request_ts: datetime,
        response_latency: float,
        status: str = "success",
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        row = {
            "session_id": session_id,
            "request_count": extra["request_count"],
            "request_ts": request_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "response_latency": round(response_latency, 2),
            "status": status,
            "extra_json": extra or {}
        }
        self.client.load_json(
            rows=[row],
            table_name=self.session_table,
            schema=self._session_schema(),
            create_if_needed=True
        )

    def record_usage(
        self,
        *,
        session_id: str,
        usage: Dict[str, int],
        model: str
    ) -> None:
        row = {
            "session_id": session_id,
            "model": model,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.client.load_json(
            rows=[row],
            table_name=self.usage_table,
            schema=self._usage_schema(),
            create_if_needed=True
        )