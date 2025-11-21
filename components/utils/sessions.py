from google.cloud.bigquery import SchemaField, ScalarQueryParameter
from agents.memory.session import SessionABC
from components.utils import BigQueryClient
from collections import deque, defaultdict
from typing import List, Optional
from datetime import datetime
from config import PH_TZ
from uuid import uuid4
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SessionHandler(SessionABC):
    """
    Session orchestrator with wired `BigQueryClient` functionality.

    :param session_id: Unique key for the conversation; Generated if empty (`uuid4`).
    :type session_id: str

    :param dataset_id: (Optional) BigQuery dataset name.
    :type dataset_id: str

    :param table_name: (Optional) BigQuery table name.
    :type table_name: str

    :param credentials_file: (Default: `google_creds.json`) File path for stored Google service credentials.
    :type credentials_file: str

    :param schema: (Optional) Table schema; Generates default schema if empty.
    :type schema: List[SchemaField]

    Example usage:
    >>> session = SessionHandler(
    ...           session_id="convo_123",
    ...           dataset_id="dataset_id",
    ...           table_id="table_id",
    ...           schema=[...]
    ...         )
    >>> agent = Agent(name="General Assistant")
    >>> result = await Runner.runner(agent, input, session)
    >>> items = await session.get_items() # retrieves items from session
    """ 
    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        dataset_id: Optional[str] = None,
        table_name: Optional[str] = None,
        credentials_file: str = "google_creds.json",
        schema: Optional[List[SchemaField]] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.session_id = session_id if session_id is not None else str(uuid4())
        self.dataset_id = dataset_id
        self.table_name = table_name
        self.cache = deque()
        self.schema = schema if schema is not None else self.get_schema()
        self.bq = BigQueryClient(credentials_file=credentials_file, dataset_id=dataset_id)
        self.bq.ensure_table(table_name=table_name, schema=self.schema)
        self.logger.info(f"{self.__class__.__name__}.session_id: {self.session_id}")

    @staticmethod
    def _extract_text(item):
        role = item.get("role")
        raw = item.get("content")

        if role == "assistant":
            if isinstance(raw, list):
                texts = [i.get("text") for i in raw if isinstance(i, dict)]
                return "\n\n".join(filter(None, texts))
            return str(raw)

        return raw if isinstance(raw, str) else json.dumps(raw)

    @staticmethod
    def get_schema():
        return [
            SchemaField("session_id", "STRING", mode="REQUIRED"),
            SchemaField("role", "STRING", mode="REQUIRED"),
            SchemaField("content", "STRING", mode="NULLABLE"),
            SchemaField("timestamp", "DATETIME", mode="REQUIRED")
        ]

    def _load_from_bq(self, limit=None):
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = "SELECT *" \
        "FROM `{}.{}.{}`" \
        "WHERE session_id = @session_id" \
        "{}".format(self.bq.client.project, self.bq.dataset_id, self.table_name, limit_clause)
        rows = self.bq.query_to_json(
            query,
            params=[ScalarQueryParameter("session_id", "STRING", self.session_id)]
        )
        expanded = []
        for row in rows:
            raw = row.get("content") or "[]"
            try:
                messages = json.loads(raw)
                if not isinstance(messages, list):
                    raise ValueError
            except Exception:
                messages = [raw] if raw else []
            for text in messages:
                expanded.append({"role": row["role"], "content": text})
        return expanded

    async def get_items(self, limit=None):
        items = list(self.cache)
        if limit:
            items = items[-limit:]

        need_bq = limit is None or len(items) < limit
        if need_bq:
            rows = self._load_from_bq(limit)
            items = rows + items
        return items

    async def add_items(self, items):
        self.cache.extend(items)
        turn_ts = getattr(self, "current_turn_ts", None)
        timestamp = (turn_ts or datetime.now(tz=PH_TZ)).astimezone(PH_TZ).replace(tzinfo=None)
        role_messages: dict[str, list[str]] = defaultdict(list)
        for item in items:
            role = item.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = self._extract_text(item)
            if text:
                role_messages[role].append(text)

        if not role_messages:
            return
        
        existing_rows = self._load_from_bq()
        current_content = {}
        for row in existing_rows:
            raw = row.get("content") or ""
            try:
                messages = json.loads(raw)
                if not isinstance(messages, list):
                    raise ValueError
            except Exception:
                messages = [raw] if raw else []
            current_content[row["role"]] = messages

        payload = []
        for role, new_msgs in role_messages.items():
            combined = current_content.get(role, []).copy()
            combined.extend(new_msgs)
            payload.append({
                "session_id": self.session_id,
                "role": role,
                "content": json.dumps(combined),
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            })

        self.bq.upsert_json(
            rows=payload,
            table_name=self.table_name,
            key_col=("session_id", "role"),
            schema=self.schema
        )

    async def pop_item(self):
        try:
            item = self.cache.pop()
        except IndexError as ie:
            self.logger.error(f"IndexError: {ie}")
            return None
        return item

    async def clear_session(self):
        self.cache.clear()