from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from google.cloud import bigquery
from typing import List, Optional
from schemas import User
import pandas as pd
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SCOPE = [
    'https://www.googleapis.com/auth/bigquery',
    'https://www.googleapis.com/auth/drive'
]

class BigQueryClient:
    def __init__(self, credentials_file: str, dataset_id: str):
        self.logger = logging.getLogger(__name__)
        self.creds = self._authenticate(credentials_file)
        self.dataset_id = dataset_id
        self.client = bigquery.Client(
            credentials=self.creds,
            project=self.creds.project_id
        )

    def _authenticate(self, credentials_file: str):
        self.logger.info("BigQuery authenticated!")
        return service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=SCOPE
        )

    def ensure_dataset(self):
        dataset_ref = self.client.dataset(self.dataset_id)
        try:
            self.client.get_dataset(dataset_ref=dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref=dataset_ref)
            dataset.location = "asia-southeast1"
            self.client.create_dataset(dataset)

    def ensure_table(
        self,
        table_name: str,
        schema: List[bigquery.SchemaField] = None
    ):
        table_id = f"{self.client.project}.{self.dataset_id}.{table_name}"
        try:
            self.client.get_table(table_id)
        except NotFound:
            table = bigquery.Table(table_id, schema=schema)
            created_table = self.client.create_table(table)
            created_table.expires = None
            self.client.update_table(created_table, ["expires"])

    def execute_query(self, query: str, return_data: bool = True) -> Optional[pd.DataFrame]:
        query_job = self.client.query(query)
        if return_data:
            df = query_job.to_dataframe()
            return df
        else:
            query_job.result()
            return None

    def insert_user(self, table_name: str, user: User):
        user_dict = user.model_dump()
        row = {
            "name": user_dict.get("name"),
            "address": user_dict.get("address"),
            "contact_num": user_dict.get("contact_num"),
            "schedule_date": user_dict.get("schedule_date"),
            "schedule_time": user_dict.get("schedule_time"),
            "payment": str(user_dict.get("payment")) if user_dict.get("payment") else None,
            "car": json.dumps(user_dict.get("car")) if user_dict.get("car") else None,
            "raw_json": json.dumps(user_dict),
        }

        table_id = f"{self.client.project}.{self.dataset_id}.{table_name}"
        errors = self.client.insert_rows_json(table_id, [row])

        if errors:
            self.logger.error(f"Error inserting user: {errors}")
            raise RuntimeError(errors)
        else:
            self.logger.info(f"Inserted user {row['name']} into {table_name}")