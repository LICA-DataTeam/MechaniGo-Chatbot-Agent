from typing import List, Dict, Optional, Sequence, Iterable, Union, Any
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from google.cloud import bigquery
from schemas import User
import pandas as pd
import logging
import json
import uuid

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
    
    def get_user_by_uid(self, table_name: str, uid: str) -> Optional[User]:
        query = f"""
            SELECT * FROM `{self.dataset_id}.{table_name}`
            WHERE uid = @uid
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
        )
        results = self.client.query(query, job_config=job_config).result()
        row = list(results)
        if row:
            return User(**dict(row[0]))
        return None

    def get_user_by_contact_num(self, table_name: str, contact_num: str) -> Optional[User]:
        query = f"""
        SELECT * FROM `{self.dataset_id}.{table_name}`
        WHERE contact_num = @contact_num
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("contact_num", "STRING", contact_num)
            ]
        )
        results = self.client.query(query, job_config=job_config).result()
        rows = list(results)
        if rows:
            return User(**dict(rows[0]))
        return None

    def get_user_by_email(self, table_name: str, email: str) -> Optional[User]:
        query = f"""
        SELECT * FROM `{self.dataset_id}.{table_name}`
        WHERE email = @email
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("email", "STRING", email)
            ]
        )
        results = self.client.query(query, job_config=job_config).result()
        rows = list(results)
        if rows:
            return User(**dict(rows[0]))
        return None

    def upsert_user(self, table_name: str, user: User):
        user_dict = user.model_dump()
        table_id = f"{self.client.project}.{self.dataset_id}.{table_name}"
        row = {
            "uid": user_dict.get("uid"),
            "name": user_dict.get("name"),
            "email": user_dict.get("email"),
            "address": user_dict.get("address"),
            "contact_num": user_dict.get("contact_num"),
            "service_type": user_dict.get("service_type"),
            "schedule_date": user_dict.get("schedule_date"),
            "schedule_time": user_dict.get("schedule_time"),
            "payment": str(user_dict.get("payment")) if user_dict.get("payment") else None,
            "car": user_dict.get("car"),
            "raw_json": json.dumps(user_dict),
        }
        
        query = """
        MERGE `{}` T
        USING (SELECT
            @uid AS uid,
            @name AS name,
            @email AS email,
            @address AS address,
            @contact_num AS contact_num,
            @service_type AS service_type,
            @schedule_date AS schedule_date,
            @schedule_time AS schedule_time,
            @payment AS payment,
            @car AS car,
            @raw_json AS raw_json
        ) S
        ON T.uid = S.uid
        WHEN MATCHED THEN
        UPDATE SET
            name = S.name,
            email = S.email,
            address = S.address,
            contact_num = S.contact_num,
            schedule_date = S.schedule_date,
            schedule_time = S.schedule_time,
            payment = S.payment,
            car = S.car,
            raw_json = S.raw_json
        WHEN NOT MATCHED THEN
        INSERT (uid, name, email, address, contact_num, service_type, schedule_date, schedule_time, payment, car, raw_json)
        VALUES(S.uid, S.name, S.email, S.address, S.contact_num, S.service_type, S.schedule_date, S.schedule_time, S.payment, S.car, S.raw_json)
        """.format(table_id)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("uid", "STRING", row["uid"]),
                bigquery.ScalarQueryParameter("name", "STRING", row["name"]),
                bigquery.ScalarQueryParameter("email", "STRING", row["email"]),
                bigquery.ScalarQueryParameter("address", "STRING", row["address"]),
                bigquery.ScalarQueryParameter("service_type", "STRING", row["service_type"]),
                bigquery.ScalarQueryParameter("contact_num", "STRING", row["contact_num"]),
                bigquery.ScalarQueryParameter("schedule_date", "STRING", row["schedule_date"]),
                bigquery.ScalarQueryParameter("schedule_time", "STRING", row["schedule_time"]),
                bigquery.ScalarQueryParameter("payment", "STRING", row["payment"]),
                bigquery.ScalarQueryParameter("car", "STRING", row["car"]),
                bigquery.ScalarQueryParameter("raw_json", "STRING", row["raw_json"]),
            ]
        )

        try:
            self.client.query(query, job_config=job_config).result()
            self.logger.info(f"Upserted user {row['name']} ({row['email']}) into {table_name}")
        except Exception as e:
            self.logger.error(f"Failed to upsert user: {e}")
            raise

    def query_to_json(self, sql: str, params: Optional[Sequence[bigquery.ScalarQueryParameter]] = None):
        job_config = bigquery.QueryJobConfig(query_parameters=params or [])
        rows = self.client.query(sql, job_config=job_config).result()
        return [dict(row.items()) for row in rows]

    def load_json(
        self,
        rows: Union[List[Dict[str, Any]], Iterable[Dict[str, Any]]],
        write_disposition: str = "WRITE_APPEND",
        table_name: str = None,
        schema: Optional[Sequence[bigquery.SchemaField]] = None,
        autodetect: bool = False,
        create_if_needed: bool = True,
        ignore_unknown_values: bool = True
    ):
        if not table_name:
            raise ValueError("Table name is required!")

        if create_if_needed:
            if not schema and not autodetect:
                raise ValueError("Table missing. Provide `schema` or set `autodetect=True`.")
            self.ensure_table(table_name=table_name, schema=schema)

        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            ignore_unknown_values=ignore_unknown_values,
            autodetect=autodetect
        )

        if schema is not None:
            job_config.schema = list(schema)

        dataset_ref = bigquery.DatasetReference(self.client.project, self.dataset_id)
        table_ref = bigquery.TableReference(dataset_ref, table_name)

        job = self.client.load_table_from_json(list(rows), table_ref, job_config=job_config)
        return job

    def upsert_json(
        self,
        rows: List[Dict[str, Any]],
        table_name: str,
        key_col: Sequence[str],
        schema: Sequence[bigquery.SchemaField]
    ):
        if not rows:
            return None

        dataset_ref = bigquery.DatasetReference(self.client.project, self.dataset_id)
        temp_table = bigquery.TableReference(dataset_ref, f"tmp_{str(uuid.uuid4().hex)}")

        self.ensure_table(temp_table.table_id, schema)
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", schema=list(schema))
        self.client.load_table_from_json(rows, temp_table, job_config=job_config).result()

        key_join = " AND ".join([f"T.{col}=S.{col}" for col in key_col])
        update_set = ", ".join([f"{field.name}=S.{field.name}" for field in schema if field.name not in key_col])
        insert_cols = ", ".join([field.name for field in schema])
        insert_vals = ", ".join([f"S.{field.name}" for field in schema])

        merge_sql = """
        MERGE `{}.{}.{}` AS T
        USING `{}.{}.{}` AS S
        ON {}
        WHEN MATCHED THEN UPDATE SET {}
        WHEN NOT MATCHED THEN INSERT ({}) VALUES ({})
        """.format(
            self.client.project,
            self.dataset_id,
            table_name,
            temp_table.project,
            temp_table.dataset_id,
            temp_table.table_id,
            key_join,
            update_set,
            insert_cols,
            insert_vals
        )
        self.client.query(merge_sql).result()
        self.client.delete_table(temp_table, not_found_ok=True)