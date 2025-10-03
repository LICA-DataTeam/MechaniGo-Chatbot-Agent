from components.agent_tools import UserInfoAgent, AgentContext
from agents import Runner, RunContextWrapper
from components.utils import BigQueryClient
from google.cloud import bigquery
from dotenv import load_dotenv
from schemas import User
import traceback
import asyncio
import logging
import uuid
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def test():
    logging.info("Initializing BigQuery Client")
    table_name = "testing_users"
    try:
        bq = BigQueryClient('google_creds.json', 'conversations')
        bq.ensure_dataset()
        schema = [
            bigquery.SchemaField("uid", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("address", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("contact_num", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("schedule_date", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("schedule_time", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("payment", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("car", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("raw_json", "STRING", mode="NULLABLE"),
        ]
        bq.ensure_table(table_name, schema)
        logging.info("BigQuery client initialized!")
    except Exception as e:
        logging.error(f"BigQuery initialization failed: {e}")
        return

    logging.info("Initializing UserInfoAgent")
    try:
        user_info_agent = UserInfoAgent(
            api_key=os.getenv("OPENAI_API_KEY"),
            bq_client=bq,
            table_name=table_name
        )
        logging.info("UserInfoAgent created!")
        logging.info(f"  - Agent name: {user_info_agent.agent.name}")
        logging.info(f"  - Number of tools: {len(user_info_agent.agent.tools)}")
        logging.info(f"  - Tools: {[type(t).__name__ for t in user_info_agent.agent.tools]}")
    except Exception as e:
        logging.error(f"UserInfoAgent creation failed: {e}")
        traceback.print_exc()
        return

    logging.info("Testing UserInfoAgent with Context Manager")
    runner = Runner()
    try:
        context = AgentContext(
            user_memory=User(uid=str(uuid.uuid4())),
            bq_client=bq,
            table_name=table_name
        )

        response = await runner.run(
            user_info_agent.agent,
            "My name is Walter Hartwell White and I drive a white Toyota Vios 2019. My contact num is 09171234567",
            context=context
        )
        print(response.final_output)

        response2 = await runner.run(
            user_info_agent.agent,
            "What do you know about me?",
            context=context
        )
        print(response2.final_output)

        response3 = await runner.run(
            user_info_agent.agent,
            "My preferred schedule is Oct 25, 2025 at around 10 am and I prefer gcash as my mode of payment",
            context=context
        )
        print(response3.final_output)
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error testing using ctx manager: {e}")
        return

asyncio.run(test())