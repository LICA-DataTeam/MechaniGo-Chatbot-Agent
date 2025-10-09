from components.agent_tools import MechanicAgentContext, UserInfoAgentContext, BookingAgentContext
from components import MechaniGoAgent, MechaniGoContext
from components.utils import BigQueryClient
from schemas import User, UserCarDetails
from google.cloud import bigquery
from dotenv import load_dotenv
from agents import Runner
import traceback
import logging
import asyncio
import uuid
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def test():
    logging.info("Initializing BigQuery...")
    table_name = "chatbot_users_test"
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

    logging.info("Initializing context...")
    try:
        ctx1 = MechaniGoContext(
            user_ctx=UserInfoAgentContext(
                user_memory=User(uid=str(uuid.uuid4())),
                bq_client=bq,
                table_name=table_name
            ),
            mechanic_ctx=MechanicAgentContext(
                car_memory=UserCarDetails()
            ),
            booking_ctx=BookingAgentContext()
        )
    except Exception as e:
        logging.error(f"Error initializing context: {e}")
        return

    logging.info("Initializing MechaniGoAgent...")
    try:
        agent1 = MechaniGoAgent(
            api_key=os.getenv("OPENAI_API_KEY"),
            bq_client=bq,
            context=ctx1
        )
        logging.info("MechaniGoAgent created!")
        logging.info(f"Agent name: {agent1.name}")
    except Exception as e:
        logging.error(f"BookingAgent creation failed: {e}")
        traceback.print_exc()
        return

    runner = Runner()
    try:
        input1 = """
        My name is Dave Grohl, I live in Seattle, Washington and I drive a Toyota Vios 2012.

        I would like to book a schedule for PMS at December 20, 2025 at around 9 am. Thank you.

        My preferred payment is gcash.
        """
        logging.info("========== Sample Conversation ==========")
        logging.info(f"user: {input1}\n")
        response1 = await runner.run(
            agent1.agent,
            input1,
            context=ctx1
        )
        print(response1.final_output)

        input2 = """
        Confirmed, the details are correct.
        """
        logging.info(f"user: {input2}")
        response2 = await runner.run(
            agent1.agent,
            input2,
            context=ctx1
        )
        print(response2.final_output)
        
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error getting response: {e}")
        return

asyncio.run(test())