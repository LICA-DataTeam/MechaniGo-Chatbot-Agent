from components.agent_tools import UserInfoAgent
from components.utils import BigQueryClient
from google.cloud import bigquery
from dotenv import load_dotenv
from agents import Runner
from schemas import User
import traceback
import asyncio
import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv()

# How to run debug script:
# enter 'python -m tests.test_user_info_agent' in terminal
# IMPORTANT: make sure you are in root directory when entering the above command
# Otherwise, cd into tests directory then run the script directly

async def test():
    logging.info("========== Starting Debug Test ==========")
    logging.info("Step 1: Testing User schema...")
    
    try:
        test_user = User(
            name="Test User",
            address="123 Makati City",
            contact_num="091712345678"
        )
        logging.info(f"User schema works: {test_user.model_dump()}")
    except Exception as e:
        logging.error(f"User schema failed: {e}")
        return

    logging.info("Step 2: Initializing BigQuery client...")
    table_name = "test_users"
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

    logging.info("Step 3: Creating UserInfoAgent...") 
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

    logging.info("Step 4: Running agent with test message...")
    message = """
    My name is Walter White, I live at 308 Negra Arroyo Lane, I drive a white 2004 Pontiac Aztek.

    My contact number is 123-456-789 and I need Car-buying assistance.
    """
    try:
        result = await Runner.run(
            user_info_agent.agent,
            message
        )
        logging.info("Agent completed")
        logging.info(f"  - Result type: {type(result)}")
        logging.info(f"  - Final output: {result.final_output}")
        logging.info(f"  - Messages: {len(result.messages) if hasattr(result, 'messages') else 'N/A'}")

        if hasattr(result, 'messages'):
            logging.info("========== ALL MESSAGES ==========")
            for i, msg in enumerate(result.messages):
                logging.info(f"Message {i}:")
                logging.info(f"  Role: {msg.role if hasattr(msg, 'role') else 'N/A'}")
                logging.info(f"  Content: {str(msg)[:200]}...")

    except Exception as e:
        logging.error(f"Agent run failed: {e}")
        traceback.print_exc()
        return

    logging.info("\n========== TEST COMPLETE ==========")

if __name__ == "__main__":
    asyncio.run(test())