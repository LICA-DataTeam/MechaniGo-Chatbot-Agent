from components.agent_tools.user_info_agent import UserInfoAgent, UserInfoAgentContext
from components.agent_tools.mechanic_agent import MechanicAgent, MechanicAgentContext
from agents import Agent, Runner, RunContextWrapper, function_tool
from components.utils import BigQueryClient, create_agent
from schemas import User, UserCarDetails
from typing import List, Optional, Any
from google.cloud import bigquery
from dotenv import load_dotenv
from pydantic import BaseModel
import logging
import asyncio
import uuid
import os

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

HANDOFF_DESC = """
A helpful assistant for MechaniGo.ph.
"""

INSTRUCTIONS = """
You are the main orchestrator agent for MechaniGo.ph, a business that offers home maintenance (PMS) and car-buying assistance.

- You are the customer-facing agent which handles responding to customer inquiries.
- Use the tools given to you to accomplish your tasks:
    - For car-related issues delegate to mechanic_agent tool
    - For user information delegate to the user_info_agent tool
- Do not attempt to solve the tasks directly; always use the tools to accomplish the task.
- Provide a clear and concise response back to the customer.
"""

TABLE_NAME = "chatbot_users_test"

class MechaniGoContext(BaseModel):
    user_ctx: UserInfoAgentContext
    mechanic_ctx: MechanicAgentContext
    model_config = {
        "arbitrary_types_allowed": True
    }

# Orchestrator Agent
class Orchestrator:
    def __init__(
        self,
        api_key,
        bq_client: BigQueryClient,
        name: str = "MechaniGo Bot",
        model: str = "gpt-4o-mini",
        context: Optional[MechaniGoContext] = None
    ):
        self.api_key = api_key
        self.bq_client = bq_client
        self.name = name
        self.handoff_description = HANDOFF_DESC
        self.instructions = INSTRUCTIONS
        self.model = model

        if not context:
            context = MechaniGoContext(
                user_ctx=UserInfoAgentContext(
                    user_memory=User(uid=str(uuid.uuid4())),
                    bq_client=self.bq_client,
                    table_name=TABLE_NAME
                ),
                mechanic_ctx=MechanicAgentContext(
                    car_memory=UserCarDetails()
                )
            )

        self.context = context

        user_info_agent = UserInfoAgent(
            api_key=self.api_key,
            bq_client=self.bq_client,
            table_name=TABLE_NAME,
            model=self.model
        )

        mechanic_agent = MechanicAgent(
            api_key=self.api_key
        )

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.handoff_description,
            instructions=self.instructions,
            model=self.model,
            tools=[user_info_agent.as_tool, mechanic_agent.as_tool]
        )
        self.logger = logging.getLogger(__name__)

    async def inquire(self, inquiry: str):
        response = await Runner.run(
            starting_agent=self.agent,
            input=inquiry,
            context=self.context
        )
        return response.final_output

async def test():
    logging.info("Initializing BigQuery...")
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
    bq.ensure_table(TABLE_NAME, schema)

    mgo_context1 = MechaniGoContext(
        user_ctx=UserInfoAgentContext(
            user_memory=User(uid=str(uuid.uuid4())),
            bq_client=bq,
            table_name=TABLE_NAME
        ),
        mechanic_ctx=MechanicAgentContext(
            car_memory=UserCarDetails()
        )
    )

    orchestrator = Orchestrator(
        api_key=os.getenv("OPENAI_API_KEY"),
        bq_client=bq,
        context=mgo_context1
    )

    runner = Runner()

    response1 = await runner.run(
        orchestrator.agent,
        "My name is Douglas McArthur, I live at Tondo Manila. I have a Toyota GR86.",
        context=mgo_context1
    )

    print(response1.final_output)

    response2 = await runner.run(
        orchestrator.agent,
        "What do you know about me?",
        context=mgo_context1
    )
    print(response2.final_output)

    response3 = await runner.run(
        orchestrator.agent,
        "I would like to update my payment method to Gcash, thanks.",
        context=mgo_context1
    )
    print(response3.final_output)

    mgo_context2 = MechaniGoContext(
        user_ctx=UserInfoAgentContext(
            user_memory=User(uid=str(uuid.uuid4())), # panibagong UUID
            bq_client=bq,
            table_name=TABLE_NAME
        ),
        mechanic_ctx=MechanicAgentContext(
            car_memory=UserCarDetails()
        )
    )

    response4 = await runner.run(
        orchestrator.agent,
        "What do you know about me?",
        context=mgo_context2
    )
    print(response4.final_output)

asyncio.run(test())