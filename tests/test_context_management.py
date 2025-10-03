"""
Test script provided by ChatGPT
- To test and understand usage of context management
"""
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
from agents import Agent, tool, Runner, RunContextWrapper, function_tool

from components.utils import BigQueryClient
from google.cloud import bigquery

from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Your user schema (simplified)
class UserMemory(BaseModel):
    uid: str
    name: Optional[str] = None
    address: Optional[str] = None
    contact_num: Optional[str] = None
    schedule_date: Optional[str] = None
    schedule_time: Optional[str] = None
    payment: Optional[str] = None
    car: Optional[str] = None

class AgentContext(BaseModel):
    memory: UserMemory
    bq_client: BigQueryClient  # your BigQuery client
    table_name: str = "test_users"

    model_config = {
        "arbitrary_types_allowed": True
    }

# ---- Tools ----
@function_tool
def extract_user_info(
    ctx: RunContextWrapper[AgentContext],
    name: Optional[str] = None,
    address: Optional[str] = None,
    contact_num: Optional[str] = None,
    schedule_date: Optional[str] = None,
    schedule_time: Optional[str] = None,
    payment: Optional[str] = None,
    car: Optional[str] = None,
):
    """
    Update memory with partial user info. 
    """
    user = ctx.context.memory
    if name: user.name = name
    if address: user.address = address
    if contact_num: user.contact_num = contact_num
    if schedule_date: user.schedule_date = schedule_date
    if schedule_time: user.schedule_time = schedule_time
    if payment: user.payment = payment
    if car: user.car = car

    print(f"Updated user memory: {user}")
    # Also persist in BigQuery if you want:
    print(f"Inserting user {user.name} into {ctx.context.table_name}")
    # record = user.model_dump(
    #     include={
    #         "uid", "name", "contact_num", "address", "car", "payment"
    #     }
    # )
    try:
        # ctx.context.bq_client.test_insert_user(ctx.context.table_name, record) # expects user to be of type dict
        ctx.context.bq_client.insert_user(ctx.context.table_name, user)
    except Exception as e:
        print(f"Failed to insert: {e}")
        return {"status": "failed", "user": user}

    return {"status": "updated", "user": user}

@function_tool
def get_user_info(ctx: RunContextWrapper[AgentContext]):
    """
    Retrieve user info from memory (local), fallback to BigQuery.
    """
    import traceback
    try:
        user = ctx.context.memory
        if user and user.contact_num:
            return user.__dict__

        user = ctx.context.bq_client.get_user_by_contact(ctx.context.table_name, user.contact_num)
        if user:
            return {"status": "success", "user": user}
    except Exception as e:
        print(f"Failed to retrieve user: {e}")
        traceback.print_exc()
        return {"message": "No user data yet."}

async def main():
    # ---- Agent ----
    agent = Agent(
        name="UserInfoAgent",
        instructions="""
        You are a user info memory agent.
        Always call `extract_user_info` when the user provides details (name, contact, car, etc.).
        You can call `get_user_info` when asked to recall details.
        """,
        tools=[extract_user_info, get_user_info]
    )

    # ---- Runner with context ----
    runner = Runner()

    # Example: initialize context with empty memory + clients
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
    bq.ensure_table("test_users", schema)
    context = AgentContext(
        memory=UserMemory(uid="1234"), 
        bq_client=bq,  # pass your BigQueryClient instance
    )

    # Run an example conversation
    response = await runner.run(agent, "My name is Walter Hartwell White, I live at 308 Negra Arroyo Lane and my car is a Toyota Vios 2019. My contact num is 09171234567", context=context)
    print(response.final_output)

    response2 = await runner.run(agent, "What do you know about me?", context=context)
    print(response2.final_output)

    response3 = await runner.run(
        agent,
        "My mode of payment is gcash", context=context
    )
    print(response3.final_output)

    context2 = AgentContext(
        memory=UserMemory(uid="5678"),
        bq_client=bq
    )

    # Dapat wala since new context
    response4 = await runner.run(
        agent,
        "What do you know about me?",
        context=context2
    )
    print(response4.final_output)

asyncio.run(main())