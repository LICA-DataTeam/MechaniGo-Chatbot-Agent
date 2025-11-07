import os
import json
import asyncio
import logging
from uuid import uuid4
from dotenv import load_dotenv

from agents import Runner
from pydantic import BaseModel
from typing import Optional, Literal

from components.agent_tools import (
    UserInfoAgentContext,
    MechanicAgentContext,
    BookingAgentContext
)
from components import (
    MechaniGoContext,
    MechaniGoAgent
)
from schemas import (
    UserCarDetails,
    User
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

class Tests(BaseModel):
    question: Optional[str] = None
    category: Literal["booking", "faq", "mechanic"] = None

def init_contexts() -> MechaniGoContext:
    return MechaniGoContext(
        user_ctx=UserInfoAgentContext(
            user_memory=User(uid=str(uuid4()))
        ),
        mechanic_ctx=MechanicAgentContext(
            car_memory=UserCarDetails()
        ),
        booking_ctx=BookingAgentContext()
    )

def booking_scenario():
    tests = [
        # Chatbot will ask details
        Tests(
            question="Hi, I want to book an appointment for PMS.",
            category="booking"
        )
    ]
    return tests

def faq_scenario():
    tests = [
        Tests(
            question="Where are you located po?",
            category="faq"
        ),
        Tests(
            question="Ano po service nyo?",
            category="faq"
        ),
        Tests(
            question="May branch po ba kayo sa Pampanga?",
            category="faq"
        ),
        Tests(
            question="Where are the parts coming from?", # Intended as vague (will check if chatbot understands the context here)
            category="faq"
        )
    ]
    return tests

def mechanic_scenario():
    tests = [
        Tests(
            question="What does acid rain damage look like on a car?",
            category="mechanic"
        ),
        Tests(
            question="How do I know the cost of transmission oil change?",
            category="mechanic"
        ),
        # In this question, chatbot should:
        # Provide answer agad, at the very end, will ask for the user's car details because
        # the issue is car-specific (electric car or not)
        Tests(
            question="My car is having battery issues. Can you help me?",
            category="mechanic"
        ),
        Tests(
            question="Mahina na po AC ng Toyota Vios 2019 ko. Ano po kaya problem?",
            category="mechanic"
        )
    ]
    return tests

def load_tests(category: Literal["booking", "faq", "mechanic"]):
    scenarios = {
        "booking": booking_scenario,
        "faq": faq_scenario,
        "mechanic": mechanic_scenario
    }
    func = scenarios.get(category)
    if not func:
        raise ValueError(f"Unknown test category: {category}")
    return func()

async def main(delete_table: bool = False):
    logging.info("Initializing context...")
    context = init_contexts()
    logging.info("Done initializing context! Starting MechaniGo agent...") 
    agent = MechaniGoAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        table_name="chatbot_test_cases",
        context=context
    )

    input = None # all test cases here
    run1 = await Runner.run(
        agent.agent,
        input=input,
        context=agent.context,
        session=agent.session
    )

    print(f"\n\n\t{run1.final_output}\n\n")

    if delete_table:
        agent._delete_table()

asyncio.run(main())