from components.agent_tools import MechanicAgentContext
from components import MechaniGoAgent, MechaniGoContext
from schemas import UserCarDetails
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Runner
import asyncio
import json
import os

load_dotenv()

class TestCase(BaseModel):
    question: str
    category: str = None
    car_specific: bool
    car_details: str | None = None

async def main():
    ctx = MechaniGoContext(
        mechanic_ctx=MechanicAgentContext(
            car_memory=UserCarDetails()
        )
    )

    agent = MechaniGoAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        context=ctx
    )

    tests = [
        # Non-car specific
        TestCase(
            question="How much does it cost to change transmission fluid?",
            car_specific=False,
            car_details=None
        ),
        TestCase(
            question="How often should I change my oil?",
            car_specific=False,
            car_details=None
        ),
        # Car specific
        # Scenario 1 and 2: No car mentioned
        # LLM should ask the about car details (it could either be electric or non-electric)
        TestCase(
            question="My car's battery is not working. What should I do?",
            car_specific=True,
            car_details=None
        ),
        TestCase(
            question="Nagdadrain battery ng car ko. Bakit po kaya?",
            car_specific=True,
            car_details=None
        ),
        # Scenario 3 and 4: Car mentioned
        TestCase(
            question="Yung AC po ng 2016 Dodge Charger ko humihina na.",
            car_specific=True,
            car_details="2016 Dodge Charger"
        ),
        TestCase(
            question="I have a Mazda CX-5, and it's having trouble starting. What should I do?",
            car_specific=True,
            car_details="Mazda CX-5"
        )
    ]

    results = {}
    try:
        for i, test in enumerate(tests, start=1):
            result = await Runner.run(agent.agent, test.question)
            results[i] = {
                "response": getattr(result, "final_output", "No Output")
            }
    except Exception as e:
        print(f"Exception: {e}")
    
    print(json.dumps(results, indent=2, ensure_ascii=False))

asyncio.run(main())