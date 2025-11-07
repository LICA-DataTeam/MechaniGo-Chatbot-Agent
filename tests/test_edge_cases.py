import os
import json
import asyncio
import logging
from uuid import uuid4
from datetime import datetime
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
    expected_behavior: Optional[str] = None

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

def booking_scenario(option: Literal["A", "B", "C", "D", "all"] = "A"):
    # Blind test
    # Provide details into one chat
    test1 = [
        # Chatbot will ask details
        Tests(
            question="Hi, I want to book an appointment for PMS.",
            category="booking",
            expected_behavior="Bot should ask for more details (e.g., name, contact, address, car details, date/time, etc.)"
        ),
        Tests(
            question=(
                "My name is Walter Hartwell White. "
                "I live at 308 Negra Arroyo Lane, Albuquerque, New Mexico, 87104. "
                "My contact num is 09171111111, I want to book a PMS appointment at "
                "December 25, 2025 at around 9 am. I have a Honda Civic 2020. "
                "And my preferred payment is gcash. Thank you."
            ),
            category="booking",
            expected_behavior="Bot should confirm the details and save to BigQuery."
        )
    ]

    # Scenario where there is an incorrect date
    test2 = [
        Tests(
            question="Hi, I want to book an appointment for PMS.",
            category="booking"
        ),
        Tests(
            question="I want to book a schedule on January 1, 1970 at around 12 am.",
            category="booking",
            expected_behavior="Two outcomes: either tripwire will trigger or bot should reject the date as invalid."
        )
    ]

    # Scenario where car is invalid
    test3 = [
        Tests(
            question="Hi, I want to book a PMS appointment for my Nissan Vios 1969.",
            category="booking",
            expected_behavior="Bot should confirm if car detail is correct."
        )
    ]

    # Scenario where there is a multi-turn booking (realistic conversation where info is added gradually)
    test4 = [
        Tests(
            question="Hello po pede po ba mag book PMS?",
            expected_behavior="Bot should ask for customer details."
        ),
        Tests(
            question="Car: Ford Fiesta 2015.\nSchedule: December 5, 2025 mga 2pm po.",
            expected_behavior="Bot should ask for other missing details."
        ),
        Tests(
            question="Sa Pasig po ako malapit sa Tiendesitas.",
            expected_behavior="Bot should validate the location, and ask for remaining important details like the name, and contact."
        ),
        Tests(
            question="Name: Juan Santos\nContact: 091333333. Gcash po thank you.",
            expected_behavior="Bot should confirm booking details and close gracefully."
        )
    ]

    scenarios = {"A": test1, "B": test2, "C": test3, "D": test4}

    if option == "all":
        combined = []
        for s in scenarios.values():
            combined.extend(s)
        return combined

    if option not in scenarios:
        raise ValueError(f"Invalid booking option: {option}. Options are from A-D or all.")

    return scenarios[option]

def faq_scenario():
    tests = [
        Tests(
            question="Where are you located po?",
            category="faq",
            expected_behavior="Bot should answer using appropriate tools (FAQAgent)"
        ),
        Tests(
            question="Ano po service nyo?",
            category="faq",
            expected_behavior="Bot should answer using appropriate tools (FAQAgent)"
        ),
        Tests(
            question="May branch po ba kayo sa Pampanga?",
            category="faq",
            expected_behavior="Bot should answer using appropriate tools (FAQAgent) and let the user know there is NO branch in Pampanga."
        ),
        Tests(
            question="Where are the parts coming from?", # Intended as vague (will check if chatbot understands the context here)
            category="faq",
            expected_behavior="Bot should answer using appropriate tools (FAQAgent). Bot should at least have a context for the question."
        )
    ]
    return tests

def mechanic_scenario():
    tests = [
        Tests(
            question="What does acid rain damage look like on a car?",
            category="mechanic",
            expected_behavior="Bot should answer accordingly, using the appropriate tools (MechanicAgent)"
        ),
        Tests(
            question="How do I know the cost of transmission oil change?",
            category="mechanic",
            expected_behavior="Bot should answer accordingly, using the appropriate tools (MechanicAgent)"
        ),
        # In this question, chatbot should:
        # Provide answer agad, at the very end, will ask for the user's car details because
        # the issue is car-specific (electric car or not)
        Tests(
            question="My car is having battery issues. Can you help me?",
            category="mechanic",
            expected_behavior="Bot should still provide an answer despite the issue being a car-specific problem. At the very end, chatbot should optionally ask for the user's car details."
        ),
        Tests(
            question="Mahina na po AC ng Toyota Vios 2019 ko. Ano po kaya problem?",
            category="mechanic",
            expected_behavior="Bot should answer accordingly."
        )
    ]
    return tests

def save_test_result(test_category: str, test_question: str, response: str, expected_behavior: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "test_results")
    os.makedirs(output_dir, exist_ok=True)

    date = datetime.now().strftime("%Y-%m-%d")

    filepath = os.path.join(output_dir, "results.json")

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {
                    "datetime": date,
                    "data": []
                }
    else:
        existing_data = {
            "datetime": date,
            "data": []
        }

    new_entry = {
        "test_category": test_category,
        "test_question": test_question,
        "response": response,
        "expected_behavior": expected_behavior
    }
    existing_data["data"].append(new_entry)
    existing_data["datetime"] = date

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Exception while saving: {e}")
    logging.info(f"Saved test result: {filepath}")

def load_tests(category: Literal["booking", "faq", "mechanic", "all"], booking_option: str = "A"):
    scenarios = {
        "booking": lambda: booking_scenario(booking_option),
        "faq": faq_scenario,
        "mechanic": mechanic_scenario
    }

    if category == "all":
        all_tests = []
        for func in scenarios.values():
            all_tests.extend(func())
        return all_tests

    func = scenarios.get(category)
    if not func:
        raise ValueError(f"Unknown test category: {category}")
    return func()

async def main(test_category: Literal["booking", "faq", "mechanic", "all"] = "booking", booking_option: str = "A", delete_table: bool = False):
    logging.info("Initializing context...")
    context = init_contexts()
    logging.info("Done initializing context! Starting MechaniGo agent...") 
    agent = MechaniGoAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        table_name="chatbot_test_cases",
        context=context
    )

    tests = load_tests(test_category, booking_option)
    prev_category = None
    scenario_marker = None
    for _, test in enumerate(tests, start=1):
        logging.info(f"Running test {_}: {test.category}")
        current_marker = f"{test.category}_{booking_option}"

        if scenario_marker != current_marker:
            logging.info(f"========== New session for {current_marker} ==========")
            agent.session = None
            scenario_marker = current_marker

        if prev_category != test.category:
            agent.session = None
            prev_category = test.category

        run = await Runner.run(
            agent.agent,
            input=test.question,
            context=agent.context,
            session=agent.session
        )
        logging.info("Saving test results...")
        save_test_result(test.category, test.question, run.final_output, test.expected_behavior)

    if delete_table:
        agent._delete_table()

# OPTIONS:
# pass in asyncio.run() the following options:
# main(test_category="booking", booking_option="A")
# - (test_category: booking, faq, mechanic, or all, booking_option: A-E or all)
asyncio.run(main(test_category="all", booking_option="all"))