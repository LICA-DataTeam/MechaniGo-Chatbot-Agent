from agents import Agent, RunContextWrapper, function_tool
from components.utils import create_agent
from schemas import UserCarDetails
from typing import Optional, Any
from pydantic import BaseModel
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MechanicAgentContext(BaseModel):
    car_memory: UserCarDetails
    model_config = {
        "arbitrary_types_allowed": True
    }

class MechanicAgent:
    """Handles all car related inquiries."""
    def __init__(
        self,
        api_key: str,
        name: str = "mechanic_agent",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key
        self.name = name
        self.model = model
        self.description = "Handles car related inquiries."
        self.logger = logging.getLogger(__name__)

        extract_car_info = self._create_extract_car_info()

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=self._dynamic_instructions,
            model=self.model,
            tools=[extract_car_info]
        )

    @property
    def as_tool(self):
        return self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )

    def _dynamic_instructions(
        self,
        ctx: RunContextWrapper[Any],
        agent: Agent
    ):
        user_car_string = getattr(ctx.context.user_ctx.user_memory, "car", None)
        car = ctx.context.mechanic_ctx.car_memory
        self.logger.info(f"User.car string: {user_car_string}")
        self.logger.info(f"Current car_memory: {car.model_dump()}")
        
        needs_parsing = user_car_string and not (car.make and car.model)
        if needs_parsing:
            prompt = (
                f"You are {agent.name}, a car mechanic sub-agent.\n\n"
                f"IMPORTANT: The user has provided the following car description: '{user_car_string}'\n\n"
                "However, this needs to be converted into structured details.\n\n"
                "Your task: \n"
                "1. Parse the car description into make, model, year, fuel_type, and transmission\n\n"
                "2. Call extract_car_info with the parsed details.\n\n"
                "Examples:\n"
                "- 'Toyota AE86' -> extract_car_info(make='Toyota', model='AE86')\n"
                "- 'Ford Fiesta' -> extract_car_info(make='Ford', model='Fiesta')\n"
                "Your other tasks:\n\n"
                "- Car diagnosis and troubleshooting\n"
                "- Maintenance recommendations\n"
                "- Updating car details if user provides new information\n\n"
                "After extracting, ask for confirmation.\n\n"
            )
        else:
            car_details = f"{car.make or 'Unknown'} {car.model or ''} {car.year or ''}".strip()
            prompt = (
                f"You are {agent.name}, a car mechanic sub-agent.\n\n"
                f"User's car details: {car_details}\n\n"
                "You can help with: \n"
                "- Car diagnosis and troubleshooting\n"
                "- Maintenance recommendations\n"
                "- Updating car details if user provides new information\n\n"
                "If user mentions different car details, call extract_car_info to update.\n"
            )
        return prompt

    def _create_extract_car_info(self):
        @function_tool
        def extract_car_info(
            ctx: RunContextWrapper[MechanicAgentContext],
            make: Optional[str] = None,
            model: Optional[str] = None,
            year: Optional[str] = None,
            fuel_type: Optional[str] = None,
            transmission: Optional[str] = None
        ):
            return self._extract_car_info(
                ctx, make, model, year, fuel_type, transmission
            )
        return extract_car_info

    def _extract_car_info(
        self,
        ctx: RunContextWrapper[Any], # MechaniGoContext
        make: Optional[str] = None,
        model: Optional[str] = None,
        year: Optional[str] = None,
        fuel_type: Optional[str] = None,
        transmission: Optional[str] = None,
    ):
        self.logger.info("========== Extracting Car Info ==========")
        self.logger.info(f"Received: make={make}, model={model}, year={year}")

        car = ctx.context.mechanic_ctx.car_memory
        if make: car.make = make
        if model: car.model = model
        if year:
            try:
                car.year = int(year) if isinstance(year, str) else year
            except ValueError:
                self.logger.warning(f"Invalid year format for vehicle: {year}")
        if fuel_type: car.fuel_type = fuel_type
        if transmission: car.transmission = transmission

        self.logger.info(f"Updated user car details: {car}")

        if hasattr(ctx.context.user_ctx.user_memory, "car"):
            car_string = f"{car.year or ''} {car.make or ''} {car.model or ''}".strip()
            ctx.context.user_ctx.user_memory.car = car_string
            self.logger.info(f"Updated User.car string: '{car_string}'")
        return {
            "status": "extracted",
            "car_details": car.model_dump(),
            "message": f"Please confirm if the following car details are correct: {car.model_dump()}"
        }