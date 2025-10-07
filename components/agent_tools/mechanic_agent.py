from agents import RunContextWrapper, function_tool
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

        self.agent = create_agent(
            api_key=self.api_key,
            name=self.name,
            handoff_description=self.description,
            instructions=(
                "You are a car mechanic agent.\n\n"
                "Your main task is to aid the clients with car related inquiries such as car diagnosis.\n\n"
                "Always ask clients to provide their car details.\n\n"
                "Call extract_car_info when client provides their car information.\n\n"
                "Output their car details for confirmation.\n\n"
                "Example: Please confirm if the following car details are correct: (Ford Fiesta...)\n\n"
                "Important: Show the error logs if any.\n\n"
            ),
            model=self.model,
            tools=[]
        )

    @property
    def as_tool(self):
        return self.agent.as_tool(
            tool_name=self.name,
            tool_description=self.description
        )

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
        car = ctx.context.mechanic_ctx.car_memory
        if make: car.make = make
        if model: car.model = model
        if year: car.year = year
        if fuel_type: car.fuel_type = fuel_type
        if transmission: car.transmission = transmission

        self.logger.info(f"Updated user car details: {car}")
        return {
            "status": "extracted",
            "car_details": car.model_dump(),
            "message": f"Please confirm if the following car details are correct: {car.model_dump()}"
        }