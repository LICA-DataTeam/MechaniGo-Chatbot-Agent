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
        car = ctx.context.mechanic_ctx.car_memory
        self.logger.info(f"Current car_memory: {car.model_dump()}")
        
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
        
        def norm_str(x): return (x or "").strip()
        def norm_int_str(x):
            if x is None:
                return None
            try:
                return int(str(x).strip())
            except (ValueError, TypeError):
                return None

        incoming = {
            "make": norm_str(make) or None,
            "model": norm_str(model) or None,
            "year": norm_int_str(year) or None,
            "fuel_type": norm_str(fuel_type) or None,
            "transmission": norm_str(transmission) or None
        }

        current = {
            "make": norm_str(car.make) or None,
            "model": norm_str(car.model) or None,
            "year": car.year if isinstance(car.year, int) else norm_int_str(car.year),
            "fuel_type": norm_str(car.fuel_type) or None,
            "transmission": norm_str(car.transmission) or None
        }

        changed_fields = {}
        if incoming["make"] is not None and incoming["make"] != current["make"]:
            car.make = incoming["make"]; changed_fields["make"] = car.make
        if incoming["model"] is not None and incoming["model"] != current["model"]:
            car.model = incoming["model"]; changed_fields["model"] = car.model
        if incoming["year"] is not None and incoming["year"] != current["year"]:
            car.year = incoming["year"]; changed_fields["year"] = car.year
        if incoming["fuel_type"] is not None and incoming["fuel_type"] != current["fuel_type"]:
            car.fuel_type = incoming["fuel_type"]; changed_fields["fuel_type"] = car.fuel_type
        if incoming["transmission"] is not None and incoming["transmission"] != current["transmission"]:
            car.transmission = incoming["transmission"]; changed_fields["transmission"] = car.transmission

        if not changed_fields:
            self.logger.info("========== _extract_car_info() ==========")
            self.logger.info(f"Current car_memory: {car.model_dump()}")
            return {
                "status": "no_change",
                "message": "Car details unchanged.",
                "car_details": car.model_dump()
            }
        self.logger.info(f"Updated user car details: {car}")

        return {
            "status": "success",
            "changed_fields": changed_fields,
            "car_details": car.model_dump(),
            "message": f"Updated car details: {changed_fields}. "
                        f"Please confirm if these are correct."
        }