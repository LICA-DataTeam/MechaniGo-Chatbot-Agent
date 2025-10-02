from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import uuid

class PaymentType(str, Enum):
    GCASH = "gcash"
    CASH = "cash"
    CREDIT = "credit"

class TransmissionType(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"

class UserCarDetails(BaseModel):
    make: str
    model: str
    year: Optional[int] = None
    fuel_type: Optional[str] = None
    transmission: Optional[TransmissionType] = None

class User(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    address: str
    contact_num: str
    schedule_date: str = None
    schedule_time: str = None
    payment: Optional[PaymentType] = None
    car: Optional[UserCarDetails] = None
