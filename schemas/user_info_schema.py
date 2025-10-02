from pydantic import BaseModel
from typing import Optional
from enum import Enum

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
    name: str
    address: str
    contact_num: str
    schedule_date: str = None
    schedule_time: str = None
    payment: Optional[PaymentType] = None
    car: Optional[UserCarDetails] = None
