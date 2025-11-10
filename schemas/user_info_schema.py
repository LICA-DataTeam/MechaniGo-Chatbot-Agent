from pydantic import BaseModel, EmailStr
from typing import Optional
from enum import Enum

class ServiceType(str, Enum):
    PMS = "pms"
    SECONDHAND_INSPECTION = "secondhand_car_buying_inspection"
    PARTS_REPLACEMENT = "parts_replacement"
    CAR_DIAGNOSIS = "car_diagnosis"

class PaymentType(str, Enum):
    GCASH = "gcash"
    CASH = "cash"
    CREDIT = "credit"

class TransmissionType(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"

class UserCarDetails(BaseModel):
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    fuel_type: Optional[str] = None
    transmission: Optional[TransmissionType] = None

class User(BaseModel):
    uid: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[str] = None
    contact_num: Optional[str] = None
    service_type: Optional[str] = None
    schedule_date: Optional[str] = None
    schedule_time: Optional[str] = None
    payment: Optional[str] = None
    car: Optional[str] = None
