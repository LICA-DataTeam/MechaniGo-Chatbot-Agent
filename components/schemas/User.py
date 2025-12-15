from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCarDetails(BaseModel):
    make: str = None
    model: str = None
    year: int = None

class User(BaseModel, EmailStr):
    uid: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[str] = None
    contact_num: Optional[str] = None
    service_type: Optional[str] = None
    schedule_date: Optional[str] = None
    schedule_time: Optional[str] = None
    payment: Optional[str] = None
    car: Optional[UserCarDetails] = None