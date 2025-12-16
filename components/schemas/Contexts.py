from components.schemas import User
from pydantic import BaseModel

class UserInfoContext(BaseModel):
    user_memory: User
    model_config = {"arbitrary_types_allowed": True}


class MechaniGoContext(BaseModel):
    user_ctx: UserInfoContext
    model_config = {"arbitrary_types_allowed": True}