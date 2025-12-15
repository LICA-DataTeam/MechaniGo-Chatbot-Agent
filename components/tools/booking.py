from components.common import RunContextWrapper, function_tool
from components.utils import ToolRegistry, get_supabase_client
from components import MechaniGoContext
from typing import Optional

@function_tool(name_override="save_user_info")
async def save_user_info(
    ctx: RunContextWrapper[MechaniGoContext],
    name: Optional[str] = None,
    email: Optional[str] = None,
    address: Optional[str] = None,
    contact_num: Optional[str] = None,
    service_type: Optional[str] = None,
    schedule: Optional[str] = None,
    payment: Optional[str] = None,
    car_make: Optional[str] = None,
    car_model: Optional[str] = None,
    car_year: Optional[int] = None
):
    """
    Docstring for save_user_info
    
    :param ctx: Context for session memory.
    :type ctx: RunContextWrapper[MechaniGoContext]
    :param name: User name.
    :type name: Optional[str]
    :param email: User email.
    :type email: Optional[str]
    :param address: User address.
    :type address: Optional[str]
    :param contact_num: User contact number.
    :type contact_num: Optional[str]
    :param service_type: User service type (PMS, Secondhand Car inspection, PMS Oil-Change, Parts Replacement).
    :type service_type: Optional[str]
    :param schedule: User schedule date and time.
    :type schedule: Optional[str]
    :param payment: User preferred payment type (GCash, Cash, Card).
    :type payment: Optional[str]
    :param car_make: User car make.
    :type car_make: Optional[str]
    :param car_model: User car model.
    :type car_model: Optional[str]
    :param car_year: User car year of manufacture.
    :type car_year: Optional[int]
    """

    client = await get_supabase_client()
    user_id = ctx.context.user_ctx.user_memory.uid

    existing = await (
        client.table("user_bookings")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    current = existing.data[0] if existing.data else {}

    def pick(val, key):
        return val if val is not None else current.get(key)

    payload = {
        "user_id": user_id,
        "name": pick(name, "name"),
        "email": pick(email, "email"),
        "address": pick(address, "address"),
        "contact_num": pick(contact_num, "contact_num"),
        "service_type": pick(service_type, "service_type"),
        "schedule": pick(schedule, "schedule"),
        "payment": pick(payment, "payment"),
        "car_make": pick(car_make, "car_make"),
        "car_model": pick(car_model, "car_model"),
        "car_year": pick(car_year, "car_year"),
    }

    await client.table("user_bookings").upsert(
        payload, on_conflict="user_id"
    ).execute()
    updated_fields = [k for k, v in payload.items() if v is not None]
    return {"status": "saved", "updated_fields": updated_fields}

ToolRegistry.register_tool(
    "booking.save_user_info",
    save_user_info,
    category="booking",
    description="Saves/updates user booking info in Supabase."
)