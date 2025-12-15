from supabase import acreate_client
from functools import lru_cache
from config import settings

@lru_cache
def _supabase_settings():
    return settings.SUPABASE_URL, settings.SUPABASE_API_KEY

_supabase_client = None

async def get_supabase_client():
    global _supabase_client
    if _supabase_client is None:
        url, api_key = _supabase_settings()
        _supabase_client = await acreate_client(url, api_key)
    return _supabase_client