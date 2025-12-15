"""
Shared OpenAI client across the function tools.

The model type is `gpt-4o-mini`.
"""
from components.common import AsyncOpenAI
from config import settings

_client = None
MODEL_TYPE = "gpt-4o-mini"

async def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client