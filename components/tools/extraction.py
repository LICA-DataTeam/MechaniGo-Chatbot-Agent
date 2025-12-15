"""
The extraction tools library.
"""
from components.tools.clients import get_openai_client, MODEL_TYPE
from components.common import function_tool
from components.utils import ToolRegistry
import json

@function_tool(name_override="extract_user_info")
async def extract_user_info(text: str):
    client = await get_openai_client()
    response = await client.responses.create(
        model=MODEL_TYPE,
        input=[
            {"role": "assistant", "content": "Extract explicit user details only."},
            {"role": "user", "content": text}
        ],
        tools=[
            {
                "type": "function",
                "name": "return_extracted_data",
                "description": "Return extracted user info in structured JSON.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "nullable": True},
                        "email": {"type": "string", "nullable": True},
                        "address": {"type": "string", "nullable": True},
                        "contact_num": {"type": "string", "nullable": True},
                        "service_type": {"type": "string", "nullable": True},
                        "date": {"type": "string", "format": "date-time", "nullable": True},
                        "payment": {
                            "type": "string",
                            "enum": ["GCash", "Cash", "Credit"],
                            "nullable": True
                        },
                        "car": {
                            "type": "object",
                            "properties": {
                                "car_make": {"type": "string", "nullable": True},
                                "car_model": {"type": "string", "nullable": True},
                                "car_year": {"type": "integer", "nullable": True},
                            },
                            "nullable": True
                        }
                    },
                    "required": []
                }
            }
        ]
    )

    tool_calls = [
        item for item in response.output if item.type == "function_call"
    ]

    if not tool_calls:
        return {}

    return json.loads(
        tool_calls[0].arguments
    )

ToolRegistry.register_tool(
    "extract.user_info",
    extract_user_info,
    category="extraction",
    description="Extracts user information from text."
)