from fastapi import Query, APIRouter, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse
from components.agent_tools import (
    UserInfoAgentContext, MechanicAgentContext, BookingAgentContext
)
from agents import InputGuardrailTripwireTriggered, set_default_openai_key, SQLiteSession
from helpers import ensure_chat_history_table_ready, save_convo
from components import MechaniGoAgent, MechaniGoContext
from config import TEST_TABLE_NAME, DATASET_NAME
from components.utils import BigQueryClient
from schemas import User, UserCarDetails
from pydantic import BaseModel
import pytz

PH_TZ = pytz.timezone("Asia/Manila")

__all__ = [
    "HTTPException", "JSONResponse", "APIRouter", "status", "Query", "Depends", "Request",
    "UserInfoAgentContext", "MechanicAgentContext", "BookingAgentContext", "SQLiteSession",
    "BigQueryClient", "User", "UserCarDetails", "set_default_openai_key",
    "ensure_chat_history_table_ready", "save_convo",
    "MechaniGoAgent", "MechaniGoContext",
    "InputGuardrailTripwireTriggered",
    "TEST_TABLE_NAME", "DATASET_NAME",
    "BaseModel",
    "PH_TZ"
]