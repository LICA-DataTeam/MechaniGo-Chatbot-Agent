from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from typing import Optional
from enum import Enum
import os

class Environment(str, Enum):
    DEV = "development"
    PROD = "production"


class BaseConfiguration(BaseSettings):
    """
    MechaniGo Chatbot API base configuration class for managing project settings and environment.
    """
    ENV: Environment = Field(
        default=Environment.DEV,
        description="The environment the project is running (e.g., development or production)."
    )

    APP_NAME: str = "MechaniGo Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    API_PREFIX: str = "/mgo-chatbot-api/v1"
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    API_RELOAD: bool = Field(default=True, description="Enable auto-reload")

    # OpenAI configurations
    OPENAI_API_KEY: str = Field(...)
    OPENAI_MODEL: str = Field(
        default="gpt-4.1",
        description="Default OpenAI model"
    )
    OPENAI_MAX_TOKENS: Optional[int] = Field(
        default=500,
        description="Max tokens (ModelSettings)"
    )
    MAIN_AGENT_TEMPERATURE: Optional[float] = Field(default=0.2, description="Control for determining the model's response.")
    SUB_AGENT_TEMPERATURE: Optional[float] = Field(default=0.1, description="Control for determining the model's response.")
    FAQ_VECTOR_STORE_ID: Optional[str] = Field(default=None, description="Chatbot knowledgebase for FAQs.")
    MECHANIC_VECTOR_STORE_ID: Optional[str] = Field(default=None, description="Chatbot knowledgebase for mechanic.")

    # Supabase configurations
    SUPABASE_API_KEY: str = Field(..., description="The unique Supabase Key which is supplied when you create a new project in your project dashboard.")
    SUPABASE_URL: str = Field(..., description="The unique Supabase URL which is supplied when you create a new project in your project dashboard.")

    LOG_LEVEL: str = Field(default="INFO", description="logging level")
    LOG_FORMAT: str = "%(asctime)s - %(name)s  - %(levelname)s - %(message)s"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def is_development(self) -> bool:
        return self.env == Environment.DEV

    @property
    def is_production(self) -> bool:
        return self.env == Environment.PROD


class DevelopmentSettings(BaseConfiguration):
    ENV: Environment = Environment.DEV
    LOG_LEVEL: str = "DEBUG"

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class ProductionSettings(BaseConfiguration):
    ENV: Environment = Environment.PROD
    API_RELOAD: bool = False
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"

    model_config = SettingsConfigDict(
        env_file=".env.prod",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache
def get_settings() -> BaseConfiguration:
    env = os.getenv("ENVIRONMENT", "development").lower()
    settings_map = {
        "development": DevelopmentSettings,
        "production": ProductionSettings
    }

    settings_class = settings_map.get(env, DevelopmentSettings)
    return settings_class()