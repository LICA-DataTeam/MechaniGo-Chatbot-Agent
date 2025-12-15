from config.settings import BaseConfiguration, Environment, ProductionSettings, DevelopmentSettings, get_settings

settings = get_settings()

__all__ = [
    "DevelopmentSettings",
    "ProductionSettings",
    "BaseConfiguration",
    "Environment",
    "get_settings",
    "settings"
]