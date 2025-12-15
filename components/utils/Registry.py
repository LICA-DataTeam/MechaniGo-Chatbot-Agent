from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class ToolEntry:
    func: Callable[..., Any]
    category: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ToolRegistry:
    _tools: Dict[str, ToolEntry] = {}
    _agents: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register_tool(
        cls,
        name: str,
        func: Callable[..., Any],
        *,
        category: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        cls._tools[name] = ToolEntry(
            func=func,
            category=category,
            description=description,
            metadata=metadata or {}
        )

    @classmethod
    def get_tool(cls, name: str) -> Callable[..., Any]:
        return cls._tools[name].func

    @classmethod
    def list_tools(cls, category: Optional[str] = None) -> Dict[str, ToolEntry]:
        if category is None:
            return cls._tools.copy()
        return {k: v for k, v in cls._tools.items() if v.category == category}

    @classmethod
    def register_agent(cls, name: str, factory: Callable[..., Any]) -> None:
        cls._agents[name] = factory

    @classmethod
    def get_agent(cls, name: str) -> Callable[..., Any]:
        return cls._agents[name]