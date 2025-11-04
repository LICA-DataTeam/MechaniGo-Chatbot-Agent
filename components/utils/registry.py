from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping,  MutableMapping, Sequence
from dataclasses import dataclass, field

GuardRailType = Any
ToolType = Callable[..., Any]

@dataclass(slots=True)
class RegistryItem:
    name: str
    target: Any
    description: str | None = None
    scopes: tuple[str, ...] = ()
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def matches_scope(self, scope: str | None) -> bool:
        if scope is None or not self.scopes:
            return True
        return scope in self.scopes

class _BaseRegistry:
    """Shared logic for tool / guardrail registries."""
    def __init__(self, kind: str):
        self._kind = kind
        self._items: MutableMapping[str, RegistryItem] = {}

    def register(
        self,
        name: str,
        target: Any,
        *,
        description: str | None = None,
        scopes: Sequence[str] | None = None,
        enabled: bool = True,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> RegistryItem:
        key = name.strip()
        if not key:
            raise ValueError(f"{self._kind} name cannot be empty")

        if not overwrite and key in self._items:
            raise ValueError(f"{self._kind} '{key}' is already registered")

        entry = RegistryItem(
            name=key,
            target=target,
            description=description,
            scopes=tuple(scopes or ()),
            enabled=enabled,
            metadata=metadata or {},
        )
        self._items[key] = entry
        return entry

    def get(self, name: str, *, require_enabled: bool = True) -> Any:
        try:
            entry = self._items[name]
        except KeyError as exc:
            raise KeyError(f"{self._kind} '{name}' is not registered") from exc

        if require_enabled and not entry.enabled:
            raise ValueError(f"{self._kind} '{name}' is disabled")
        return entry.target

    def get_many(
        self,
        names: Iterable[str],
        *,
        require_enabled: bool = True,
    ) -> list[Any]:
        return [self.get(n, require_enabled=require_enabled) for n in names]

    def list_names(
        self,
        *,
        scope: str | None = None,
        include_disabled: bool = False,
    ) -> list[str]:
        return [
            name
            for name, entry in self._items.items()
            if (include_disabled or entry.enabled) and entry.matches_scope(scope)
        ]

    def items(
        self,
        *,
        scope: str | None = None,
        include_disabled: bool = False,
    ) -> list[RegistryItem]:
        return [
            entry
            for entry in self._items.values()
            if (include_disabled or entry.enabled) and entry.matches_scope(scope)
        ]

    def enable(self, name: str) -> None:
        self._items[name].enabled = True

    def disable(self, name: str) -> None:
        self._items[name].enabled = False

    def clear(self) -> None:
        self._items.clear()


class ToolRegistry(_BaseRegistry):
    def __init__(self):
        super().__init__(kind="tool")


class GuardrailRegistry(_BaseRegistry):
    def __init__(self):
        super().__init__(kind="guardrail")


# Singleton instances used throughout the app
tool_registry = ToolRegistry()
guardrail_registry = GuardrailRegistry()


# Convenience helpers (optional)
def register_tool(*args, **kwargs) -> RegistryItem:
    return tool_registry.register(*args, **kwargs)


def register_guardrail(*args, **kwargs) -> RegistryItem:
    return guardrail_registry.register(*args, **kwargs)