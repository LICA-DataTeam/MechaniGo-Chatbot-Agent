from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, List, Optional, Dict
import asyncio
import time

from components.common import SessionABC, TResponseInputItem
from components.utils import get_supabase_client


class SessionHandler(SessionABC):
    DEFAULT_TTL_SECONDS = 10  # cache Supabase history for 10s

    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        table: str = "session_history",
    ):
        self.session_id = session_id
        self.user_id = user_id or session_id
        self.table = table

        self._pending_items: List[TResponseInputItem] = []
        self._cache: Dict[Optional[int], tuple[float, list[TResponseInputItem]]] = {}
        self._cache_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    @staticmethod
    def _extract_role(item: TResponseInputItem) -> Optional[str]:
        return item.get("role") if isinstance(item, dict) else None

    @staticmethod
    def _extract_message(item: TResponseInputItem) -> Optional[str]:
        if not isinstance(item, dict):
            return None
        content = item.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict):
                    text_value = block.get("text") or block.get("content")
                    if isinstance(text_value, str):
                        texts.append(text_value)
            joined = " ".join(texts)
            return joined or None

        return None

    @staticmethod
    def _ensure_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
                return parsed if isinstance(parsed, list) else [stripped]
            except json.JSONDecodeError:
                return [stripped]
        return []

    async def _get_cached(self, limit: Optional[int]) -> Optional[list[TResponseInputItem]]:
        async with self._cache_lock:
            entry = self._cache.get(limit)
            if not entry:
                return None
            ts, data = entry
            if (time.time() - ts) < self.DEFAULT_TTL_SECONDS:
                return data
            # expired
            del self._cache[limit]
            return None

    async def _set_cached(self, limit: Optional[int], data: list[TResponseInputItem]):
        async with self._cache_lock:
            self._cache[limit] = (time.time(), data)

    async def _invalidate_cache(self):
        async with self._cache_lock:
            self._cache.clear()

    async def collect_items(self, items: list[TResponseInputItem]):
        if items:
            self._pending_items.extend(items)

    async def persist_items(self):
        async with self._write_lock:
            if not self._pending_items:
                return

            items = self._pending_items
            self._pending_items = []

            await self._write_items_to_supabase(items)
            await self._invalidate_cache()

    async def _write_items_to_supabase(self, items: list[TResponseInputItem]):
        role_messages: Dict[str, List[str]] = defaultdict(list)

        # group by role
        for item in items:
            role = self._extract_role(item)
            message = self._extract_message(item)
            if role and message:
                role_messages[role].append(message)

        if not role_messages:
            return

        client = await get_supabase_client()

        for role, messages in role_messages.items():
            existing = await (
                client.table(self.table)
                .select("id, content")
                .eq("session_id", self.session_id)
                .eq("user_id", self.user_id)
                .eq("role", role)
                .limit(1)
                .execute()
            )

            if existing.data:
                record = existing.data[0]
                current_content = self._ensure_list(record.get("content"))
                updated = current_content + messages

                await (
                    client.table(self.table)
                    .update({"content": updated})
                    .eq("id", record["id"])
                    .execute()
                )

            else:
                await client.table(self.table).insert(
                    {
                        "session_id": self.session_id,
                        "user_id": self.user_id,
                        "role": role,
                        "content": messages,
                    }
                ).execute()

    async def get_items(self, limit: Optional[int] = None) -> list[TResponseInputItem]:
        cached = await self._get_cached(limit)
        if cached is not None:
            return cached

        client = await get_supabase_client()
        query = (
            client.table(self.table)
            .select("role, content")
            .eq("session_id", self.session_id)
            .eq("user_id", self.user_id)
            .order("created_at", desc=False)
        )

        if limit is not None:
            query = query.limit(limit)

        rows = await query.execute()

        history: list[TResponseInputItem] = []
        for row in rows.data or []:
            role = row.get("role")
            messages = self._ensure_list(row.get("content"))
            for msg in messages:
                history.append({"role": role, "content": msg})

        if limit is not None:
            history = history[-limit:]

        await self._set_cached(limit, history)

        return history

    async def add_items(self, items):
        await self.collect_items(items)
        await self.persist_items()

    async def pop_item(self) -> Optional[TResponseInputItem]:
        history = await self.get_items()
        if not history:
            return None

        last = history[-1]
        role = self._extract_role(last)
        if not role:
            return last

        client = await get_supabase_client()
        existing = await (
            client.table(self.table)
            .select("id, content")
            .eq("session_id", self.session_id)
            .eq("user_id", self.user_id)
            .eq("role", role)
            .limit(1)
            .execute()
        )

        if existing.data:
            record = existing.data[0]
            content = self._ensure_list(record.get("content"))
            if content:
                content.pop()
                await (
                    client.table(self.table)
                    .update({"content": content})
                    .eq("id", record["id"])
                    .execute()
                )

        await self._invalidate_cache()
        return last

    async def clear_session(self) -> None:
        client = await get_supabase_client()
        await (
            client.table(self.table)
            .delete()
            .eq("session_id", self.session_id)
            .eq("user_id", self.user_id)
            .execute()
        )
        await self._invalidate_cache()