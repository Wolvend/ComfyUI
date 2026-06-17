from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"entries": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("entries"), dict):
                return data
        except Exception:
            pass
        return {"entries": {}}

    def _save(self, data: dict[str, Any]) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def get(self, key: str) -> Any | None:
        data = self._load()
        entry = data.get("entries", {}).get(key)
        if not entry:
            return None
        expires_at = entry.get("expires_at")
        if expires_at is not None and expires_at < time.time():
            data["entries"].pop(key, None)
            self._save(data)
            return None
        return entry.get("value")

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        data = self._load()
        data.setdefault("entries", {})[key] = {
            "expires_at": time.time() + ttl_seconds if ttl_seconds > 0 else None,
            "value": value,
        }
        self._save(data)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def list_keys(self) -> list[str]:
        data = self._load()
        return sorted(data.get("entries", {}).keys())

    def stats(self) -> dict[str, Any]:
        data = self._load()
        entries = data.get("entries", {})
        active = 0
        expired = 0
        for entry in entries.values():
            expires_at = entry.get("expires_at")
            if expires_at is not None and expires_at < time.time():
                expired += 1
            else:
                active += 1
        return {
            "path": str(self.path),
            "entry_count": len(entries),
            "active_count": active,
            "expired_count": expired,
            "keys": sorted(entries.keys()),
        }

    def delete(self, key: str) -> bool:
        data = self._load()
        entries = data.get("entries", {})
        if key not in entries:
            return False
        entries.pop(key, None)
        self._save(data)
        return True
