from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def estimate_tokens(text: str) -> int:
    # Coarse token estimate for deterministic penalties.
    return max(1, len(text) // 4)
