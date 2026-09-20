"""Log store JSONL determinista (SIMULATED, sim-only 0EUR).

Design #171: `app/log_store.py` append/replay JSONL determinista para
replay igual a live-run (hash idéntico). Sim-time `t` explícito en cada
evento. Solo escribe JSONL sim; nunca toca B1-4 ni yamls reales.

Contratos:
  LogStore(path).append(ev) -> None
  LogStore(path).replay(offset=0) -> list[dict]
  state_hash(obj) -> sha256 hex (json sort_keys)
"""

import hashlib
import json
from pathlib import Path

SIMULATED = True


def state_hash(obj) -> str:
    """Hash determinista de estado dashboard (json sort_keys)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


class LogStore:
    """Append/replay JSONL determinista (una línea = un evento)."""

    def __init__(self, path):
        self.path = Path(path)

    def append(self, event: dict) -> None:
        """Añade evento con `t` explícito (json sort_keys + \\n)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def replay(self, offset: int = 0):
        """Lee eventos desde offset (lista de dicts, orden archivo)."""
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        events = [json.loads(ln) for ln in lines]
        return events[int(offset):]
