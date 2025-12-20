# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Tuple


def _base_dir() -> Path:
    # Persistente si Fly monta /data; si no, cae a ./logs
    if Path("/data").exists():
        return Path("/data")
    return Path("logs")


BASE_DIR = _base_dir()
BASE_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = BASE_DIR / "learning_state.json"


def _load() -> dict:
    if not STATE_PATH.exists():
        return {"stats": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _save(state: dict) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _key(base: str, tf: str, side: str) -> str:
    return f"{base}|{tf}|{side}"


def update_result(base: str, tf: str, side: str, outcome: str) -> None:
    """
    outcome: 'TP' o 'SL'
    EWMA: pondera más lo reciente (aprende rápido).
    """
    state = _load()
    stats = state["stats"]
    k = _key(base, tf, side)

    if k not in stats:
        stats[k] = {"ewma_wr": 0.5, "n": 0}

    alpha = 0.20  # aprende rápido
    win = 1.0 if outcome == "TP" else 0.0

    prev = float(stats[k]["ewma_wr"])
    stats[k]["ewma_wr"] = (1 - alpha) * prev + alpha * win
    stats[k]["n"] = int(stats[k]["n"]) + 1

    _save(state)


def should_trade(
    base: str,
    tf: str,
    side: str,
    min_trades: int = 8,
    min_ewma_wr: float = 0.45,
) -> Tuple[bool, str]:
    """
    Bloqueo automático:
    - si n >= min_trades y ewma_wr < min_ewma_wr -> bloquear.
    """
    state = _load()
    s = state["stats"].get(_key(base, tf, side))
    if not s:
        return True, "sin histórico (permitido)"

    n = int(s.get("n", 0))
    ewma = float(s.get("ewma_wr", 0.5))

    if n >= min_trades and ewma < min_ewma_wr:
        return False, f"bloqueado por bajo rendimiento (ewma_wr={ewma:.2f}, n={n})"

    return True, f"ok (ewma_wr={ewma:.2f}, n={n})"
