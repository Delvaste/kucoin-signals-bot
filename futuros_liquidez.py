# -*- coding: utf-8 -*-
import os
import time
import json
import threading
import signal
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import requests
import talib
import yaml
import mplfinance as mpf
from flask import Flask
from zoneinfo import ZoneInfo

from learning import update_result, should_trade

# =======================
# ENTORNO / CREDENCIALES
# =======================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

# =======================
# CONFIG
# =======================
def load_config() -> dict:
    path = Path("config.yml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

CONFIG = load_config()

TZ_NAME = CONFIG.get("settings", {}).get("timezone", "Europe/Madrid")
TZ = ZoneInfo(TZ_NAME)

PRIMARY_TIMEFRAME = CONFIG.get("settings", {}).get("timeframe", "15m")
TIMEFRAME_CANDIDATES = CONFIG.get("settings", {}).get(
    "timeframe_candidates", [PRIMARY_TIMEFRAME, "1h"]
)

UPDATE_INTERVAL = int(CONFIG.get("settings", {}).get("update_interval", 30))
MAX_STATE_SIZE = int(CONFIG.get("settings", {}).get("max_state_size", 500))

def _log_root() -> Path:
    # Persistencia en Fly volume si existe
    if Path("/data").exists():
        return Path("/data/logs")
    return Path(CONFIG.get("settings", {}).get("log_dir", "logs"))

LOG_DIR = _log_root()
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Estrategia (más estricta)
MIN_SCORE = float(CONFIG.get("strategy", {}).get("min_score_for_entry", 92))

ADX_PERIOD = int(CONFIG.get("strategy", {}).get("adx_period", 14))
MIN_ADX = float(CONFIG.get("strategy", {}).get("min_adx", 28))

RSI_PERIOD = int(CONFIG.get("strategy", {}).get("rsi_period", 14))
RSI_LONG_MIN = float(CONFIG.get("strategy", {}).get("rsi_long_min", 55))
RSI_SHORT_MAX = float(CONFIG.get("strategy", {}).get("rsi_short_max", 45))

ATR_PERIOD = int(CONFIG.get("strategy", {}).get("atr_period", 14))
ATR_SL_MULT = float(CONFIG.get("strategy", {}).get("atr_sl_mult", 1.2))
ATR_TP_MULT = float(CONFIG.get("strategy", {}).get("atr_tp_mult", 2.0))
MAX_SL_PCT = float(CONFIG.get("strategy", {}).get("max_sl_pct", 0.02))
MAX_TP_PCT = float(CONFIG.get("strategy", {}).get("max_tp_pct", 0.05))

ATR_PCT_MIN = float(CONFIG.get("strategy", {}).get("atr_pct_min", 0.003))  # 0.3%
ATR_PCT_MAX = float(CONFIG.get("strategy", {}).get("atr_pct_max", 0.018))  # 1.8%

# Rango mínimo en lookback (evita laterales sin espacio)
RANGE_LOOKBACK = int(CONFIG.get("strategy", {}).get("range_lookback", 20))
MIN_RANGE_PCT = float(CONFIG.get("strategy", {}).get("min_range_pct", 0.01))  # 1%

# RR mínimo
MIN_RR = float(CONFIG.get("strategy", {}).get("min_rr", 1.5))

# Confirmación breakout entrada
BREAKOUT_CONFIRM = bool(CONFIG.get("strategy", {}).get("breakout_confirm", True))

ALIGNMENT_ENABLED = bool(CONFIG.get("strategy", {}).get("alignment_enabled", True))
ALIGNMENT_TFS = CONFIG.get("strategy", {}).get("alignment_timeframes", ["15m", "1h"])
ALIGNMENT_MIN_ABS_SCORE = float(CONFIG.get("strategy", {}).get("alignment_min_abs_score", 70))
ALIGNED_NEUTRAL = bool(CONFIG.get("strategy", {}).get("aligned_neutral", True))

SIGNAL_COOLDOWN_MIN = int(CONFIG.get("strategy", {}).get("signal_cooldown_min", 45))
MIN_MOVE_PCT = float(CONFIG.get("strategy", {}).get("min_move_pct", 0.001))  # 0.1%

LEARNING_MIN_TRADES = int(CONFIG.get("learning", {}).get("min_trades", 8))
LEARNING_MIN_EWMA_WR = float(CONFIG.get("learning", {}).get("min_ewma_wr", 0.45))

DAILY_SUMMARY_AT = CONFIG.get("settings", {}).get("daily_summary_at", "00:00")
TICKERS = CONFIG.get("markets", {}).get("base_tickers", ["XRP"])

# =======================
# SHUTDOWN LIMPIO (Fly)
# =======================
SHUTDOWN = False

def _handle_shutdown(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    print(f"[shutdown] Señal recibida: {signum}")

# =======================
# TELEGRAM
# =======================
def enviar_mensaje(texto: str, chart_path: Optional[str] = None) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] No configurado.")
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
    try:
        if chart_path and Path(chart_path).exists():
            with open(chart_path, "rb") as f:
                requests.post(
                    base_url + "sendPhoto",
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": texto, "parse_mode": "Markdown"},
                    files={"photo": f},
                    timeout=20,
                )
            try:
                os.remove(chart_path)
            except Exception:
                pass
        else:
            requests.post(
                base_url + "sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": texto, "parse_mode": "Markdown"},
                timeout=15,
            )
    except Exception as e:
        print(f"[telegram] Error: {e}")

# =======================
# GRÁFICOS
# =======================
def generar_grafico(symbol: str, timeframe: str, ohlcv: List[List[float]]) -> Optional[str]:
    try:
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)

        df["ema20"] = talib.EMA(df["close"], timeperiod=20)
        df["ema50"] = talib.EMA(df["close"], timeperiod=50)

        df_plot = df.tail(80)
        safe_symbol = symbol.split(":")[0].replace("/", "_")
        filename = f"chart_{safe_symbol}_{timeframe}.png"

        ap = [
            mpf.make_addplot(df_plot["ema20"]),
            mpf.make_addplot(df_plot["ema50"]),
        ]
        mpf.plot(df_plot, type="candle", addplot=ap, savefig=filename, style="charles", tight_layout=True)
        return filename
    except Exception as e:
        print(f"[chart] Error: {e}")
        return None

# =======================
# UTILIDADES
# =======================
def now_local() -> datetime:
    return datetime.now(TZ)

def local_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def day_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def log_path_for_day(dt: datetime) -> Path:
    return LOG_DIR / f"signals_{day_key(dt)}.jsonl"

def append_log(sig: "Signal") -> None:
    p = log_path_for_day(now_local())
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(sig), ensure_ascii=False) + "\n")

def load_day_signals(dt: datetime) -> List[dict]:
    p = log_path_for_day(dt)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)

def resolve_symbol(exchange, base: str) -> Optional[str]:
    for s in (f"{base}USDTM", f"{base}/USDT:USDT"):
        if s in exchange.markets:
            return s
    return None

def candle_ts(ohlcv: List[List[float]]) -> int:
    return int(ohlcv[-1][0])

def round_price(exchange, symbol: str, price: float) -> float:
    """Ajuste a la precisión del mercado (evita TP/SL iguales en memes)."""
    try:
        m = exchange.market(symbol)
        prec = (m.get("precision") or {}).get("price", None)
        if prec is None:
            return float(price)
        return float(round(float(price), int(prec)))
    except Exception:
        return float(price)

def clamp_levels(entry: float, sl: float, tp: float, side: str) -> Tuple[float, float]:
    """Caps por % para 10x."""
    entry = float(entry)
    if side == "LONG":
        sl_min = entry * (1 - MAX_SL_PCT)
        tp_max = entry * (1 + MAX_TP_PCT)
        sl = max(sl, sl_min)
        tp = min(tp, tp_max)
    else:
        sl_max = entry * (1 + MAX_SL_PCT)
        tp_min = entry * (1 - MAX_TP_PCT)
        sl = min(sl, sl_max)
        tp = max(tp, tp_min)
    return float(sl), float(tp)

# =======================
# MODELO DE SEÑALES
# =======================
@dataclass
class Signal:
    symbol: str
    base: str
    side: str         # LONG/SHORT
    timeframe: str
    entry: float
    tp: float
    sl: float
    score: float
    reasons: List[str]
    opened_ts: int
    opened_at_local: str
    status: str = "OPEN"  # OPEN / TP / SL / CANCELLED
    closed_ts: Optional[int] = None
    closed_at_local: Optional[str] = None
    notes: Optional[str] = None

def detectar_signal_alta_precision(ohlcv: List[List[float]]) -> dict:
    """
    Señal estricta + más asertiva:
    - ATR% en rango (evita chop o locura)
    - ADX mínimo
    - EMA20/EMA50 tendencia + cruce suma
    - RSI confirmación dura (si no acompaña, NO TRADE)
    - Stoch timing suma
    - Filtro rango 20 velas
    - Confirmación breakout entrada (prev_high/prev_low)
    - RR mínimo (>= MIN_RR)
    """
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
    if len(df) < 80:
        return {"signal": "NO_TRADE", "score": 0, "reasons": [], "entry": None, "sl": None, "tp": None}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    df["ema20"] = talib.EMA(close, timeperiod=20)
    df["ema50"] = talib.EMA(close, timeperiod=50)
    df["adx"] = talib.ADX(high, low, close, timeperiod=ADX_PERIOD)
    df["rsi"] = talib.RSI(close, timeperiod=RSI_PERIOD)
    df["atr"] = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)
    df["k"], df["d"] = talib.STOCH(high, low, close, 9, 3, 3)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    entry_market = float(last["close"])
    adx = float(last["adx"])
    rsi = float(last["rsi"])
    atr = float(last["atr"])
    k = float(last["k"])
    d = float(last["d"])

    reasons: List[str] = []
    score = 0.0

    # ATR% filtro
    if atr <= 0 or np.isnan(atr):
        return {"signal": "NO_TRADE", "score": 0, "reasons": ["ATR inválido"], "entry": entry_market, "sl": None, "tp": None}

    atr_pct = atr / entry_market
    if atr_pct < ATR_PCT_MIN or atr_pct > ATR_PCT_MAX:
        return {
            "signal": "NO_TRADE",
            "score": 0,
            "reasons": [f"ATR% fuera de rango ({atr_pct*100:.2f}%)"],
            "entry": entry_market,
            "sl": None,
            "tp": None,
        }
    reasons.append(f"ATR% OK ({atr_pct*100:.2f}%)")
    score += 10

    # ADX mínimo
    if np.isnan(adx) or adx < MIN_ADX:
        return {
            "signal": "NO_TRADE",
            "score": 0,
            "reasons": [f"ADX bajo ({adx:.1f} < {MIN_ADX})"],
            "entry": entry_market,
            "sl": None,
            "tp": None,
        }
    reasons.append(f"ADX OK ({adx:.1f})")
    score += 25

    # Filtro rango (evita estrecho/lateral)
    lookback = min(RANGE_LOOKBACK, len(df))
    hh = float(df["high"].tail(lookback).max())
    ll = float(df["low"].tail(lookback).min())
    range_pct = (hh - ll) / entry_market
    if range_pct < MIN_RANGE_PCT:
        return {
            "signal": "NO_TRADE",
            "score": 0,
            "reasons": [f"Rango bajo {lookback} velas ({range_pct*100:.2f}% < {MIN_RANGE_PCT*100:.2f}%)"],
            "entry": entry_market,
            "sl": None,
            "tp": None,
        }
    reasons.append(f"Rango {lookback} velas OK ({range_pct*100:.2f}%)")
    score += 10

    # EMA tendencia
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    prev_ema20 = float(prev["ema20"])
    prev_ema50 = float(prev["ema50"])

    bullish = ema20 > ema50
    bearish = ema20 < ema50

    if bullish:
        score += 35
        reasons.append("Tendencia alcista EMA20>EMA50")
        if prev_ema20 <= prev_ema50:
            score += 10
            reasons.append("Cruce alcista reciente")
    elif bearish:
        score -= 35
        reasons.append("Tendencia bajista EMA20<EMA50")
        if prev_ema20 >= prev_ema50:
            score -= 10
            reasons.append("Cruce bajista reciente")

    # RSI confirmación dura
    if bullish and rsi >= RSI_LONG_MIN:
        score += 20
        reasons.append(f"RSI confirma LONG ({rsi:.1f})")
    elif bearish and rsi <= RSI_SHORT_MAX:
        score -= 20
        reasons.append(f"RSI confirma SHORT ({rsi:.1f})")
    else:
        return {
            "signal": "NO_TRADE",
            "score": 0,
            "reasons": reasons + [f"RSI no acompaña ({rsi:.1f})"],
            "entry": entry_market,
            "sl": None,
            "tp": None,
        }

    # Stoch timing (solo suma)
    prev_k = float(prev["k"])
    prev_d = float(prev["d"])
    if bullish and (k > d) and (prev_k <= prev_d) and (k < 35):
        score += 10
        reasons.append("Timing Stoch alcista")
    elif bearish and (k < d) and (prev_k >= prev_d) and (k > 65):
        score -= 10
        reasons.append("Timing Stoch bajista")
    else:
        reasons.append("Stoch sin timing claro")

    # Decide señal por score
    signal_out = "NO_TRADE"
    if score >= MIN_SCORE:
        signal_out = "LONG"
    elif score <= -MIN_SCORE:
        signal_out = "SHORT"
    else:
        return {"signal": "NO_TRADE", "score": float(score), "reasons": reasons, "entry": entry_market, "sl": None, "tp": None}

    # Confirmación breakout (entrada recomendada)
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])

    entry = entry_market
    if BREAKOUT_CONFIRM:
        if signal_out == "LONG":
            if entry_market <= prev_high:
                return {
                    "signal": "NO_TRADE",
                    "score": 0,
                    "reasons": reasons + ["Falta confirmación: no rompe high previo"],
                    "entry": entry_market,
                    "sl": None,
                    "tp": None,
                }
            entry = prev_high
            reasons.append("Entrada confirmada: breakout high previo")
        elif signal_out == "SHORT":
            if entry_market >= prev_low:
                return {
                    "signal": "NO_TRADE",
                    "score": 0,
                    "reasons": reasons + ["Falta confirmación: no rompe low previo"],
                    "entry": entry_market,
                    "sl": None,
                    "tp": None,
                }
            entry = prev_low
            reasons.append("Entrada confirmada: breakout low previo")

    # SL/TP por ATR usando ENTRY confirmada
    if signal_out == "LONG":
        sl = entry - ATR_SL_MULT * atr
        tp = entry + ATR_TP_MULT * atr
    else:
        sl = entry + ATR_SL_MULT * atr
        tp = entry - ATR_TP_MULT * atr

    # RR mínimo
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0 or reward <= 0:
        return {"signal": "NO_TRADE", "score": 0, "reasons": reasons + ["Riesgo/reward inválido"], "entry": entry, "sl": None, "tp": None}

    rr = reward / risk
    if rr < MIN_RR:
        return {
            "signal": "NO_TRADE",
            "score": 0,
            "reasons": reasons + [f"RR bajo ({rr:.2f} < {MIN_RR})"],
            "entry": entry,
            "sl": None,
            "tp": None,
        }
    reasons.append(f"RR OK ({rr:.2f})")

    return {"signal": signal_out, "score": float(score), "reasons": reasons, "entry": float(entry), "sl": float(sl), "tp": float(tp)}

# =======================
# MULTI-TF ALIGNMENT
# =======================
def evaluate_tf(exchange, symbol: str, tf: str) -> Tuple[dict, List[List[float]]]:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=140)
    if not ohlcv or len(ohlcv) < 80:
        return {"signal": "NO_TRADE", "score": 0, "reasons": [f"Datos insuficientes {tf}"], "entry": None, "sl": None, "tp": None}, []
    res = detectar_signal_alta_precision(ohlcv)
    return res, ohlcv

def aligned_direction(res_a: dict, res_b: dict) -> bool:
    sa = res_a.get("signal")
    sb = res_b.get("signal")

    # La primaria debe tener señal
    if sa not in ("LONG", "SHORT"):
        return False

    # Si el marco mayor no tiene señal, lo tratamos como neutral (no bloquea)
    if sb == "NO_TRADE":
        return True

    # Si el marco mayor sí tiene señal, debe coincidir
    if sb in ("LONG", "SHORT"):
        return sa == sb

    return False

def choose_best_timeframe(exchange, symbol: str) -> Tuple[str, dict, List[List[float]]]:
    cached: Dict[str, Tuple[dict, List[List[float]]]] = {}

    for tf in set(TIMEFRAME_CANDIDATES + ALIGNMENT_TFS):
        try:
            cached[tf] = evaluate_tf(exchange, symbol, tf)
        except Exception as e:
            print(f"[tf] error {symbol} {tf}: {e}")

    if ALIGNMENT_ENABLED:
        tf_a, tf_b = ALIGNMENT_TFS[0], ALIGNMENT_TFS[1]
        ra, _ = cached.get(tf_a, ({"signal": "NO_TRADE", "score": 0}, []))
        rb, _ = cached.get(tf_b, ({"signal": "NO_TRADE", "score": 0}, []))

        if not aligned_direction(ra, rb):
            return PRIMARY_TIMEFRAME, {"signal": "NO_TRADE", "score": 0, "reasons": ["No alineación 15m↔1h"]}, []

        if abs(float(ra.get("score", 0))) < ALIGNMENT_MIN_ABS_SCORE or abs(float(rb.get("score", 0))) < ALIGNMENT_MIN_ABS_SCORE:
            return PRIMARY_TIMEFRAME, {"signal": "NO_TRADE", "score": 0, "reasons": [f"Alineación débil (<{ALIGNMENT_MIN_ABS_SCORE})"]}, []

    best = None  # (abs_score, tf, res, ohlcv)
    for tf in TIMEFRAME_CANDIDATES:
        res, ohlcv = cached.get(tf, ({"signal": "NO_TRADE", "score": 0}, []))
        sig = res.get("signal", "NO_TRADE")
        score = float(res.get("score", 0))
        if sig == "NO_TRADE":
            continue

        cand = (abs(score), tf, res, ohlcv)
        if best is None:
            best = cand
        else:
            if cand[0] > best[0] + 1e-9:
                best = cand
            elif abs(cand[0] - best[0]) <= 1e-9 and best[1] != PRIMARY_TIMEFRAME and tf == PRIMARY_TIMEFRAME:
                best = cand

    if best is None:
        return PRIMARY_TIMEFRAME, {"signal": "NO_TRADE", "score": 0, "reasons": ["NO_TRADE"]}, []
    return best[1], best[2], best[3]

# =======================
# MENSAJES
# =======================
def format_signal_message(sig: Signal) -> str:
    emoji = "🟢" if sig.side == "LONG" else "🔴"
    return (
        f"{emoji} **SEÑAL (ALTA CONVICCIÓN): {sig.side}**\n\n"
        f"🪙 **Activo:** `{sig.symbol}`\n"
        f"⏱️ **Timeframe:** `{sig.timeframe}`\n"
        f"💰 **Entrada (recomendada):** `{sig.entry:.10f}`\n"
        f"🎯 **TP:** `{sig.tp:.10f}`\n"
        f"🛑 **SL:** `{sig.sl:.10f}`\n\n"
        f"📊 **Score:** `{sig.score:.1f}` (min `{MIN_SCORE}`)\n"
        f"📝 **Motivos:** {', '.join(sig.reasons)}\n"
        f"🕒 **Hora local:** `{sig.opened_at_local}`"
    )

def format_cancel_message(symbol: str, reason: str) -> str:
    return (
        f"🟠 **CANCELAR ORDEN**\n\n"
        f"🪙 **Activo:** `{symbol}`\n"
        f"🧠 **Motivo:** {reason}\n"
        f"🕒 **Hora local:** `{local_str(now_local())}`"
    )

def format_close_message(sig: Signal) -> str:
    emoji = "✅" if sig.status == "TP" else "🛑" if sig.status == "SL" else "⚪"
    return (
        f"{emoji} **CIERRE: {sig.status}**\n\n"
        f"🪙 `{sig.symbol}` | `{sig.side}` | `{sig.timeframe}`\n"
        f"Entrada `{sig.entry:.10f}` | TP `{sig.tp:.10f}` | SL `{sig.sl:.10f}`\n"
        f"Apertura `{sig.opened_at_local}`\n"
        f"Cierre `{sig.closed_at_local}`\n"
        f"Notas: {sig.notes or '-'}"
    )

def send_daily_summary(for_day: date) -> None:
    dt = datetime(for_day.year, for_day.month, for_day.day, 12, 0, tzinfo=TZ)
    items = load_day_signals(dt)
    if not items:
        enviar_mensaje(f"📌 **Resumen diario {for_day.isoformat()}**\n\nSin señales registradas.")
        return

    total = len(items)
    tp = sum(1 for x in items if x.get("status") == "TP")
    sl = sum(1 for x in items if x.get("status") == "SL")
    canc = sum(1 for x in items if x.get("status") == "CANCELLED")
    open_ = sum(1 for x in items if x.get("status") == "OPEN")

    lines = []
    for x in items[-25:]:
        lines.append(
            f"- `{x.get('symbol')}` {x.get('side')} `{x.get('timeframe')}` → **{x.get('status')}** | "
            f"Entry `{float(x.get('entry',0)):.6f}` TP `{float(x.get('tp',0)):.6f}` SL `{float(x.get('sl',0)):.6f}`"
        )

    msg = (
        f"📌 **Resumen diario {for_day.isoformat()}**\n\n"
        f"Total: **{total}** | ✅ TP: **{tp}** | 🛑 SL: **{sl}** | 🟠 Canceladas: **{canc}** | ⏳ Abiertas: **{open_}**\n\n"
        f"📋 Últimas {min(total, 25)}:\n" + "\n".join(lines)
    )
    enviar_mensaje(msg)

# =======================
# TRACKING TP/SL (más realista)
# =======================
def check_signal_outcome(exchange, sig: Signal) -> Optional[str]:
    """
    - TP: por toque (high/low)
    - SL: por cierre (close) -> reduce falsos SL por mecha
    """
    try:
        ohlcv = exchange.fetch_ohlcv(sig.symbol, timeframe=sig.timeframe, limit=3)
        if not ohlcv:
            return None
        last = ohlcv[-1]
        high = float(last[2])
        low = float(last[3])
        close = float(last[4])

        if sig.side == "LONG":
            if high >= sig.tp:
                return "TP"
            if close <= sig.sl:
                return "SL"
        else:
            if low <= sig.tp:
                return "TP"
            if close >= sig.sl:
                return "SL"
        return None
    except Exception as e:
        print(f"[outcome] Error {sig.symbol}: {e}")
        return None

# =======================
# LOOP PRINCIPAL
# =======================
def main_loop() -> None:
    print("[bot] Iniciado (Alta Precisión + Autoaprendizaje + Filtros extra)")

    exchange = ccxt.kucoinfutures({
        "apiKey": KUCOIN_API_KEY,
        "secret": KUCOIN_API_SECRET,
        "password": KUCOIN_API_PASSPHRASE,
        "enableRateLimit": True,
    })
    exchange.load_markets()

    last_signal_time_by_base: Dict[str, datetime] = {}
    active_by_base: Dict[str, Signal] = {}
    sent_ids: Dict[str, bool] = {}
    last_no_trade_log: Dict[str, float] = {}

    while not SHUTDOWN:
        # heartbeat cada ~60s (para ver que el loop está vivo)
        if int(time.time()) % 60 == 0:
            print(f"[heartbeat] {local_str(now_local())} alive")
        # 1) Revisar abiertas
        for base, sig in list(active_by_base.items()):
            if SHUTDOWN:
                break
            if sig.status != "OPEN":
                continue

            outcome = check_signal_outcome(exchange, sig)
            if outcome in ("TP", "SL"):
                sig.status = outcome
                sig.closed_ts = int(time.time() * 1000)
                sig.closed_at_local = local_str(now_local())
                sig.notes = "TP por toque; SL por cierre (close)."
                append_log(sig)

                # Autoaprendizaje
                update_result(sig.base, sig.timeframe, sig.side, outcome)

                enviar_mensaje(format_close_message(sig))
                del active_by_base[base]

        if SHUTDOWN:
            break

        # 2) Generar nuevas señales (selectivo)
        for base in TICKERS:
            if SHUTDOWN:
                break

            time.sleep(1.2)

            symbol = resolve_symbol(exchange, base)
            if not symbol:
                continue

            # cooldown
            last_t = last_signal_time_by_base.get(base)
            if last_t and (now_local() - last_t) < timedelta(minutes=SIGNAL_COOLDOWN_MIN):
                continue

            tf, res, ohlcv = choose_best_timeframe(exchange, symbol)
            sig_type = res.get("signal", "NO_TRADE")
            if sig_type == "NO_TRADE" or not ohlcv:
                if int(time.time()) % 300 == 0:  # cada ~5 min
                    print(f"[no_trade] {base} tf={tf} reasons={(res.get('reasons') or [])[:2]}")
                continue


            score = float(res.get("score", 0))
            entry = float(res["entry"])
            sl = float(res["sl"])
            tp = float(res["tp"])
            reasons = list(res.get("reasons", []))

            # Aprendizaje: bloquear combos malos
            ok, why = should_trade(
                base, tf, sig_type,
                min_trades=LEARNING_MIN_TRADES,
                min_ewma_wr=LEARNING_MIN_EWMA_WR
            )
            if not ok:
                print(f"[learning] bloqueado {base} {tf} {sig_type}: {why}")
                continue

            # Caps + precision
            sl, tp = clamp_levels(entry, sl, tp, sig_type)

            entry = round_price(exchange, symbol, entry)
            sl = round_price(exchange, symbol, sl)
            tp = round_price(exchange, symbol, tp)
            # --- GUARD: niveles inválidos (evita ZeroDivision y señales rotas) ---
            if not np.isfinite(entry) or not np.isfinite(sl) or not np.isfinite(tp):
                print(f"[guard] {symbol} niveles no finitos: entry={entry} sl={sl} tp={tp}")
                continue

            # Algunos mercados/memes pueden devolver 0 o redondear a 0 con precision rara
            if entry <= 0:
                print(f"[guard] {symbol} entry inválida (<=0): entry={entry} (raw levels tp={tp}, sl={sl})")
                continue

            # Evita niveles idénticos (puede ocurrir por redondeo)
            if tp == entry or sl == entry or tp == sl:
                print(f"[guard] {symbol} niveles degenerados: entry={entry} sl={sl} tp={tp}")
                continue

            # Evitar niveles absurdamente cerca
            move_tp = abs(tp - entry) / max(entry, 1e-12)
            move_sl = abs(entry - sl) / max(entry, 1e-12)
            if move_tp < MIN_MOVE_PCT or move_sl < MIN_MOVE_PCT:
                print(f"[levels] {base} {tf}: movimientos muy pequeños TP={move_tp:.6f} SL={move_sl:.6f}. NO_TRADE")
                continue


            last_ts = candle_ts(ohlcv)
            unique_id = f"{symbol}|{tf}|{last_ts}|{sig_type}"
            if unique_id in sent_ids:
                continue

            # Regla: no 2 órdenes seguidas misma moneda salvo CANCEL por cambio de sesgo
            if base in active_by_base and active_by_base[base].status == "OPEN":
                prev_sig = active_by_base[base]
                if prev_sig.side != sig_type:
                    enviar_mensaje(format_cancel_message(symbol, f"Cambio de sesgo: {prev_sig.side} → {sig_type}. Cancela la anterior."))
                    prev_sig.status = "CANCELLED"
                    prev_sig.closed_ts = int(time.time() * 1000)
                    prev_sig.closed_at_local = local_str(now_local())
                    prev_sig.notes = "Cancelada por señal contraria posterior."
                    append_log(prev_sig)
                    enviar_mensaje(format_close_message(prev_sig))
                    del active_by_base[base]
                else:
                    continue

            opened_dt = now_local()
            sig = Signal(
                symbol=symbol,
                base=base,
                side=sig_type,
                timeframe=tf,
                entry=entry,
                tp=tp,
                sl=sl,
                score=score,
                reasons=reasons + [f"Learning: {why}"],
                opened_ts=last_ts,
                opened_at_local=local_str(opened_dt),
            )

            chart = generar_grafico(symbol, tf, ohlcv)
            enviar_mensaje(format_signal_message(sig), chart_path=chart)

            active_by_base[base] = sig
            last_signal_time_by_base[base] = opened_dt
            sent_ids[unique_id] = True
            append_log(sig)

            if len(sent_ids) > MAX_STATE_SIZE:
                sent_ids.clear()

        # sleep interrumpible
        for _ in range(UPDATE_INTERVAL):
            if SHUTDOWN:
                break
            time.sleep(1)

# =======================
# RESUMEN DIARIO
# =======================
def daily_summary_worker() -> None:
    hh, mm = parse_hhmm(DAILY_SUMMARY_AT)
    last_sent_day: Optional[date] = None

    while not SHUTDOWN:
        dt = now_local()
        if dt.hour == hh and dt.minute == mm:
            target_day = (dt - timedelta(days=1)).date() if (hh == 0 and mm == 0) else dt.date()
            if last_sent_day != target_day:
                send_daily_summary(target_day)
                last_sent_day = target_day
        time.sleep(30)

# =======================
# FLASK (healthcheck)
# =======================
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot Online", 200

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()
    threading.Thread(target=daily_summary_worker, daemon=True).start()
    main_loop()
