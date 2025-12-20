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
LOG_DIR = Path(CONFIG.get("settings", {}).get("log_dir", "logs"))
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

ALIGNMENT_ENABLED = bool(CONFIG.get("strategy", {}).get("alignment_enabled", True))
ALIGNMENT_TFS = CONFIG.get("strategy", {}).get("alignment_timeframes", ["15m", "1h"])
ALIGNMENT_MIN_ABS_SCORE = float(CONFIG.get("strategy", {}).get("alignment_min_abs_score", 70))

SIGNAL_COOLDOWN_MIN = int(CONFIG.get("strategy", {}).get("signal_cooldown_min", 90))
MIN_MOVE_PCT = float(CONFIG.get("strategy", {}).get("min_move_pct", 0.001))  # 0.1% mínimo entry->tp/sl

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
    print(f"?? Señal de apagado recibida ({signum}). Cerrando limpio…")

# =======================
# TELEGRAM
# =======================
def enviar_mensaje(texto: str, chart_path: Optional[str] = None) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("?? Telegram no configurado (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
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
        print(f"Error Telegram: {e}")

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
        print(f"Error gráfico: {e}")
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
    """
    Ajuste a la precisión del mercado (evita TP/SL iguales en memes con muchos decimales).
    """
    try:
        m = exchange.market(symbol)
        prec = (m.get("precision") or {}).get("price", None)
        if prec is None:
            return float(price)
        return float(round(float(price), int(prec)))
    except Exception:
        return float(price)

def clamp_levels(entry: float, sl: float, tp: float, side: str) -> Tuple[float, float]:
    """
    Caps por % para 10x.
    """
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
    Señal estricta:
    - ADX mínimo (fuerza tendencia)
    - EMA20/EMA50 (tendencia)
    - RSI confirmación
    - Stoch para timing (suma)
    - ATR% filtro de volatilidad
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

    entry = float(last["close"])
    adx = float(last["adx"])
    rsi = float(last["rsi"])
    atr = float(last["atr"])
    k = float(last["k"])
    d = float(last["d"])

    reasons: List[str] = []
    score = 0.0

    # ATR% filtro (evita chop / locura)
    if atr <= 0 or np.isnan(atr):
        return {"signal": "NO_TRADE", "score": 0, "reasons": ["ATR inválido"], "entry": entry, "sl": None, "tp": None}

    atr_pct = atr / entry
    if atr_pct < ATR_PCT_MIN or atr_pct > ATR_PCT_MAX:
        return {
            "signal": "NO_TRADE",
            "score": 0,
            "reasons": [f"? ATR% fuera de rango ({atr_pct*100:.2f}%, rango {ATR_PCT_MIN*100:.2f}-{ATR_PCT_MAX*100:.2f}%)"],
            "entry": entry,
            "sl": None,
            "tp": None,
        }
    reasons.append(f"??? ATR% OK ({atr_pct*100:.2f}%)")
    score += 10

    # ADX mínimo
    if np.isnan(adx) or adx < MIN_ADX:
        return {
            "signal": "NO_TRADE",
            "score": 0,
            "reasons": [f"? ADX bajo ({adx:.1f} < {MIN_ADX})"],
            "entry": entry,
            "sl": None,
            "tp": None,
        }
    reasons.append(f"?? ADX OK ({adx:.1f})")
    score += 25

    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    prev_ema20 = float(prev["ema20"])
    prev_ema50 = float(prev["ema50"])

    bullish = ema20 > ema50
    bearish = ema20 < ema50

    # Tendencia + cruce
    if bullish:
        score += 35
        reasons.append("?? Tendencia alcista (EMA20>EMA50)")
        if prev_ema20 <= prev_ema50:
            score += 10
            reasons.append("?? Cruce alcista reciente")
    elif bearish:
        score -= 35
        reasons.append("?? Tendencia bajista (EMA20<EMA50)")
        if prev_ema20 >= prev_ema50:
            score -= 10
            reasons.append("?? Cruce bajista reciente")

    # RSI confirmación (más duro)
    if bullish and rsi >= RSI_LONG_MIN:
        score += 20
        reasons.append(f"? RSI confirma long ({rsi:.1f} = {RSI_LONG_MIN})")
    elif bearish and rsi <= RSI_SHORT_MAX:
        score -= 20
        reasons.append(f"? RSI confirma short ({rsi:.1f} = {RSI_SHORT_MAX})")
    else:
        # si RSI no acompaña, prácticamente descartamos
        reasons.append(f"?? RSI no acompaña ({rsi:.1f})")
        return {"signal": "NO_TRADE", "score": 0, "reasons": reasons, "entry": entry, "sl": None, "tp": None}

    # Timing Stoch (solo suma, no obliga)
    prev_k = float(prev["k"])
    prev_d = float(prev["d"])
    if bullish and (k > d) and (prev_k <= prev_d) and (k < 35):
        score += 10
        reasons.append("?? Timing alcista (Stoch cruce en zona baja)")
    elif bearish and (k < d) and (prev_k >= prev_d) and (k > 65):
        score -= 10
        reasons.append("?? Timing bajista (Stoch cruce en zona alta)")
    else:
        reasons.append("? Stoch sin timing claro")

    # SL/TP por ATR + caps
    sl_long = entry - ATR_SL_MULT * atr
    tp_long = entry + ATR_TP_MULT * atr
    sl_short = entry + ATR_SL_MULT * atr
    tp_short = entry - ATR_TP_MULT * atr

    signal = "NO_TRADE"
    sl = None
    tp = None

    if score >= MIN_SCORE:
        signal = "LONG"
        sl = float(sl_long)
        tp = float(tp_long)
    elif score <= -MIN_SCORE:
        signal = "SHORT"
        sl = float(sl_short)
        tp = float(tp_short)

    return {"signal": signal, "score": float(score), "reasons": reasons, "entry": entry, "sl": sl, "tp": tp}

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
    if sa in ("LONG", "SHORT") and sb in ("LONG", "SHORT"):
        return sa == sb
    return False

def choose_best_timeframe(exchange, symbol: str) -> Tuple[str, dict, List[List[float]]]:
    """
    Evalúa timeframes candidatos y elige el de mayor |score|.
    Si alignment está activado:
      - exige que 15m y 1h estén alineados (mismo LONG/SHORT)
      - y que ambos tengan |score| >= ALIGNMENT_MIN_ABS_SCORE
    """
    cached: Dict[str, Tuple[dict, List[List[float]]]] = {}

    # precache alignment tfs
    for tf in set(TIMEFRAME_CANDIDATES + ALIGNMENT_TFS):
        try:
            cached[tf] = evaluate_tf(exchange, symbol, tf)
        except Exception as e:
            print(f"TF error {symbol} {tf}: {e}")

    if ALIGNMENT_ENABLED:
        tf_a, tf_b = ALIGNMENT_TFS[0], ALIGNMENT_TFS[1]
        ra, _ = cached.get(tf_a, ({"signal": "NO_TRADE", "score": 0}, []))
        rb, _ = cached.get(tf_b, ({"signal": "NO_TRADE", "score": 0}, []))

        if not aligned_direction(ra, rb):
            return PRIMARY_TIMEFRAME, {"signal": "NO_TRADE", "score": 0, "reasons": ["?? No alineación 15m?1h"]}, []

        if abs(float(ra.get("score", 0))) < ALIGNMENT_MIN_ABS_SCORE or abs(float(rb.get("score", 0))) < ALIGNMENT_MIN_ABS_SCORE:
            return PRIMARY_TIMEFRAME, {"signal": "NO_TRADE", "score": 0, "reasons": [f"?? Alineación débil (<{ALIGNMENT_MIN_ABS_SCORE})"]}, []

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
    emoji = "??" if sig.side == "LONG" else "??"
    return (
        f"{emoji} **ENTRADA MUY ASEGURADA: {sig.side}**\n\n"
        f"?? **Activo:** `{sig.symbol}`\n"
        f"?? **Timeframe:** `{sig.timeframe}` (prioridad 15m)\n"
        f"?? **Entrada (recomendada):** `{sig.entry:.10f}`\n"
        f"?? **Salida (TP recomendada):** `{sig.tp:.10f}`\n"
        f"?? **Stop (SL):** `{sig.sl:.10f}`\n\n"
        f"?? **Score:** `{sig.score:.1f}` (min `{MIN_SCORE}`)\n"
        f"?? **Motivos:** {', '.join(sig.reasons)}\n"
        f"?? **Hora local:** `{sig.opened_at_local}`"
    )

def format_cancel_message(symbol: str, reason: str) -> str:
    return (
        f"?? **CANCELAR ORDEN**\n\n"
        f"?? **Activo:** `{symbol}`\n"
        f"?? **Motivo:** {reason}\n"
        f"?? **Hora local:** `{local_str(now_local())}`"
    )

def format_close_message(sig: Signal) -> str:
    emoji = "?" if sig.status == "TP" else "??" if sig.status == "SL" else "?"
    return (
        f"{emoji} **CIERRE SEÑAL: {sig.status}**\n\n"
        f"?? **Activo:** `{sig.symbol}`\n"
        f"?? **Lado:** `{sig.side}` | ?? `{sig.timeframe}`\n"
        f"?? **Entrada:** `{sig.entry:.10f}`\n"
        f"?? **TP:** `{sig.tp:.10f}` | ?? **SL:** `{sig.sl:.10f}`\n"
        f"?? **Apertura:** `{sig.opened_at_local}`\n"
        f"?? **Cierre:** `{sig.closed_at_local}`\n"
        f"?? **Notas:** {sig.notes or '-'}"
    )

def send_daily_summary(for_day: date) -> None:
    dt = datetime(for_day.year, for_day.month, for_day.day, 12, 0, tzinfo=TZ)
    items = load_day_signals(dt)
    if not items:
        enviar_mensaje(f"?? **Resumen diario {for_day.isoformat()}**\n\nSin señales registradas.")
        return

    total = len(items)
    tp = sum(1 for x in items if x.get("status") == "TP")
    sl = sum(1 for x in items if x.get("status") == "SL")
    canc = sum(1 for x in items if x.get("status") == "CANCELLED")
    open_ = sum(1 for x in items if x.get("status") == "OPEN")

    lines = []
    for x in items[-25:]:
        lines.append(
            f"- `{x.get('symbol')}` {x.get('side')} `{x.get('timeframe')}` ? **{x.get('status')}** | "
            f"Entry `{float(x.get('entry',0)):.6f}` TP `{float(x.get('tp',0)):.6f}` SL `{float(x.get('sl',0)):.6f}`"
        )

    msg = (
        f"?? **Resumen diario {for_day.isoformat()}**\n\n"
        f"Total: **{total}** | ? TP: **{tp}** | ?? SL: **{sl}** | ?? Canceladas: **{canc}** | ? Abiertas: **{open_}**\n\n"
        f"?? **Últimas {min(total, 25)} señales:**\n" + "\n".join(lines)
    )
    enviar_mensaje(msg)

# =======================
# TRACKING TP/SL (verificación)
# =======================
def check_signal_outcome(exchange, sig: Signal) -> Optional[str]:
    try:
        ohlcv = exchange.fetch_ohlcv(sig.symbol, timeframe=sig.timeframe, limit=3)
        if not ohlcv:
            return None
        last = ohlcv[-1]
        high = float(last[2])
        low = float(last[3])

        if sig.side == "LONG":
            if high >= sig.tp:
                return "TP"
            if low <= sig.sl:
                return "SL"
        else:
            if low <= sig.tp:
                return "TP"
            if high >= sig.sl:
                return "SL"
        return None
    except Exception as e:
        print(f"Outcome check error {sig.symbol}: {e}")
        return None

# =======================
# LOOP PRINCIPAL
# =======================
def main_loop() -> None:
    print("?? Bot (Alta Precisión + Autoaprendizaje) iniciado…")

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

    while not SHUTDOWN:
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
                sig.notes = "Verificado por toque de vela (high/low) en el timeframe de la señal."
                append_log(sig)

                # Autoaprendizaje
                update_result(sig.base, sig.timeframe, sig.side, outcome)

                enviar_mensaje(format_close_message(sig))
                del active_by_base[base]

        if SHUTDOWN:
            break

        # 2) Generar nuevas señales (muy selectivo)
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
                continue

            score = float(res.get("score", 0))
            entry = float(res["entry"])
            sl = float(res["sl"])
            tp = float(res["tp"])
            reasons = list(res.get("reasons", []))

            # Autoaprendizaje: bloquear si ese combo rinde mal
            ok, why = should_trade(
                base, tf, sig_type,
                min_trades=LEARNING_MIN_TRADES,
                min_ewma_wr=LEARNING_MIN_EWMA_WR
            )
            if not ok:
                print(f"?? {base} {tf} {sig_type}: {why}")
                continue

            # Caps SL/TP y redondeo a tick/precision
            sl, tp = clamp_levels(entry, sl, tp, sig_type)
            entry = round_price(exchange, symbol, entry)
            sl = round_price(exchange, symbol, sl)
            tp = round_price(exchange, symbol, tp)

            # Evitar señales “absurdas” (tp/sl demasiado cerca por precision)
            if abs(tp - entry) / entry < MIN_MOVE_PCT or abs(entry - sl) / entry < MIN_MOVE_PCT:
                print(f"?? {base} {tf}: niveles demasiado cerca (precision/tick). NO_TRADE")
                continue

            last_ts = candle_ts(ohlcv)
            unique_id = f"{symbol}|{tf}|{last_ts}|{sig_type}"
            if unique_id in sent_ids:
                continue

            # Regla: no 2 seguidas misma moneda salvo CANCEL por cambio de sesgo
            if base in active_by_base and active_by_base[base].status == "OPEN":
                prev_sig = active_by_base[base]
                if prev_sig.side != sig_type:
                    enviar_mensaje(format_cancel_message(symbol, f"Cambio de sesgo: {prev_sig.side} ? {sig_type}. Cancela la anterior."))
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
                reasons=reasons + [f"?? Learning: {why}"],
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
