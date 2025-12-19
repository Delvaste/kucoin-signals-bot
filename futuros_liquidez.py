import os
import time
import json
import threading
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
    "timeframe_candidates",
    [PRIMARY_TIMEFRAME, "1h", "5m"]
)

UPDATE_INTERVAL = int(CONFIG.get("settings", {}).get("update_interval", 30))
MAX_STATE_SIZE = int(CONFIG.get("settings", {}).get("max_state_size", 300))

# “Solo señales muy asertivas”
MIN_SCORE = float(CONFIG.get("strategy", {}).get("min_score_for_entry", 85))

# Gestión riesgo por ATR (más lógico que % fijo con 10x)
ATR_PERIOD = int(CONFIG.get("strategy", {}).get("atr_period", 14))
ATR_SL_MULT = float(CONFIG.get("strategy", {}).get("atr_sl_mult", 1.2))
ATR_TP_MULT = float(CONFIG.get("strategy", {}).get("atr_tp_mult", 2.0))
MAX_SL_PCT = float(CONFIG.get("strategy", {}).get("max_sl_pct", 0.02))  # cap SL a 2% (por 10x)
MAX_TP_PCT = float(CONFIG.get("strategy", {}).get("max_tp_pct", 0.05))  # cap TP a 5%

# Filtros anti-mercado “chop”
ADX_PERIOD = int(CONFIG.get("strategy", {}).get("adx_period", 14))
MIN_ADX = float(CONFIG.get("strategy", {}).get("min_adx", 20))          # fuerza tendencia
RSI_PERIOD = int(CONFIG.get("strategy", {}).get("rsi_period", 14))
RSI_LONG_MIN = float(CONFIG.get("strategy", {}).get("rsi_long_min", 52))
RSI_SHORT_MAX = float(CONFIG.get("strategy", {}).get("rsi_short_max", 48))

# Evitar 2 señales seguidas de la misma moneda salvo CANCEL
SIGNAL_COOLDOWN_MIN = int(CONFIG.get("strategy", {}).get("signal_cooldown_min", 30))

# Resumen diario
DAILY_SUMMARY_AT = CONFIG.get("settings", {}).get("daily_summary_at", "00:00")  # HH:MM

# Markets
TICKERS = CONFIG.get("markets", {}).get("base_tickers", ["XRP"])

LOG_DIR = Path(CONFIG.get("settings", {}).get("log_dir", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# TELEGRAM
# =======================
def enviar_mensaje(texto: str, chart_path: Optional[str] = None) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram no configurado (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
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

        df_plot = df.tail(60)
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
# MODELO DE SEÑALES
# =======================
@dataclass
class Signal:
    symbol: str
    base: str
    side: str  # LONG/SHORT
    timeframe: str
    entry: float
    tp: float
    sl: float
    score: float
    reasons: List[str]
    opened_ts: int  # ms
    opened_at_local: str
    status: str = "OPEN"  # OPEN / TP / SL / CANCELLED / EXPIRED
    closed_ts: Optional[int] = None
    closed_at_local: Optional[str] = None
    notes: Optional[str] = None

def now_local() -> datetime:
    return datetime.now(TZ)

def local_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def clamp_pct_levels(entry: float, sl: float, tp: float) -> Tuple[float, float]:
    """
    Con 10x, capamos SL/TP para evitar niveles absurdos en coins volátiles.
    """
    max_sl = entry * (1 - MAX_SL_PCT)
    min_sl = entry * (1 + MAX_SL_PCT)
    max_tp = entry * (1 + MAX_TP_PCT)
    min_tp = entry * (1 - MAX_TP_PCT)

    # Para LONG: SL debajo (pero no más de MAX_SL_PCT), TP arriba (no más de MAX_TP_PCT)
    # Para SHORT: SL arriba, TP debajo
    return (max_sl, max_tp, min_tp, min_sl)

def detectar_signal_alta_precision(ohlcv: List[List[float]]) -> dict:
    """
    Señales “muy asertivas”:
    - Tendencia: EMA20/EMA50 + cruce reciente (bonifica)
    - Fuerza: ADX mínimo
    - Momentum: RSI filtro + Estocástico en zonas
    - Evitar chop: si ADX bajo -> no trade
    """
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
    if len(df) < 60:
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

    score = 0.0
    reasons: List[str] = []

    # 0) Filtro ADX: sin fuerza -> fuera (reduce señales falsas)
    if np.isnan(adx) or adx < MIN_ADX:
        return {
            "signal": "NO_TRADE",
            "score": 0,
            "reasons": [f"✋ ADX bajo ({adx:.1f} < {MIN_ADX}) → mercado sin fuerza"],
            "entry": entry,
            "sl": None,
            "tp": None,
        }

    score += 20
    reasons.append(f"💪 ADX OK ({adx:.1f})")

    # 1) Tendencia EMA
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    prev_ema20 = float(prev["ema20"])
    prev_ema50 = float(prev["ema50"])

    bullish = ema20 > ema50
    bearish = ema20 < ema50

    if bullish:
        score += 35
        reasons.append("🌊 Tendencia alcista (EMA20>EMA50)")
        if prev_ema20 <= prev_ema50:
            score += 25
            reasons.append("🔥 Cruce alcista reciente")
    elif bearish:
        score -= 35
        reasons.append("🌊 Tendencia bajista (EMA20<EMA50)")
        if prev_ema20 >= prev_ema50:
            score -= 25
            reasons.append("🔥 Cruce bajista reciente")

    # 2) RSI filtro (confirmación)
    if bullish and rsi >= RSI_LONG_MIN:
        score += 15
        reasons.append(f"✅ RSI confirma long ({rsi:.1f} ≥ {RSI_LONG_MIN})")
    elif bearish and rsi <= RSI_SHORT_MAX:
        score -= 15
        reasons.append(f"✅ RSI confirma short ({rsi:.1f} ≤ {RSI_SHORT_MAX})")
    else:
        # si RSI no acompaña, penalizamos fuerte (para “solo asertivas”)
        score *= 0.55
        reasons.append(f"⚠️ RSI no acompaña ({rsi:.1f}) → penalización")

    # 3) Estocástico en zonas (timing)
    prev_k = float(prev["k"])
    prev_d = float(prev["d"])

    if bullish and (k > d) and (prev_k <= prev_d) and (k < 35):
        score += 15
        reasons.append("🚀 Timing alcista (Stoch K>D en zona baja)")
    elif bearish and (k < d) and (prev_k >= prev_d) and (k > 65):
        score -= 15
        reasons.append("🔻 Timing bajista (Stoch K<D en zona alta)")
    else:
        # no mata la señal, pero no suma
        reasons.append("⏳ Stoch sin señal clara")

    # 4) Construcción de SL/TP por ATR con caps
    if np.isnan(atr) or atr <= 0:
        return {"signal": "NO_TRADE", "score": 0, "reasons": ["ATR inválido"], "entry": entry, "sl": None, "tp": None}

    # Cálculo base
    sl_long = entry - ATR_SL_MULT * atr
    tp_long = entry + ATR_TP_MULT * atr
    sl_short = entry + ATR_SL_MULT * atr
    tp_short = entry - ATR_TP_MULT * atr

    # Caps por % máximo
    max_sl_long, max_tp_long, min_tp_short, min_sl_short = clamp_pct_levels(entry, 0, 0)

    sl_long = max(sl_long, max_sl_long)  # no más lejos del 2%
    tp_long = min(tp_long, max_tp_long)  # no más lejos del 5%
    tp_short = max(tp_short, min_tp_short)  # no más lejos del 5% hacia abajo
    sl_short = min(sl_short, min_sl_short)  # no más lejos del 2% hacia arriba

    # Señal final
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
# UTILIDADES DE ESTADO / LOG
# =======================
def day_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def log_path_for_day(dt: datetime) -> Path:
    return LOG_DIR / f"signals_{day_key(dt)}.jsonl"

def append_log(sig: Signal) -> None:
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

# =======================
# MENSAJES DE SEÑAL / CANCEL / CIERRE / RESUMEN
# =======================
def format_signal_message(sig: Signal) -> str:
    emoji = "🟢" if sig.side == "LONG" else "🔴"
    return (
        f"{emoji} **ENTRADA MUY ASEGURADA: {sig.side}**\n\n"
        f"🪙 **Activo:** `{sig.symbol}`\n"
        f"⏱️ **Timeframe:** `{sig.timeframe}` (prioridad 15m)\n"
        f"💰 **Entrada (recomendada):** `{sig.entry:.6f}`\n"
        f"🎯 **Salida (TP recomendada):** `{sig.tp:.6f}`\n"
        f"🛑 **Stop (SL):** `{sig.sl:.6f}`\n\n"
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
        f"{emoji} **CIERRE SEÑAL: {sig.status}**\n\n"
        f"🪙 **Activo:** `{sig.symbol}`\n"
        f"📌 **Lado:** `{sig.side}` | ⏱️ `{sig.timeframe}`\n"
        f"💰 **Entrada:** `{sig.entry:.6f}`\n"
        f"🎯 **TP:** `{sig.tp:.6f}` | 🛑 **SL:** `{sig.sl:.6f}`\n"
        f"🕒 **Apertura:** `{sig.opened_at_local}`\n"
        f"🕒 **Cierre:** `{sig.closed_at_local}`\n"
        f"📝 **Notas:** {sig.notes or '-'}"
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
    other = total - tp - sl - canc - open_

    lines = []
    # compacto pero útil
    for x in items[-25:]:  # evita mensajes enormes
        lines.append(
            f"- `{x.get('symbol')}` {x.get('side')} `{x.get('timeframe')}` → **{x.get('status')}** | "
            f"Entry `{x.get('entry'):.6f}` TP `{x.get('tp'):.6f}` SL `{x.get('sl'):.6f}`"
        )

    msg = (
        f"📌 **Resumen diario {for_day.isoformat()}**\n\n"
        f"Total: **{total}** | ✅ TP: **{tp}** | 🛑 SL: **{sl}** | 🟠 Canceladas: **{canc}** | ⏳ Abiertas: **{open_}**"
        + (f" | ⚪ Otros: **{other}**" if other else "")
        + "\n\n"
        f"📋 **Últimas {min(total, 25)} señales:**\n"
        + "\n".join(lines)
    )
    enviar_mensaje(msg)

# =======================
# CORE: SELECCIÓN TF + REGLA NO-2 SEGUIDAS MISMA MONEDA
# =======================
def choose_best_timeframe(exchange, symbol: str) -> Tuple[str, dict, List[List[float]]]:
    """
    Evalúa múltiples timeframes y elige el que tenga:
    - señal (LONG/SHORT) válida y
    - mayor |score|
    Prioriza 15m si hay empate.
    """
    best = None  # (abs_score, tf, res, ohlcv)
    for tf in TIMEFRAME_CANDIDATES:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=120)
            if not ohlcv or len(ohlcv) < 60:
                continue
            res = detectar_signal_alta_precision(ohlcv)
            sig = res.get("signal", "NO_TRADE")
            score = float(res.get("score", 0))
            if sig == "NO_TRADE":
                continue

            candidate = (abs(score), tf, res, ohlcv)
            if best is None:
                best = candidate
            else:
                # Mayor abs(score) gana; empate -> preferir PRIMARY_TIMEFRAME
                if candidate[0] > best[0] + 1e-9:
                    best = candidate
                elif abs(candidate[0] - best[0]) <= 1e-9:
                    if best[1] != PRIMARY_TIMEFRAME and tf == PRIMARY_TIMEFRAME:
                        best = candidate
        except Exception as e:
            print(f"TF eval error {symbol} {tf}: {e}")

    if best is None:
        return PRIMARY_TIMEFRAME, {"signal": "NO_TRADE", "score": 0, "reasons": []}, []
    return best[1], best[2], best[3]

def resolve_symbol(exchange, base: str) -> Optional[str]:
    # KuCoin Futures suele usar "XRPUSDTM". Mantenemos fallback a formato ccxt alternativo.
    for s in (f"{base}USDTM", f"{base}/USDT:USDT"):
        if s in exchange.markets:
            return s
    return None

def candle_ts(ohlcv: List[List[float]]) -> int:
    return int(ohlcv[-1][0])

# =======================
# TRACKING TP/SL (sin ejecutar trades, solo “verificación”)
# =======================
def check_signal_outcome(exchange, sig: Signal) -> Optional[str]:
    """
    Verifica si se tocó TP o SL usando la última vela del timeframe de la señal.
    """
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
            # SHORT
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
    print("🤖 Bot (Alta Precisión) iniciado…")

    exchange = ccxt.kucoinfutures({
        "apiKey": KUCOIN_API_KEY,
        "secret": KUCOIN_API_SECRET,
        "password": KUCOIN_API_PASSPHRASE,
        "enableRateLimit": True,
    })
    exchange.load_markets()

    # Estado
    last_signal_time_by_base: Dict[str, datetime] = {}
    active_by_base: Dict[str, Signal] = {}  # 1 activa por moneda
    sent_ids: Dict[str, bool] = {}          # anti-duplicado (symbol+ts+side+tf)

    while True:
        # 1) Revisar señales abiertas (TP/SL)
        for base, sig in list(active_by_base.items()):
            if sig.status != "OPEN":
                continue
            outcome = check_signal_outcome(exchange, sig)
            if outcome in ("TP", "SL"):
                sig.status = outcome
                sig.closed_ts = int(time.time() * 1000)
                sig.closed_at_local = local_str(now_local())
                sig.notes = "Verificado por toque de vela (high/low) en el timeframe de la señal."
                append_log(sig)
                enviar_mensaje(format_close_message(sig))
                # liberar la moneda
                del active_by_base[base]

        # 2) Generar nuevas señales (muy selectivo)
        for base in TICKERS:
            time.sleep(1.2)  # rate limit friendly

            symbol = resolve_symbol(exchange, base)
            if not symbol:
                continue

            # cooldown: evita 2 señales seguidas del mismo base, salvo CANCEL explícito por cambio de sesgo
            last_t = last_signal_time_by_base.get(base)
            if last_t and (now_local() - last_t) < timedelta(minutes=SIGNAL_COOLDOWN_MIN):
                continue

            tf, res, ohlcv = choose_best_timeframe(exchange, symbol)
            if not ohlcv:
                continue

            sig_type = res.get("signal", "NO_TRADE")
            if sig_type == "NO_TRADE":
                continue

            entry = float(res["entry"])
            sl = float(res["sl"])
            tp = float(res["tp"])
            score = float(res["score"])
            reasons = list(res.get("reasons", []))

            last_ts = candle_ts(ohlcv)
            unique_id = f"{symbol}|{tf}|{last_ts}|{sig_type}"
            if unique_id in sent_ids:
                continue

            # Regla: no 2 órdenes seguidas misma moneda.
            # Si hay activa y la nueva es OPUESTA => enviar CANCELAR la anterior + lanzar la nueva.
            if base in active_by_base and active_by_base[base].status == "OPEN":
                prev_sig = active_by_base[base]
                if prev_sig.side != sig_type:
                    enviar_mensaje(format_cancel_message(symbol, f"Cambio de sesgo: {prev_sig.side} → {sig_type}. Cancela la anterior antes de continuar."))
                    prev_sig.status = "CANCELLED"
                    prev_sig.closed_ts = int(time.time() * 1000)
                    prev_sig.closed_at_local = local_str(now_local())
                    prev_sig.notes = "Cancelada por señal contraria posterior."
                    append_log(prev_sig)
                    enviar_mensaje(format_close_message(prev_sig))
                    del active_by_base[base]
                else:
                    # misma dirección mientras hay una abierta -> no hacemos nada
                    continue

            # Crear señal nueva
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
                reasons=reasons,
                opened_ts=last_ts,
                opened_at_local=local_str(opened_dt),
            )

            # Enviar señal + gráfico
            chart = generar_grafico(symbol, tf, ohlcv)
            enviar_mensaje(format_signal_message(sig), chart_path=chart)

            # Registrar
            active_by_base[base] = sig
            last_signal_time_by_base[base] = opened_dt
            sent_ids[unique_id] = True
            append_log(sig)

            # limpieza de memoria
            if len(sent_ids) > MAX_STATE_SIZE:
                sent_ids.clear()

        time.sleep(UPDATE_INTERVAL)

# =======================
# RESUMEN DIARIO A LAS 00:00 (Europe/Madrid)
# =======================
def daily_summary_worker() -> None:
    hh, mm = parse_hhmm(DAILY_SUMMARY_AT)
    last_sent_day: Optional[date] = None

    while True:
        dt = now_local()
        # Disparar a las 00:00 exacto (con tolerancia de 60s por el sleep)
        if dt.hour == hh and dt.minute == mm:
            # El resumen es del “día que acaba de terminar” si es 00:00
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
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()
    threading.Thread(target=daily_summary_worker, daemon=True).start()
    main_loop()
