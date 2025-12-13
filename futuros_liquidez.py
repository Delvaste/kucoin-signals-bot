import time
import json
from pathlib import Path

import ccxt
import requests

# =======================
# CONFIGURACIÓN
# =======================

# --- Telegram ---
import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

# Timeframe
TIMEFRAME = "15m"

# Lista de ALTCOINS (por base); el símbolo exacto de futuros lo detectamos en runtime
BASE_TICKERS = [
    "XRP", "ADA", "DOGE", "AVAX", "INJ",
    "NEAR", "DOT", "LINK", "LTC", "BCH",
    "ARB", "OP", "SUI", "WLD", "FIL",
    "TRX", "HBAR", "SHIB", "XLM", "ZEC",
]

# Fichero para guardar el estado por símbolo real de futures (ej. "XRP/USDT:USDT")
STATE_FILE = Path("state_kucoin_signals.json")


# ==================================
# PARÁMETROS DE LA ESTRATEGIA
# ==================================
EMA_RAPIDA_PERIOD = 9
EMA_LENTA_PERIOD = 21
EMA_TENDENCIA_PERIOD = 100  # Filtro de tendencia
ATR_PERIOD = 14             # Período para el cálculo del ATR
ATR_SL_MULT = 1.5           # Multiplicador para el Stop Loss
ATR_TP_MULT = 3.0           # Multiplicador para el Take Profit


# ==================================
# FUNCIONES DE INDICADORES
# ==================================

def ema(values, period):
    """
    Media Móvil Exponencial (EMA)
    """
    if not values:
        return []

    k = 2 / (period + 1)
    ema_vals = []

    ema_prev = values[0]
    ema_vals.append(ema_prev)

    for i in range(1, len(values)):
        v = values[i]
        ema_prev = v * k + ema_prev * (1 - k)
        ema_vals.append(ema_prev)

    return ema_vals


def tr(high, low, close_prev):
    """
    True Range (Rango Verdadero)
    """
    r1 = high - low
    r2 = abs(high - close_prev)
    r3 = abs(low - close_prev)
    return max(r1, r2, r3)


def atr(ohlcv, period=ATR_PERIOD):
    """
    Average True Range (ATR) con suavizado tipo EMA.
    Devuelve una lista de la misma longitud que ohlcv.
    """
    if len(ohlcv) < 2:
        return []

    trs = []
    for i in range(1, len(ohlcv)):
        high = ohlcv[i][2]
        low = ohlcv[i][3]
        close_prev = ohlcv[i - 1][4]
        trs.append(tr(high, low, close_prev))

    if len(trs) < period:
        return [0.0] * len(ohlcv)

    initial_atr = sum(trs[:period]) / period

    atr_vals = [0.0] * period
    atr_vals.append(initial_atr)

    atr_prev = initial_atr

    for i in range(period, len(trs)):
        atr_prev = (atr_prev * (period - 1) + trs[i]) / period
        atr_vals.append(atr_prev)

    if len(atr_vals) < len(ohlcv):
        atr_vals.extend([atr_prev] * (len(ohlcv) - len(atr_vals)))
    elif len(atr_vals) > len(ohlcv):
        atr_vals = atr_vals[:len(ohlcv)]

    return atr_vals


# ==================================
# FUNCIÓN DE SEÑALES MEJORADA
# ==================================

def generar_senal(ohlcv):
    """
    Estrategia:
    - Entrada: cruce EMA 9/21 + filtro EMA 100 + SL/TP por ATR
    - Señal sobre la ÚLTIMA VELA CERRADA (penúltima del array).
    Devuelve también flags de cruce para trailing exit.
    """
    if len(ohlcv) < EMA_TENDENCIA_PERIOD + 2:
        return {
            "senal": "NO_TRADE",
            "mensaje": "Datos insuficientes",
        }

    closes = [c[4] for c in ohlcv]

    ema_rapida = ema(closes, EMA_RAPIDA_PERIOD)
    ema_lenta = ema(closes, EMA_LENTA_PERIOD)
    ema_tendencia = ema(closes, EMA_TENDENCIA_PERIOD)
    atr_vals = atr(ohlcv, ATR_PERIOD)

    try:
        ema_r_1 = ema_rapida[-2]
        ema_r_2 = ema_rapida[-3]
        ema_l_1 = ema_lenta[-2]
        ema_l_2 = ema_lenta[-3]
        ema_t_1 = ema_tendencia[-2]
        atr_1 = atr_vals[-2]
        precio = closes[-2]
        ts_ultima_cerrada = ohlcv[-2][0]
    except IndexError:
        return {
            "senal": "NO_TRADE",
            "mensaje": "Error de indexación en indicadores.",
        }

    senal = "NO_TRADE"
    stop_loss = None
    take_profit = None
    distancia_sl = atr_1 * ATR_SL_MULT

    cruce_alcista = ema_r_2 < ema_l_2 and ema_r_1 > ema_l_1
    cruce_bajista = ema_r_2 > ema_l_2 and ema_r_1 < ema_l_1

    # LONG
    precio_confirmado_long = precio > ema_r_1 and precio > ema_l_1
    filtro_tendencia_long = precio > ema_t_1

    if cruce_alcista and precio_confirmado_long and filtro_tendencia_long:
        senal = "LONG"
        stop_loss = precio - distancia_sl
        take_profit = precio + (atr_1 * ATR_TP_MULT)

    # SHORT
    precio_confirmado_short = precio < ema_r_1 and precio < ema_l_1
    filtro_tendencia_short = precio < ema_t_1

    if cruce_bajista and precio_confirmado_short and filtro_tendencia_short:
        senal = "SHORT"
        stop_loss = precio + distancia_sl
        take_profit = precio - (atr_1 * ATR_TP_MULT)

    return {
        "senal": senal,
        "precio": precio,
        "ema_rapida": ema_r_1,
        "ema_lenta": ema_l_1,
        "ema_tendencia": ema_t_1,
        "atr": atr_1,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "timestamp_candle": ts_ultima_cerrada,
        "cruce_alcista": cruce_alcista,
        "cruce_bajista": cruce_bajista,
    }


# ==================================
# GESTIÓN DE RIESGO
# ==================================

BALANCE_TEORICO = 50.0
RIESGO_POR_OPERACION = 0.01   # 1%

def calcular_posicion(precio_entrada, stop_loss):
    balance = BALANCE_TEORICO
    riesgo_dolares = balance * RIESGO_POR_OPERACION

    if not stop_loss:
        return 0, riesgo_dolares

    distancia = abs(precio_entrada - stop_loss)
    if distancia == 0:
        return 0, riesgo_dolares

    tamaño = riesgo_dolares / distancia
    return tamaño, riesgo_dolares


# ==================================
# EXCHANGE KUCOIN FUTURES
# ==================================

def get_exchange():
    exchange = ccxt.kucoinfutures({
        "apiKey": KUCOIN_API_KEY,
        "secret": KUCOIN_API_SECRET,
        "password": KUCOIN_API_PASSPHRASE,
        "enableRateLimit": True,
    })
    return exchange


def build_symbol_map(exchange, base_tickers):
    """
    Mapea cada base (XRP, ADA, etc.) al símbolo real de futuros USDT-M en CCXT.
    Ejemplo: "INJ" -> "INJ/USDT:USDT"
    Incluye alias como 1000SHIB -> SHIB si fuera el caso.
    """
    markets = exchange.load_markets()
    symbol_map = {}

    # Posible alias de base raras
    alias = {
        "1000SHIB": "SHIB",
    }

    for m in markets.values():
        # Queremos SOLO futuros lineales (USDT-Margined)
        if not m.get("future", False) and m.get("type") != "swap":
            continue
        if not m.get("linear", False):
            continue
        if m.get("quote") not in ("USDT", "USDC"):
            continue

        base_ccxt = m.get("base")
        base_normalizada = alias.get(base_ccxt, base_ccxt)

        if base_normalizada in base_tickers and base_normalizada not in symbol_map:
            symbol_map[base_normalizada] = m["symbol"]

    print("Mapa de símbolos detectado:")
    for b, s in symbol_map.items():
        print(f"  {b} -> {s}")

    return symbol_map


def get_ohlcv(exchange, symbol, limit=300):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    return ohlcv


# ==================================
# TELEGRAM
# ==================================

def enviar_mensaje(texto: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": texto,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        print("Respuesta Telegram:", r.status_code, r.text)
        r.raise_for_status()
    except Exception as e:
        print("Error enviando mensaje a Telegram:", e)


# ==================================
# STATE
# ==================================

def cargar_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def guardar_state(state):
    try:
        STATE_FILE.write_text(json.dumps(state))
    except Exception as e:
        print(f"No se pudo guardar estado: {e}")


# ==================================
# LOOP PRINCIPAL
# ==================================

def main_loop():
    state = cargar_state()
    exchange = get_exchange()

    # Construimos mapa base -> símbolo de futuros
    symbol_map = build_symbol_map(exchange, BASE_TICKERS)

    if not symbol_map:
        print("No se encontraron símbolos de futuros para las bases especificadas.")
        return

    while True:
        try:
            for base in BASE_TICKERS:
                symbol = symbol_map.get(base)
                if not symbol:
                    # No se encontró mercado de futuros para este base
                    continue

                try:
                    print(f"\n--- Procesando {base} ({symbol}) ---")
                    ohlcv = get_ohlcv(exchange, symbol, limit=300)
                    if not ohlcv:
                        print("No se han recibido datos OHLCV para", symbol)
                        continue

                    info = generar_senal(ohlcv)
                    senal = info.get("senal", "NO_TRADE")
                    ts_candle = info.get("timestamp_candle")
                    cruce_alcista = info.get("cruce_alcista", False)
                    cruce_bajista = info.get("cruce_bajista", False)

                    if ts_candle is None:
                        print("No hay timestamp en la señal. Info:", info)
                        continue

                    sym_state = state.get(symbol, {})
                    last_candle_ts = sym_state.get("last_candle_ts")
                    last_signal = sym_state.get("last_signal")

                    if last_candle_ts is None or ts_candle > last_candle_ts:
                        print(f"Vela nueva en {symbol}. Señal entrada: {senal}, last_signal: {last_signal}")
                        last_candle_ts = ts_candle

                        # 1) TRAILING EXIT POR CRUCE EMA
                        if last_signal == "LONG" and cruce_bajista:
                            mensaje_salida = (
                                f"SALIDA LONG (Trailing EMA)\n"
                                f"Par: {symbol}\n"
                                f"Timeframe: {TIMEFRAME}\n\n"
                                f"Motivo: EMA9 cruza por DEBAJO de EMA21.\n"
                                f"EMA9: {info.get('ema_rapida'):.4f}\n"
                                f"EMA21: {info.get('ema_lenta'):.4f}\n"
                                f"EMA100: {info.get('ema_tendencia'):.4f}\n"
                            )
                            enviar_mensaje(mensaje_salida)
                            last_signal = None

                        elif last_signal == "SHORT" and cruce_alcista:
                            mensaje_salida = (
                                f"SALIDA SHORT (Trailing EMA)\n"
                                f"Par: {symbol}\n"
                                f"Timeframe: {TIMEFRAME}\n\n"
                                f"Motivo: EMA9 cruza por ENCIMA de EMA21.\n"
                                f"EMA9: {info.get('ema_rapida'):.4f}\n"
                                f"EMA21: {info.get('ema_lenta'):.4f}\n"
                                f"EMA100: {info.get('ema_tendencia'):.4f}\n"
                            )
                            enviar_mensaje(mensaje_salida)
                            last_signal = None

                        # 2) ENTRADA NUEVA
                        stop_loss = info.get("stop_loss")
                        take_profit = info.get("take_profit")
                        precio = info.get("precio")

                        if senal in ("LONG", "SHORT") and stop_loss and take_profit and precio:
                            if last_signal != senal:
                                tamaño, riesgo = calcular_posicion(precio, stop_loss)

                                mensaje_entrada = (
                                    f"ENTRADA {senal} Futuros KuCoin\n"
                                    f"Par: {symbol}\n"
                                    f"Timeframe: {TIMEFRAME}\n\n"
                                    f"Precio entrada (aprox): {precio:.6f}\n"
                                    f"Stop Loss: {stop_loss:.6f}\n"
                                    f"Take Profit: {take_profit:.6f}\n\n"
                                    f"Tamaño teórico sugerido: {tamaño:.6f} unidades\n"
                                    f"Riesgo teórico ({RIESGO_POR_OPERACION*100:.1f}% de {BALANCE_TEORICO}$): {riesgo:.2f} USDT\n\n"
                                    f"EMA9: {info.get('ema_rapida'):.6f}\n"
                                    f"EMA21: {info.get('ema_lenta'):.6f}\n"
                                    f"EMA100: {info.get('ema_tendencia'):.6f}\n"
                                    f"ATR: {info.get('atr'):.6f}\n\n"
                                    f"Nota: Bot educativo. No es recomendación de inversión."
                                )

                                enviar_mensaje(mensaje_entrada)
                                last_signal = senal
                            else:
                                print("Señal igual a la anterior, no se envía nueva entrada.")
                        else:
                            print("Sin nueva entrada operable en esta vela.")

                        state[symbol] = {
                            "last_candle_ts": int(last_candle_ts),
                            "last_signal": last_signal,
                        }

                    else:
                        print(f"Misma vela en {symbol}, esperando cierre...")

                    time.sleep(0.5)

                except Exception as e_sym:
                    print(f"Error procesando {symbol}: {e_sym}")

            guardar_state(state)

        except Exception as e:
            print(f"Error en loop principal: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main_loop()

from flask import Flask
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot running!", 200

def run_flask():
    app.run(host="0.0.0.0", port=8080)

threading.Thread(target=run_flask).start()

