import time
import json
from pathlib import Path

import ccxt
import requests

# =======================
# CONFIGURACIÓN
# =======================

# --- Variables de Entorno (Se asume que ya están configuradas) ---
import os
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

# Timeframe
TIMEFRAME = "15m"

# Lista de ALTCOINS (Alta Liquidez, Precio < $5 USD)
BASE_TICKERS = [
    "XRP", "ADA", "LINK", "NEAR", "WLD",
    "FIL", "ARB", "OP", "SUI", "SEI",
    "DOGE", "TRX", "XLM", "FTM", "TIA",
    "MINA", "MANA", "SAND", "GALA", "CHZ",
    "SHIB", "FLOKI"
]

# Fichero para guardar el estado por símbolo real de futures (ej. "XRP/USDT:USDT")
STATE_FILE = Path("state_kucoin_signals.json")


# ==================================
# PARÁMETROS DE LA ESTRATEGIA (ACTUALIZADOS)
# ==================================
EMA_RAPIDA_PERIOD = 9
EMA_LENTA_PERIOD = 21
EMA_TENDENCIA_PERIOD = 100  # Filtro de tendencia
ATR_PERIOD = 14  
ATR_SL_MULT = 1.5  
ATR_TP_MULT = 3.0  

# PARÁMETROS DE GESTIÓN DE CAPITAL (ACTUALIZADOS)
BALANCE_TEORICO = 50.0
RIESGO_POR_OPERACION = 0.05    # 5% de riesgo por operación (ANTES ERA 0.01)
APALANCAMIENTO_FIJO = 10       # Apalancamiento deseado (x10)


# ==================================
# FUNCIONES DE INDICADORES (SIN CAMBIOS)
# ==================================

def ema(values, period):
    # ... (código de ema sin cambios)
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
    # ... (código de tr sin cambios)
    r1 = high - low
    r2 = abs(high - close_prev)
    r3 = abs(low - close_prev)
    return max(r1, r2, r3)


def atr(ohlcv, period=ATR_PERIOD):
    # ... (código de atr sin cambios)
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
# FUNCIÓN DE SEÑALES (SIN CAMBIOS)
# ==================================

def generar_senal(ohlcv):
    """
    Estrategia:
    - Entrada: cruce EMA 9/21 + filtro EMA 100 + SL/TP por ATR
    - Señal sobre la ÚLTIMA VELA CERRADA (penúltima del array).
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
# GESTIÓN DE RIESGO (ACTUALIZADO)
# ==================================

def calcular_posicion(precio_entrada, stop_loss):
    """
    Calcula el tamaño de la posición basado en un riesgo fijo (5% del balance).
    """
    balance = BALANCE_TEORICO
    riesgo_dolares = balance * RIESGO_POR_OPERACION # 5% de $50 = $2.50

    if not stop_loss:
        return 0, riesgo_dolares

    distancia = abs(precio_entrada - stop_loss)
    if distancia == 0:
        return 0, riesgo_dolares

    # tamaño = Riesgo en USD / Distancia al SL en precio
    tamaño = riesgo_dolares / distancia  # tamaño en unidades del activo

    return tamaño, riesgo_dolares


# ==================================
# RESTO DEL LOOP PRINCIPAL (SIN CAMBIOS)
# ==================================
# ... (Funciones get_exchange, build_symbol_map, get_ohlcv, TELEGRAM, STATE, y main_loop)
# ... (El código restante del loop principal se mantiene idéntico, ya que la lógica de entrada/salida y estado es correcta)

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
    """
    markets = exchange.load_markets()
    symbol_map = {}

    alias = {
        "1000SHIB": "SHIB",
        "1000PEPE": "PEPE", # Añadido por seguridad si incluyes más tarde
    }

    for m in markets.values():
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


def main_loop():
    state = cargar_state()
    exchange = get_exchange()

    symbol_map = build_symbol_map(exchange, BASE_TICKERS)

    if not symbol_map:
        print("No se encontraron símbolos de futuros para las bases especificadas.")
        return

    while True:
        try:
            for base in BASE_TICKERS:
                symbol = symbol_map.get(base)
                if not symbol:
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
                                    f"ENTRADA {senal} Futuros KuCoin (x{APALANCAMIENTO_FIJO} sugerido)\n"
                                    f"Par: {symbol}\n"
                                    f"Timeframe: {TIMEFRAME}\n\n"
                                    f"Precio entrada (aprox): {precio:.6f}\n"
                                    f"Stop Loss (ATR {ATR_SL_MULT}x): {stop_loss:.6f}\n"
                                    f"Take Profit (ATR {ATR_TP_MULT}x): {take_profit:.6f}\n\n"
                                    f"Tamaño teórico sugerido: {tamaño:.6f} unidades\n"
                                    f"Riesgo teórico ({RIESGO_POR_OPERACION*100:.1f}% de {BALANCE_TEORICO}$): {riesgo:.2f} USDT\n\n"
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
