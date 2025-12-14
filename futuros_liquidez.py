import time
import json
from pathlib import Path

import ccxt
import requests

import os
from flask import Flask
import threading

# =======================
# CONFIGURACIÓN
# =======================

# --- Variables de Entorno ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

# Timeframe -> AHORA 1H
TIMEFRAME = "1h"

# Slippage máximo permitido (0.5% = 0.005)
MAX_SLIPPAGE_PCT = 0.005

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
# PARÁMETROS DE LA ESTRATEGIA
# ==================================
EMA_RAPIDA_PERIOD = 9
EMA_LENTA_PERIOD = 21
EMA_TENDENCIA_PERIOD = 100  # Filtro de tendencia
ATR_PERIOD = 14
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0

# PARÁMETROS DE GESTIÓN DE CAPITAL
BALANCE_TEORICO = 50.0
RIESGO_POR_OPERACION = 0.05    # 5% de riesgo por operación
APALANCAMIENTO_FIJO = 10       # Apalancamiento deseado (x10)


# ==================================
# FUNCIONES DE INDICADORES
# ==================================

def ema(values, period):
    """
    Cálculo de EMA simple
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
    True Range
    """
    r1 = high - low
    r2 = abs(high - close_prev)
    r3 = abs(low - close_prev)
    return max(r1, r2, r3)


def atr(ohlcv, period=ATR_PERIOD):
    """
    ATR suavizado tipo RMA
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
# FUNCIÓN DE SEÑALES
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
# GESTIÓN DE RIESGO
# ==================================

def calcular_posicion(precio_entrada, stop_loss):
    """
    Calcula el tamaño de la posición basado en un riesgo fijo (5% del balance).
    """
    balance = BALANCE_TEORICO
    riesgo_dolares = balance * RIESGO_POR_OPERACION  # 5% de $50 = $2.50

    if not stop_loss:
        return 0, riesgo_dolares

    distancia = abs(precio_entrada - stop_loss)
    if distancia == 0:
        return 0, riesgo_dolares

    tamaño = riesgo_dolares / distancia  # tamaño en unidades del activo
    return tamaño, riesgo_dolares


# ==================================
# EXCHANGE / DATA / TELEGRAM / STATE
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
    """
    markets = exchange.load_markets()
    symbol_map = {}

    alias = {
        "1000SHIB": "SHIB",
        "1000PEPE": "PEPE",
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
    return exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)


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


# ==================================
# LOOP PRINCIPAL
# ==================================

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

                    # ¿Vela nueva?
                    if last_candle_ts is None or ts_candle > last_candle_ts:
                        print(f"Vela nueva en {symbol}. Señal entrada: {senal}, last_signal: {last_signal}")
                        last_candle_ts = ts_candle

                        # 1) SALIDA TRAILING POR CRUCE EMA
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

                        # 2) ENTRADA NUEVA CON FILTRO DE SLIPPAGE
                        stop_loss = info.get("stop_loss")
                        take_profit = info.get("take_profit")
                        precio_senal = info.get("precio")  # cierre de la vela de señal

                        if senal in ("LONG", "SHORT") and stop_loss and take_profit and precio_senal:
                            # --- FILTRO DE PRECIO / SLIPPAGE ---
                            es_valido = True
                            precio_entrada_real = precio_senal

                            try:
                                ticker = exchange.fetch_ticker(symbol)
                                ask = ticker.get("ask") or ticker.get("last")
                                bid = ticker.get("bid") or ticker.get("last")

                                if senal == "LONG":
                                    if ask is None:
                                        raise ValueError("ask vacío en ticker")
                                    precio_entrada_real = ask
                                    if precio_entrada_real > precio_senal * (1 + MAX_SLIPPAGE_PCT):
                                        print(
                                            f"LONG RECHAZADO: ask {precio_entrada_real:.6f} "
                                            f"muy por encima del precio de señal {precio_senal:.6f}"
                                        )
                                        es_valido = False
                                else:  # SHORT
                                    if bid is None:
                                        raise ValueError("bid vacío en ticker")
                                    precio_entrada_real = bid
                                    if precio_entrada_real < precio_senal * (1 - MAX_SLIPPAGE_PCT):
                                        print(
                                            f"SHORT RECHAZADO: bid {precio_entrada_real:.6f} "
                                            f"muy por debajo del precio de señal {precio_senal:.6f}"
                                        )
                                        es_valido = False

                            except Exception as e_ticker:
                                print(
                                    f"Error al obtener ticker en tiempo real para slippage: {e_ticker}. "
                                    f"Usando precio de vela."
                                )
                                es_valido = True
                                precio_entrada_real = precio_senal

                            # --- EJECUTAR SEÑAL SI ES VÁLIDA ---
                            if es_valido:
                                if last_signal != senal:
                                    tamaño, riesgo = calcular_posicion(precio_entrada_real, stop_loss)

                                    mensaje_entrada = (
                                        f"ENTRADA {senal} Futuros KuCoin (x{APALANCAMIENTO_FIJO} sugerido)\n"
                                        f"Par: {symbol}\n"
                                        f"Timeframe: {TIMEFRAME}\n\n"
                                        f"Precio entrada (real-ask/bid): {precio_entrada_real:.6f}\n"
                                        f"Precio de Señal (Cierre Vela): {precio_senal:.6f}\n"
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
                                print("Señal rechazada por slippage.")
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

        # Espera general entre ciclos sobre todos los símbolos
        time.sleep(30)


# ==================================
# FLASK PARA HEALTHCHECK EN FLY.IO
# ==================================

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot running!", 200


def run_flask():
    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    # Lanzamos Flask en un hilo en segundo plano (para Fly.io)
    threading.Thread(target=run_flask, daemon=True).start()
    # Y arrancamos el loop principal del bot
    main_loop()
