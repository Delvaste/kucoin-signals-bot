import time
import json
from pathlib import Path

import ccxt
import requests

import os
from flask import Flask
import threading

import pandas as pd
import numpy as np
import talib
import yaml

# =======================
# CONFIGURACIÓN
# =======================

# --- Variables de Entorno ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

CONFIG_PATH = Path("config.yml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """
    Carga la configuración desde config.yml si existe.
    Si no existe, devuelve un dict vacío y se usan los defaults del código.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CONFIG = load_config()

# Timeframe -> 1H (Confirmado) (valor por defecto, se puede sobreescribir por config.yml)
TIMEFRAME = "1h"

# Slippage máximo permitido (0.5% = 0.005)
MAX_SLIPPAGE_PCT = 0.005

# Lista de ALTCOINS (ACTUALIZADA: < $5 USD, Alta Liquidez en Futuros, Timeframe 1h)
BASE_TICKERS = [
    "XRP", "ADA", "NEAR", "WLD", "FIL", "ARB", "OP", "SUI", "SEI", 
    "DOGE", "TRX", "XLM", "FTM", "TIA", "MINA", "MANA", "SAND", 
    "GALA", "CHZ", "SHIB", "FLOKI", 
    "LDO", "WIF", "PEPE", "HBAR" # Nuevas adiciones estratégicas
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
# FUNCIONES DE INDICADORES (SIN CAMBIOS)
# ==================================

def ema(values, period):
    # ... (Código de EMA)
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
    # ... (Código de TR)
    r1 = high - low
    r2 = abs(high - close_prev)
    r3 = abs(low - close_prev)
    return max(r1, r2, r3)


def atr(ohlcv, period=ATR_PERIOD):
    # ... (Código de ATR)
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
# FUNCIÓN DE SEÑALES (ESTRATEGIA AVANZADA - SWAP)
# La lógica interna fue reemplazada por una versión simplificada de la estrategia
# de scoring (puntuación) del bot externo.
# ==================================

# --- PARÁMETROS DE LA NUEVA ESTRATEGIA (Ajustables) ---
MIN_SCORE_FOR_ENTRY = 70
# -----------------------------------------------------

# ================================
# OVERRIDE DE PARÁMETROS POR CONFIG
# ================================
settings_cfg = CONFIG.get("settings", {})
markets_cfg = CONFIG.get("markets", {})
strategy_cfg = CONFIG.get("strategy", {})

# SETTINGS
TIMEFRAME = settings_cfg.get("timeframe", TIMEFRAME)
MAX_SLIPPAGE_PCT = settings_cfg.get("max_slippage_pct", MAX_SLIPPAGE_PCT)
BALANCE_TEORICO = settings_cfg.get("balance_teorico", BALANCE_TEORICO)
RIESGO_POR_OPERACION = settings_cfg.get("riesgo_por_operacion", RIESGO_POR_OPERACION)
APALANCAMIENTO_FIJO = settings_cfg.get("apalancamiento", APALANCAMIENTO_FIJO)
BASE_TICKERS = markets_cfg.get("base_tickers", BASE_TICKERS)

# ESTRATEGIA
MIN_SCORE_FOR_ENTRY = strategy_cfg.get("min_score_for_entry", MIN_SCORE_FOR_ENTRY)
ATR_SL_MULT = strategy_cfg.get("atr_sl_mult", ATR_SL_MULT)
ATR_TP_MULT = strategy_cfg.get("atr_tp_mult", ATR_TP_MULT)

# Intervalo general entre ciclos sobre todos los símbolos (segundos)
UPDATE_INTERVAL = settings_cfg.get("update_interval", 30)

def generar_senal(ohlcv: list, last_signal: str) -> dict:
    """
    Genera la señal de trading basada en EMA, MACD y RSI con un sistema de scoring (puntuación).
    También calcula SL/TP basado en el último ATR.
    ohlcv debe tener al menos 200 velas.
    """
    if len(ohlcv) < 200:
        return {"senal": "NO_TRADE", "precio": 0, "stop_loss": 0, "take_profit": 0, "timestamp_candle": 0}

    # 1. Preparar datos
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Valores de la última vela cerrada (entrada/referencia)
    last_close = closes[-1]
    last_timestamp = ohlcv[-1][0] # Último timestamp de cierre

    # 2. Calcular Indicadores TÉCNICOS
    ema_20 = talib.EMA(closes, timeperiod=20)
    ema_50 = talib.EMA(closes, timeperiod=50)
    ema_200 = talib.EMA(closes, timeperiod=200)
    
    macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    rsi = talib.RSI(closes, timeperiod=14)
    
    # Calcular ATR para SL/TP (manteniendo la gestión de riesgo original)
    atr_vals = talib.ATR(highs, lows, closes, timeperiod=14)
    last_atr = atr_vals[-1]

    # Tomar los últimos valores de los indicadores (vela cerrada)
    last_ema_20 = ema_20[-1]
    last_ema_50 = ema_50[-1]
    last_ema_200 = ema_200[-1]
    last_macd_hist = macdhist[-1]
    last_rsi = rsi[-1]
    
    # 3. Calcular la Puntuación (Scoring) de Señal
    # Se inicializa el score a 0
    bullish_score = 0
    bearish_score = 0

    # --- PUNTUACIÓN LONG (BULLISH) ---
    # 1. Trend Filter: EMA 50 > EMA 200 (Tendencia alcista)
    if last_ema_50 > last_ema_200:
        bullish_score += 20
        
    # 2. Entry Price: Cierre > EMA 20 (Por encima del precio medio rápido)
    if last_close > last_ema_20:
        bullish_score += 20
        
    # 3. Momentum: MACD Histograma > 0 (Momentum positivo)
    if last_macd_hist > 0:
        bullish_score += 30
        
    # 4. Strength: RSI > 55 (Fuerza alcista)
    if last_rsi > 55:
        bullish_score += 30
        
    # --- PUNTUACIÓN SHORT (BEARISH) ---
    # 1. Trend Filter: EMA 50 < EMA 200 (Tendencia bajista)
    if last_ema_50 < last_ema_200:
        bearish_score += 20
        
    # 2. Entry Price: Cierre < EMA 20 (Por debajo del precio medio rápido)
    if last_close < last_ema_20:
        bearish_score += 20
        
    # 3. Momentum: MACD Histograma < 0 (Momentum negativo)
    if last_macd_hist < 0:
        bearish_score += 30
        
    # 4. Strength: RSI < 45 (Fuerza bajista)
    if last_rsi < 45:
        bearish_score += 30
        

    # 4. Generar la Señal Final
    senal = "NO_TRADE"
    stop_loss = 0
    take_profit = 0
    
    # La nueva estrategia requiere un score MÍNIMO para entrar
    if bullish_score >= MIN_SCORE_FOR_ENTRY and bearish_score < MIN_SCORE_FOR_ENTRY:
        senal = "LONG"
    elif bearish_score >= MIN_SCORE_FOR_ENTRY and bullish_score < MIN_SCORE_FOR_ENTRY:
        senal = "SHORT"

    # 5. Calcular SL/TP basado en ATR (Usando 2x ATR para SL, 4x ATR para TP)
    if senal != "NO_TRADE":
        atr_multiplier_sl = 2.0
        atr_multiplier_tp = 4.0 # Esto es una configuración de riesgo conservadora (Ratio 1:2)
        
        if senal == "LONG":
            stop_loss = last_close - (last_atr * atr_multiplier_sl)
            take_profit = last_close + (last_atr * atr_multiplier_tp)
        elif senal == "SHORT":
            stop_loss = last_close + (last_atr * atr_multiplier_sl)
            take_profit = last_close - (last_atr * atr_multiplier_tp)

    # Devolver el resultado en el formato esperado
    return {
        "senal": senal,
        "precio": last_close,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "timestamp_candle": last_timestamp,
        "debug_score": f"LONG:{bullish_score}, SHORT:{bearish_score}" # DEBUG
    }


# ==================================
# GESTIÓN DE RIESGO (SIN CAMBIOS)
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
        "1000PEPE": "PEPE", # Añadido alias para el nuevo ticker PEPE
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
    # Carga inicial de estado y conexión
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

                # --- INICIALIZACIÓN POR SÍMBOLO ---
                # 1. Cargar el estado anterior (garantiza que sym_state siempre existe)
                sym_state = state.get(symbol, {"last_candle_ts": 0, "last_signal": "NO_TRADE"})
                
                # 2. Definir last_signal y last_candle_ts (evita 'referenced before assignment')
                last_signal = sym_state.get("last_signal", "NO_TRADE")
                last_candle_ts = sym_state.get("last_candle_ts", 0)
                
                # Usaremos new_signal para guardar el estado al final
                new_last_signal = last_signal 
                new_last_candle_ts = last_candle_ts

                try:
                    print(f"\n--- Procesando {base} ({symbol}) ---")
                    ohlcv = get_ohlcv(exchange, symbol, limit=300)
                    if not ohlcv:
                        print("No se han recibido datos OHLCV para", symbol)
                        continue
                    
                    # Llamada a generar_senal CORREGIDA: Se pasa el estado anterior
                    info = generar_senal(ohlcv, last_signal)
                    senal = info.get("senal", "NO_TRADE")
                    ts_candle = info.get("timestamp_candle")
                    cruce_alcista = info.get("cruce_alcista", False)
                    cruce_bajista = info.get("cruce_bajista", False)

                    if ts_candle is None:
                        print("No hay timestamp en la señal. Info:", info)
                        continue
                    
                    # --- LÓGICA DE NUEVA VELA ---
                    if ts_candle > last_candle_ts:
                        print(f"Vela nueva en {symbol}. Señal entrada: {senal}, last_signal: {last_signal}")
                        new_last_candle_ts = ts_candle
                        
                        # 1) SALIDA TRAILING POR CRUCE EMA
                        if last_signal == "LONG" and cruce_bajista:
                            mensaje_salida = (
                                f"SALIDA LONG (Trailing EMA)\n"
                                f"Par: {symbol}\n"
                                f"Timeframe: {TIMEFRAME}\n\n"
                                f"Motivo: EMA9 cruza por DEBAJO de EMA21.\n"
                                f"EMA9: {info.get('ema_rapida'):.6f}\n"
                                f"EMA21: {info.get('ema_lenta'):.6f}\n"
                            )
                            enviar_mensaje(mensaje_salida)
                            new_last_signal = "NO_TRADE" # Cierra la posición anterior
                            
                        elif last_signal == "SHORT" and cruce_alcista:
                            mensaje_salida = (
                                f"SALIDA SHORT (Trailing EMA)\n"
                                f"Par: {symbol}\n"
                                f"Timeframe: {TIMEFRAME}\n\n"
                                f"Motivo: EMA9 cruza por ENCIMA de EMA21.\n"
                                f"EMA9: {info.get('ema_rapida'):.6f}\n"
                                f"EMA21: {info.get('ema_lenta'):.6f}\n"
                            )
                            enviar_mensaje(mensaje_salida)
                            new_last_signal = "NO_TRADE" # Cierra la posición anterior

                        # 2) ENTRADA NUEVA
                        stop_loss = info.get("stop_loss")
                        take_profit = info.get("take_profit")
                        precio_senal = info.get("precio") # Precio de CIERRE de la vela de señal

                        if senal in ("LONG", "SHORT") and last_signal != senal and stop_loss and take_profit and precio_senal:
                            es_valido = True
                            precio_entrada_real = precio_senal # valor por defecto
                            
                            # --- FILTRO DE PRECIO (SLIPPAGE + SL) ---
                            try:
                                ticker = exchange.fetch_ticker(symbol)

                                # Aseguramos tener ask/bid; si no, usamos last
                                ask = ticker.get("ask") or ticker.get("last")
                                bid = ticker.get("bid") or ticker.get("last")

                                if ask is None or bid is None:
                                    raise Exception("Ticker sin ask/bid válidos")

                                # 1. Comprobación contra el Stop Loss (señal invalidada)
                                if senal == "LONG" and ask < stop_loss:
                                    print(f"LONG RECHAZADO: Precio actual ({ask:.6f}) ya está por debajo del SL ({stop_loss:.6f}).")
                                    es_valido = False
                                elif senal == "SHORT" and bid > stop_loss:
                                    print(f"SHORT RECHAZADO: Precio actual ({bid:.6f}) ya está por encima del SL ({stop_loss:.6f}).")
                                    es_valido = False

                                # 2. Comprobación de slippage sólo si sigue siendo válida
                                if es_valido:
                                    if senal == "LONG":
                                        if ask > precio_senal * (1 + MAX_SLIPPAGE_PCT):
                                            print(
                                                f"LONG RECHAZADO: ask {ask:.6f} muy por encima del precio de señal {precio_senal:.6f} (slippage > {MAX_SLIPPAGE_PCT*100:.1f}%)."
                                            )
                                            es_valido = False
                                        else:
                                            precio_entrada_real = ask
                                    elif senal == "SHORT":
                                        if bid < precio_senal * (1 - MAX_SLIPPAGE_PCT):
                                            print(
                                                f"SHORT RECHAZADO: bid {bid:.6f} muy por debajo del precio de señal {precio_senal:.6f} (slippage > {MAX_SLIPPAGE_PCT*100:.1f}%)."
                                            )
                                            es_valido = False
                                        else:
                                            precio_entrada_real = bid
                                            
                            except Exception as e_ticker:
                                print(
                                    f"Error al obtener ticker en tiempo real para slippage: {e_ticker}. "
                                    "Usando precio de cierre de la vela."
                                )
                                es_valido = True
                                precio_entrada_real = precio_senal

                            # --- Ejecución de la Señal si es Válida ---
                            if es_valido:
                                # Recalculamos el tamaño de la posición con el nuevo precio de entrada más preciso
                                tamaño, riesgo = calcular_posicion(precio_entrada_real, stop_loss)

                                mensaje_entrada = (
                                    f"ENTRADA {senal} Futuros KuCoin (x{APALANCAMIENTO_FIJO} sugerido)\n"
                                    f"Par: {symbol}\n"
                                    f"Timeframe: {TIMEFRAME}\n\n"
                                    f"Precio entrada (real ask/bid): {precio_entrada_real:.6f}\n"
                                    f"Precio de Señal (Cierre Vela): {precio_senal:.6f}\n"
                                    f"Stop Loss (ATR {ATR_SL_MULT}x): {stop_loss:.6f}\n"
                                    f"Take Profit (ATR {ATR_TP_MULT}x): {take_profit:.6f}\n\n"
                                    f"Tamaño teórico sugerido: {tamaño:.6f} unidades\n"
                                    f"Riesgo teórico ({RIESGO_POR_OPERACION*100:.1f}% de {BALANCE_TEORICO}$): {riesgo:.2f} USDT\n\n"
                                    f"Nota: Bot educativo. No es recomendación de inversión."
                                )

                                enviar_mensaje(mensaje_entrada)
                                new_last_signal = senal # Se actualiza la señal
                            else:
                                print("Señal rechazada por slippage/SL.")
                        else:
                            # La señal era NO_TRADE, o la señal es igual a la anterior y no se sale.
                            print("Sin nueva entrada operable en esta vela.")
                            if senal == "NO_TRADE":
                                new_last_signal = "NO_TRADE"
                                

                    else:
                        print(f"Misma vela en {symbol}, esperando cierre...")
                    
                    # --- ACTUALIZAR ESTADO DEL SÍMBOLO ---
                    state[symbol] = {
                        "last_candle_ts": int(new_last_candle_ts),
                        "last_signal": new_last_signal,
                    }

                    time.sleep(0.5)

                except Exception as e_sym:
                    print(f"Error procesando {symbol}: {e_sym}")
                    # Si hay un error, el estado que se guarda es el que se cargó al inicio del loop del símbolo, 
                    # ya que new_last_signal y new_last_candle_ts no se actualizaron con éxito.

            guardar_state(state)

        except Exception as e:
            print(f"Error en loop principal: {e}")

        # Espera general entre ciclos sobre todos los símbolos
        time.sleep(UPDATE_INTERVAL)


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
