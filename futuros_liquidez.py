import time
import json
from datetime import datetime
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
import matplotlib.pyplot as plt
import mplfinance as mpf

# =======================
# CONFIGURACIÓN
# =======================

# --- Variables de Env ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

CONFIG_PATH = Path("config.yml")

def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

CONFIG = load_config()

# Parámetros Base
TIMEFRAME = CONFIG.get("settings", {}).get("timeframe", "1h")
UPDATE_INTERVAL = CONFIG.get("settings", {}).get("update_interval", 30)
MAX_SLIPPAGE_PCT = CONFIG.get("settings", {}).get("max_slippage_pct", 0.005)
MIN_SCORE = CONFIG.get("strategy", {}).get("min_score_for_entry", 70)

# --- FUNCIONES DE TELEGRAM ---

def enviar_mensaje(texto: str, chart_path: str = None):
    """Envía un mensaje o una foto (con caption) a Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
    
    if chart_path and Path(chart_path).exists():
        send_photo_url = url + "sendPhoto"
        try:
            with open(chart_path, 'rb') as f:
                response = requests.post(
                    send_photo_url,
                    data={'chat_id': TELEGRAM_CHAT_ID, 'caption': texto, 'parse_mode': 'Markdown'},
                    files={'photo': f}
                )
            os.remove(chart_path) # Limpiar archivo local
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error enviando foto: {e}")
            
    send_text_url = url + "sendMessage"
    try:
        response = requests.post(send_text_url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': texto, 'parse_mode': 'Markdown'})
        return response.json()
    except Exception as e:
        print(f"Error enviando texto: {e}")
        return None

# --- FUNCIÓN DE GRÁFICO ---

def generar_grafico(symbol: str, ohlcv: list, info: dict) -> str | None:
    """Genera un gráfico actualizado estilo NextWave."""
    try:
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Indicadores para el gráfico
        df['ema20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema50'] = talib.EMA(df['close'], timeperiod=50)
        df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                             fastk_period=14, slowk_period=5, slowd_period=3)
        
        df_plot = df.tail(60).copy()
        mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

        add_plots = [
            mpf.make_addplot(df_plot['ema20'], color='blue', panel=0),
            mpf.make_addplot(df_plot['ema50'], color='orange', panel=0),
            mpf.make_addplot(df_plot['slowk'], color='purple', panel=2, title='NextWave Stoch'),
            mpf.make_addplot(df_plot['slowd'], color='red', panel=2)
        ]
        
        filename = f"chart_{symbol.replace('/', '_')}.png"
        mpf.plot(df_plot, type='candle', style=s, addplot=add_plots, savefig=filename, volume=True, figsize=(10, 7))
        return filename
    except Exception as e:
        print(f"Error gráfico: {e}")
        return None

# --- LÓGICA DE ESTRATEGIA NEXTWAVE ---

def generar_senal(ohlcv: list, last_signal: str) -> dict:
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Indicadores NextWave
    df['ema_rapida'] = talib.EMA(df['close'], timeperiod=20)
    df['ema_lenta'] = talib.EMA(df['close'], timeperiod=50)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                         fastk_period=14, slowk_period=5, slowd_period=3)
    
    last_row = df.iloc[-1]
    close = last_row['close']
    ema20 = last_row['ema_rapida']
    ema50 = last_row['ema_lenta']
    k, d = last_row['slowk'], last_row['slowd']
    adx = last_row['adx']

    score = 0
    razones = []

    # 1. Filtro de Tendencia (Ola principal)
    if close > ema20 > ema50:
        score += 50
        razones.append("🌊 Ola Alcista Confirmada")
    elif close < ema20 < ema50:
        score -= 50
        razones.append("🌊 Ola Bajista Confirmada")

    # 2. Impulso y Fuerza (Stoch + ADX)
    if adx > 20:
        if k < 25 and k > d:
            score += 40
            razones.append("🚀 Impulso: Cruce K>D en Sobreventa")
        elif k > 75 and k < d:
            score -= 40
            razones.append("🔻 Impulso: Cruce K<D en Sobrecompra")
    
    # Determinar señal final
    senal = "NO_TRADE"
    if score >= MIN_SCORE: senal = "LONG"
    elif score <= -MIN_SCORE: senal = "SHORT"

    return {
        "senal": senal,
        "score": score,
        "razones": razones,
        "precio_entrada": close,
        "stop_loss": close * 0.95 if senal == "LONG" else close * 1.05,
        "take_profit": close * 1.10 if senal == "LONG" else close * 0.90
    }

# --- CÁLCULO DE POSICIÓN ---

def calcular_posicion(precio_entrada: float, stop_loss: float) -> tuple[float, float]:
    balance = CONFIG.get("settings", {}).get("balance_teorico", 100)
    riesgo_pct = CONFIG.get("settings", {}).get("riesgo_por_operacion", 0.05)
    distancia = abs(precio_entrada - stop_loss) / precio_entrada
    if distancia == 0: return 0, 0
    tamano_posicion = (balance * riesgo_pct) / distancia
    return round(tamano_posicion, 2), round(balance * riesgo_pct, 2)

# --- BUCLE PRINCIPAL ---

def main_loop():
    print("Iniciando Bot estilo NextWave...")
    tickers = CONFIG.get("markets", {}).get("base_tickers", ["XRP", "ADA"])
    exchange = ccxt.kucoin()
    state = {}

    while True:
        try:
            for symbol in tickers:
                full_symbol = f"{symbol}/USDT:USDT"
                ohlcv = exchange.fetch_ohlcv(full_symbol, timeframe=TIMEFRAME, limit=100)
                if not ohlcv: continue
                
                last_candle_ts = ohlcv[-1][0]
                symbol_state = state.get(symbol, {"last_candle_ts": 0, "last_signal": "NO_TRADE"})
                
                if last_candle_ts > symbol_state["last_candle_ts"]:
                    info = generar_senal(ohlcv, symbol_state["last_signal"])
                    
                    if info["senal"] != "NO_TRADE" and info["senal"] != symbol_state["last_signal"]:
                        tam, ries = calcular_posicion(info["precio_entrada"], info["stop_loss"])
                        chart = generar_grafico(full_symbol, ohlcv, info)
                        
                        msg = (f"🌊 **NEXTWAVE SIGNAL: {info['senal']}**\n"
                               f"🪙 Símbolo: {symbol}\n"
                               f"💰 Entrada: {info['precio_entrada']}\n"
                               f"🛡️ SL: {info['stop_loss']} | 🎯 TP: {info['take_profit']}\n"
                               f"📊 Score: {info['score']}\n"
                               f"📝 Motivo: {', '.join(info['razones'])}")
                        
                        enviar_mensaje(msg, chart)
                        symbol_state["last_signal"] = info["senal"]
                    
                    symbol_state["last_candle_ts"] = last_candle_ts
                    state[symbol] = symbol_state
                
                time.sleep(1)
            time.sleep(UPDATE_INTERVAL)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

# --- FLASK ---
app = Flask(__name__)
@app.route("/")
def home(): return "Bot NextWave activo", 200

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()
    main_loop()
