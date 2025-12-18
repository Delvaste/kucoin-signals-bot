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
import mplfinance as mpf

# =======================
# CONFIGURACIÓN
# =======================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

CONFIG_PATH = Path("config.yml")

def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

CONFIG = load_config()

TIMEFRAME = CONFIG.get("settings", {}).get("timeframe", "1h")
UPDATE_INTERVAL = CONFIG.get("settings", {}).get("update_interval", 30)
MIN_SCORE = CONFIG.get("strategy", {}).get("min_score_for_entry", 70)

# --- FUNCIONES AUXILIARES ---

def enviar_mensaje(texto: str, chart_path: str = None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
    if chart_path and Path(chart_path).exists():
        try:
            with open(chart_path, 'rb') as f:
                requests.post(url + "sendPhoto", 
                              data={'chat_id': TELEGRAM_CHAT_ID, 'caption': texto, 'parse_mode': 'Markdown'},
                              files={'photo': f})
            os.remove(chart_path)
            return
        except Exception as e:
            print(f"Error enviando foto: {e}")
            
    requests.post(url + "sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': texto, 'parse_mode': 'Markdown'})

def generar_grafico(symbol: str, ohlcv: list) -> str | None:
    try:
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df['ema20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema50'] = talib.EMA(df['close'], timeperiod=50)
        df['k'], df['d'] = talib.STOCH(df['high'], df['low'], df['close'], 14, 5, 3)
        
        df_plot = df.tail(60).copy()
        filename = f"chart_{symbol.replace('/', '_').replace(':', '_')}.png"
        
        aplots = [
            mpf.make_addplot(df_plot['ema20'], color='blue'),
            mpf.make_addplot(df_plot['ema50'], color='orange'),
            mpf.make_addplot(df_plot['k'], color='purple', panel=2),
            mpf.make_addplot(df_plot['d'], color='red', panel=2)
        ]
        
        mpf.plot(df_plot, type='candle', addplot=aplots, savefig=filename, style='yahoo', figsize=(10, 7))
        return filename
    except Exception as e:
        print(f"Error gráfico {symbol}: {e}")
        return None

# --- ESTRATEGIA NEXTWAVE ---

def generar_senal(ohlcv: list) -> dict:
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['ema20'] = talib.EMA(df['close'], timeperiod=20)
    df['ema50'] = talib.EMA(df['close'], timeperiod=50)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['k'], df['d'] = talib.STOCH(df['high'], df['low'], df['close'], 14, 5, 3)
    
    last = df.iloc[-1]
    score = 0
    razones = []

    if last['close'] > last['ema20'] > last['ema50']:
        score += 50
        razones.append("🌊 Tendencia Alcista")
    elif last['close'] < last['ema20'] < last['ema50']:
        score -= 50
        razones.append("🌊 Tendencia Bajista")

    if not pd.isna(last['adx']) and last['adx'] > 20:
        if last['k'] < 25 and last['k'] > last['d']:
            score += 40
            razones.append("🚀 Impulso Alcista")
        elif last['k'] > 75 and last['k'] < last['d']:
            score -= 40
            razones.append("🔻 Impulso Bajista")

    senal = "NO_TRADE"
    if score >= MIN_SCORE: senal = "LONG"
    elif score <= -MIN_SCORE: senal = "SHORT"

    return {"senal": senal, "score": score, "razones": razones, "price": last['close']}

# --- BUCLE PRINCIPAL ---

def main_loop():
    print("Iniciando Bot estilo NextWave...")
    exchange = ccxt.kucoinfutures({
        'apiKey': KUCOIN_API_KEY,
        'secret': KUCOIN_API_SECRET,
        'password': KUCOIN_API_PASSPHRASE,
        'enableRateLimit': True
    })

    try:
        print("Cargando mercados de Kucoin Futures...")
        exchange.load_markets()
    except Exception as e:
        print(f"Error crítico cargando mercados: {e}")
        return

    tickers_config = CONFIG.get("markets", {}).get("base_tickers", [])
    state = {}

    while True:
        try:
            for base_symbol in tickers_config:
                time.sleep(1) # SIEMPRE esperar 1s para evitar bloqueos y spam en logs

                possible_symbols = [
                    f"{base_symbol}/USDT:USDT", 
                    f"{base_symbol}USDTM",
                    f"{base_symbol}/USDT"
                ]
    
                full_symbol = next((s for s in possible_symbols if s in exchange.markets), None)
                
                if not full_symbol:
                    print(f"⚠️ Símbolo {base_symbol} no encontrado. Saltando...")
                    continue

                try:
                    # Aumentado a 200 velas para mayor precisión en indicadores
                    ohlcv = exchange.fetch_ohlcv(full_symbol, timeframe=TIMEFRAME, limit=200)
                    if not ohlcv: continue
                    
                    last_ts = ohlcv[-1][0]
                    symbol_state = state.get(full_symbol, {"ts": 0, "signal": "NO_TRADE"})

                    info = generar_senal(ohlcv)
                    
                    # Log de monitoreo (se ve en fly logs)
                    print(f"🔍 {full_symbol} | Score: {info['score']} | Senal: {info['senal']}")

                    if last_ts > symbol_state["ts"]:
                        if info["senal"] != "NO_TRADE" and info["senal"] != symbol_state["signal"]:
                            chart = generar_grafico(full_symbol, ohlcv)
                            msg = (f"🌊 **NextWave: {info['senal']}**\n"
                                   f"🪙 {full_symbol}\n"
                                   f"💰 Precio: {info['price']}\n"
                                   f"📊 Score: {info['score']}\n"
                                   f"📝 {', '.join(info['razones'])}")
                            enviar_mensaje(msg, chart)
                            symbol_state["signal"] = info["senal"]

                        symbol_state["ts"] = last_ts
                        state[full_symbol] = symbol_state
                
                except Exception as e:
                    print(f"Error procesando {full_symbol}: {e}")
            
            print(f"✅ Ciclo completado. Esperando {UPDATE_INTERVAL}s...")
            time.sleep(UPDATE_INTERVAL)
            
        except Exception as e:
            print(f"Error en loop: {e}")
            time.sleep(10)

app = Flask(__name__)
@app.route("/")
def home(): return "Bot Active", 200

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()
    main_loop()
