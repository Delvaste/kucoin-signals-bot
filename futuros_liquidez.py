import time
import os
import threading
import pandas as pd
import numpy as np
import talib
import yaml
import requests
import ccxt
from pathlib import Path
from flask import Flask

# =======================
# CONFIGURACIÓN CARGA
# =======================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")

def load_config():
    path = Path("config.yml")
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

CONFIG = load_config()

# Forzamos 15m para que sea funcional
TIMEFRAME = "15m" 
UPDATE_INTERVAL = CONFIG.get("settings", {}).get("update_interval", 30)
MIN_SCORE = 60  # Bajamos un poco el umbral para mayor sensibilidad

# --- TELEGRAM ---
def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error Telegram: {e}")

# --- LÓGICA NEXTWAVE ADAPTADA ---
def calcular_nextwave(df):
    # Indicadores rápidos para 15m
    df['ema20'] = talib.EMA(df['close'], timeperiod=20)
    df['ema50'] = talib.EMA(df['close'], timeperiod=50)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    # Estocástico más rápido (9,3,3)
    df['k'], df['d'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                 fastk_period=9, slowk_period=3, slowd_period=3)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    razones = []

    # 1. Filtro de Ola (Tendencia) - 40 puntos
    if last['close'] > last['ema20'] > last['ema50']:
        score += 40
        razones.append("🌊 Tendencia Alcista")
    elif last['close'] < last['ema20'] < last['ema50']:
        score -= 40
        razones.append("🌊 Tendencia Bajista")

    # 2. Filtro de Impulso (Cruce Estocástico) - 40 puntos
    # Long: Cruce arriba en sobreventa (<30)
    if last['k'] > last['d'] and prev['k'] <= prev['d'] and last['k'] < 30:
        score += 40
        razones.append("🚀 Cruce Estocástico Alcista")
    # Short: Cruce abajo en sobrecompra (>70)
    elif last['k'] < last['d'] and prev['k'] >= prev['d'] and last['k'] > 70:
        score -= 40
        razones.append("🔻 Cruce Estocástico Bajista")

    # 3. Filtro Volatilidad - 20 puntos
    if last['adx'] > 22:
        score = score + 20 if score > 0 else score - 20
        razones.append("🔥 Volatilidad Alta (ADX)")

    return score, razones, last['close']

# --- BUCLE PRINCIPAL ---
def main_loop():
    print(f"🚀 Bot NextWave 2.0 Iniciado [TF: {TIMEFRAME}]")
    exchange = ccxt.kucoinfutures({
        'apiKey': KUCOIN_API_KEY, 'secret': KUCOIN_API_SECRET, 'password': KUCOIN_API_PASSPHRASE,
        'enableRateLimit': True
    })

    exchange.load_markets()
    base_tickers = CONFIG.get("markets", {}).get("base_tickers", [])
    last_signals = {} # Para no repetir mensajes en la misma vela

    while True:
        for ticker in base_tickers:
            # IMPORTANTE: Espera de 1.5s para no saturar y permitir logs limpios
            time.sleep(1.5) 
            
            symbol = f"{ticker}USDTM"
            if symbol not in exchange.markets:
                symbol = f"{ticker}/USDT:USDT"
                if symbol not in exchange.markets:
                    print(f"❌ {ticker} no hallado.")
                    continue

            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
                
                score, razones, precio = calcular_nextwave(df)
                
                # Log de seguimiento en Fly.io
                print(f"🔍 {ticker.ljust(5)} | Score: {str(score).ljust(4)} | Precio: {precio}")

                current_ts = df.iloc[-1]['ts']
                
                if abs(score) >= MIN_SCORE:
                    signal_key = f"{symbol}_{current_ts}"
                    if signal_key not in last_signals:
                        tipo = "🟢 LONG" if score > 0 else "🔴 SHORT"
                        msg = (f"{tipo} #NextWave\n"
                               f"🪙 Moneda: {symbol}\n"
                               f"💰 Precio: {precio}\n"
                               f"📊 Score: {score}\n"
                               f"📝 {', '.join(razones)}")
                        enviar_telegram(msg)
                        last_signals[signal_key] = True
                        print(f"📣 SEÑAL ENVIADA: {symbol}")

            except Exception as e:
                print(f"⚠️ Error en {ticker}: {e}")

        print(f"✅ Ciclo terminado. Esperando {UPDATE_INTERVAL}s...")
        time.sleep(UPDATE_INTERVAL)

# --- FLASK HEALTHCHECK ---
app = Flask(__name__)
@app.route("/")
def health(): return "OK", 200

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()
    main_loop()
