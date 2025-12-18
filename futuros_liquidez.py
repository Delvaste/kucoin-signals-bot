import time
import os
import threading
import pandas as pd
import numpy as np
import talib
import yaml
import requests
import mplfinance as mpf
from pathlib import Path
from flask import Flask
import ccxt

# =======================
# CONFIGURACIÓN
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

# Parámetros extraídos de config.yml
TIMEFRAME = CONFIG.get("settings", {}).get("timeframe", "1h")
UPDATE_INTERVAL = CONFIG.get("settings", {}).get("update_interval", 60)
MIN_SCORE = CONFIG.get("strategy", {}).get("min_score_for_entry", 80) # Subimos a 80 para precisión
SL_PCT = CONFIG.get("strategy", {}).get("sl_pct", 0.05)
TP_PCT = CONFIG.get("strategy", {}).get("tp_pct", 0.10)

# --- NOTIFICACIONES ---
def enviar_mensaje(texto, chart_path=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
    try:
        if chart_path and Path(chart_path).exists():
            with open(chart_path, 'rb') as f:
                requests.post(url + "sendPhoto", 
                              data={'chat_id': TELEGRAM_CHAT_ID, 'caption': texto, 'parse_mode': 'Markdown'},
                              files={'photo': f}, timeout=15)
            os.remove(chart_path)
        else:
            requests.post(url + "sendMessage", 
                          data={'chat_id': TELEGRAM_CHAT_ID, 'text': texto, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"Error Telegram: {e}")

# --- GRÁFICOS ---
def generar_grafico(symbol, ohlcv):
    try:
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        
        df['ema20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema50'] = talib.EMA(df['close'], timeperiod=50)
        
        df_plot = df.tail(40)
        filename = f"chart_{symbol.split(':')[0].replace('/', '_')}.png"
        
        ap = [mpf.make_addplot(df_plot['ema20'], color='cyan'),
              mpf.make_addplot(df_plot['ema50'], color='magenta')]
        
        mpf.plot(df_plot, type='candle', addplot=ap, savefig=filename, style='charles', tight_layout=True)
        return filename
    except Exception as e:
        print(f"Error gráfico: {e}")
        return None

# --- LÓGICA DE CRUCE PRECISO ---
def detectar_cruce_y_score(ohlcv):
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ema20'] = talib.EMA(df['close'], timeperiod=20)
    df['ema50'] = talib.EMA(df['close'], timeperiod=50)
    df['k'], df['d'] = talib.STOCH(df['high'], df['low'], df['close'], 9, 3, 3)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    razones = []

    # 1. CRUCE DE TENDENCIA (EMA 20/50)
    # Solo damos puntos si el cruce es RECIENTE (en las últimas 2 velas)
    trend_bullish = last['ema20'] > last['ema50']
    trend_bearish = last['ema20'] < last['ema50']
    
    if trend_bullish:
        score += 50
        if prev['ema20'] <= prev['ema50']:
            score += 30 # Bonus por cruce fresco
            razones.append("🔥 CRUCE ALCISTA RECIENTE")
        else:
            razones.append("🌊 Tendencia Alcista")
            
    elif trend_bearish:
        score -= 50
        if prev['ema20'] >= prev['ema50']:
            score -= 30 # Bonus por cruce fresco
            razones.append("🔥 CRUCE BAJISTA RECIENTE")
        else:
            razones.append("🌊 Tendencia Bajista")

    # 2. MOMENTO (ESTOCÁSTICO)
    # Cruce de K sobre D en zonas extremas
    if last['k'] > last['d'] and prev['k'] <= prev['d'] and last['k'] < 30:
        score += 20
        razones.append("🚀 Impulso Alcista (Stoch)")
    elif last['k'] < last['d'] and prev['k'] >= prev['d'] and last['k'] > 70:
        score -= 20
        razones.append("🔻 Impulso Bajista (Stoch)")

    # Cálculo de niveles
    price = last['close']
    sl, tp, senal = 0, 0, "NO_TRADE"

    if score >= MIN_SCORE:
        senal = "LONG"
        sl = price * (1 - SL_PCT)
        tp = price * (1 + TP_PCT)
    elif score <= -MIN_SCORE:
        senal = "SHORT"
        sl = price * (1 + SL_PCT)
        tp = price * (1 - TP_PCT)

    return {"senal": senal, "score": score, "razones": razones, "price": price, "sl": sl, "tp": tp}

# --- LOOP PRINCIPAL ---
def main_loop():
    print("🤖 Bot NextWave Preciso Iniciado...")
    exchange = ccxt.kucoinfutures({
        'apiKey': KUCOIN_API_KEY, 'secret': KUCOIN_API_SECRET, 
        'password': KUCOIN_API_PASSPHRASE, 'enableRateLimit': True
    })
    exchange.load_markets()

    tickers = CONFIG.get("markets", {}).get("base_tickers", ["XRP"])
    # Estado para no repetir señales en la misma vela
    state = {}

    while True:
        for base in tickers:
            time.sleep(1.5)
            symbol = next((s for s in [f"{base}USDTM", f"{base}/USDT:USDT"] if s in exchange.markets), None)
            if not symbol: continue

            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
                if not ohlcv: continue
                
                res = detectar_cruce_y_score(ohlcv)
                last_ts = ohlcv[-1][0]
                
                # MONITOR EN CONSOLA
                print(f"🔍 {base.ljust(5)} | Score: {str(res['score']).ljust(4)} | Vela: {last_ts}")

                # EVITAR SPAM: Solo si es una vela nueva O una señal distinta
                signal_id = f"{symbol}_{last_ts}_{res['senal']}"
                if res['senal'] != "NO_TRADE" and signal_id not in state:
                    chart = generar_grafico(symbol, ohlcv)
                    emoji = "🟢" if res['senal'] == "LONG" else "🔴"
                    
                    msg = (
                        f"{emoji} **NUEVA ENTRADA: {res['senal']}**\n\n"
                        f"🪙 **Activo:** `{symbol}`\n"
                        f"💰 **Entrada:** `{res['price']:.5f}`\n"
                        f"🎯 **Objetivo:** `{res['tp']:.5f}`\n"
                        f"🛑 **Stop Loss:** `{res['sl']:.5f}`\n\n"
                        f"📊 **Score:** {res['score']}\n"
                        f"📝 **Motivos:** {', '.join(res['razones'])}"
                    )
                    enviar_mensaje(msg, chart)
                    state[signal_id] = True
                    # Limpiar estados antiguos para no saturar memoria
                    if len(state) > 100: state.clear()

            except Exception as e:
                print(f"Error en {base}: {e}")

        time.sleep(UPDATE_INTERVAL)

# --- FLASK ---
app = Flask(__name__)
@app.route("/")
def home(): return "Bot Online", 200

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()
    main_loop()
