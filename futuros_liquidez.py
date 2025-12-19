settings:
  timeframe: "15m"
  timeframe_candidates: ["15m", "1h", "5m"]   # el bot elige el mejor; desempata a favor de 15m
  update_interval: 30
  timezone: "Europe/Madrid"
  daily_summary_at: "00:00"
  log_dir: "logs"
  max_state_size: 300

markets:
  base_tickers:
    - XRP
    - ADA
    - NEAR
    - WLD
    - FIL
    - ARB
    - WIF
    - PEPE
    - HBAR

strategy:
  # Solo señales de alta convicción
  min_score_for_entry: 85

  # SL/TP por ATR (más natural que % fijo), con caps por % (por apalancamiento 10x)
  atr_period: 14
  atr_sl_mult: 1.2
  atr_tp_mult: 2.0
  max_sl_pct: 0.02   # SL máximo 2% desde entrada
  max_tp_pct: 0.05   # TP máximo 5% desde entrada

  # Anti-chop / confirmación
  adx_period: 14
  min_adx: 20
  rsi_period: 14
  rsi_long_min: 52
  rsi_short_max: 48

  # Evitar 2 señales seguidas del mismo ticker (salvo CANCEL por cambio de sesgo)
  signal_cooldown_min: 30

notifications:
  telegram:
    enabled: true
