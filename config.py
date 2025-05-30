# Конфигурация для crypto_signal_bot

# --- Таймфреймы ---
TIMEFRAME        = '5m'      # основной ТФ для сигналов
BACKUP_TIMEFRAME = '1h'      # старший ТФ для фильтра тренда

# --- EMA периоды ---
MA_FAST = 20    # быстрая EMA
MA_SLOW = 50    # медленная EMA

# --- Лимиты данных ---
LIMIT = 600     # количество свечей для анализа (должно быть > MA_SLOW*6)

# --- RSI ---
RSI_WINDOW = 14
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60

# --- ATR ---
ATR_WINDOW = 20
TRAIL_ATR_MULT = 2.5   # множитель для trailing-стопа

# --- Объём ---
MIN_VOLUME_USDT = 2_000_000  # минимальный объём торгов за 24ч (USDT)

# --- Signal Cooldown ---
SIGNAL_COOLDOWN_MINUTES = 10

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# --- Take Profit ---
TP_MIN = 0.007   # минимум 0.7%
TP_MAX = 0.15    # максимум 15%
TP_ATR_MULT = 3.0

# --- Stop Loss ---
SL_MIN = 0.007   # минимум 0.7%
SL_MAX = 0.15    # максимум 15%
SL_ATR_MULT = 2.0

# --- Fee Rate ---
FEE_RATE = 0.0006  # комиссия биржи (0.06%) 