# Конфигурация для crypto_signal_bot

# --- Таймфреймы ---
TIMEFRAME        = '15m'     # основной ТФ для сигналов
BACKUP_TIMEFRAME = '4h'      # старший ТФ для фильтра тренда

# --- EMA периоды ---
MA_FAST = 9    # быстрая EMA
MA_SLOW = 21    # медленная EMA

# --- Лимиты данных ---
LIMIT = 400     # количество свечей для анализа (должно быть > MA_SLOW*6)

# --- RSI ---
RSI_WINDOW = 14
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60

# --- ATR ---
ATR_WINDOW = 20
TRAIL_ATR_MULT = 2.5   # множитель для trailing-стопа

# --- Объём ---
MIN_VOLUME_USDT = 1_000_000  # минимальный объём торгов за 24ч (USDT)

# --- Signal Cooldown ---
SIGNAL_COOLDOWN_MINUTES = 15  # уменьшаем кулдаун для 15м

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# --- Take Profit ---
TP_MIN = 0.02   # минимум 2%
TP_MAX = 0.25   # максимум 25%
TP_ATR_MULT = 2.5

# --- Stop Loss ---
SL_MIN = 0.02   # минимум 2%
SL_MAX = 0.20   # максимум 20%
SL_ATR_MULT = 1.8

# --- Минимальное расстояние между TP и SL ---
MIN_TP_SL_DISTANCE = 0.01  # минимум 1% между TP и SL

# --- Fee Rate ---
FEE_RATE = 0.0006  # комиссия биржи (0.06%)

# --- Фильтры сигналов ---
MIN_ADX = 15  # уменьшаем минимальное значение ADX для более мягкого фильтра
MAX_SPREAD_PCT = 0.008  # увеличиваем максимальный спред для более частых входов
VOLUME_SPIKE_MULT = 1.5  # множитель для объёма (1.5 = объём должен быть выше на 50% от среднего)
MACD_SIGNAL_WINDOW = 9  # период для сигнальной линии MACD

# --- Дополнительные индикаторы ---
USE_VOLUME_FILTER = True  # включаем фильтр по объёму
USE_VOLATILITY_FILTER = True  # включаем фильтр по волатильности 