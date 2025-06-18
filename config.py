# Конфигурация для crypto_signal_bot (строгая фильтрация для надежных сигналов)
# Цель: 5-10 высококачественных сигналов в сутки

# --- Таймфреймы ---
TIMEFRAME        = '15m'     # основной ТФ для сигналов (15 минут)
BACKUP_TIMEFRAME = '1h'      # 1ч для подтверждения тренда

# --- EMA периоды (оптимизированы для качества, а не количества) ---
MA_FAST = 12    # увеличено с 9 до 12 для более стабильных сигналов
MA_SLOW = 26    # увеличено с 21 до 26 для лучшего тренда

# --- Лимиты данных ---
LIMIT = 400     # ~4 дня истории на 15м

# --- RSI (немного ослабляем) ---
RSI_WINDOW = 14
RSI_OVERSOLD = 20    # ослаблено с 18 до 20
RSI_OVERBOUGHT = 80  # ослаблено с 82 до 80

# --- ATR ---
ATR_WINDOW = 14
TRAIL_ATR_MULT = 1.5

# --- Bollinger Bands ---
BB_WINDOW = 20
BB_STD_DEV = 2.0
BB_SQUEEZE_THRESHOLD = 0.06  # оставляем строго

# --- MACD (оставляем как есть) ---
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- VWAP (немного ослабляем) ---
USE_VWAP = True
VWAP_DEVIATION_THRESHOLD = 0.015  # ослаблено с 1.2% до 1.5%

# --- Объём (оставляем строго) ---
MIN_VOLUME_USDT = 800_000  # оставляем строго для качества

# --- Частота сигналов ---
SIGNAL_COOLDOWN_MINUTES = 30

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# --- Take Profit и Stop Loss ---
TP_ATR_MULT = 1.8
SL_ATR_MULT = 0.9
TP_MIN = 0.008
TP_MAX = 0.025
SL_MIN = 0.005
SL_MAX = 0.015

# --- Минимальное расстояние между TP и SL ---
MIN_TP_SL_DISTANCE = 0.006

# --- Fee Rate ---
FEE_RATE = 0.0006

# --- Строгие фильтры сигналов (ослабляем умеренно) ---
MIN_ADX = 25  # ослаблено с 30 до 25 - все еще сильные тренды
MAX_SPREAD_PCT = 0.008  # оставляем строго
VOLUME_SPIKE_MULT = 1.0
VOLUME_BOOST_THRESHOLD = 1.5
MACD_SIGNAL_WINDOW = 9

# Минимум свечей для анализа
MIN_15M_CANDLES = 150

# КРИТИЧЕСКИ ВАЖНО: минимальный композитный скор для сигнала
MIN_COMPOSITE_SCORE = 7.5  # Только очень сильные сигналы (7.5 и выше)

# --- Адаптивные настройки по времени ---
ACTIVE_HOURS_UTC = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
ACTIVE_HOURS_MULTIPLIER = 0.98

# --- Система адаптации к волатильности ---
VOLATILITY_LOOKBACK = 48
HIGH_VOLATILITY_THRESHOLD = 0.02
LOW_VOLATILITY_THRESHOLD = 0.018

# При высокой волатильности - строгие фильтры
HIGH_VOL_ADX_MIN = 30  # ослаблено с 35 до 30
HIGH_VOL_RSI_EXTREME = 12  # ослаблено с 8 до 12

# При низкой волатильности - умеренные фильтры
LOW_VOL_ADX_MIN = 20  # ослаблено с 25 до 20
LOW_VOL_RSI_RANGE = 5  # ослаблено с 2 до 5

# --- Дополнительные ОЧЕНЬ строгие фильтры ---
MIN_MOMENTUM = 0.03   # увеличено
MAX_BB_WIDTH = 0.12   # еще строже

# --- Фильтры ---
USE_VOLUME_FILTER = True
USE_VOLATILITY_FILTER = True

# --- Стохастический RSI ---
STOCH_RSI_K = 3
STOCH_RSI_D = 3
STOCH_RSI_LENGTH = 14
STOCH_RSI_SMOOTH = 3

# --- Дивергенции ---
USE_DIVERGENCE_DETECTION = True
DIVERGENCE_LOOKBACK = 20

# --- Мультитаймфреймовый анализ ---
USE_MULTI_TIMEFRAME = True
MTF_CONFLUENCE_WEIGHT = 0.6  # еще больше вес подтверждения

# --- ОЧЕНЬ СТРОГАЯ система скоринга ---
WEIGHT_RSI = 1.0
WEIGHT_MACD = 1.4      # еще больше вес
WEIGHT_BB = 1.3        # еще больше вес
WEIGHT_VWAP = 1.6      # еще больше вес VWAP
WEIGHT_VOLUME = 1.1    # еще больше вес объёма
WEIGHT_ADX = 1.3       # еще больше вес ADX

# --- Адаптивные настройки по времени ---
ACTIVE_HOURS_UTC = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
ACTIVE_HOURS_MULTIPLIER = 0.98  # почти без изменений

# --- Система адаптации к волатильности ---
VOLATILITY_LOOKBACK = 48
HIGH_VOLATILITY_THRESHOLD = 0.02   # снижено - более строго
LOW_VOLATILITY_THRESHOLD = 0.018   # увеличено

# При высокой волатильности - МАКСИМАЛЬНО строгие фильтры
HIGH_VOL_ADX_MIN = 35  # еще строже с 32 до 35
HIGH_VOL_RSI_EXTREME = 8  # только критические уровни

# При низкой волатильности - строгие фильтры
LOW_VOL_ADX_MIN = 25  # увеличено с 22 до 25
LOW_VOL_RSI_RANGE = 2  # очень узкий диапазон 