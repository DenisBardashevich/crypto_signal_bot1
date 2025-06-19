# Конфигурация для crypto_signal_bot (ОПТИМИЗИРОВАННАЯ для 15+ сигналов/день)
# Цель: 15+ качественных сигналов в день с закрытием по TP

# --- Таймфреймы ---
TIMEFRAME        = '15m'     # основной ТФ для сигналов (15 минут)
BACKUP_TIMEFRAME = '1h'      # 1ч для подтверждения тренда

# --- EMA периоды (более консервативные для качества) ---
MA_FAST = 16    # увеличено с 12 до 16 для более стабильных сигналов
MA_SLOW = 34    # увеличено с 26 до 34 для лучшего тренда

# --- Лимиты данных ---
LIMIT = 400     # ~4 дня истории на 15м

# --- RSI (более строгие фильтры) ---
RSI_WINDOW = 14
RSI_OVERSOLD = 30    # повышено с 25 до 30 для избежания ложных сигналов
RSI_OVERBOUGHT = 70  # понижено с 75 до 70
RSI_EXTREME_OVERSOLD = 20  # новый фильтр для экстремальных условий
RSI_EXTREME_OVERBOUGHT = 80

# --- ATR (увеличиваем расстояния для TP/SL) ---
ATR_WINDOW = 14
TRAIL_ATR_MULT = 2.0  # увеличено с 1.5

# --- Bollinger Bands ---
BB_WINDOW = 20
BB_STD_DEV = 2.0
BB_SQUEEZE_THRESHOLD = 0.06

# --- MACD ---
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- VWAP (более строгие фильтры) ---
USE_VWAP = True
VWAP_DEVIATION_THRESHOLD = 0.012  # уменьшено с 1.5% до 1.2%

# --- Объём (сбалансированные требования) ---
MIN_VOLUME_USDT = 600_000  # снижено с 800К до 600К - больше доступных пар

# --- Частота сигналов (увеличиваем кулдаун) ---
SIGNAL_COOLDOWN_MINUTES = 60  # увеличено с 45 до 60 для избежания ложных сигналов

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# --- Take Profit и Stop Loss (ОПТИМИЗИРОВАННЫЕ ДЛЯ 65% TP) ---
TP_ATR_MULT = 0.9   # УМЕНЬШЕНО с 1.0 до 0.9 - максимально агрессивные TP
SL_ATR_MULT = 1.9   # УВЕЛИЧЕНО с 1.8 до 1.9 - еще больше пространства
TP_MIN = 0.005      # уменьшено с 0.006 до 0.005 - ультра близкие цели
TP_MAX = 0.012      # УМЕНЬШЕНО с 0.014 до 0.012 - быстрые мини-прибыли
SL_MIN = 0.012      # увеличено с 0.011 до 0.012 - больше терпения
SL_MAX = 0.024      # увеличено с 0.022 до 0.024 - максимальное терпение

# --- Минимальное расстояние между TP и SL ---
MIN_TP_SL_DISTANCE = 0.008  # уменьшено с 0.010 для более гибких настроек

# --- Fee Rate ---
FEE_RATE = 0.0006

# --- СМЯГЧЕННЫЕ фильтры для 15+ сигналов ---
MIN_ADX = 10          # снижено с 12 до 10 - очень мягкое требование
MAX_SPREAD_PCT = 0.010  # увеличено с 0.008 до 0.010 - больше пар
VOLUME_SPIKE_MULT = 1.5  # уменьшено с 2.0 до 1.5
VOLUME_BOOST_THRESHOLD = 2.0  # уменьшено с 2.5 до 2.0
MACD_SIGNAL_WINDOW = 9

# Минимум свечей для анализа
MIN_15M_CANDLES = 80  # уменьшено со 120 до 80

# СНИЖЕН минимальный скор для большего количества сигналов
MIN_COMPOSITE_SCORE = 4.5  # КРИТИЧНО снижено с 6.0 до 4.5 - больше сигналов!

# --- Пороги триггеров (ОЧЕНЬ МЯГКИЕ) ---
MIN_TRIGGERS_ACTIVE_HOURS = 0.3   # снижено с 0.4 - еще мягче
MIN_TRIGGERS_INACTIVE_HOURS = 0.4  # снижено с 0.5 - еще мягче

# --- Адаптивные настройки по времени ---
ACTIVE_HOURS_UTC = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
ACTIVE_HOURS_MULTIPLIER = 0.90  # более консервативно

# --- Система адаптации к волатильности ---
VOLATILITY_LOOKBACK = 48
HIGH_VOLATILITY_THRESHOLD = 0.025   # более строго
LOW_VOLATILITY_THRESHOLD = 0.015

# При высокой волатильности - мягкие фильтры
HIGH_VOL_ADX_MIN = 20  # снижено с 25 до 20
HIGH_VOL_RSI_EXTREME = 15  # смягчено с 10 до 15

# При низкой волатильности - супер мягкие фильтры  
LOW_VOL_ADX_MIN = 8   # снижено с 10 до 8
LOW_VOL_RSI_RANGE = 8   # увеличено с 5 до 8

# --- Дополнительные строгие фильтры ---
MIN_MOMENTUM = 0.045   # увеличено с 0.035
MAX_BB_WIDTH = 0.08    # уменьшено с 0.10

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
MTF_CONFLUENCE_WEIGHT = 0.8  # увеличено с 0.5

# --- УЛУЧШЕННАЯ система скоринга (акцент на качество) ---
WEIGHT_RSI = 1.8       # увеличено
WEIGHT_MACD = 2.0      # увеличено - главный драйвер
WEIGHT_BB = 1.5        # увеличено
WEIGHT_VWAP = 1.2      # увеличено
WEIGHT_VOLUME = 1.8    # увеличено
WEIGHT_ADX = 2.2       # максимально увеличено - тренд критичен

# --- УСИЛЕННАЯ система скоринга для SHORT (они работают лучше) ---
SHORT_BOOST_MULTIPLIER = 2.2  # увеличено с 1.8 до 2.2 - SHORT отлично работают!
LONG_PENALTY_IN_DOWNTREND = 0.3  # усилен штраф с 0.4 до 0.3

# --- Дополнительные фильтры качества ---
MIN_CANDLE_BODY_PCT = 0.4    # увеличено с 0.3
MAX_WICK_TO_BODY_RATIO = 2.5  # уменьшено с 3.0
MIN_VOLUME_MA_RATIO = 1.5     # увеличено с 1.2

# --- Фильтр времени (избегаем низкую активность) ---
AVOID_WEEKEND_SIGNALS = True  # избегаем сигналов в выходные
MIN_MARKET_ACTIVITY_SCORE = 0.7  # увеличено с 0.6

# --- НОВЫЕ ЭКСТРЕМАЛЬНЫЕ ФИЛЬТРЫ ---
MIN_EMA_SEPARATION = 0.008  # минимальное расстояние между EMA (0.8%)
MAX_RSI_VOLATILITY = 15     # максимальная волатильность RSI за 5 свечей
MIN_VOLUME_CONSISTENCY = 0.8  # минимальная консистентность объема
REQUIRE_MACD_HISTOGRAM_CONFIRMATION = True  # требуем подтверждение гистограммы MACD 