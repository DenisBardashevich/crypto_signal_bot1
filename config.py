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
RSI_OVERSOLD = 25    # ослаблено с 20 до 25 для более частых сигналов покупки
RSI_OVERBOUGHT = 75  # ослаблено с 80 до 75 для более частых сигналов продажи

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

# --- Объём (ослабляем для больше сигналов) ---
MIN_VOLUME_USDT = 500_000  # снижено с 800К для больше возможностей

# --- Частота сигналов ---
SIGNAL_COOLDOWN_MINUTES = 20  # снижено с 30 до 20 для более частых сигналов

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
MIN_ADX = 20  # ослаблено с 22 до 20 - принимаем умеренные тренды на 15м
MAX_SPREAD_PCT = 0.008  # оставляем строго
VOLUME_SPIKE_MULT = 1.0
VOLUME_BOOST_THRESHOLD = 1.5
MACD_SIGNAL_WINDOW = 9

# Минимум свечей для анализа
MIN_15M_CANDLES = 100  # снижено со 150 для более быстрого анализа

# КРИТИЧЕСКИ ВАЖНО: минимальный композитный скор для сигнала
MIN_COMPOSITE_SCORE = 6.0  # Снижено с 6.5 для достижения 10+ сигналов в день

# --- Адаптивные настройки по времени ---
ACTIVE_HOURS_UTC = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
ACTIVE_HOURS_MULTIPLIER = 0.95  # Больше снижение в активные часы

# --- Система адаптации к волатильности ---
VOLATILITY_LOOKBACK = 48
HIGH_VOLATILITY_THRESHOLD = 0.02
LOW_VOLATILITY_THRESHOLD = 0.018

# При высокой волатильности - строгие фильтры
HIGH_VOL_ADX_MIN = 25  # снижено с 28 до 25
HIGH_VOL_RSI_EXTREME = 18  # ослаблено с 15 до 18

# При низкой волатильности - мягкие фильтры
LOW_VOL_ADX_MIN = 15  # снижено с 18 до 15
LOW_VOL_RSI_RANGE = 10  # увеличено с 8 до 10

# --- Дополнительные ОЧЕНЬ строгие фильтры ---
MIN_MOMENTUM = 0.025   # снижено с 0.03
MAX_BB_WIDTH = 0.15   # ослаблено с 0.12

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
MTF_CONFLUENCE_WEIGHT = 0.5  # снижено с 0.6 для больше сигналов

# --- СБАЛАНСИРОВАННАЯ система скоринга для 15м ---
WEIGHT_RSI = 1.0       # Снижен с 1.2 для баланса
WEIGHT_MACD = 1.4      # Основной драйвер моментума
WEIGHT_BB = 1.0        # Увеличен с 0.8 для лучшего входа
WEIGHT_VWAP = 0.8      # Снижен с 1.0
WEIGHT_VOLUME = 1.2    # Увеличен - объём важен для фьючерсов
WEIGHT_ADX = 1.1       # Снижен с 1.3

# --- Адаптивные настройки по времени ---
ACTIVE_HOURS_UTC = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
ACTIVE_HOURS_MULTIPLIER = 0.95  # Больше снижение в активные часы

# --- Система адаптации к волатильности ---
VOLATILITY_LOOKBACK = 48
HIGH_VOLATILITY_THRESHOLD = 0.022   # увеличено - более строго
LOW_VOLATILITY_THRESHOLD = 0.016   # снижено

# При высокой волатильности - строгие фильтры
HIGH_VOL_ADX_MIN = 28  # снижено с 35
HIGH_VOL_RSI_EXTREME = 15  # ослаблено

# При низкой волатильности - мягкие фильтры
LOW_VOL_ADX_MIN = 18  # снижено с 25
LOW_VOL_RSI_RANGE = 8  # увеличено 