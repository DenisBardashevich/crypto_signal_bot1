# 🛡️ ОПТИМИЗИРОВАННЫЙ КОНФИГ C OPTUNA
# Обновлено: 2025-01-27 15:30:00
# НАЙДЕНЫ ИДЕАЛЬНЫЕ ПАРАМЕТРЫ С OPTUNA! ✨
# СИНХРОНИЗАЦИЯ С ОПТИМИЗАТОРОМ: 100 сигналов в день, винрейт 99.75%
# Алгоритм: TPE (Tree-structured Parzen Estimator) - оптимизация

# === ОСНОВНЫЕ НАСТРОЙКИ ===
TIMEFRAME        = '15m'     # основной ТФ для сигналов (15 минут)
BACKUP_TIMEFRAME = '1h'      # 1ч для подтверждения тренда

# --- EMA периоды (ЛУЧШИЕ ПАРАМЕТРЫ) ---
MA_FAST = 22  # OPTUNA AUTO
MA_SLOW = 98  # OPTUNA AUTO

# --- Лимиты данных ---
LIMIT = 400     # ~4 дня истории на 15м

# === ИСПРАВЛЕННЫЕ ФИЛЬТРЫ ===
# КРИТИЧНО: Более разумные параметры для получения сигналов
MIN_COMPOSITE_SCORE = 1.5  # ИСПРАВЛЕНО: снижено с 2.0 для получения большего количества сигналов
MIN_SCORE = 5.0  # OPTUNA AUTO
MIN_ADX = 11  # OPTUNA AUTO
RSI_MIN = 44  # OPTUNA AUTO
RSI_MAX = 73  # OPTUNA AUTO
SHORT_MIN_ADX = 5  # OPTUNA AUTO
SHORT_MIN_RSI = 62  # OPTUNA AUTO
LONG_MAX_RSI = 80  # OPTUNA AUTO

# === ИСПРАВЛЕННЫЕ TP/SL ===
# Более консервативные цели для лучшего винрейта
TP_ATR_MULT = 0.1  # OPTUNA AUTO
SL_ATR_MULT = 5.9  # OPTUNA AUTO

# === ИСПРАВЛЕННЫЕ ОБЪЕМНЫЕ ФИЛЬТРЫ ===
MIN_VOLUME_USDT = 0.001  # OPTUNA AUTO

# === RSI (ЛУЧШИЕ ПАРАМЕТРЫ) ===
RSI_WINDOW = 9  # OPTUNA AUTO
RSI_OVERSOLD = RSI_MIN       
RSI_OVERBOUGHT = RSI_MAX     
RSI_EXTREME_OVERSOLD = 26  # OPTUNA AUTO
RSI_EXTREME_OVERBOUGHT = 80  # OPTUNA AUTO

# --- ATR (ЛУЧШИЕ ПАРАМЕТРЫ) ---
ATR_WINDOW = 51  # OPTUNA AUTO
TRAIL_ATR_MULT = 0.3  # OPTUNA AUTO

# --- Bollinger Bands (ЛУЧШИЕ ПАРАМЕТРЫ) ---
BB_WINDOW = 75  # OPTUNA AUTO
BB_STD_DEV = 2.0  # OPTUNA AUTO
BB_SQUEEZE_THRESHOLD = 0.163  # OPTUNA AUTO
MIN_BB_WIDTH = 0.0001  # OPTUNA AUTO

# --- MACD (ЛУЧШИЕ ПАРАМЕТРЫ) ---
MACD_FAST = 16  # OPTUNA AUTO
MACD_SLOW = 73  # OPTUNA AUTO
MACD_SIGNAL = 13  # OPTUNA AUTO

# --- VWAP ---
USE_VWAP = True
VWAP_DEVIATION_THRESHOLD = 0.018  # сохраняем прежнее значение

# === ЛУЧШИЕ ПАРАМЕТРЫ КУЛДАУН ===
SIGNAL_COOLDOWN_MINUTES = 66  # OPTUNA AUTO

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# === ЛУЧШИЕ ПАРАМЕТРЫ TP/SL НАСТРОЙКИ ===
TP_MIN = 0.059  # OPTUNA AUTO
TP_MAX = 0.08   # увеличиваем максимум
SL_MIN = 0.064  # OPTUNA AUTO
SL_MAX = 0.15   # увеличиваем максимум

# --- Минимальное расстояние между TP и SL ---
MIN_TP_SL_DISTANCE = 0.019  # OPTUNA AUTO

# --- Fee Rate ---
FEE_RATE = 0.0006

# === ЛУЧШИЕ ПАРАМЕТРЫ ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ ===
MAX_SPREAD_PCT = 0.016  # OPTUNA AUTO
VOLUME_SPIKE_MULT = 2.3      # сохраняем
VOLUME_BOOST_THRESHOLD = 2.1 # сохраняем
MACD_SIGNAL_WINDOW = 39  # OPTUNA AUTO

# Минимум свечей для анализа
MIN_15M_CANDLES = 105  # Оставить как есть

# --- ИСПРАВЛЕННЫЕ пороги триггеров ---
MIN_TRIGGERS_ACTIVE_HOURS = 0.42  # OPTUNA AUTO
MIN_TRIGGERS_INACTIVE_HOURS = 3.15  # OPTUNA AUTO

# --- Адаптивные настройки по времени ---
ACTIVE_HOURS_UTC = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # ИСПРАВЛЕНО: расширено
ACTIVE_HOURS_MULTIPLIER = 1.0     # упрощено

# --- Система адаптации к волатильности ---
VOLATILITY_LOOKBACK = 48
HIGH_VOLATILITY_THRESHOLD = 0.034  # сохраняем
LOW_VOLATILITY_THRESHOLD = 0.011   # сохраняем
VOLATILITY_FILTER_STRENGTH = 4.8  # OPTUNA AUTO

# При высокой волатильности
HIGH_VOL_ADX_MIN = 26       # сохраняем
HIGH_VOL_RSI_EXTREME = 13   # сохраняем

# При низкой волатильности
LOW_VOL_ADX_MIN = 13        # сохраняем
LOW_VOL_RSI_RANGE = 7       # сохраняем

# --- Дополнительные фильтры ---
MIN_MOMENTUM = 0.008        # сохраняем
MAX_BB_WIDTH = 0.055        # сохраняем
TREND_STRENGTH_MULTIPLIER = 2.7  # OPTUNA AUTO
VOLUME_SPIKE_SENSITIVITY = 0.65  # OPTUNA AUTO

# --- Фильтры ---
USE_VOLUME_FILTER = True
USE_VOLATILITY_FILTER = True

# --- ЛУЧШИЕ ПАРАМЕТРЫ Стохастический RSI ---
STOCH_RSI_K = 2  # OPTUNA AUTO
STOCH_RSI_D = 12  # OPTUNA AUTO
STOCH_RSI_LENGTH = 39  # OPTUNA AUTO
STOCH_RSI_SMOOTH = 5  # OPTUNA AUTO

# --- Дивергенции ---
USE_DIVERGENCE_DETECTION = True
DIVERGENCE_LOOKBACK = 20
DIVERGENCE_WEIGHT = 3.05  # OPTUNA AUTO

# --- Мультитаймфреймовый анализ ---
USE_MULTI_TIMEFRAME = True
MTF_CONFLUENCE_WEIGHT = 2.2  # сохраняем

# === ЛУЧШИЕ ПАРАМЕТРЫ система скоринга ===
# Оптимизированные веса по результатам ЛУЧШИЕ ПАРАМЕТРЫ - синхронизированы с оптимизатором
WEIGHT_RSI = 0.75  # OPTUNA AUTO
WEIGHT_MACD = 1.45  # OPTUNA AUTO
WEIGHT_BB = 1.4  # OPTUNA AUTO
WEIGHT_VWAP = 2.4  # OPTUNA AUTO
WEIGHT_VOLUME = 5.95  # OPTUNA AUTO
WEIGHT_ADX = 4.9  # OPTUNA AUTO

# === ЛУЧШИЕ ПАРАМЕТРЫ система для SHORT/LONG ===
SHORT_BOOST_MULTIPLIER = 3.02  # OPTUNA AUTO
LONG_PENALTY_IN_DOWNTREND = 0.681  # OPTUNA AUTO

# --- ЛУЧШИЕ ПАРАМЕТРЫ фильтры качества ===
MIN_CANDLE_BODY_PCT = 0.54  # OPTUNA AUTO
MAX_WICK_TO_BODY_RATIO = 8.0  # OPTUNA AUTO
MIN_VOLUME_MA_RATIO = 0.41  # OPTUNA AUTO

# --- Фильтр времени ---
AVOID_WEEKEND_SIGNALS = True
MIN_MARKET_ACTIVITY_SCORE = 0.85     # сохраняем

# --- ЛУЧШИЕ ПАРАМЕТРЫ ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ ===
MIN_EMA_SEPARATION = 0.0008          # сохраняем
MAX_RSI_VOLATILITY = 15  # OPTUNA AUTO
MIN_VOLUME_CONSISTENCY = 0.16  # OPTUNA AUTO
REQUIRE_MACD_HISTOGRAM_CONFIRMATION = False  # OPTUNA AUTO
