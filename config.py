# 🛡️ СИНХРОНИЗИРОВАННЫЙ КОНФИГ С ОПТИМИЗАТОРОМ
# Обновлено: 2025-01-27 16:00:00
# ИСПРАВЛЕНЫ ЭКСТРЕМАЛЬНЫЕ ПАРАМЕТРЫ TP/SL
# УДАЛЕНЫ НЕСИНХРОНИЗИРОВАННЫЕ ПАРАМЕТРЫ

# === ОСНОВНЫЕ НАСТРОЙКИ ===
TIMEFRAME        = '15m'     # основной ТФ для сигналов (15 минут)
BACKUP_TIMEFRAME = '1h'      # 1ч для подтверждения тренда

# --- EMA периоды ---
MA_FAST = 22  # OPTUNA FIXED
MA_SLOW = 98  # OPTUNA FIXED

# --- Лимиты данных ---
LIMIT = 400     # ~4 дня истории на 15м

# === ПАРАМЕТРЫ ДЛЯ ОПТИМИЗАЦИИ ===
MIN_COMPOSITE_SCORE = 3.0  # OPTUNA FIXED
MIN_SCORE = 5.0  # OPTUNA FIXED
MIN_ADX = 11  # OPTUNA FIXED
RSI_MIN = 44  # OPTUNA FIXED
RSI_MAX = 73  # OPTUNA FIXED
SHORT_MIN_ADX = 5  # OPTUNA FIXED
SHORT_MIN_RSI = 62  # OPTUNA FIXED
LONG_MAX_RSI = 80  # OPTUNA FIXED

# === ИСПРАВЛЕННЫЕ TP/SL ===
# КРИТИЧНО: Более реалистичные значения вместо экстремальных
TP_ATR_MULT = 1.2  # OPTUNA FIXED
SL_ATR_MULT = 2.0  # OPTUNA FIXED

# === ОБЪЕМНЫЕ ФИЛЬТРЫ ===
MIN_VOLUME_USDT = 0.001  # OPTUNA FIXED

# === RSI ПАРАМЕТРЫ ===
RSI_WINDOW = 9  # OPTUNA FIXED
RSI_OVERSOLD = RSI_MIN       
RSI_OVERBOUGHT = RSI_MAX     
RSI_EXTREME_OVERSOLD = 26  # OPTUNA FIXED
RSI_EXTREME_OVERBOUGHT = 80  # OPTUNA FIXED

# --- ATR ---
ATR_WINDOW = 51  # OPTUNA FIXED
TRAIL_ATR_MULT = 0.3  # OPTUNA FIXED

# --- Bollinger Bands ---
BB_WINDOW = 75  # OPTUNA FIXED
BB_STD_DEV = 2.0  # OPTUNA FIXED
BB_SQUEEZE_THRESHOLD = 0.163  # OPTUNA FIXED
MIN_BB_WIDTH = 0.0001  # OPTUNA FIXED

# --- MACD ---
MACD_FAST = 16  # OPTUNA FIXED
MACD_SLOW = 73  # OPTUNA FIXED
MACD_SIGNAL = 13  # OPTUNA FIXED
MACD_SIGNAL_WINDOW = 39  # OPTUNA FIXED

# --- VWAP ---
USE_VWAP = True
VWAP_DEVIATION_THRESHOLD = 0.018

# === ВРЕМЕННЫЕ ФИЛЬТРЫ ===
SIGNAL_COOLDOWN_MINUTES = 66  # OPTUNA FIXED
MIN_TRIGGERS_ACTIVE_HOURS = 0.42  # OPTUNA FIXED
MIN_TRIGGERS_INACTIVE_HOURS = 3.15  # OPTUNA FIXED

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# === TP/SL НАСТРОЙКИ ===
TP_MIN = 0.059  # OPTUNA FIXED
TP_MAX = 0.08   
SL_MIN = 0.064  # OPTUNA FIXED
SL_MAX = 0.15   

# --- Минимальное расстояние между TP и SL ---
MIN_TP_SL_DISTANCE = 0.008  # OPTUNA FIXED

# --- Fee Rate ---
FEE_RATE = 0.0006

# === ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ ===
MAX_SPREAD_PCT = 0.016  # OPTUNA FIXED
MIN_CANDLE_BODY_PCT = 0.54  # OPTUNA FIXED
MAX_WICK_TO_BODY_RATIO = 8.0  # OPTUNA FIXED
MIN_VOLUME_MA_RATIO = 0.41  # OPTUNA FIXED
MIN_VOLUME_CONSISTENCY = 0.16  # OPTUNA FIXED
MAX_RSI_VOLATILITY = 15  # OPTUNA FIXED
REQUIRE_MACD_HISTOGRAM_CONFIRMATION = False  # OPTUNA FIXED

# --- Минимум свечей для анализа ---
MIN_15M_CANDLES = 105

# --- Адаптивные настройки по времени ---
ACTIVE_HOURS_UTC = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
ACTIVE_HOURS_MULTIPLIER = 1.0

# --- Система адаптации к волатильности ---
VOLATILITY_LOOKBACK = 48
HIGH_VOLATILITY_THRESHOLD = 0.034
LOW_VOLATILITY_THRESHOLD = 0.011

# При высокой волатильности
HIGH_VOL_ADX_MIN = 26
HIGH_VOL_RSI_EXTREME = 13

# При низкой волатильности
LOW_VOL_ADX_MIN = 13
LOW_VOL_RSI_RANGE = 7

# --- Дополнительные фильтры ---
MIN_MOMENTUM = 0.008
MAX_BB_WIDTH = 0.055

# --- Фильтры ---
USE_VOLUME_FILTER = True
USE_VOLATILITY_FILTER = True

# --- Стохастический RSI ---
STOCH_RSI_K = 2  # OPTUNA FIXED
STOCH_RSI_D = 12  # OPTUNA FIXED
STOCH_RSI_LENGTH = 39  # OPTUNA FIXED
STOCH_RSI_SMOOTH = 5  # OPTUNA FIXED

# --- Мультитаймфреймовый анализ ---
USE_MULTI_TIMEFRAME = True
MTF_CONFLUENCE_WEIGHT = 2.2

# === СИСТЕМА СКОРИНГА ===
# Веса компонентов системы оценки сигналов
WEIGHT_RSI = 0.75  # OPTUNA FIXED
WEIGHT_MACD = 1.45  # OPTUNA FIXED
WEIGHT_BB = 1.4  # OPTUNA FIXED
WEIGHT_VWAP = 2.4  # OPTUNA FIXED
WEIGHT_VOLUME = 5.95  # OPTUNA FIXED
WEIGHT_ADX = 4.9  # OPTUNA FIXED

# === СИСТЕМА ДЛЯ SHORT/LONG ===
SHORT_BOOST_MULTIPLIER = 3.02  # OPTUNA FIXED
LONG_PENALTY_IN_DOWNTREND = 0.681  # OPTUNA FIXED

# --- Фильтр времени ---
AVOID_WEEKEND_SIGNALS = True
MIN_MARKET_ACTIVITY_SCORE = 0.85

# --- ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ ---
MIN_EMA_SEPARATION = 0.0008

# УДАЛЕНЫ НЕСИНХРОНИЗИРОВАННЫЕ ПАРАМЕТРЫ:
# - VOLATILITY_FILTER_STRENGTH (есть только в оптимизаторе)
# - TREND_STRENGTH_MULTIPLIER (есть только в оптимизаторе) 
# - VOLUME_SPIKE_SENSITIVITY (есть только в оптимизаторе)
# - DIVERGENCE_WEIGHT (есть только в оптимизаторе)
# - VOLUME_SPIKE_MULT (есть только в config.py)
# - VOLUME_BOOST_THRESHOLD (есть только в config.py)
# - USE_DIVERGENCE_DETECTION (есть только в config.py)
# - DIVERGENCE_LOOKBACK (есть только в config.py)
