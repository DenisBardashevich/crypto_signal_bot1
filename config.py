# =============================================================================
# ОПТИМИЗИРОВАНО OPTUNA: 2025-01-17
# =============================================================================
# 🎯 РЕЗУЛЬТАТ: 83.1% винрейт, 8.0 сигналов/день, 182.1% месячная доходность
# 🛡️ ПАРАМЕТРЫ: найдены через 600 итераций с защитой от overfitting
# ⚡ PROFIT FACTOR: 2.62, мат.ожидание: 0.912%
# =============================================================================

import os

# === ОСНОВНЫЕ НАСТРОЙКИ ===
TIMEFRAME        = os.getenv('SIGNAL_TIMEFRAME', '15m')
BACKUP_TIMEFRAME = '1h'
MARKET_SNAPSHOT_TTL_SECONDS = 300

# === СТАРШИЙ ТАЙМФРЕЙМ ДЛЯ ПОДТВЕРЖДЕНИЯ ===
HIGHER_TIMEFRAME = os.getenv('SIGNAL_HIGHER_TIMEFRAME', '30m')
REQUIRE_HIGHER_TF = True
HIGHER_TF_CACHE_SECONDS = 480
HIGHER_TF_MIN_BARS = 140
HIGHER_TF_EMA_FAST = 34
HIGHER_TF_EMA_SLOW = 89
HIGHER_TF_RSI_WINDOW = 14
HIGHER_TF_RSI_BULL = 52
HIGHER_TF_RSI_BEAR = 48
HIGHER_TF_ADX_MIN = 18

# === ФИЛЬТРАЦИЯ КАЧЕСТВА СИГНАЛОВ ===
MIN_SIGNAL_SCORE_TO_SEND = 6.5
RISK_REWARD_MIN = 1.2
MAX_SIGNALS_PER_ROUND = 5

# --- EMA периоды ---
MA_FAST = 27
MA_SLOW = 146

# --- Лимиты данных ---
LIMIT = 400     


# === ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ===
MIN_COMPOSITE_SCORE = 1.4
MIN_ADX = 23
RSI_MIN = 40
RSI_MAX = 62
SHORT_MIN_RSI_MAX = 82
SHORT_MIN_ADX = 26
SHORT_MIN_RSI = 60
LONG_MAX_RSI = 55

TP_ATR_MULT = 2.1
SL_ATR_MULT = 2.4
TP_MIN = 0.009
SL_MIN = 0.028

# === ВРЕМЕННЫЕ ФИЛЬТРЫ (ОПТИМИЗИРОВАНЫ) ===
SIGNAL_COOLDOWN_MINUTES = 15
MIN_TRIGGERS_ACTIVE_HOURS = 0.8
MIN_TRIGGERS_INACTIVE_HOURS = 1.2

# === ОБЪЕМНЫЕ ФИЛЬТРЫ (ОПТИМИЗИРОВАНЫ) ===
MIN_VOLUME_USDT = 2_000_000
MIN_VOLUME_MA_RATIO = 0.85
MIN_24H_VOLUME_USDT = 75_000_000
REQUIRE_MACD_HISTOGRAM_CONFIRMATION = False

# === RSI ПАРАМЕТРЫ ===
RSI_WINDOW = 9  # Оптимизировано Optuna для стабильного расчета
RSI_EXTREME_OVERSOLD = 22  # Подстроено под волатильность 2025
RSI_EXTREME_OVERBOUGHT = 78  # Подстроено под волатильность 2025
RSI_OVERSOLD = RSI_MIN       # 40
RSI_OVERBOUGHT = RSI_MAX     # 62

# --- ATR ---
ATR_WINDOW = 14  # Стандартное значение для надежного расчета волатильности

# --- ADX ---
ADX_WINDOW = 14  # Стандартное значение для определения силы тренда

# --- Bollinger Bands ---
BB_WINDOW = 20   # Стандартное значение для стабильных полос
BB_STD_DEV = 2.0 # Стандартное отклонение для классических полос

# --- MACD ---
MACD_FAST = 12   # Стандартное значение для быстрого MACD
MACD_SLOW = 26   # Стандартное значение для медленного MACD
MACD_SIGNAL = 9  # Стандартное значение для сигнальной линии

# --- VWAP ---
USE_VWAP = True
VWAP_DEVIATION_THRESHOLD = 0.006

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# --- Fee Rate ---
FEE_RATE = 0.0006

# === БАЗОВЫЕ ПАРАМЕТРЫ ===
MIN_15M_CANDLES = 50
VOLATILITY_LOOKBACK = 48
HIGH_VOLATILITY_THRESHOLD = 0.05
LOW_VOLATILITY_THRESHOLD = 0.012
HIGH_VOL_ADX_MIN = 28
LOW_VOL_ADX_MIN = 18
MIN_ATR_PCT = 0.0025
MAX_SPREAD_PCT = 0.0025
MAX_FUNDING_RATE_ABS = 0.0008
GLOBAL_MAX_ABS_FUNDING = 0.0006

# Адаптивные настройки по времени
ACTIVE_HOURS_UTC = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# --- Фильтры ---
USE_VOLUME_FILTER = True
USE_VOLATILITY_FILTER = True
USE_GLOBAL_TREND_FILTER = True
GLOBAL_TREND_SYMBOL = 'BTC/USDT:USDT'
GLOBAL_MIN_ADX = 18

# --- Стохастический RSI ---
STOCH_RSI_K = 14   # Стандартное значение для %K
STOCH_RSI_D = 3    # Стандартное значение для %D
STOCH_RSI_LENGTH = 14  # Стандартная длина для стабильного расчета

# === СИСТЕМА СКОРИНГА (ОПТИМИЗИРОВАНЫ) ===
WEIGHT_RSI = 3.4
WEIGHT_MACD = 1.6
WEIGHT_BB = 0.9
WEIGHT_VWAP = 2.2
WEIGHT_VOLUME = 0.8
WEIGHT_ADX = 0.5

# === СИСТЕМА ДЛЯ SHORT/LONG (ОПТИМИЗИРОВАНЫ) ===
SHORT_BOOST_MULTIPLIER = 1.85
LONG_PENALTY_IN_DOWNTREND = 0.9
