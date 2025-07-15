
# =============================================================================
# ОБНОВЛЕНО ИСПРАВЛЕННЫМ ОПТИМИЗАТОРОМ: 2025-07-16 00:54:16
# =============================================================================
# 🎯 РЕАЛИСТИЧНЫЕ РЕЗУЛЬТАТЫ:
#   📊 Winrate: 68.0%
#   💰 Мат. ожидание: 0.003%
#   📈 TP/SL Count Ratio: 2.12
#   ⚡ Сигналов/день: 39.0
#   🎯 TP: 221, SL: 104
#   💸 Чистый TP: 0.845%, Чистый SL: -1.787%
# =============================================================================

# 🛡️ СИНХРОНИЗИРОВАННЫЙ КОНФИГ С ОПТИМИЗАТОРОМ
# Обновлено: 2025-01-27 16:00:00
# ИСПРАВЛЕНЫ ЭКСТРЕМАЛЬНЫЕ ПАРАМЕТРЫ TP/SL
# УДАЛЕНЫ НЕСИНХРОНИЗИРОВАННЫЕ ПАРАМЕТРЫ

# === ОСНОВНЫЕ НАСТРОЙКИ ===
TIMEFRAME        = '15m'     # основной ТФ для сигналов (15 минут)
BACKUP_TIMEFRAME = '1h'      # 1ч для подтверждения тренда

# --- EMA периоды ---
MA_FAST = 16
MA_SLOW = 40

# --- Лимиты данных ---
LIMIT = 400     # ~4 дня истории на 15м

# === ПАРАМЕТРЫ ДЛЯ ОПТИМИЗАЦИИ ===
MIN_COMPOSITE_SCORE = 3.0  # OPTUNA FIXED
MIN_SCORE = 6.5
MIN_ADX = 30
RSI_MIN = 40
RSI_MAX = 70
SHORT_MIN_ADX = 24
SHORT_MIN_RSI = 60
LONG_MAX_RSI = 55

# === ИСПРАВЛЕННЫЕ TP/SL ===
# КРИТИЧНО: Более реалистичные значения вместо экстремальных
TP_ATR_MULT = 1.4
SL_ATR_MULT = 2.7

# === ОБЪЕМНЫЕ ФИЛЬТРЫ ===
MIN_VOLUME_USDT = 0.01

# === RSI ПАРАМЕТРЫ ===
RSI_WINDOW = 24
RSI_OVERSOLD = RSI_MIN       
RSI_OVERBOUGHT = RSI_MAX     
RSI_EXTREME_OVERSOLD = 25
RSI_EXTREME_OVERBOUGHT = 80

# --- ATR ---
ATR_WINDOW = 16
TRAIL_ATR_MULT = 2.2

# --- Bollinger Bands ---
BB_WINDOW = 18
BB_STD_DEV = 2.3
BB_SQUEEZE_THRESHOLD = 0.15000000000000002
MIN_BB_WIDTH = 0.006

# --- MACD ---
MACD_FAST = 18
MACD_SLOW = 35
MACD_SIGNAL = 10
MACD_SIGNAL_WINDOW = 13

# --- VWAP ---
USE_VWAP = True
VWAP_DEVIATION_THRESHOLD = 0.018

# === ВРЕМЕННЫЕ ФИЛЬТРЫ ===
SIGNAL_COOLDOWN_MINUTES = 90
MIN_TRIGGERS_ACTIVE_HOURS = 1.4
MIN_TRIGGERS_INACTIVE_HOURS = 1.5

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# === TP/SL НАСТРОЙКИ ===
TP_MIN = 0.02
TP_MAX = 0.08   
SL_MIN = 0.01
SL_MAX = 0.15   

# --- Минимальное расстояние между TP и SL ---
MIN_TP_SL_DISTANCE = 0.01

# --- Fee Rate ---
FEE_RATE = 0.0006

# === ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ ===
MAX_SPREAD_PCT = 0.025
MIN_CANDLE_BODY_PCT = 0.2
MAX_WICK_TO_BODY_RATIO = 4.0
MIN_VOLUME_MA_RATIO = 0.8
MIN_VOLUME_CONSISTENCY = 0.6000000000000001
MAX_RSI_VOLATILITY = 20
REQUIRE_MACD_HISTOGRAM_CONFIRMATION = True

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
STOCH_RSI_K = 6
STOCH_RSI_D = 8
STOCH_RSI_LENGTH = 14
STOCH_RSI_SMOOTH = 4

# --- Мультитаймфреймовый анализ ---
USE_MULTI_TIMEFRAME = True
MTF_CONFLUENCE_WEIGHT = 2.2

# === СИСТЕМА СКОРИНГА ===
# Веса компонентов системы оценки сигналов
WEIGHT_RSI = 0.8
WEIGHT_MACD = 2.9
WEIGHT_BB = 1.5000000000000002
WEIGHT_VWAP = 0.7
WEIGHT_VOLUME = 3.8
WEIGHT_ADX = 2.0

# === СИСТЕМА ДЛЯ SHORT/LONG ===
SHORT_BOOST_MULTIPLIER = 0.8
LONG_PENALTY_IN_DOWNTREND = 0.9

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
