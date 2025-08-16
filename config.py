# =============================================================================
# ОПТИМИЗИРОВАНО OPTUNA: 2025-08-16
# =============================================================================
# 🎯 РЕЗУЛЬТАТ: 80.3% винрейт, 6.3 сигналов/день, 84% месячная доходность
# 🛡️ ПАРАМЕТРЫ: найдены через 600 итераций с требованием 5+ сигналов
# ⚡ PROFIT FACTOR: 1.80, мат.ожидание: 0.575%
# =============================================================================

# === ОСНОВНЫЕ НАСТРОЙКИ ===
TIMEFRAME        = '15m'     
BACKUP_TIMEFRAME = '1h'      

# --- EMA периоды ---
MA_FAST = 27
MA_SLOW = 146

# --- Лимиты данных ---
LIMIT = 400     

# === ПАРАМЕТРЫ (ОПТИМИЗИРОВАНЫ OPTUNA) ===
MIN_COMPOSITE_SCORE = 4.0
MIN_SCORE = 2.0  
MIN_ADX = 21
RSI_MIN = 15
RSI_MAX = 65
SHORT_MIN_ADX = 23
SHORT_MIN_RSI = 80
LONG_MAX_RSI = 30

# === TP/SL (ОПТИМИЗИРОВАНЫ) ===
TP_ATR_MULT = 2.5
SL_ATR_MULT = 1.9

# === ОБЪЕМНЫЕ ФИЛЬТРЫ ===
MIN_VOLUME_USDT = 0.0001  

# === RSI ПАРАМЕТРЫ ===
RSI_WINDOW = 8
RSI_OVERSOLD = RSI_MIN       
RSI_OVERBOUGHT = RSI_MAX     
RSI_EXTREME_OVERSOLD = 12
RSI_EXTREME_OVERBOUGHT = 89

# --- ATR ---
ATR_WINDOW = 41
TRAIL_ATR_MULT = 7.3

# --- Bollinger Bands ---
BB_WINDOW = 10
BB_STD_DEV = 5.8
# УДАЛЕНО: BB_SQUEEZE_THRESHOLD, MIN_BB_WIDTH - не используются  

# --- MACD ---
MACD_FAST = 18
MACD_SLOW = 38
MACD_SIGNAL = 18
# УДАЛЕНО: MACD_SIGNAL_WINDOW - не используется

# --- VWAP ---
USE_VWAP = True
VWAP_DEVIATION_THRESHOLD = 0.5  

# === ВРЕМЕННЫЕ ФИЛЬТРЫ (ОПТИМИЗИРОВАНЫ) ===
SIGNAL_COOLDOWN_MINUTES = 15
MIN_TRIGGERS_ACTIVE_HOURS = 1.9
MIN_TRIGGERS_INACTIVE_HOURS = 2.1

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# === TP/SL НАСТРОЙКИ (ОПТИМИЗИРОВАНЫ) ===
TP_MIN = 0.014
SL_MIN = 0.034
# УДАЛЕНО: TP_MAX, SL_MAX, MIN_TP_SL_DISTANCE - не используются

# --- Fee Rate ---
FEE_RATE = 0.0006

# === ОБЪЕМНЫЕ ФИЛЬТРЫ (ОПТИМИЗИРОВАНЫ) ===
MIN_VOLUME_MA_RATIO = 1.5
REQUIRE_MACD_HISTOGRAM_CONFIRMATION = False
# УДАЛЕНО: MAX_SPREAD_PCT, MIN_CANDLE_BODY_PCT, MAX_WICK_TO_BODY_RATIO, 
#          MIN_VOLUME_CONSISTENCY, MAX_RSI_VOLATILITY - не используются  

# === НЕЙТРАЛИЗОВАННЫЕ ПАРАМЕТРЫ ===
MIN_15M_CANDLES = 50  

VOLATILITY_LOOKBACK = 48  
HIGH_VOLATILITY_THRESHOLD = 0.99   
LOW_VOLATILITY_THRESHOLD = 0.001   
HIGH_VOL_ADX_MIN = 1    
LOW_VOL_ADX_MIN = 1     

# Адаптивные настройки по времени
ACTIVE_HOURS_UTC = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# --- Фильтры ---
USE_VOLUME_FILTER = True
USE_VOLATILITY_FILTER = True

# --- Стохастический RSI ---
STOCH_RSI_K = 13
STOCH_RSI_D = 14
STOCH_RSI_LENGTH = 4
# УДАЛЕНО: STOCH_RSI_SMOOTH - не используется

# === СИСТЕМА СКОРИНГА (ОПТИМИЗИРОВАНЫ) ===
WEIGHT_RSI = 4.0
WEIGHT_MACD = 6.5
WEIGHT_BB = 2.5
WEIGHT_VWAP = 5.5
WEIGHT_VOLUME = 5.0
WEIGHT_ADX = 8.0

# === СИСТЕМА ДЛЯ SHORT/LONG (ОПТИМИЗИРОВАНЫ) ===
SHORT_BOOST_MULTIPLIER = 1.2
LONG_PENALTY_IN_DOWNTREND = 0.35
