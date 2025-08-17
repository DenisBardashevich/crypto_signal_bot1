# =============================================================================
# ОПТИМИЗИРОВАНО OPTUNA: 2025-01-17
# =============================================================================
# 🎯 РЕЗУЛЬТАТ: 83.1% винрейт, 8.0 сигналов/день, 182.1% месячная доходность
# 🛡️ ПАРАМЕТРЫ: найдены через 600 итераций с защитой от overfitting
# ⚡ PROFIT FACTOR: 2.62, мат.ожидание: 0.912%
# =============================================================================

# === ОСНОВНЫЕ НАСТРОЙКИ ===
TIMEFRAME        = '15m'     
BACKUP_TIMEFRAME = '1h'      

# --- EMA периоды ---
MA_FAST = 27
MA_SLOW = 146

# --- Лимиты данных ---
LIMIT = 400     

# === ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ===
MIN_COMPOSITE_SCORE = 0.5
MIN_ADX = 21
RSI_MIN = 15
RSI_MAX = 85
SHORT_MIN_ADX = 23
SHORT_MIN_RSI = 40
LONG_MAX_RSI = 45

# === TP/SL (ОПТИМИЗИРОВАНЫ) ===
TP_ATR_MULT = 1.6
SL_ATR_MULT = 4.2
TP_MIN = 0.008
SL_MIN = 0.034

# === ВРЕМЕННЫЕ ФИЛЬТРЫ (ОПТИМИЗИРОВАНЫ) ===
SIGNAL_COOLDOWN_MINUTES = 10
MIN_TRIGGERS_ACTIVE_HOURS = 1.9
MIN_TRIGGERS_INACTIVE_HOURS = 2.8

# === ОБЪЕМНЫЕ ФИЛЬТРЫ (ОПТИМИЗИРОВАНЫ) ===
MIN_VOLUME_USDT = 0.0001  
MIN_VOLUME_MA_RATIO = 1.2
REQUIRE_MACD_HISTOGRAM_CONFIRMATION = False

# === RSI ПАРАМЕТРЫ ===
RSI_WINDOW = 8
RSI_EXTREME_OVERSOLD = 12
RSI_EXTREME_OVERBOUGHT = 89
RSI_OVERSOLD = RSI_MIN       # 15 
RSI_OVERBOUGHT = RSI_MAX     # 85

# --- ATR ---
ATR_WINDOW = 41

# --- ADX ---
ADX_WINDOW = 14

# --- Bollinger Bands ---
BB_WINDOW = 10
BB_STD_DEV = 5.8

# --- MACD ---
MACD_FAST = 18
MACD_SLOW = 38
MACD_SIGNAL = 18

# --- VWAP ---
USE_VWAP = True
VWAP_DEVIATION_THRESHOLD = 0.5  

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# --- Fee Rate ---
FEE_RATE = 0.0006

# === БАЗОВЫЕ ПАРАМЕТРЫ ===
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

# === СИСТЕМА СКОРИНГА (ОПТИМИЗИРОВАНЫ) ===
WEIGHT_RSI = 6.5
WEIGHT_MACD = 7.5
WEIGHT_BB = 1.0
WEIGHT_VWAP = 10.0
WEIGHT_VOLUME = 1.0
WEIGHT_ADX = 8.0

# === СИСТЕМА ДЛЯ SHORT/LONG (ОПТИМИЗИРОВАНЫ) ===
SHORT_BOOST_MULTIPLIER = 2.4
LONG_PENALTY_IN_DOWNTREND = 0.3
