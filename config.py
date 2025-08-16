# =============================================================================
# ИСПРАВЛЕН ДЛЯ ГЕНЕРАЦИИ НАДЕЖНЫХ СИГНАЛОВ: 2025-08-16
# =============================================================================
# 🎯 ЦЕЛЬ: Генерировать 8-15 качественных сигналов в день  
# 🛡️ ПРИНЦИП: Сохранить надежность, но сделать фильтры реалистичными
# ⚡ РЕЗУЛЬТАТ: лучшие параметры на текущем рынке
# =============================================================================

# === ОСНОВНЫЕ НАСТРОЙКИ ===
TIMEFRAME        = '15m'     
BACKUP_TIMEFRAME = '1h'      

# --- EMA периоды ---
MA_FAST = 27
MA_SLOW = 146

# --- Лимиты данных ---
LIMIT = 400     

# === ПАРАМЕТРЫ ===
MIN_COMPOSITE_SCORE = 2.0  
MIN_SCORE = 2.0  
MIN_ADX = 11
RSI_MIN = 45
RSI_MAX = 65
SHORT_MIN_ADX = 19
SHORT_MIN_RSI = 45
LONG_MAX_RSI = 10

# === TP/SL ===
TP_ATR_MULT = 1.7
SL_ATR_MULT = 3.7

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
BB_SQUEEZE_THRESHOLD = 0.10
MIN_BB_WIDTH = 0.008  

# --- MACD ---
MACD_FAST = 18
MACD_SLOW = 38
MACD_SIGNAL = 18
MACD_SIGNAL_WINDOW = 29

# --- VWAP ---
USE_VWAP = True
VWAP_DEVIATION_THRESHOLD = 0.5  

# === ВРЕМЕННЫЕ ФИЛЬТРЫ ===
SIGNAL_COOLDOWN_MINUTES = 15
MIN_TRIGGERS_ACTIVE_HOURS = 1.9
MIN_TRIGGERS_INACTIVE_HOURS = 2.5

# --- Telegram ---
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'
TELEGRAM_CHAT_ID = 931346988

# === TP/SL НАСТРОЙКИ ===
TP_MIN = 0.016
TP_MAX = 0.08   
SL_MIN = 0.034
SL_MAX = 0.15   

# --- Минимальное расстояние между TP и SL ---
MIN_TP_SL_DISTANCE = 0.02

# --- Fee Rate ---
FEE_RATE = 0.0006

# === ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ ===
MAX_SPREAD_PCT = 0.035    
MIN_CANDLE_BODY_PCT = 0.03  
MAX_WICK_TO_BODY_RATIO = 15.0  
MIN_VOLUME_MA_RATIO = 0.9    
MIN_VOLUME_CONSISTENCY = 0.3  
MAX_RSI_VOLATILITY = 30      
REQUIRE_MACD_HISTOGRAM_CONFIRMATION = False  

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
STOCH_RSI_SMOOTH = 15

# === СИСТЕМА СКОРИНГА ===
WEIGHT_RSI = 8.0
WEIGHT_MACD = 7.5
WEIGHT_BB = 1.0
WEIGHT_VWAP = 10.0
WEIGHT_VOLUME = 2.5
WEIGHT_ADX = 8.0

# === СИСТЕМА ДЛЯ SHORT/LONG ===
SHORT_BOOST_MULTIPLIER = 0.9
LONG_PENALTY_IN_DOWNTREND = 0.85
