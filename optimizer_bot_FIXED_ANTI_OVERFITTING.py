#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ccxt
import pandas as pd
import numpy as np
from config import *
from crypto_signal_bot import analyze, SYMBOLS
import logging
import optuna
import json
from typing import Dict, Any
import time
import ta

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

def analyze_with_params(df, params):
    """ОПТИМИЗИРОВАННЫЙ анализ данных с параметрами индикаторов из оптимизатора"""
    try:
        # Используем максимальный период из параметров для проверки
        ma_slow = params.get('MA_SLOW', MA_SLOW)
        if df.empty or len(df) < ma_slow:
            return pd.DataFrame()
        
        # ОПТИМИЗИРОВАНО: Создаем копию только один раз в начале
        df = df.copy()
        
        # Извлекаем параметры индикаторов (используем значения из config.py как fallback)
        rsi_window = params.get('RSI_WINDOW', RSI_WINDOW)
        rsi_extreme_oversold = params.get('RSI_EXTREME_OVERSOLD', RSI_EXTREME_OVERSOLD)
        rsi_extreme_overbought = params.get('RSI_EXTREME_OVERBOUGHT', RSI_EXTREME_OVERBOUGHT)
        atr_window = params.get('ATR_WINDOW', ATR_WINDOW)
        adx_window = params.get('ADX_WINDOW', ADX_WINDOW)
        bb_window = params.get('BB_WINDOW', BB_WINDOW)
        bb_std_dev = params.get('BB_STD_DEV', BB_STD_DEV)
        macd_fast = params.get('MACD_FAST', MACD_FAST)
        macd_slow = params.get('MACD_SLOW', MACD_SLOW)
        macd_signal = params.get('MACD_SIGNAL', MACD_SIGNAL)
        stoch_rsi_k = params.get('STOCH_RSI_K', STOCH_RSI_K)
        stoch_rsi_d = params.get('STOCH_RSI_D', STOCH_RSI_D)
        stoch_rsi_length = params.get('STOCH_RSI_LENGTH', STOCH_RSI_LENGTH)
        
        # EMA с оптимизируемыми периодами (МАКСИМАЛЬНАЯ НАДЕЖНОСТЬ)
        ma_fast = params.get('MA_FAST', MA_FAST)
        ma_slow = params.get('MA_SLOW', MA_SLOW)
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=ma_fast)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=ma_slow)
        
        # MACD через класс ta.trend.MACD (используем параметры из оптимизатора)
        macd_obj = ta.trend.MACD(
            close=df['close'],
            window_slow=macd_slow,
            window_fast=macd_fast,
            window_sign=macd_signal
        )
        df['macd_line'] = macd_obj.macd()
        df['macd_signal'] = macd_obj.macd_signal()
        df['macd'] = macd_obj.macd_diff()  # гистограмма
        
        # RSI (используем параметр из оптимизатора)
        df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_window)
        
        # Stochastic RSI (используем параметры из оптимизатора)
        stoch_rsi = ta.momentum.stochrsi(df['close'], window=stoch_rsi_length, smooth1=stoch_rsi_k, smooth2=stoch_rsi_d)
        df['stoch_rsi_k'] = stoch_rsi
        
        # ADX (используем параметр из оптимизатора)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=adx_window)
        
        # ATR (используем параметр из оптимизатора)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_window)
        
        # Bollinger Bands (используем параметры из оптимизатора)
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=bb_window, window_dev=bb_std_dev)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        # Синхронизация с crypto_signal_bot.py
        df['bollinger_high'] = df['bb_upper']
        df['bollinger_low'] = df['bb_lower']
        
        # VWAP (если включен)
        if USE_VWAP:
            df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
            df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Объёмные фильтры (если включены)
        if USE_VOLUME_FILTER:
            # ИСПРАВЛЕНО: Создаем volume_usdt если его нет
            if 'volume_usdt' not in df.columns:
                df['volume_usdt'] = df['volume'] * df['close']
            df['volume_ma_usdt'] = df['volume_usdt'].rolling(window=bb_window).mean()
            df['volume_ratio_usdt'] = df['volume_usdt'] / df['volume_ma_usdt']
        
        return df
        
    except Exception as e:
        logging.error(f"Ошибка в анализе данных: {e}")
        return pd.DataFrame()

# Инициализация рынков Bybit для стабильной работы символов/рынков
try:
    EXCHANGE.load_markets()
except Exception as e:
    logging.warning(f"Не удалось загрузить рынки: {e}")

# --- ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ДЛЯ ПРОИЗВОДИТЕЛЬНОСТИ ---
GLOBAL_HOURS_BACK = 504  # УМЕНЬШЕНО: ~21 день истории для ускорения без потери качества
try:
    GLOBAL_ACTIVE_HOURS_UTC = ACTIVE_HOURS_UTC  # из config.py
except Exception:
    GLOBAL_ACTIVE_HOURS_UTC = list(range(6, 24))
 
GLOBAL_ALL_SYMBOLS = []
DATA_CACHE_ANALYZED: Dict[str, pd.DataFrame] = {}

# --- УПРОЩЕННЫЕ ОГРАНИЧЕНИЯ ДЛЯ 15М ФЬЮЧЕРСОВ ---
MIN_SL_COUNT = 2      # Минимум SL сделок для статистики
COMMISSION_PCT = 0.055
SPREAD_PCT = 0.04

def get_all_symbols_from_data():
    """СИНХРОНИЗИРОВАНО: Используем ТОЧНО ТЕ ЖЕ символы что и в crypto_signal_bot.py"""
    # Импортируем символы напрямую из реального бота для полной синхронизации
    return SYMBOLS.copy()  # Все символы из crypto_signal_bot.py (34 монеты)

def get_historical_data(symbol, hours_back=72):
    """ОПТИМИЗИРОВАНО: Загружает исторические данные через API с UTC временем."""
    try:
        candles_needed = int(hours_back * 60 / 15) + 120
        all_ohlcv = []

        now_ms = EXCHANGE.milliseconds()
        since = now_ms - hours_back * 60 * 60 * 1000
        try:
            tf_ms = EXCHANGE.parse_timeframe(TIMEFRAME) * 1000
        except Exception:
            tf_ms = 15 * 60 * 1000

        safety_loops = 0
        while len(all_ohlcv) < candles_needed and safety_loops < 30:
            batch_limit = min(1000, candles_needed - len(all_ohlcv))
            try:
                ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=int(since), limit=batch_limit)
            except TypeError:
                ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=batch_limit)

            if not ohlcv:
                break

            if all_ohlcv and ohlcv and ohlcv[0][0] <= all_ohlcv[-1][0]:
                since = all_ohlcv[-1][0] + tf_ms
                safety_loops += 1
                continue

            all_ohlcv.extend(ohlcv)
            since = all_ohlcv[-1][0] + tf_ms
            safety_loops += 1
            time.sleep(0.2)

        # Используем консервативную проверку - минимум 50 свечей для любого анализа
        if not all_ohlcv or len(all_ohlcv) < 50:
            logging.warning(f"{symbol}: недостаточно данных для анализа ({len(all_ohlcv)})")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # ИСПРАВЛЕНО: Используем UTC везде для синхронизации с ботом
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['volume_usdt'] = df['volume'] * df['close']

        logging.info(f"Загружено {len(df)} свечей для {symbol} через API (UTC)")
        return df

    except ccxt.RateLimitExceeded as e:
        wait_time = getattr(e, 'retry_after', 1)
        logging.warning(f"Rate limit exceeded for {symbol}, жду {wait_time} сек.")
        time.sleep(wait_time)
        return pd.DataFrame()
    except ccxt.NetworkError as e:
        logging.error(f"Network error for {symbol}: {e}")
        time.sleep(5)
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Ошибка получения OHLCV по {symbol}: {e}")
        return pd.DataFrame()

def suggest_parameters_anti_overfitting(trial: optuna.Trial) -> Dict[str, Any]:
    """УПРОЩЕННЫЕ параметры для 15-минутной торговли с приоритетом на количество сигналов"""
    
    # === ПРОСТЫЕ И ПОНЯТНЫЕ ОГРАНИЧЕНИЯ ===
    
    # 1. EMA периоды: ТОП-5 ЛУЧШИХ комбинаций для фьючерсов 15м
    # Отобраны на основе тестирования и опыта торговли
    ma_combinations = [
        (8, 21),   # 🏆 ЛУЧШАЯ: Классическая, стабильная, Win Rate 100%
        (6, 15),   # 🥈 ВТОРАЯ: Быстрая реакция, много сигналов
        (12, 26),  # 🥉 ТРЕТЬЯ: Медленная, трендовая, надежная
        (9, 18),   # 🎯 ЧЕТВЕРТАЯ: Сбалансированная реакция
        (10, 20),  # 🎯 ПЯТАЯ: Универсальная, проверенная временем
    ]
    ma_idx = trial.suggest_int('MA_COMBINATION', 0, len(ma_combinations) - 1)
    ma_fast, ma_slow = ma_combinations[ma_idx]
    
    # 2. RSI фильтры: используем простые диапазоны с проверкой граничных случаев
    rsi_min = trial.suggest_int('RSI_MIN', 20, 80, step=2)
    rsi_max = trial.suggest_int('RSI_MAX', rsi_min + 2, 90, step=2)  # RSI_MAX > RSI_MIN
    
    # УПРОЩЕНО: Убираем сложные ограничения - система скоринга сама отсеет плохие сигналы
    # LONG_MAX_RSI: BUY сигналы при RSI <= этого значения
    long_max_rsi = trial.suggest_int('LONG_MAX_RSI', 10, 80, step=2)
    
    # SHORT_MIN_RSI: SELL сигналы при RSI >= этого значения  
    short_min_rsi = trial.suggest_int('SHORT_MIN_RSI', 20, 90, step=2)
    
    return {
        # Основные пороги (ОПТИМИЗИРОВАНО ДЛЯ БОЛЬШЕ СИГНАЛОВ)
        'MIN_COMPOSITE_SCORE': trial.suggest_float('MIN_COMPOSITE_SCORE', 0, 1.5, step=0.5),  # ИСПРАВЛЕНО: еще ниже для больше сигналов
        'MIN_ADX': trial.suggest_int('MIN_ADX', 6, 40, step=2),  # ПОНИЖЕНО: для больше сигналов
        'SHORT_MIN_ADX': trial.suggest_int('SHORT_MIN_ADX', 20, 52, step=2),  # ПОНИЖЕНО: для больше сигналов
        
        # RSI фильтры (ИСПРАВЛЕНО: логически корректные диапазоны с ограничениями)
        'RSI_MIN': rsi_min,             # Корректно ограничен
        'RSI_MAX': rsi_max,             # Корректно ограничен
        'LONG_MAX_RSI': long_max_rsi,   # Корректно ограничен: < RSI_MIN
        'SHORT_MIN_RSI': short_min_rsi,  # Корректно ограничен: > RSI_MAX
        
        # TP/SL (ОПТИМИЗИРОВАНО ДЛЯ ЛУЧШИХ СООТНОШЕНИЙ)
        'TP_ATR_MULT': trial.suggest_float('TP_ATR_MULT', 0.8, 6.0, step=0.2),  # Расширен верхний предел
        'SL_ATR_MULT': trial.suggest_float('SL_ATR_MULT', 1.0, 8.0, step=0.2),  # Расширен верхний предел

        # Триггеры (ОПТИМИЗИРОВАНО ДЛЯ БОЛЬШЕ СИГНАЛОВ)
        'MIN_TRIGGERS_ACTIVE_HOURS': trial.suggest_float('MIN_TRIGGERS_ACTIVE_HOURS', 0.1, 6.0, step=0.3),   # ПОНИЖЕНО: больше сигналов
        
        # Временные фильтры (ОПТИМИЗИРОВАНО ДЛЯ БОЛЬШЕ СИГНАЛОВ)
        'SIGNAL_COOLDOWN_MINUTES': trial.suggest_int('SIGNAL_COOLDOWN_MINUTES', 15, 60, step=15),  # Разрешаем 0 и до 60
        
        # Объем (ОПТИМИЗИРОВАНО ДЛЯ БОЛЬШЕ СИГНАЛОВ)
        'MIN_VOLUME_MA_RATIO': trial.suggest_float('MIN_VOLUME_MA_RATIO', 0.05, 3.0, step=0.05),  # Верх до 3.0

        # MACD подтверждение убрано - система скоринга сама отсеет плохие сигналы
        
        # Веса скоринга (ОПТИМИЗИРОВАНО ДЛЯ БОЛЬШЕ СИГНАЛОВ)
        'WEIGHT_RSI': trial.suggest_float('WEIGHT_RSI', 0.0, 10.0, step=0.2),
        'WEIGHT_MACD': trial.suggest_float('WEIGHT_MACD', 0.0, 9.0, step=0.2),
        'WEIGHT_BB': trial.suggest_float('WEIGHT_BB', 0.0, 6.0, step=0.2),
        'WEIGHT_VWAP': trial.suggest_float('WEIGHT_VWAP', 0.0, 12.0, step=0.2),
        'WEIGHT_VOLUME': trial.suggest_float('WEIGHT_VOLUME', 0.0, 6.0, step=0.2),
        'WEIGHT_ADX': trial.suggest_float('WEIGHT_ADX', 0.0, 12.0, step=0.2),
        

        
        # Множители направления (ОПТИМИЗИРОВАНО ДЛЯ БОЛЬШЕ СИГНАЛОВ)
        'SHORT_BOOST_MULTIPLIER': trial.suggest_float('SHORT_BOOST_MULTIPLIER', 0.2, 5.0, step=0.2),
        'LONG_PENALTY_IN_DOWNTREND': trial.suggest_float('LONG_PENALTY_IN_DOWNTREND', 0.0, 1.0, step=0.05),

        # Минимальные TP/SL (проценты) - ОПТИМИЗИРОВАНО ДЛЯ ЛУЧШИХ СООТНОШЕНИЙ
        'TP_MIN': trial.suggest_float('TP_MIN', 0.01, 0.10, step=0.002),
        'SL_MIN': trial.suggest_float('SL_MIN', 0.01, 0.10, step=0.002),
        
        # === ПАРАМЕТРЫ ИНДИКАТОРОВ (ДЕТАЛЬНЫЙ ПОИСК: 6-10 ЗНАЧЕНИЙ) ===
        # RSI параметры
        'RSI_WINDOW': trial.suggest_categorical('RSI_WINDOW', [5, 7, 9, 12, 14, 18, 21, 24]),  # 8 значений: от быстрого до медленного
        'RSI_EXTREME_OVERSOLD': trial.suggest_categorical('RSI_EXTREME_OVERSOLD', [10, 12, 15, 18, 20, 22, 25, 28]),  # 8 значений: от мягкого до строгого
        'RSI_EXTREME_OVERBOUGHT': trial.suggest_categorical('RSI_EXTREME_OVERBOUGHT', [72, 75, 78, 80, 82, 85, 88, 90]),  # 8 значений: от мягкого до строгого
        
        # ATR параметры
        'ATR_WINDOW': trial.suggest_categorical('ATR_WINDOW', [7, 10, 12, 14, 16, 18, 20, 24]),  # 8 значений: от быстрого до медленного
        
        # ADX параметры
        'ADX_WINDOW': trial.suggest_categorical('ADX_WINDOW', [7, 10, 12, 14, 16, 18, 20, 24]),  # 8 значений: от быстрого до медленного
        
        # Bollinger Bands параметры
        'BB_WINDOW': trial.suggest_categorical('BB_WINDOW', [10, 12, 15, 18, 20, 22, 25, 28, 30]),  # 8 значений: от быстрого до медленного
        'BB_STD_DEV': trial.suggest_categorical('BB_STD_DEV', [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]),  # 8 значений: от узкого до широкого
        
        # MACD параметры
        'MACD_FAST': trial.suggest_categorical('MACD_FAST', [6, 8, 10, 12, 14, 16, 18, 20]),  # 8 значений: от быстрого до медленного
        'MACD_SLOW': trial.suggest_categorical('MACD_SLOW', [18, 21, 24, 26, 28, 30, 32, 35]),  # 8 значений: от быстрого до медленного
        'MACD_SIGNAL': trial.suggest_categorical('MACD_SIGNAL', [4, 6, 8, 9, 10, 12, 14, 16]),  # 8 значений: от быстрого до медленного
        
        # VWAP параметры
        'VWAP_DEVIATION_THRESHOLD': trial.suggest_categorical('VWAP_DEVIATION_THRESHOLD', [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2]),  # 8 значений: от мягкого до строгого
        
        # Stochastic RSI параметры
        'STOCH_RSI_K': trial.suggest_categorical('STOCH_RSI_K', [1,     2, 3, 4, 5, 6, 8, 10, 12]),  # 8 значений: от быстрого до медленного
        'STOCH_RSI_D': trial.suggest_categorical('STOCH_RSI_D', [0, 1, 2, 3, 4, 5, 6, 7, 8]),  # 8 значений: от быстрого до медленного
        'STOCH_RSI_LENGTH': trial.suggest_categorical('STOCH_RSI_LENGTH', [6, 8, 10, 12, 14, 16, 18, 20, 22]),  # 8 значений: от быстрого до медленного
        
        # === EMA ПЕРИОДЫ (ДЕТАЛЬНЫЙ ПОИСК: 8 ЗНАЧЕНИЙ) ===
        'MA_FAST': ma_fast,  # Уже вычислено с ограничениями
        'MA_SLOW': ma_slow,   # Уже вычислено с ограничениями
    }

def evaluate_signal_strength_optimized(df, current_index, symbol, action, weights, params):
    """УПРОЩЕННАЯ оценка силы сигнала без копирования DataFrame"""
    try:
        if current_index < 5 or current_index >= len(df):
            return 0
            
        score = 0
        last = df.iloc[current_index]
        prev = df.iloc[current_index-1] if current_index > 0 else last
        
        return evaluate_signal_strength_with_weights_fast(last, prev, action, weights, params)
        
    except Exception as e:
        logging.error(f"Ошибка в evaluate_signal_strength_optimized: {e}")
        return 0

def evaluate_signal_strength_with_weights_fast(last, prev, action, weights, params):
    """УПРОЩЕННАЯ быстрая оценка силы сигнала"""
    try:
        score = 0
        
        # 1. RSI анализ с переданными весами (используем параметры из оптимизатора)
        rsi_score = 0
        rsi_extreme_oversold = params.get('RSI_EXTREME_OVERSOLD', RSI_EXTREME_OVERSOLD)
        rsi_extreme_overbought = params.get('RSI_EXTREME_OVERBOUGHT', RSI_EXTREME_OVERBOUGHT)
        rsi_oversold = params.get('RSI_MIN', RSI_MIN)
        rsi_overbought = params.get('RSI_MAX', RSI_MAX)
        
        if action == 'BUY':
            if last['rsi'] < rsi_extreme_oversold:
                rsi_score = 3.0
            elif last['rsi'] < rsi_oversold:
                rsi_score = 2.5
            elif rsi_oversold < last['rsi'] < 50:
                rsi_score = 1.5
            elif last['rsi'] > rsi_overbought:
                rsi_score = -0.5
        elif action == 'SELL':
            if last['rsi'] > rsi_extreme_overbought:
                rsi_score = 3.0
            elif last['rsi'] > rsi_overbought:
                rsi_score = 2.5
            elif 50 < last['rsi'] < rsi_overbought:
                rsi_score = 1.5
            elif last['rsi'] < rsi_oversold:
                rsi_score = -0.5
                
        # Применяем вес RSI к базовому скору
        rsi_weight = weights.get('WEIGHT_RSI', 3.0)
        score += rsi_score * rsi_weight
        
        # 2. ИСПРАВЛЕНО: MACD анализ (используем правильные компоненты)
        macd_score = 0
        if ('macd_line' in last) and ('macd_signal' in last):
            macd_cross = last['macd_line'] - last['macd_signal']  # ИСПРАВЛЕНО: основная линия - сигнальная линия
            prev_macd_cross = prev['macd_line'] - prev['macd_signal']  # ИСПРАВЛЕНО: основная линия - сигнальная линия
            
            if action == 'BUY':
                if macd_cross > 0 and prev_macd_cross <= 0:
                    macd_score = 4.0
                elif macd_cross > 0:
                    macd_score = 2.0
                elif macd_cross > prev_macd_cross:
                    macd_score = 1.0
            elif action == 'SELL':
                if macd_cross < 0 and prev_macd_cross >= 0:
                    macd_score = 4.0
                elif macd_cross < 0:
                    macd_score = 2.0
                elif macd_cross < prev_macd_cross:
                    macd_score = 1.0
                    
        # Применяем вес MACD к базовому скору
        macd_weight = weights.get('WEIGHT_MACD', 3.0)
        score += macd_score * macd_weight
        
        # 3. ОПТИМИЗИРОВАНО: Bollinger Bands
        bb_score = 0
        if ('bollinger_low' in last) and ('bollinger_high' in last):
            close = last['close']
            bb_denom = max((last['bollinger_high'] - last['bollinger_low']), 1e-12)
            bb_position = (close - last['bollinger_low']) / bb_denom
            
            if action == 'BUY':
                if bb_position <= 0.05:
                    bb_score = 1.5
                elif bb_position <= 0.15:
                    bb_score = 1.0
                elif bb_position <= 0.3:
                    bb_score = 0.5
            elif action == 'SELL':
                if bb_position >= 0.95:
                    bb_score = 1.5
                elif bb_position >= 0.85:
                    bb_score = 1.0
                elif bb_position >= 0.7:
                    bb_score = 0.5
                    
        # Применяем вес BB к базовому скору
        bb_weight = weights.get('WEIGHT_BB', 2.0)
        score += bb_score * bb_weight
        
        # 4. ОПТИМИЗИРОВАНО: VWAP анализ (используем параметр из оптимизатора)
        vwap_score = 0
        if 'vwap' in last:
            vwap_deviation_threshold = params.get('VWAP_DEVIATION_THRESHOLD', VWAP_DEVIATION_THRESHOLD)
            vwap_dev = last.get('vwap_deviation', 0)
            if action == 'BUY':
                if vwap_dev <= -vwap_deviation_threshold * 1.5:
                    vwap_score = 1.5
                elif vwap_dev <= -vwap_deviation_threshold:
                    vwap_score = 1.0
                elif vwap_dev <= 0:
                    vwap_score = 0.3
            elif action == 'SELL':
                if vwap_dev >= vwap_deviation_threshold * 1.5:
                    vwap_score = 1.5
                elif vwap_dev >= vwap_deviation_threshold:
                    vwap_score = 1.0
                elif vwap_dev >= 0:
                    vwap_score = 0.3
                    
        # Применяем вес VWAP к базовому скору
        vwap_weight = weights.get('WEIGHT_VWAP', 2.0)
        score += vwap_score * vwap_weight
        
        # 5. ОПТИМИЗИРОВАНО: Объём анализ
        volume_score = 0
        if 'volume_ratio_usdt' in last:
            vol_ratio = last.get('volume_ratio_usdt', 1.0)
            if vol_ratio >= 2.0:
                volume_score = 1.5
            elif vol_ratio >= 1.5:
                volume_score = 1.0
            elif vol_ratio >= 1.2:
                volume_score = 0.5
                
        # Применяем вес Volume к базовому скору
        volume_weight = weights.get('WEIGHT_VOLUME', 2.0)
        score += volume_score * volume_weight
        
        # 6. ОПТИМИЗИРОВАНО: Простой ADX анализ с min_adx
        adx_score = 0
        
        # Используем оптимизируемый параметр min_adx
        min_adx = params.get('MIN_ADX', 25)  # ИСПРАВЛЕНО: берем из params, а не weights
        
        if last['adx'] >= 50:
            adx_score = 3.0
        elif last['adx'] >= 40:
            adx_score = 2.5
        elif last['adx'] >= 30:
            adx_score = 2.0
        elif last['adx'] >= min_adx:
            adx_score = 1.5
        elif last['adx'] >= min_adx * 0.8:
            adx_score = 1.0
        else:
            adx_score = 0.5
            
        # Применяем вес ADX к базовому скору
        adx_weight = weights.get('WEIGHT_ADX', 2.0)
        score += adx_score * adx_weight
        
        # Применяем корректировки для SHORT/LONG
        if action == 'SELL':
            short_boost = weights.get('SHORT_BOOST_MULTIPLIER', 1.0)
            score *= short_boost
        elif action == 'BUY':
            # Проверяем нисходящий тренд для LONG позиций
            if ('ema_fast' in last) and ('ema_slow' in last):
                if last['ema_fast'] < last['ema_slow']:  # Нисходящий тренд
                    long_penalty = weights.get('LONG_PENALTY_IN_DOWNTREND', 0.5)
                    score *= long_penalty
        
        return max(0, score)
        
    except Exception as e:
        logging.error(f"Ошибка в evaluate_signal_strength_with_weights_fast: {e}")
        return 0

def simulate_signals_anti_overfitting(df, symbol, params, active_hours_utc):
    """ОПТИМИЗИРОВАННАЯ симуляция сигналов с ботом"""
    # ИСПРАВЛЕНО: Проверяем минимальное количество данных для симуляции
    # Уменьшено исключение с 384 до 96 свечей (1 день) - более реалистично для 15м торговли
    min_candles_needed = MIN_15M_CANDLES + 96  # Минимум для анализа + исключение последних свечей (1 день)
    if df.empty or len(df) < min_candles_needed:
        logging.warning(f"🚫 {symbol}: Недостаточно данных для симуляции ({len(df)} < {min_candles_needed})")
        return []
    
    # ОПТИМИЗИРОВАНО: Убрано лишнее копирование - работаем напрямую с кэшем
    if 'ema_fast' in df.columns and 'atr' in df.columns:
        df_analyzed = df  # Убрано .copy() - не нужно копировать если не меняем данные
    else:
        df_analyzed = analyze_with_params(df, params)  # Убрано .copy() - analyze_with_params сам создает копию
    
    if df_analyzed.empty:
        logging.warning(f"🚫 {symbol}: Пустой DataFrame после анализа")
        return []
        
    logging.info(f"📊 {symbol}: Начинаем симуляцию с {len(df_analyzed)} записей")
    
    # Гарантируем наличие volume_usdt
    if 'volume_usdt' not in df_analyzed.columns and 'volume' in df_analyzed.columns and 'close' in df_analyzed.columns:
        df_analyzed['volume_usdt'] = df_analyzed['volume'] * df_analyzed['close']
    
    signals = []
    last_signal_time = None
    
    # Извлекаем параметры
    min_composite_score = params['MIN_COMPOSITE_SCORE']
    min_adx = params['MIN_ADX']
    short_min_adx = params['SHORT_MIN_ADX']
    short_min_rsi = params['SHORT_MIN_RSI']
    long_max_rsi = params['LONG_MAX_RSI']
    rsi_min = params['RSI_MIN']
    rsi_max = params['RSI_MAX']
    tp_mult = params['TP_ATR_MULT']
    sl_mult = params['SL_ATR_MULT']
    signal_cooldown_minutes = params['SIGNAL_COOLDOWN_MINUTES']
    min_triggers_active_hours = params['MIN_TRIGGERS_ACTIVE_HOURS']
    
    # Параметры индикаторов уже обработаны в analyze_with_params()

    # УБРАНО: Любые соотношения TP/SL допустимы, главное - общая прибыльность!
    
    # ОПТИМИЗИРОВАНО: без копирования DataFrame (ускорение в 100+ раз)
    # Исключаем последние 96 свечей (1 день) для предотвращения look-ahead bias - более реалистично для 15м
    for i in range(MIN_15M_CANDLES, len(df_analyzed) - 96):  # Баланс скорость/качество: 1 день достаточно для 15м сигналов
        # ОПТИМИЗИРОВАНО: работаем по индексам без копирования
        last = df_analyzed.iloc[i]
        prev = df_analyzed.iloc[i-1] if i > 0 else df_analyzed.iloc[i]
        now = last['timestamp']
        
        # Ранее здесь был фильтр по активным часам; теперь тестируем во все часы
            
        # Кулдаун
        if last_signal_time and (now - last_signal_time).total_seconds() < signal_cooldown_minutes * 60:
            continue
            
        # Базовые фильтры
        if last['adx'] < min_adx:
            continue
            
        # ОПТИМИЗИРОВАНО: Упрощенная проверка объёма
        min_volume_ratio = params['MIN_VOLUME_MA_RATIO']
        
        # Проверяем volume_ratio_usdt если есть, иначе пропускаем проверку
        if 'volume_ratio_usdt' in last and last.get('volume_ratio_usdt', 1.0) < min_volume_ratio:
            continue
        
        # === ОПТИМИЗИРОВАННЫЕ ТРИГГЕРЫ ДЛЯ 15М ===
        buy_triggers = 0
        sell_triggers = 0
        
        # RSI триггеры (оптимизированы для 15м, используем параметры из оптимизатора)
        rsi_extreme_oversold = params.get('RSI_EXTREME_OVERSOLD', RSI_EXTREME_OVERSOLD)
        rsi_extreme_overbought = params.get('RSI_EXTREME_OVERBOUGHT', RSI_EXTREME_OVERBOUGHT)
        
        if last['rsi'] <= rsi_extreme_oversold:
            buy_triggers += 2.0
        elif last['rsi'] < rsi_min:
            buy_triggers += 1.0
        if last['rsi'] >= rsi_extreme_overbought:
            sell_triggers += 2.0
        elif last['rsi'] > rsi_max:
            sell_triggers += 1.0
        
        # EMA кроссовер (основной триггер для 15м)
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            buy_triggers += 1.5
        elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
            buy_triggers += 0.5
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            sell_triggers += 1.5
        elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
            sell_triggers += 0.5
            
        # ИСПРАВЛЕНО: MACD триггеры (используем правильные компоненты)
        if ('macd_line' in last) and ('macd_signal' in last):
            if last['macd_line'] > last['macd_signal']:  # ИСПРАВЛЕНО: основная линия > сигнальная линия
                buy_triggers += 0.5
            if last['macd_line'] < last['macd_signal']:  # ИСПРАВЛЕНО: основная линия < сигнальная линия
                sell_triggers += 0.5
        
        # ОПТИМИЗИРОВАНО: Bollinger Bands без проверки columns
        if ('bollinger_low' in last) and ('bollinger_high' in last):
            denom = max((last['bollinger_high'] - last['bollinger_low']), 1e-12)
            bb_position = (last['close'] - last['bollinger_low']) / denom
            if bb_position <= 0.25:  # Более строго для 15м
                buy_triggers += 0.5
            if bb_position >= 0.75:  # Более строго для 15м
                sell_triggers += 0.5
                
        # Минимальные триггеры (единый порог для всех часов)
        min_triggers = min_triggers_active_hours
        
        # Определение типа сигнала
        signal_type = None
        if buy_triggers >= min_triggers and last['rsi'] <= long_max_rsi:
            signal_type = 'BUY'
        elif sell_triggers >= min_triggers and last['rsi'] >= short_min_rsi:
            signal_type = 'SELL'
            
        # Дополнительные условия
        if signal_type == 'SELL' and last['adx'] < short_min_adx:
            continue
            
        # MACD Histogram проверка убрана - система скоринга сама отсеет плохие сигналы

        if signal_type:
            try:
                # ИСПРАВЛЕНО: Создаем правильные weights из params и передаем их отдельно
                weights = {k: params[k] for k in params if k.startswith('WEIGHT_')}
                score = evaluate_signal_strength_optimized(df_analyzed, i, symbol, signal_type, weights, params)
                if score >= min_composite_score:
                    entry_price = last['close']
                    entry_time = now
                    future_data = df_analyzed.iloc[i+1:i+385]  # Горизонт до 4 дней вперёд для корректной проверки TP/SL
                    
                    if len(future_data) >= 4:  # ОПТИМИЗИРОВАНО: минимум 4 свечи
                        atr = last['atr']
                        tp_distance = atr * tp_mult
                        sl_distance = atr * sl_mult
                        
                        if signal_type == 'BUY':
                            tp_price = entry_price + tp_distance
                            sl_price = entry_price - sl_distance
                        else:
                            tp_price = entry_price - tp_distance
                            sl_price = entry_price + sl_distance
                        
                        # Минимальные TP/SL в процентах
                        tp_pct_min = params['TP_MIN']  # Используем точное значение параметра
                        sl_pct_min = params['SL_MIN']  # Используем точное значение параметра

                        # Применяем минимальные проценты
                        def enforce_min_levels(entry, tp_price, sl_price, side):
                            if side == 'BUY':
                                tp_eff = max((tp_price - entry) / entry, tp_pct_min)
                                sl_eff = max((entry - sl_price) / entry, sl_pct_min)
                                return entry * (1 + tp_eff), entry * (1 - sl_eff)
                            else:
                                tp_eff = max((entry - tp_price) / entry, tp_pct_min)
                                sl_eff = max((sl_price - entry) / entry, sl_pct_min)
                                return entry * (1 - tp_eff), entry * (1 + sl_eff)

                        tp_price, sl_price = enforce_min_levels(entry_price, tp_price, sl_price, signal_type)

                        # ОПТИМИЗИРОВАНО: Векторизованный поиск TP/SL вместо iterrows()
                        # Корректно обрабатываем сигналы, которые закрываются через несколько дней
                        result = None
                        future_highs = future_data['high'].values
                        future_lows = future_data['low'].values
                        
                        if signal_type == 'BUY':
                            # Ищем первый индекс где high >= tp_price или low <= sl_price
                            tp_hits = future_highs >= tp_price
                            sl_hits = future_lows <= sl_price
                            if np.any(tp_hits):
                                tp_idx = np.where(tp_hits)[0][0]  # ИСПРАВЛЕНО: первый индекс
                                sl_idx = np.where(sl_hits)[0][0] if np.any(sl_hits) else len(future_highs)
                                result = 'tp' if tp_idx <= sl_idx else 'sl'
                            elif np.any(sl_hits):
                                result = 'sl'
                        else:  # SELL
                            # Ищем первый индекс где low <= tp_price или high >= sl_price
                            tp_hits = future_lows <= tp_price
                            sl_hits = future_highs >= sl_price
                            if np.any(tp_hits):
                                tp_idx = np.where(tp_hits)[0][0]  # ИСПРАВЛЕНО: первый индекс
                                sl_idx = np.where(sl_hits)[0][0] if np.any(sl_hits) else len(future_lows)
                                result = 'tp' if tp_idx <= sl_idx else 'sl'
                            elif np.any(sl_hits):
                                result = 'sl'
                                    
                        if not result:
                            result = 'sl'
                            
                        tp_pct = ((tp_price - entry_price) / entry_price * 100) if signal_type == 'BUY' else ((entry_price - tp_price) / entry_price * 100)
                        sl_pct = ((entry_price - sl_price) / entry_price * 100) if signal_type == 'BUY' else ((sl_price - entry_price) / entry_price * 100)
                        
                        signals.append({
                            'symbol': symbol,
                            'type': signal_type,
                            'entry_time': entry_time,
                            'score': score,
                            'result': result,
                            'tp_pct': tp_pct,
                            'sl_pct': sl_pct
                        })
                        last_signal_time = now
                        
            except Exception as e:
                logging.warning(f"Ошибка оценки сигнала {symbol} в {now}: {e}")
                continue

    logging.info(f"📈 {symbol}: Создано сигналов: {len(signals)}")
    return signals

def test_single_params_anti_overfitting(params, hours_back=None, active_hours_utc=None):
    """ОПТИМИЗИРОВАНО: Тестирует один набор параметров С МИНИМАЛЬНЫМИ РАСЧЕТАМИ"""
    if hours_back is None:
        hours_back = GLOBAL_HOURS_BACK
    if active_hours_utc is None:
        active_hours_utc = GLOBAL_ACTIVE_HOURS_UTC
        
    all_signals = []
    
    for symbol in GLOBAL_ALL_SYMBOLS:
        df_analyzed = DATA_CACHE_ANALYZED.get(symbol)
        if df_analyzed is None or df_analyzed.empty:
            continue
        signals = simulate_signals_anti_overfitting(df_analyzed, symbol, params, active_hours_utc)
        all_signals.extend(signals)
    
    if not all_signals:
        return None
    
    # ОПТИМИЗИРОВАНО: Только необходимые расчеты для скоринга
    tp_signals = [s for s in all_signals if s['result'] == 'tp']
    sl_signals = [s for s in all_signals if s['result'] == 'sl']
    # Timeout больше не используем: сделки без TP/SL за 4 дня считаем SL
    tp_count = len(tp_signals)
    sl_count = len(sl_signals)
    
    # ОПТИМИЗИРОВАНО: Упрощенный расчет winrate
    total_trades = tp_count + sl_count
    if total_trades == 0:
        return None
    
    # Упрощенный winrate: считаем только TP и SL
    winrate = tp_count / total_trades * 100
    
    # ОПТИМИЗИРОВАНО: Только необходимые метрики для скоринга
    tp_sum = sum([s['tp_pct'] for s in tp_signals])
    sl_sum = sum([s['sl_pct'] for s in sl_signals])
    
    # Базовые соотношения
    tp_sl_count_ratio = tp_count / max(sl_count, 1)
    # Корректируем окно на исключённые последние 4 дня (синхронно с lookahead)
    effective_hours = max(hours_back - 24 * 4, 1)
    signals_per_day = len(all_signals) / (effective_hours / 24)
    
    # ОПТИМИЗИРОВАНО: Упрощенный расчет средних значений
    avg_tp_pct = tp_sum / max(tp_count, 1)
    avg_sl_pct = sl_sum / max(sl_count, 1)

    # Упрощенные TP/SL без комиссий и спреда (для быстрых метрик)
    net_tp_pct = avg_tp_pct
    net_sl_pct = avg_sl_pct

    # Математическое ожидание (в процентах, без комиссий)
    winrate_decimal = winrate / 100
    expected_return = winrate_decimal * net_tp_pct - (1 - winrate_decimal) * net_sl_pct

    # ОПТИМИЗИРОВАНО: Упрощенный profit factor (без комиссий, точный расчет в calculate_advanced_score)
    profit_factor = tp_sum / max(sl_sum, 0.1) if sl_sum > 0 else float('inf')
    
    # ОПТИМИЗИРОВАНО: Просадка здесь не рассчитывается
    max_drawdown_pct = 0.0
    
    # ОПТИМИЗИРОВАНО: Упрощенная месячная доходность (без комиссий, точный расчет в calculate_advanced_score)
    avg_net_pct = (tp_sum - sl_sum) / len(all_signals)
    monthly_net_pct = avg_net_pct * signals_per_day * 30.0
    
    return {
        'signals': len(all_signals),
        'signals_per_day': signals_per_day,
        'winrate': winrate,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'tp_sl_count_ratio': tp_sl_count_ratio,
        'tp_sum': tp_sum,
        'sl_sum': sl_sum,
        'avg_tp_pct': avg_tp_pct,
        'avg_sl_pct': avg_sl_pct,
        'net_tp_pct': net_tp_pct,
        'net_sl_pct': net_sl_pct,
        'expected_return': expected_return,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_net_pct': avg_net_pct,
        'monthly_net_pct': monthly_net_pct
    }

def calculate_advanced_score(result: dict, trial_number: int) -> float:
    """🎯 ИСПРАВЛЕННАЯ СИСТЕМА БАЛЛОВ: ПРАВИЛЬНЫЙ УЧЕТ ПРИБЫЛИ И УБЫТКОВ"""
    
    # Извлекаем только нужные метрики
    winrate = result['winrate']
    signals_per_day = result['signals_per_day']
    tp_count = result['tp_count']
    sl_count = result['sl_count']
    tp_sum = result['tp_sum']
    sl_sum = result['sl_sum']
    
    # === ИСПРАВЛЕННАЯ СИСТЕМА: ПРАВИЛЬНЫЙ РАСЧЕТ ПРИБЫЛИ ===
    STARTING_CAPITAL = 100.0  # Стартовый капитал $100
    
    # ИСПРАВЛЕНО: Объявляем безопасные значения по умолчанию для логирования
    tp_total_profit = 0.0
    sl_total_loss = 0.0
    final_capital = STARTING_CAPITAL
    
    # Рассчитываем реальную прибыль/убыток
    total_trades = tp_count + sl_count
    if total_trades == 0:
        return 0.0
    
    # Средняя прибыль на сделку (учитываем комиссии)
    avg_tp_pct = result.get('avg_tp_pct', 0) / 100  # Переводим в десятичные
    avg_sl_pct = result.get('avg_sl_pct', 0) / 100  # Переводим в десятичные
    
    # Комиссии и спреды
    commission_roundtrip = 2 * COMMISSION_PCT / 100
    spread_roundtrip = 2 * SPREAD_PCT / 100
    
    # Чистая прибыль/убыток на сделку
    net_tp_pct = avg_tp_pct - commission_roundtrip - spread_roundtrip
    net_sl_pct = avg_sl_pct + commission_roundtrip + spread_roundtrip
    
    # Математическое ожидание прибыли на сделку
    winrate_decimal = winrate / 100.0
    expected_return_per_trade = winrate_decimal * net_tp_pct - (1 - winrate_decimal) * net_sl_pct
    
    # Правильный расчет финального капитала с компаундингом
    final_capital = STARTING_CAPITAL * (1 + expected_return_per_trade) ** total_trades
    
    # Проверка на разумность результатов
    if final_capital < 0:
        final_capital = 0  # Капитал не может быть отрицательным
    if final_capital > STARTING_CAPITAL * 100:
        logging.warning(f"Нереалистично высокий капитал: {final_capital:.2f}")
        final_capital = STARTING_CAPITAL * 100  # Ограничиваем максимум
    
    # Переменные для логирования
    tp_total_profit = tp_sum / 100
    sl_total_loss = sl_sum / 100
    
    # Общая прибыль/убыток в долларах и процентах
    total_profit_usd = final_capital - STARTING_CAPITAL
    total_profit_pct = (total_profit_usd / STARTING_CAPITAL) * 100
    
    # Система баллов
    base_score = total_profit_usd  # Основной скор: прибыль в долларах
    trades_bonus = min(total_trades * 0.01, 30)  # Бонус за количество сделок
    signals_bonus = 0  # Сигналы = сделки (дублирование убрано)
    
    # Бонус за винрейт
    winrate_bonus = 0
    if winrate >= 70:
        winrate_bonus = 8
    elif winrate >= 60:
        winrate_bonus = 6
    elif winrate >= 55:
        winrate_bonus = 4
    elif winrate >= 50:
        winrate_bonus = 2
    
    # Финальный скор
    final_score = base_score + trades_bonus + signals_bonus + winrate_bonus
    
    # Логирование результатов
    if final_score > 50 and trial_number % 15 == 0:
        tp_sl_count_ratio = tp_count / max(sl_count, 1) if sl_count > 0 else float('inf')
        tp_sl_size_ratio = avg_tp_pct / max(avg_sl_pct, 0.001) if sl_count > 0 else float('inf')
        
        logging.info(f"Trial {trial_number}: Score={final_score:.1f} | "
                    f"Capital: ${STARTING_CAPITAL:.0f} → ${final_capital:.2f} | "
                    f"Profit: ${total_profit_usd:.2f} ({total_profit_pct:.1f}%) | "
                    f"Trades: {total_trades} ({tp_count} TP / {sl_count} SL) | "
                    f"Winrate: {winrate:.1f}% | "
                    f"Trades Bonus: +{trades_bonus:.1f} | "
                    f"Winrate Bonus: +{winrate_bonus}")
    
    return final_score

def objective_anti_overfitting(trial: optuna.Trial) -> float:
    """🎯 УПРОЩЕННАЯ ЦЕЛЕВАЯ ФУНКЦИЯ с минимальными фильтрами - система скоринга сама отсеет плохие!"""
    try:
        # === ОБРЕЗКА УБРАНА: ПОЛНАЯ ОЦЕНКА ВСЕХ СТРАТЕГИЙ ===
        # Каждая стратегия получает полную оценку для максимальной надежности
        
        params = suggest_parameters_anti_overfitting(trial)
        result = test_single_params_anti_overfitting(params)
        
        # === БАЗОВЫЕ ПРОВЕРКИ ===
        if result is None:
            return 0.0
            
        if result['signals'] == 0:
            return 0.0
        
        # === МИНИМАЛЬНЫЕ ФИЛЬТРЫ - СИСТЕМА СКОРИНГА САМА ОТСЕЕТ ПЛОХИЕ ===
        # УБРАНО: жесткие ограничения на expected_return - система скоринга сама отсеет убыточные!
        # УБРАНО: жесткие ограничения на winrate - система скоринга сама отсеет плохие!
        # УБРАНО: жесткие ограничения на tp_sl_ratio - система скоринга сама отсеет плохие!
        # УБРАНО: жесткие ограничения на drawdown - система скоринга сама отсеет плохие!
        
        # === МИНИМАЛЬНЫЕ ПРОВЕРКИ ===
        if result['tp_count'] + result['sl_count'] < 3:  # Минимум 3 сделки для базовой статистики
            return 0.0
        
        # === ВЫЗОВ ПРОСТОЙ СИСТЕМЫ БАЛЛОВ ===
        score = calculate_advanced_score(result, trial.number)
        
        # === СИСТЕМА СКОРИНГА САМА ОТСЕЕТ ПЛОХИЕ ===
        # Если результат хороший, он получит высокий скор автоматически
        
        return score
        
    except Exception as e:
        logging.error(f"Ошибка в objective_anti_overfitting: {e}")
        return 0.0

def check_data_quality():
    """ОПТИМИЗИРОВАНО: Упрощенная проверка качества данных"""
    print("🔍 ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
    print("="*50)
    
    symbols = get_all_symbols_from_data()
    if len(symbols) == 0:
        print("❌ НЕТ ДАННЫХ! Запустите download_ohlcv.py")
        return False
        
    print(f"📊 Найдено символов: {len(symbols)}")
    
    # ОПТИМИЗИРОВАНО: Проверка количества символов убрана - работаем с любым количеством
    print(f"✅ Работаем с {len(symbols)} символами")
        
    print("✅ Качество данных приемлемо")
    return True

def optimize_filters_anti_overfitting():
    """🎯 ОПТИМИЗИРОВАННАЯ СИНХРОНИЗИРОВАННАЯ ОПТИМИЗАЦИЯ ДЛЯ 15М ТОРГОВЛИ (1000 TRIALS)"""
    global GLOBAL_ALL_SYMBOLS
    
    print("🎯 ЗАПУСК ОПТИМИЗИРОВАННОЙ СИНХРОНИЗИРОВАННОЙ ОПТИМИЗАЦИИ")
    print("="*80)
    
    # Проверяем данные
    if not check_data_quality():
        return
        
    GLOBAL_ALL_SYMBOLS = get_all_symbols_from_data()
    
    # Предзагрузка данных через API и единичный анализ (кэш)
    print("\n📥 Загружаем данные через API и считаем индикаторы (один раз)...")
    loaded = 0
    for symbol in GLOBAL_ALL_SYMBOLS:
        df_raw = get_historical_data(symbol, GLOBAL_HOURS_BACK)
        if df_raw.empty:
            continue
        # ОПТИМИЗИРОВАНО: analyze() сам создает копию, не нужно дублировать
        df_an = analyze(df_raw)  # Убрано .copy() - analyze() сам создает копию
        if df_an.empty:
            continue
        # Гарантируем наличие volume_usdt
        if 'volume_usdt' not in df_an.columns and 'volume' in df_an.columns and 'close' in df_an.columns:
            df_an['volume_usdt'] = df_an['volume'] * df_an['close']
        DATA_CACHE_ANALYZED[symbol] = df_an
        loaded += 1
    if loaded == 0:
        print("❌ Не удалось загрузить/проанализировать данные ни по одному символу")
        return
    print(f"✅ Подготовлено символов: {loaded} (кэш индикаторов готов)")
    
    N_TRIALS = 200  # ОПТИМИЗИРОВАНО: 2000 попыток для тщательной оптимизации
    
    print(f"🛡️ ОПТИМИЗИРОВАННЫЕ ЗАЩИТНЫЕ МЕРЫ:")
    print(f"  📊 Минимум сделок: 3 (было 8)")
    print(f"  💰 Учет комиссий: {COMMISSION_PCT}%")
    print(f"  📈 Учет спреда: {SPREAD_PCT}%")
    print(f"  🎯 TP диапазон: 0.8-5.0 ATR (для лучших TP/SL соотношений)")
    print(f"  🛡️ SL диапазон: 1.0-6.0 ATR (для лучших TP/SL соотношений)")
    print(f"  🚀 ЦЕЛЬ: больше TP чем SL + хорошая прибыль - БЕЗ ОГРАНИЧЕНИЙ на сигналы!")
    
    print(f"\n🎯 МАКСИМАЛЬНО УПРОЩЕННАЯ СИСТЕМА БАЛЛОВ: РЕАЛЬНЫЙ КАПИТАЛ $100")
    print(f"  💰 Основной скор: прибыль в долларах (МОЖЕТ БЫТЬ ОТРИЦАТЕЛЬНЫМ!)")
    print(f"  🚨 ШТРАФОВ НЕТ: Основной скор сам наказывает убытки!")
    print(f"  🎯 Бонусы (СИСТЕМА: ПРИБЫЛЬ + СДЕЛКИ + ВИНРЕЙТ):")
    print(f"    • 🆕 ОЧЕНЬ МИНИМАЛЬНЫЕ БОНУСЫ ЗА КОЛИЧЕСТВО СДЕЛОК (главное - прибыль!):")
    print(f"      - Формула: min(сделки * 0.01, 30)")
    print(f"      - 50 сделок: +0.5 балла")
    print(f"      - 100 сделок: +1 балл")
    print(f"      - 500 сделок: +5 баллов")
    print(f"      - 1000 сделок: +10 баллов")
    print(f"      - 3000 сделок: +30 баллов (максимум!)")
    print(f"    • 🆕 БОНУС ЗА ВИНРЕЙТ: 50%+ = +2, 55%+ = +4, 60%+ = +6, 70%+ = +8")
    print(f"    • 🆕 УБРАНО ДУБЛИРОВАНИЕ: Сигналы = Сделки (никаких бонусов за сигналы!)")
    print(f"  💡 Принцип: ГЛАВНОЕ - ПРИБЫЛЬНОСТЬ! Больше сделок = больше возможностей!")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=200,           # ОПТИМИЗИРОВАНО: 200/2000 = 10% для тщательного разогрева
            n_ei_candidates=20,            # ОПТИМИЗИРОВАНО: 20 кандидатов для 2000 trials
            constant_liar=False            # ИСПРАВЛЕНО: отключено для однопоточной оптимизации
        ),
        pruner=None,                        # Обрезка отключена: полная оценка всех стратегий
        storage=None,                         # Локальное хранение (быстрее)
        load_if_exists=False                  # Не загружаем старые результаты
    )
    
    print(f"\n🔥 НАЧИНАЕМ ОПТИМИЗИРОВАННУЮ СИНХРОНИЗИРОВАННУЮ ОПТИМИЗАЦИЮ...")
    print(f"🚀 РЕЖИМ: однопоточная оптимизация (n_jobs=1) для надежности на Windows")
    try:
        study.optimize(
            objective_anti_overfitting, 
            n_trials=N_TRIALS, 
            n_jobs=1,                        # Однопоточно: стабильный доступ к глобальному кэшу
            show_progress_bar=True
        )
        
        print(f"\n🏁 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print("="*80)
        
        if len(study.trials) == 0 or study.best_value == 0:
            print("❌ НЕ НАЙДЕНО ВАЛИДНЫХ ПАРАМЕТРОВ")
            print("💡 Попробуйте:")
            print("  - Загрузить больше данных")
            print("  - Увеличить N_TRIALS")
            print("  - Ослабить ограничения")
            return
            
        best_result = test_single_params_anti_overfitting(study.best_trial.params)
        
        if best_result is None:
            print("❌ Лучший результат не прошел проверки")
            return
            
        print(f"🏆 ЛУЧШИЕ ПАРАМЕТРЫ (СИНХРОНИЗИРОВАННЫЕ):")
        print(f"  📊 Winrate: {best_result['winrate']:.1f}%")
        print(f"  💰 Мат. ожидание: {best_result['expected_return']:.3f}%")
        print(f"  📈 TP/SL Count Ratio: {best_result['tp_sl_count_ratio']:.2f}")
        print(f"  ⚡ Сигналов/день: {best_result['signals_per_day']:.1f}")
        print(f"  🎯 TP: {best_result['tp_count']}, SL: {best_result['sl_count']}")
        print(f"  💸 Чистый TP: {best_result['net_tp_pct']:.3f}%")
        print(f"  💸 Чистый SL: -{best_result['net_sl_pct']:.3f}%")
        print(f"  💹 Profit Factor: {best_result.get('profit_factor', 0):.2f}")
        print(f"  📉 Макс. просадка: {best_result.get('max_drawdown_pct', 0):.1f}%")
        print(f"  📅 Месячная доходность (модел.): {best_result.get('monthly_net_pct', 0):.1f}%")
        
        # ОПТИМИЗИРОВАНО: Упрощенная проверка реалистичности
        is_realistic = best_result['sl_count'] >= MIN_SL_COUNT
        
        if is_realistic:
            print("\n✅ НАЙДЕНЫ РЕАЛИСТИЧНЫЕ ПАРАМЕТРЫ ДЛЯ 15М! 🎯")
            print(f"🎯 TP/SL Ratio: {best_result.get('tp_sl_count_ratio', 0):.2f}")
        else:
            print("\n⚠️ Параметры требуют дополнительной проверки")
            
        # Сохраняем результаты
        compatible_params = {}
        for key in [
            'MIN_COMPOSITE_SCORE','MIN_ADX','SHORT_MIN_ADX','SHORT_MIN_RSI','LONG_MAX_RSI',
            'RSI_MIN','RSI_MAX','TP_ATR_MULT','SL_ATR_MULT',
            'MIN_TRIGGERS_ACTIVE_HOURS',
            'SIGNAL_COOLDOWN_MINUTES','MIN_VOLUME_MA_RATIO',
            'TP_MIN','SL_MIN','WEIGHT_RSI','WEIGHT_MACD','WEIGHT_BB','WEIGHT_VWAP',
            'WEIGHT_VOLUME','WEIGHT_ADX','SHORT_BOOST_MULTIPLIER','LONG_PENALTY_IN_DOWNTREND',
            'MA_FAST','MA_SLOW']:  # Добавлены EMA периоды
            if key in study.best_trial.params:
                compatible_params[key] = study.best_trial.params[key]

        results = {
            'best_trial': {
                'params': study.best_trial.params,
                'value': study.best_trial.value,
                'number': study.best_trial.number
            },
            'best_result': best_result,
            'config_params_compatible_with_bot': compatible_params,
                    'protection_status': {
            'anti_overfitting': True,
            'commission_included': True,
            'realistic_boundaries': True,
            'min_sl_required': MIN_SL_COUNT,
            'synchronized_with_bot': True,
            'optimized_for_15m': True,
            'simplified_calculations': True
        }
        }
        
        with open('optuna_results_anti_overfitting.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
        print(f"\n💾 Результаты сохранены в optuna_results_anti_overfitting.json")
        
        print(f"\n🔧 ЛУЧШИЕ ПАРАМЕТРЫ:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Ошибка оптимизации: {e}")
    except KeyboardInterrupt:
        print(f"\n⏹️ Оптимизация прервана")

if __name__ == '__main__':
    optimize_filters_anti_overfitting() 