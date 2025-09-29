#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ccxt
import pandas as pd
import numpy as np
from config import *
from crypto_signal_bot import SYMBOLS
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

# Глобальные EMA комбинации для оптимизации
EMA_COMBINATIONS = [
    (8, 21),   # 🏆 ЛУЧШАЯ: Классическая, стабильная, Win Rate 100%
    (6, 15),   # 🥈 ВТОРАЯ: Быстрая реакция, много сигналов
    (12, 26),  # 🥉 ТРЕТЬЯ: Медленная, трендовая, надежная
    (9, 18),   # 🎯 ЧЕТВЕРТАЯ: Сбалансированная реакция
    (10, 20),  # 🎯 ПЯТАЯ: Универсальная, проверенная временем
]

def analyze_with_params(df, params):
    """Анализ данных с параметрами индикаторов"""
    try:
        ma_slow = params['MA_SLOW']
        if df.empty or len(df) < ma_slow:
            return pd.DataFrame()
        
        df = df.copy()
        rsi_window = params['RSI_WINDOW']
        rsi_extreme_oversold = params['RSI_EXTREME_OVERSOLD']
        rsi_extreme_overbought = params['RSI_EXTREME_OVERBOUGHT']
        atr_window = params['ATR_WINDOW']
        adx_window = params['ADX_WINDOW']
        bb_window = params['BB_WINDOW']
        bb_std_dev = params['BB_STD_DEV']
        macd_fast = params['MACD_FAST']
        macd_slow = params['MACD_SLOW']
        macd_signal = params['MACD_SIGNAL']
        stoch_rsi_k = params['STOCH_RSI_K']
        stoch_rsi_d = params['STOCH_RSI_D']
        stoch_rsi_length = params['STOCH_RSI_LENGTH']
        
        # EMA с оптимизируемыми периодами
        ma_fast = params['MA_FAST']
        ma_slow = params['MA_SLOW']
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=ma_fast)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=ma_slow)
        
        # MACD через класс ta.trend.MACD
        macd_obj = ta.trend.MACD(
            close=df['close'],
            window_slow=macd_slow,
            window_fast=macd_fast,
            window_sign=macd_signal
        )
        df['macd_line'] = macd_obj.macd()
        df['macd_signal'] = macd_obj.macd_signal()
        df['macd'] = macd_obj.macd_diff()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_window)
        
        # Stochastic RSI
        stoch_rsi = ta.momentum.stochrsi(df['close'], window=stoch_rsi_length, smooth1=stoch_rsi_k, smooth2=stoch_rsi_d)
        df['stoch_rsi_k'] = stoch_rsi
        
        # ADX
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=adx_window)
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_window)
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=bb_window, window_dev=bb_std_dev)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bollinger_high'] = df['bb_upper']
        df['bollinger_low'] = df['bb_lower']
        
        # VWAP (если включен)
        if USE_VWAP:
            df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
            df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Объёмные фильтры (если включены)
        if USE_VOLUME_FILTER:
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
GLOBAL_HOURS_BACK = 1200  # УВЕЛИЧЕНО: ~50 дней истории для более надежной оптимизации
GLOBAL_ACTIVE_HOURS_UTC = ACTIVE_HOURS_UTC  # из config.py
 
GLOBAL_ALL_SYMBOLS = []
DATA_CACHE_ANALYZED: Dict[str, pd.DataFrame] = {}

# --- УПРОЩЕННЫЕ ОГРАНИЧЕНИЯ ДЛЯ 15М ФЬЮЧЕРСОВ ---
MIN_SL_COUNT = 2      # Минимум SL сделок для статистики

def get_all_symbols_from_data():
    """Используем те же символы что и в crypto_signal_bot.py"""
    return SYMBOLS.copy()

def get_historical_data(symbol, hours_back=72):
    """Загружает исторические данные через API"""
    try:
        candles_needed = int(hours_back * 60 / 15) + 120
        all_ohlcv = []

        now_ms = EXCHANGE.milliseconds()
        since = now_ms - hours_back * 60 * 60 * 1000
        tf_ms = EXCHANGE.parse_timeframe(TIMEFRAME) * 1000

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

        if not all_ohlcv or len(all_ohlcv) < 50:
            logging.warning(f"{symbol}: недостаточно данных для анализа ({len(all_ohlcv)})")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['volume_usdt'] = df['volume'] * df['close']

        logging.info(f"Загружено {len(df)} свечей для {symbol}")
        return df

    except ccxt.RateLimitExceeded as e:
        wait_time = e.retry_after
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
    """Параметры для оптимизации"""
    
    # EMA периоды: ТОП-5 комбинаций
    ma_idx = trial.suggest_int('MA_COMBINATION', 0, len(EMA_COMBINATIONS) - 1)
    ma_fast, ma_slow = EMA_COMBINATIONS[ma_idx]
    
    params = {
        'MA_FAST': ma_fast,
        'MA_SLOW': ma_slow,
    }
    
    # RSI фильтры
    rsi_min = trial.suggest_int('RSI_MIN', 20, 80, step=2)
    rsi_max = trial.suggest_int('RSI_MAX', rsi_min + 2, 96, step=2)
    
    long_max_rsi = trial.suggest_int('LONG_MAX_RSI', 10, 80, step=2)
    short_min_rsi = trial.suggest_int('SHORT_MIN_RSI', 20, 90, step=2)
    
    params.update({
        # Основные пороги
        'MIN_COMPOSITE_SCORE': trial.suggest_float('MIN_COMPOSITE_SCORE', 0, 1.5, step=0.5),
        'MIN_ADX': trial.suggest_int('MIN_ADX', 6, 40, step=2),
        'SHORT_MIN_ADX': trial.suggest_int('SHORT_MIN_ADX', 20, 52, step=2),
        
        # RSI фильтры
        'RSI_MIN': rsi_min,
        'RSI_MAX': rsi_max,
        'LONG_MAX_RSI': long_max_rsi,
        'SHORT_MIN_RSI': short_min_rsi,
        
        # TP/SL
        'TP_ATR_MULT': trial.suggest_float('TP_ATR_MULT', 0.8, 6.0, step=0.2),
        'SL_ATR_MULT': trial.suggest_float('SL_ATR_MULT', 1.0, 8.0, step=0.2),

        # Триггеры
        'MIN_TRIGGERS_ACTIVE_HOURS': trial.suggest_float('MIN_TRIGGERS_ACTIVE_HOURS', 0, 6.0, step=0.3),
        
        # Временные фильтры
        'SIGNAL_COOLDOWN_MINUTES': trial.suggest_int('SIGNAL_COOLDOWN_MINUTES', 15, 60, step=15),
        
        # Объем
        'MIN_VOLUME_MA_RATIO': trial.suggest_float('MIN_VOLUME_MA_RATIO', 0, 3.0, step=0.05),
        
        # Веса скоринга
        'WEIGHT_RSI': trial.suggest_float('WEIGHT_RSI', 0.0, 10.0, step=0.2),
        'WEIGHT_MACD': trial.suggest_float('WEIGHT_MACD', 0.0, 9.0, step=0.2),
        'WEIGHT_BB': trial.suggest_float('WEIGHT_BB', 0.0, 6.0, step=0.2),
        'WEIGHT_VWAP': trial.suggest_float('WEIGHT_VWAP', 0.0, 12.0, step=0.2),
        'WEIGHT_VOLUME': trial.suggest_float('WEIGHT_VOLUME', 0.0, 6.0, step=0.2),
        'WEIGHT_ADX': trial.suggest_float('WEIGHT_ADX', 0.0, 12.0, step=0.2),
        
        # Множители направления
        'SHORT_BOOST_MULTIPLIER': trial.suggest_float('SHORT_BOOST_MULTIPLIER', 0, 5.0, step=0.2),
        'LONG_PENALTY_IN_DOWNTREND': trial.suggest_float('LONG_PENALTY_IN_DOWNTREND', 0.0, 1.0, step=0.05),

        # Минимальные TP/SL
        'TP_MIN': trial.suggest_float('TP_MIN', 0.006, 0.10, step=0.002),
        'SL_MIN': trial.suggest_float('SL_MIN', 0.01, 0.10, step=0.002),
        
        # Параметры индикаторов
        'RSI_WINDOW': trial.suggest_categorical('RSI_WINDOW', [5, 7, 9, 12, 14, 18, 21, 24]),
        'RSI_EXTREME_OVERSOLD': trial.suggest_categorical('RSI_EXTREME_OVERSOLD', [10, 12, 15, 18, 20, 22, 25, 28]),
        'RSI_EXTREME_OVERBOUGHT': trial.suggest_categorical('RSI_EXTREME_OVERBOUGHT', [72, 75, 78, 80, 82, 85, 88, 90]),
        
        'ATR_WINDOW': trial.suggest_categorical('ATR_WINDOW', [5,7, 10, 12, 14, 16, 18, 20, 24]),
        'ADX_WINDOW': trial.suggest_categorical('ADX_WINDOW', [7, 10, 12, 14, 16, 18, 20, 24]),
        
        'BB_WINDOW': trial.suggest_categorical('BB_WINDOW', [10, 12, 15, 18, 20, 22, 25, 28, 30]),
        'BB_STD_DEV': trial.suggest_categorical('BB_STD_DEV', [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2]),
        
        'MACD_FAST': trial.suggest_categorical('MACD_FAST', [4, 6, 8, 10, 12, 14, 16, 18, 20]),
        'MACD_SLOW': trial.suggest_categorical('MACD_SLOW', [18, 21, 24, 26, 28, 30, 32, 35]),
        'MACD_SIGNAL': trial.suggest_categorical('MACD_SIGNAL', [4, 6, 8, 9, 10, 12, 14, 16]),
        
        'VWAP_DEVIATION_THRESHOLD': trial.suggest_categorical('VWAP_DEVIATION_THRESHOLD', [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2]),
        
        'STOCH_RSI_K': trial.suggest_categorical('STOCH_RSI_K', [1,     2, 3, 4, 5, 6, 8, 10, 12]),
        'STOCH_RSI_D': trial.suggest_categorical('STOCH_RSI_D', [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        'STOCH_RSI_LENGTH': trial.suggest_categorical('STOCH_RSI_LENGTH', [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]),
    })
    
    return params

def evaluate_signal_strength_optimized(df, current_index, symbol, action, weights, params):
    """Оценка силы сигнала"""
    try:
        if current_index < 5 or current_index >= len(df):
            return 0
            
        last = df.iloc[current_index]
        prev = df.iloc[current_index-1] if current_index > 0 else last
        
        return evaluate_signal_strength_with_weights_fast(last, prev, action, weights, params)
        
    except Exception as e:
        logging.error(f"Ошибка в evaluate_signal_strength_optimized: {e}")
        return 0

def evaluate_signal_strength_with_weights_fast(last, prev, action, weights, params):
    """Быстрая оценка силы сигнала"""
    try:
        score = 0
        
        # RSI анализ
        rsi_score = 0
        rsi_extreme_oversold = params['RSI_EXTREME_OVERSOLD']
        rsi_extreme_overbought = params['RSI_EXTREME_OVERBOUGHT']
        rsi_oversold = params['RSI_MIN']
        rsi_overbought = params['RSI_MAX']
        
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
                
        score += rsi_score * weights['WEIGHT_RSI']
        
        # MACD анализ
        macd_score = 0
        if ('macd_line' in last) and ('macd_signal' in last):
            macd_cross = last['macd_line'] - last['macd_signal']
            prev_macd_cross = prev['macd_line'] - prev['macd_signal']
            
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
                    
        score += macd_score * weights['WEIGHT_MACD']
        
        # Bollinger Bands
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
                    
        score += bb_score * weights['WEIGHT_BB']
        
        # VWAP анализ
        vwap_score = 0
        if 'vwap' in last:
            vwap_deviation_threshold = params['VWAP_DEVIATION_THRESHOLD']
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
                    
        score += vwap_score * weights['WEIGHT_VWAP']
        
        # Объём анализ
        volume_score = 0
        if 'volume_ratio_usdt' in last:
            vol_ratio = last.get('volume_ratio_usdt', 1.0)
            if vol_ratio >= 2.0:
                volume_score = 1.5
            elif vol_ratio >= 1.5:
                volume_score = 1.0
            elif vol_ratio >= 1.2:
                volume_score = 0.5
                
        score += volume_score * weights['WEIGHT_VOLUME']
        
        # ADX анализ
        adx_score = 0
        min_adx = params['MIN_ADX']
        
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
            
        score += adx_score * weights['WEIGHT_ADX']
        
        # Корректировки для SHORT/LONG
        if action == 'SELL':
            score *= params['SHORT_BOOST_MULTIPLIER']
        elif action == 'BUY':
            if ('ema_fast' in last) and ('ema_slow' in last):
                if last['ema_fast'] < last['ema_slow']:
                    score *= params['LONG_PENALTY_IN_DOWNTREND']
        
        return max(0, score)
        
    except Exception as e:
        logging.error(f"Ошибка в evaluate_signal_strength_with_weights_fast: {e}")
        return 0

def simulate_signals_anti_overfitting(df, symbol, params, active_hours_utc):
    """Симуляция сигналов"""
    min_candles_needed = MIN_15M_CANDLES + 96
    if df.empty or len(df) < min_candles_needed:
        logging.warning(f"🚫 {symbol}: Недостаточно данных для симуляции ({len(df)} < {min_candles_needed})")
        return []
    
    df_analyzed = analyze_with_params(df, params)
    
    if df_analyzed.empty:
        logging.warning(f"🚫 {symbol}: Пустой DataFrame после анализа")
        return []
    
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
    
    # Исключаем последние 96 свечей для предотвращения look-ahead bias
    for i in range(MIN_15M_CANDLES, len(df_analyzed) - 96):
        last = df_analyzed.iloc[i]
        prev = df_analyzed.iloc[i-1] if i > 0 else df_analyzed.iloc[i]
        now = last['timestamp']
            
        # Кулдаун
        if last_signal_time and (now - last_signal_time).total_seconds() < signal_cooldown_minutes * 60:
            continue
            
        # Базовые фильтры
        if last['adx'] < min_adx:
            continue
            
        # Проверка объёма
        min_volume_ratio = params['MIN_VOLUME_MA_RATIO']
        if 'volume_ratio_usdt' in last and last.get('volume_ratio_usdt', 1.0) < min_volume_ratio:
            continue
        
        # Триггеры
        buy_triggers = 0
        sell_triggers = 0
        
        # RSI триггеры
        rsi_extreme_oversold = params['RSI_EXTREME_OVERSOLD']
        rsi_extreme_overbought = params['RSI_EXTREME_OVERBOUGHT']
        
        if last['rsi'] <= rsi_extreme_oversold:
            buy_triggers += 2.0
        elif last['rsi'] < rsi_min:
            buy_triggers += 1.0
        if last['rsi'] >= rsi_extreme_overbought:
            sell_triggers += 2.0
        elif last['rsi'] > rsi_max:
            sell_triggers += 1.0
        
        # EMA кроссовер
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            buy_triggers += 1.5
        elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
            buy_triggers += 0.5
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            sell_triggers += 1.5
        elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
            sell_triggers += 0.5
            
        # MACD триггеры
        if ('macd_line' in last) and ('macd_signal' in last):
            if last['macd_line'] > last['macd_signal']:
                buy_triggers += 0.5
            if last['macd_line'] < last['macd_signal']:
                sell_triggers += 0.5
        
        # Bollinger Bands
        if ('bollinger_low' in last) and ('bollinger_high' in last):
            denom = max((last['bollinger_high'] - last['bollinger_low']), 1e-12)
            bb_position = (last['close'] - last['bollinger_low']) / denom
            if bb_position <= 0.25:
                buy_triggers += 0.5
            if bb_position >= 0.75:
                sell_triggers += 0.5
                
        # Минимальные триггеры
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

        if signal_type:
            try:
                weights = {k: params[k] for k in params if k.startswith('WEIGHT_')}
                score = evaluate_signal_strength_optimized(df_analyzed, i, symbol, signal_type, weights, params)
                if score >= min_composite_score:
                    entry_price = last['close']
                    entry_time = now
                    future_data = df_analyzed.iloc[i+1:i+385]
                    
                    if len(future_data) >= 4:
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
                        tp_pct_min = params['TP_MIN']
                        sl_pct_min = params['SL_MIN']

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

                        # Векторизованный поиск TP/SL
                        result = None
                        future_highs = future_data['high'].values
                        future_lows = future_data['low'].values
                        
                        if signal_type == 'BUY':
                            tp_hits = future_highs >= tp_price
                            sl_hits = future_lows <= sl_price
                            if np.any(tp_hits):
                                tp_idx = np.where(tp_hits)[0][0]
                                sl_idx = np.where(sl_hits)[0][0] if np.any(sl_hits) else len(future_highs)
                                result = 'tp' if tp_idx <= sl_idx else 'sl'
                            elif np.any(sl_hits):
                                result = 'sl'
                        else:  # SELL
                            tp_hits = future_lows <= tp_price
                            sl_hits = future_highs >= sl_price
                            if np.any(tp_hits):
                                tp_idx = np.where(tp_hits)[0][0]
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

    return signals

def restore_ema_params_from_combination(params):
    """Восстанавливает MA_FAST и MA_SLOW из MA_COMBINATION"""
    if 'MA_COMBINATION' in params:
        ma_idx = params['MA_COMBINATION']
        ma_fast, ma_slow = EMA_COMBINATIONS[ma_idx]
        params['MA_FAST'] = ma_fast
        params['MA_SLOW'] = ma_slow
    
    return params

def test_single_params_anti_overfitting(params, hours_back=None, active_hours_utc=None):
    """Тестирует один набор параметров"""
    if hours_back is None:
        hours_back = GLOBAL_HOURS_BACK
    if active_hours_utc is None:
        active_hours_utc = GLOBAL_ACTIVE_HOURS_UTC
    
    params = restore_ema_params_from_combination(params)
        
    all_signals = []
    
    for symbol in GLOBAL_ALL_SYMBOLS:
        df_raw = DATA_CACHE_ANALYZED.get(symbol)
        if df_raw is None or df_raw.empty:
            continue
        signals = simulate_signals_anti_overfitting(df_raw, symbol, params, active_hours_utc)
        all_signals.extend(signals)
    
    if not all_signals:
        return None
    
    # Подсчет результатов
    tp_signals = [s for s in all_signals if s['result'] == 'tp']
    sl_signals = [s for s in all_signals if s['result'] == 'sl']
    tp_count = len(tp_signals)
    sl_count = len(sl_signals)
    
    total_trades = tp_count + sl_count
    if total_trades == 0:
        return None
    
    # Основные метрики
    winrate = tp_count / total_trades * 100
    tp_sum = sum([s['tp_pct'] for s in tp_signals])
    sl_sum = sum([s['sl_pct'] for s in sl_signals])
    avg_tp_pct = tp_sum / max(tp_count, 1)
    avg_sl_pct = sl_sum / max(sl_count, 1)
    
    # Дополнительные метрики
    tp_sl_count_ratio = tp_count / max(sl_count, 1)
    effective_hours = max(hours_back - 24 * 4, 1)
    signals_per_day = len(all_signals) / (effective_hours / 24)
    
    # Математическое ожидание
    winrate_decimal = winrate / 100
    expected_return = winrate_decimal * avg_tp_pct - (1 - winrate_decimal) * avg_sl_pct
    
    # Месячная доходность
    avg_net_pct = (tp_sum - sl_sum) / len(all_signals)
    monthly_net_pct = avg_net_pct * signals_per_day * 30.0
    
    return {
        'signals': len(all_signals),
        'signals_per_day': signals_per_day,
        'winrate': winrate,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'tp_sl_count_ratio': tp_sl_count_ratio,
        'avg_tp_pct': avg_tp_pct,
        'avg_sl_pct': avg_sl_pct,
        'expected_return': expected_return,
        'monthly_net_pct': monthly_net_pct
    }

def calculate_advanced_score(result: dict, trial_number: int) -> float:
    """Упрощенная система баллов без комиссий"""
    
    winrate = result['winrate']
    tp_count = result['tp_count']
    sl_count = result['sl_count']
    avg_tp_pct = result.get('avg_tp_pct', 0)
    avg_sl_pct = result.get('avg_sl_pct', 0)
    
    total_trades = tp_count + sl_count
    if total_trades == 0:
        return 0.0
    
    # Простое математическое ожидание (в процентах)
    winrate_decimal = winrate / 100.0
    expected_return = winrate_decimal * avg_tp_pct - (1 - winrate_decimal) * avg_sl_pct
    
    # Базовый скор - математическое ожидание
    base_score = expected_return
    
    # Бонус за количество сделок
    trades_bonus = min(total_trades * 0.1, 20)
    
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
    final_score = base_score + trades_bonus + winrate_bonus
    
    # Логирование
    if final_score > 10 and trial_number % 20 == 0:
        logging.info(f"Trial {trial_number}: Score={final_score:.1f} | "
                    f"Expected: {expected_return:.2f}% | "
                    f"Trades: {total_trades} ({tp_count} TP / {sl_count} SL) | "
                    f"Winrate: {winrate:.1f}%")
    
    return final_score

def objective_anti_overfitting(trial: optuna.Trial) -> float:
    """Упрощенная целевая функция"""
    try:
        params = suggest_parameters_anti_overfitting(trial)
        result = test_single_params_anti_overfitting(params)
        
        if result is None or result['signals'] == 0:
            return 0.0
        
        # Минимальная проверка
        if result['tp_count'] + result['sl_count'] < 3:
            return 0.0
        
        score = calculate_advanced_score(result, trial.number)
        return score
        
    except Exception as e:
        logging.error(f"Ошибка в objective_anti_overfitting: {e}")
        return 0.0

def check_data_quality():
    """Проверка качества данных"""
    symbols = get_all_symbols_from_data()
    if len(symbols) == 0:
        print("❌ НЕТ ДАННЫХ! Запустите download_ohlcv.py")
        return False
        
    print(f"📊 Найдено символов: {len(symbols)}")
    print("✅ Качество данных приемлемо")
    return True

def optimize_filters_anti_overfitting():
    """Оптимизация параметров для 15м торговли"""
    global GLOBAL_ALL_SYMBOLS
    
    print("🎯 ЗАПУСК ОПТИМИЗАЦИИ ПАРАМЕТРОВ")
    print("="*50)
    
    # Проверяем данные
    if not check_data_quality():
        return
        
    GLOBAL_ALL_SYMBOLS = get_all_symbols_from_data()
    
    # Загружаем данные
    print("\n📥 Загружаем данные...")
    loaded = 0
    for symbol in GLOBAL_ALL_SYMBOLS:
        df_raw = get_historical_data(symbol, GLOBAL_HOURS_BACK)
        if df_raw.empty:
            continue
        DATA_CACHE_ANALYZED[symbol] = df_raw
        loaded += 1
    if loaded == 0:
        print("❌ Не удалось загрузить данные")
        return
    print(f"✅ Загружено символов: {loaded}")
    
    N_TRIALS = 3000
    
    print(f"\n🛡️ ПАРАМЕТРЫ ОПТИМИЗАЦИИ:")
    print(f"  📊 Минимум сделок: 3")
    print(f"  🎯 TP диапазон: 0.8-6.0 ATR")
    print(f"  🛡️ SL диапазон: 1.0-8.0 ATR")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=300,
            n_ei_candidates=50,
            constant_liar=False
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=50, n_warmup_steps=10),
        storage=None,
        load_if_exists=False
    )
    
    print(f"\n🔥 НАЧИНАЕМ ОПТИМИЗАЦИЮ...")
    try:
        study.optimize(
            objective_anti_overfitting, 
            n_trials=N_TRIALS, 
            n_jobs=1,
            show_progress_bar=True
        )
        
        print(f"\n🏁 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print("="*50)
        
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
            
        print(f"🏆 ЛУЧШИЕ ПАРАМЕТРЫ:")
        print(f"  📊 Winrate: {best_result['winrate']:.1f}%")
        print(f"  💰 Мат. ожидание: {best_result['expected_return']:.3f}%")
        print(f"  📈 TP/SL Ratio: {best_result['tp_sl_count_ratio']:.2f}")
        print(f"  ⚡ Сигналов/день: {best_result['signals_per_day']:.1f}")
        print(f"  🎯 TP: {best_result['tp_count']}, SL: {best_result['sl_count']}")
        print(f"  💸 TP: {best_result['avg_tp_pct']:.3f}%")
        print(f"  💸 SL: -{best_result['avg_sl_pct']:.3f}%")
        print(f"  📅 Месячная доходность: {best_result.get('monthly_net_pct', 0):.1f}%")
            
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
            'config_params_compatible_with_bot': compatible_params
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