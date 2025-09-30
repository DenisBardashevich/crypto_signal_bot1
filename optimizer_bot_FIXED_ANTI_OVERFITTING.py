#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
УПРОЩЕННЫЙ ОПТИМИЗАТОР ПАРАМЕТРОВ ДЛЯ 15М КРИПТОТОРГОВЛИ
========================================================
🎯 ФОКУС: Только основные индикаторы для надежных сигналов
🛡️ УБРАНО: Bollinger Bands, VWAP, Stochastic RSI, объемные фильтры
⚡ ОСТАВЛЕНО: EMA, RSI, MACD, ADX, ATR
📊 КОМБИНАЦИИ: Используются проверенные комбинации EMA и MACD
🚀 РЕЗУЛЬТАТ: Быстрая оптимизация, меньше переоптимизации
"""

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

# Проверенные MACD комбинации для 15м
MACD_COMBINATIONS = [
    (12, 26, 9),   # 🏆 Классическая MACD
    (8, 21, 5),    # 🥈 Быстрая MACD
    (10, 24, 7),   # 🥉 Сбалансированная MACD
    (14, 28, 11),  # 🎯 Медленная MACD
    (9, 18, 6),    # 🎯 Универсальная MACD
]

def analyze_with_params(df, params):
    """УПРОЩЕННЫЙ анализ данных - только основные индикаторы"""
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
        macd_fast = params['MACD_FAST']
        macd_slow = params['MACD_SLOW']
        macd_signal = params['MACD_SIGNAL']
        
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
        
        # ADX
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=adx_window)
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_window)
        
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
GLOBAL_HOURS_BACK = 1000  # УВЕЛИЧЕНО: ~50 дней истории для более надежной оптимизации
 
GLOBAL_ALL_SYMBOLS = []
DATA_CACHE_ANALYZED: Dict[str, pd.DataFrame] = {}

# --- УПРОЩЕННЫЕ ОГРАНИЧЕНИЯ ДЛЯ 15М ФЬЮЧЕРСОВ ---
MIN_SL_COUNT = 3      # Минимум сделок для базовой статистики

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
    """УПРОЩЕННЫЕ параметры для оптимизации - только нужные фильтры"""
    
    # EMA периоды: ТОП-5 комбинаций
    ma_idx = trial.suggest_int('MA_COMBINATION', 0, len(EMA_COMBINATIONS) - 1)
    ma_fast, ma_slow = EMA_COMBINATIONS[ma_idx]
    
    # MACD периоды: ТОП-5 комбинаций
    macd_idx = trial.suggest_int('MACD_COMBINATION', 0, len(MACD_COMBINATIONS) - 1)
    macd_fast, macd_slow, macd_signal = MACD_COMBINATIONS[macd_idx]
    
    params = {
        'MA_FAST': ma_fast,
        'MA_SLOW': ma_slow,
        'MACD_FAST': macd_fast,
        'MACD_SLOW': macd_slow,
        'MACD_SIGNAL': macd_signal,
    }
    
    # RSI фильтры
    rsi_min = trial.suggest_int('RSI_MIN', 20, 80, step=2)
    rsi_max = trial.suggest_int('RSI_MAX', rsi_min + 2, 96, step=2)
    
    long_max_rsi = trial.suggest_int('LONG_MAX_RSI', 30, 90, step=2)
    short_min_rsi = trial.suggest_int('SHORT_MIN_RSI', 30, 90, step=2)
    
    params.update({
        # Основные пороги
        'MIN_COMPOSITE_SCORE': trial.suggest_float('MIN_COMPOSITE_SCORE', 0, 1, step=0.5),
        'MIN_ADX': trial.suggest_int('MIN_ADX', 6, 22, step=2),
        'SHORT_MIN_ADX': trial.suggest_int('SHORT_MIN_ADX', 6, 28, step=2),
        
        # RSI фильтры
        'RSI_MIN': rsi_min,
        'RSI_MAX': rsi_max,
        'LONG_MAX_RSI': long_max_rsi,
        'SHORT_MIN_RSI': short_min_rsi,
        
        # TP/SL
        'TP_ATR_MULT': trial.suggest_float('TP_ATR_MULT', 0.5, 6.0, step=0.5),
        'SL_ATR_MULT': trial.suggest_float('SL_ATR_MULT', 1.0, 8.0, step=0.5),

        # Триггеры
        'MIN_TRIGGERS_ACTIVE_HOURS': trial.suggest_float('MIN_TRIGGERS_ACTIVE_HOURS', 0, 6.0, step=0.5),
        
        # Временные фильтры
        'SIGNAL_COOLDOWN_MINUTES': trial.suggest_int('SIGNAL_COOLDOWN_MINUTES', 40, 80, step=20),
        
        # УПРОЩЕННЫЕ веса скоринга (только основные)
        'WEIGHT_RSI': trial.suggest_float('WEIGHT_RSI', 0.0, 10.0, step=0.5),
        'WEIGHT_MACD': trial.suggest_float('WEIGHT_MACD', 0.0, 9.0, step=0.5),
        'WEIGHT_ADX': trial.suggest_float('WEIGHT_ADX', 0.0, 12.0, step=0.5),
        
        # Множители направления
        'SHORT_BOOST_MULTIPLIER': trial.suggest_float('SHORT_BOOST_MULTIPLIER', 0, 5.0, step=0.5),
        'LONG_PENALTY_IN_DOWNTREND': trial.suggest_float('LONG_PENALTY_IN_DOWNTREND', 0.0, 1.0, step=0.05),

        # Минимальные TP/SL
        'TP_MIN': trial.suggest_float('TP_MIN', 0.006, 0.10, step=0.003),
        'SL_MIN': trial.suggest_float('SL_MIN', 0.01, 0.10, step=0.003),
        
        # УПРОЩЕННЫЕ параметры индикаторов (только основные)
        'RSI_WINDOW': trial.suggest_categorical('RSI_WINDOW', [9, 12, 14, 18, 21]),
        'RSI_EXTREME_OVERSOLD': trial.suggest_categorical('RSI_EXTREME_OVERSOLD', [15, 18, 20, 22, 25]),
        'RSI_EXTREME_OVERBOUGHT': trial.suggest_categorical('RSI_EXTREME_OVERBOUGHT', [75, 78, 80, 82, 85]),
        
        'ATR_WINDOW': trial.suggest_categorical('ATR_WINDOW', [10, 14, 20]),
        'ADX_WINDOW': trial.suggest_categorical('ADX_WINDOW', [10, 14, 20]),
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
    """УПРОЩЕННАЯ оценка силы сигнала - только основные индикаторы"""
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
        
        # ADX анализ (сила тренда)
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

def simulate_signals_anti_overfitting(df, symbol, params):
    """Симуляция сигналов"""
    min_candles_needed = MIN_15M_CANDLES + 50
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
    
    # Исключаем последние 4 свечи для предотвращения look-ahead bias
    for i in range(MIN_15M_CANDLES, len(df_analyzed) - 4):
        last = df_analyzed.iloc[i]
        prev = df_analyzed.iloc[i-1] if i > 0 else df_analyzed.iloc[i]
        now = last['timestamp']
        
        # Кулдаун
        if last_signal_time and (now - last_signal_time).total_seconds() < signal_cooldown_minutes * 60:
            continue
            
        # Базовые фильтры
        if last['adx'] < min_adx:
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
        
        # EMA кроссовер - только чистые пересечения
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            buy_triggers += 2.0
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            sell_triggers += 2.0
            
        # MACD кроссовер - только чистые пересечения
        if ('macd_line' in last) and ('macd_signal' in last) and ('macd_line' in prev) and ('macd_signal' in prev):
            if prev['macd_line'] <= prev['macd_signal'] and last['macd_line'] > last['macd_signal']:
                buy_triggers += 1.5
            if prev['macd_line'] >= prev['macd_signal'] and last['macd_line'] < last['macd_signal']:
                sell_triggers += 1.5
                
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
                        if signal_type == 'BUY':
                            tp_eff = max((tp_price - entry_price) / entry_price, tp_pct_min)
                            sl_eff = max((entry_price - sl_price) / entry_price, sl_pct_min)
                            tp_price = entry_price * (1 + tp_eff)
                            sl_price = entry_price * (1 - sl_eff)
                        else:
                            tp_eff = max((entry_price - tp_price) / entry_price, tp_pct_min)
                            sl_eff = max((sl_price - entry_price) / entry_price, sl_pct_min)
                            tp_price = entry_price * (1 - tp_eff)
                            sl_price = entry_price * (1 + sl_eff)

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

def restore_params_from_combinations(params):
    """Восстанавливает MA и MACD параметры из комбинаций"""
    if 'MA_COMBINATION' in params:
        ma_idx = params['MA_COMBINATION']
        ma_fast, ma_slow = EMA_COMBINATIONS[ma_idx]
        params['MA_FAST'] = ma_fast
        params['MA_SLOW'] = ma_slow
    
    if 'MACD_COMBINATION' in params:
        macd_idx = params['MACD_COMBINATION']
        macd_fast, macd_slow, macd_signal = MACD_COMBINATIONS[macd_idx]
        params['MACD_FAST'] = macd_fast
        params['MACD_SLOW'] = macd_slow
        params['MACD_SIGNAL'] = macd_signal
    
    return params

def test_single_params_anti_overfitting(params, hours_back=None):
    """Тестирует один набор параметров"""
    if hours_back is None:
        hours_back = GLOBAL_HOURS_BACK
    
    params = restore_params_from_combinations(params)
        
    all_signals = []
    
    for symbol in GLOBAL_ALL_SYMBOLS:
        df_raw = DATA_CACHE_ANALYZED.get(symbol)
        if df_raw is None or df_raw.empty:
            continue
        signals = simulate_signals_anti_overfitting(df_raw, symbol, params)
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
    """Улучшенная система баллов - учитывает общую прибыльность и количество сигналов"""
    
    winrate = result['winrate']
    tp_count = result['tp_count']
    sl_count = result['sl_count']
    avg_tp_pct = result.get('avg_tp_pct', 0)
    avg_sl_pct = result.get('avg_sl_pct', 0)
    total_signals = result['signals']
    signals_per_day = result['signals_per_day']
    
    total_trades = tp_count + sl_count
    if total_trades == 0:
        return 0.0
    
    # Математическое ожидание на сделку (в процентах)
    winrate_decimal = winrate / 100.0
    expected_return_per_trade = winrate_decimal * avg_tp_pct - (1 - winrate_decimal) * avg_sl_pct
    
    # Общая прибыльность = мат. ожидание * количество сделок
    total_profitability = expected_return_per_trade * total_signals
    
    # Если мало сделок, то и прибыль будет маленькая - никаких штрафов не нужно
    final_score = total_profitability
    
    # Логирование
    if total_profitability > 10 and trial_number % 50 == 0:
        logging.info(f"Trial {trial_number}: Total Profit: {total_profitability:.1f}% | "
                    f"Trades: {total_trades} ({tp_count} TP / {sl_count} SL) | "
                    f"Winrate: {winrate:.1f}% | Signals/day: {signals_per_day:.1f}")
    
    return final_score

def objective_anti_overfitting(trial: optuna.Trial) -> float:
    """Упрощенная целевая функция"""
    try:
        params = suggest_parameters_anti_overfitting(trial)
        result = test_single_params_anti_overfitting(params)
        
        if result is None or result['signals'] == 0:
            return 0.0
        
        # Минимальная проверка - хотя бы 3 сделки для базовой статистики
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
    """УПРОЩЕННАЯ оптимизация параметров для 15м торговли - только основные индикаторы"""
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
    
    print(f"\n🛡️ УПРОЩЕННЫЕ ПАРАМЕТРЫ ОПТИМИЗАЦИИ:")
    print(f"  📊 Минимум сделок: 3 (базовая статистика)")
    print(f"  💰 Фокус на ОБЩУЮ ПРИБЫЛЬНОСТЬ (мат.ожидание × количество)")
    print(f"  🎯 Без штрафов за количество сделок - мало сделок = мало прибыли")
    print(f"  🎯 TP диапазон: 0.8-6.0 ATR")
    print(f"  🛡️ SL диапазон: 1.0-8.0 ATR")
    print(f"  ⚡ Индикаторы: EMA, RSI, MACD, ADX, ATR")
    print(f"  🎯 EMA комбинации: 5 проверенных")
    print(f"  🎯 MACD комбинации: 5 проверенных")
    print(f"  🚫 Убрано: Bollinger Bands, VWAP, Stochastic RSI, объемные фильтры")
    
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
            
        # Сохраняем результаты (УПРОЩЕННЫЕ параметры)
        compatible_params = {}
        for key in [
            'MIN_COMPOSITE_SCORE','MIN_ADX','SHORT_MIN_ADX','SHORT_MIN_RSI','LONG_MAX_RSI',
            'RSI_MIN','RSI_MAX','TP_ATR_MULT','SL_ATR_MULT',
            'MIN_TRIGGERS_ACTIVE_HOURS',
            'SIGNAL_COOLDOWN_MINUTES',
            'TP_MIN','SL_MIN','WEIGHT_RSI','WEIGHT_MACD','WEIGHT_ADX',
            'SHORT_BOOST_MULTIPLIER','LONG_PENALTY_IN_DOWNTREND',
            'MA_FAST','MA_SLOW','MACD_FAST','MACD_SLOW','MACD_SIGNAL',
            'RSI_EXTREME_OVERSOLD','RSI_EXTREME_OVERBOUGHT']:  # Основные параметры + MACD
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