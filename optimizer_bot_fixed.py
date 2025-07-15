#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
УЛУЧШЕННАЯ версия оптимизатора с OPTUNA для умной оптимизации
Используются только фильтры из config.py и crypto_signal_bot.py
Исправлены расчеты и логика
ИСПРАВЛЕНО: MIN_TP_SL_DISTANCE добавлен в оптимизацию
НОВОЕ: Optuna для интеллектуального поиска параметров

🎯 УЛУЧШЕНИЯ ДИАПАЗОНОВ ПАРАМЕТРОВ (2025-01-27):
✅ Расширены узкие диапазоны для более эффективного поиска
✅ Уменьшены шаги для критически важных параметров
✅ Добавлены больше вариантов для объемных фильтров
✅ Улучшены временные фильтры с очень мелким шагом (0.01)
✅ Расширены диапазоны для всех индикаторных параметров
✅ Оптимизированы веса системы с меньшими шагами
✅ Увеличено количество попыток до 1000 для тщательного поиска

🔥 КРИТИЧЕСКИ ВАЖНЫЕ УЛУЧШЕНИЯ:
- min_triggers_active_hours: шаг 0.01 (было 0.05)
- BB_SQUEEZE_THRESHOLD: шаг 0.002 (было 0.005)
- min_tp_sl_distance: шаг 0.0002 (было 0.0005)
- Все веса системы: шаг 0.05 (было 0.1)
- Объемные фильтры: 11 вариантов (было 6)
"""

import ccxt
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from config import *
from crypto_signal_bot import analyze, evaluate_signal_strength, SYMBOLS
import logging
import random
import multiprocessing as mp
from functools import partial
import glob
import optuna
import plotly
import json
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ OPTUNA ---
GLOBAL_HOURS_BACK = 200  # Загружаем ВСЕ доступные данные из CSV файлов
GLOBAL_ACTIVE_HOURS_UTC = list(range(6, 24))
GLOBAL_MIN_SIGNALS_PER_DAY = 12
GLOBAL_ALL_SYMBOLS = []

# --- РАСШИРЕННОЕ ПОИСКОВОЕ ПРОСТРАНСТВО ДЛЯ OPTUNA ---

def get_all_symbols_from_data():
    files = glob.glob('data/*_15m.csv')
    symbols = []
    for f in files:
        base = os.path.basename(f).replace('_15m.csv', '')
        symbols.append(base)
    return symbols

def suggest_parameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Функция для генерации параметров с помощью Optuna
    Теперь возвращает только CAPS-ключи для полной совместимости с config.py"""
    return {
        'MIN_SCORE': trial.suggest_float('MIN_SCORE', 1.0, 10.0, step=0.5),
        'MIN_ADX': trial.suggest_int('MIN_ADX', 2, 40, step=1),
        'SHORT_MIN_ADX': trial.suggest_int('SHORT_MIN_ADX', 2, 35, step=1),
        'SHORT_MIN_RSI': trial.suggest_int('SHORT_MIN_RSI', 10, 84, step=2),  # 84 = 10 + 37*2
        'LONG_MAX_RSI': trial.suggest_int('LONG_MAX_RSI', 40, 98, step=2),    # 98 = 40 + 29*2
        'RSI_MIN': trial.suggest_int('RSI_MIN', 1, 50, step=1),
        'RSI_MAX': trial.suggest_int('RSI_MAX', 50, 99, step=1),
        'TP_ATR_MULT': trial.suggest_float('TP_ATR_MULT', 0.2, 6.0, step=0.05),
        'SL_ATR_MULT': trial.suggest_float('SL_ATR_MULT', 0.5, 6.0, step=0.05),
        'MIN_VOLUME_USDT': trial.suggest_categorical('MIN_VOLUME_USDT', [0.001, 0.01, 0.1]),  # ИСПРАВЛЕНО: реалистичные объемы в миллионах USDT
        'MAX_SPREAD_PCT': trial.suggest_float('MAX_SPREAD_PCT', 0.001, 0.08, step=0.0005),  # (0.08-0.001)/0.0005=158
        'MIN_BB_WIDTH': trial.suggest_float('MIN_BB_WIDTH', 0.0001, 0.02, step=0.0005),   # ИСПРАВЛЕНО: снижен верхний предел до реалистичного
        'RSI_EXTREME_OVERSOLD': trial.suggest_int('RSI_EXTREME_OVERSOLD', 1, 40, step=1),
        'RSI_EXTREME_OVERBOUGHT': trial.suggest_int('RSI_EXTREME_OVERBOUGHT', 70, 99, step=1),
        'MIN_CANDLE_BODY_PCT': trial.suggest_float('MIN_CANDLE_BODY_PCT', 0.05, 0.8, step=0.01),  # ИСПРАВЛЕНО: снижен верхний предел
        'MAX_WICK_TO_BODY_RATIO': trial.suggest_float('MAX_WICK_TO_BODY_RATIO', 0.5, 12.0, step=0.1),
        'MIN_TRIGGERS_ACTIVE_HOURS': trial.suggest_float('MIN_TRIGGERS_ACTIVE_HOURS', 0.1, 3.0, step=0.01),  # ИСПРАВЛЕНО: снижен диапазон
        'MIN_TRIGGERS_INACTIVE_HOURS': trial.suggest_float('MIN_TRIGGERS_INACTIVE_HOURS', 0.1, 4.0, step=0.05),  # ИСПРАВЛЕНО: снижен диапазон
        'SIGNAL_COOLDOWN_MINUTES': trial.suggest_int('SIGNAL_COOLDOWN_MINUTES', 1, 90, step=1),
        'MIN_VOLUME_MA_RATIO': trial.suggest_float('MIN_VOLUME_MA_RATIO', 0.01, 4.96, step=0.05),  # (4.96-0.01)/0.05=99
        'MIN_VOLUME_CONSISTENCY': trial.suggest_float('MIN_VOLUME_CONSISTENCY', 0.01, 0.97, step=0.01),  # (0.97-0.01)/0.01=96
        'MAX_RSI_VOLATILITY': trial.suggest_int('MAX_RSI_VOLATILITY', 1, 40, step=1),
        'REQUIRE_MACD_HISTOGRAM_CONFIRMATION': trial.suggest_categorical('REQUIRE_MACD_HISTOGRAM_CONFIRMATION', [False, True]),
        'WEIGHT_RSI': trial.suggest_float('WEIGHT_RSI', 0.1, 6.0, step=0.05),
        'WEIGHT_MACD': trial.suggest_float('WEIGHT_MACD', 0.1, 6.0, step=0.05),
        'WEIGHT_BB': trial.suggest_float('WEIGHT_BB', 0.05, 4.0, step=0.05),
        'WEIGHT_VWAP': trial.suggest_float('WEIGHT_VWAP', 0.05, 4.0, step=0.05),
        'WEIGHT_VOLUME': trial.suggest_float('WEIGHT_VOLUME', 0.1, 8.0, step=0.05),
        'WEIGHT_ADX': trial.suggest_float('WEIGHT_ADX', 0.1, 12.0, step=0.1),
        'SHORT_BOOST_MULTIPLIER': trial.suggest_float('SHORT_BOOST_MULTIPLIER', 0.1, 4.0, step=0.02),
        'LONG_PENALTY_IN_DOWNTREND': trial.suggest_float('LONG_PENALTY_IN_DOWNTREND', 0.001, 0.996, step=0.005),  # (0.996-0.001)/0.005=199
        'RSI_WINDOW': trial.suggest_int('RSI_WINDOW', 2, 40, step=1),
        'MA_FAST': trial.suggest_int('MA_FAST', 2, 80, step=1),
        'MA_SLOW': trial.suggest_int('MA_SLOW', 8, 150, step=2),
        'ATR_WINDOW': trial.suggest_int('ATR_WINDOW', 2, 60, step=1),
        'TRAIL_ATR_MULT': trial.suggest_float('TRAIL_ATR_MULT', 0.1, 8.0, step=0.1),
        'TP_MIN': trial.suggest_float('TP_MIN', 0.006, 0.08, step=0.001),
        'SL_MIN': trial.suggest_float('SL_MIN', 0.006, 0.15, step=0.001),
        'BB_WINDOW': trial.suggest_int('BB_WINDOW', 4, 80, step=1),
        'BB_STD_DEV': trial.suggest_float('BB_STD_DEV', 0.5, 6.0, step=0.05),
        'MACD_FAST': trial.suggest_int('MACD_FAST', 2, 40, step=1),
        'MACD_SLOW': trial.suggest_int('MACD_SLOW', 5, 80, step=1),
        'MACD_SIGNAL': trial.suggest_int('MACD_SIGNAL', 1, 40, step=1),
        'STOCH_RSI_K': trial.suggest_int('STOCH_RSI_K', 1, 20),
        'STOCH_RSI_D': trial.suggest_int('STOCH_RSI_D', 1, 20),
        'STOCH_RSI_LENGTH': trial.suggest_int('STOCH_RSI_LENGTH', 2, 40, step=1),
        'STOCH_RSI_SMOOTH': trial.suggest_int('STOCH_RSI_SMOOTH', 1, 20),
        'MIN_TP_SL_DISTANCE': trial.suggest_float('MIN_TP_SL_DISTANCE', 0.001, 0.02, step=0.002),  # ИСПРАВЛЕНО: снижен верхний предел
        'BB_SQUEEZE_THRESHOLD': trial.suggest_float('BB_SQUEEZE_THRESHOLD', 0.005, 0.249, step=0.002),  # (0.249-0.005)/0.002=122
        'MACD_SIGNAL_WINDOW': trial.suggest_int('MACD_SIGNAL_WINDOW', 1, 40, step=1),
        'VOLATILITY_FILTER_STRENGTH': trial.suggest_float('VOLATILITY_FILTER_STRENGTH', 0.1, 5.0, step=0.05),
        'TREND_STRENGTH_MULTIPLIER': trial.suggest_float('TREND_STRENGTH_MULTIPLIER', 0.1, 3.0, step=0.02),
        'VOLUME_SPIKE_SENSITIVITY': trial.suggest_float('VOLUME_SPIKE_SENSITIVITY', 0.5, 8.0, step=0.05),
        'DIVERGENCE_WEIGHT': trial.suggest_float('DIVERGENCE_WEIGHT', 0.05, 4.0, step=0.05),
    }

def get_historical_data(symbol, hours_back=72):
    """Загружает исторические данные из CSV файлов (кэш)"""
    try:
        # Формируем имя файла
        filename = f"data/{symbol}_15m.csv"
        
        if not os.path.exists(filename):
            logging.warning(f"Файл данных не найден: {filename}")
            logging.warning(f"Сначала запустите download_ohlcv.py для загрузки данных")
            return pd.DataFrame()
        
        # Читаем данные из CSV
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Новый столбец: объём в USDT
        df['volume_usdt'] = df['volume'] * df['close']
        
        # Ограничиваем количество свечей если нужно
        if hours_back < 72:
            candles_needed = int(hours_back * 60 / 15) + 50
            if len(df) > candles_needed:
                df = df.tail(candles_needed)
        
        logging.info(f"Загружено {len(df)} свечей для {symbol} из {filename}")
        return df
        
    except Exception as e:
        logging.warning(f"Ошибка чтения данных {symbol}: {e}")
        return pd.DataFrame()

def simulate_signals(df, symbol, params, active_hours_utc):
    """Симуляция сигналов с расширенными фильтрами"""
    if df.empty or len(df) < MIN_15M_CANDLES + 50:
        return []
    
    # Анализируем объём в USDT
    df_analyzed = analyze(df.copy())
    if 'volume_usdt' not in df_analyzed.columns:
        df_analyzed['volume_usdt'] = df_analyzed['volume'] * df_analyzed['close']
    if df_analyzed.empty:
        return []
    
    signals = []
    last_signal_time = None
    
    # Извлекаем параметры
    min_score = params['MIN_SCORE']
    min_adx = params['MIN_ADX']
    short_min_adx = params['SHORT_MIN_ADX']
    short_min_rsi = params['SHORT_MIN_RSI']
    long_max_rsi = params['LONG_MAX_RSI']
    rsi_min = params['RSI_MIN']
    rsi_max = params['RSI_MAX']
    tp_mult = params['TP_ATR_MULT']
    sl_mult = params['SL_ATR_MULT']
    min_volume = params['MIN_VOLUME_USDT']
    max_spread = params['MAX_SPREAD_PCT']
    min_bb_width = params['MIN_BB_WIDTH']
    rsi_extreme_oversold = params['RSI_EXTREME_OVERSOLD']
    rsi_extreme_overbought = params['RSI_EXTREME_OVERBOUGHT']
    min_candle_body_pct = params['MIN_CANDLE_BODY_PCT']
    max_wick_to_body_ratio = params['MAX_WICK_TO_BODY_RATIO']
    signal_cooldown_minutes = params['SIGNAL_COOLDOWN_MINUTES']
    min_triggers_active_hours = params['MIN_TRIGGERS_ACTIVE_HOURS']
    min_triggers_inactive_hours = params['MIN_TRIGGERS_INACTIVE_HOURS']
    min_volume_ma_ratio = params['MIN_VOLUME_MA_RATIO']
    min_volume_consistency = params['MIN_VOLUME_CONSISTENCY']
    max_rsi_volatility = params['MAX_RSI_VOLATILITY']
    require_macd_histogram = params['REQUIRE_MACD_HISTOGRAM_CONFIRMATION']
    min_tp_sl_distance = params['MIN_TP_SL_DISTANCE']
    
    # ПРАВИЛЬНО: Оптимизатор НЕ должен видеть последние данные (будущее)!
    # Исключаем последние 384 свечи (4 дня) чтобы не "подглядывать"
    for i in range(MIN_15M_CANDLES, len(df_analyzed) - 384):  # 4 суток = 384 свечи
        current_df = df_analyzed.iloc[:i+1].copy()
        last = current_df.iloc[-1]
        prev = current_df.iloc[-2]
        now = last['timestamp']
        hour_utc = now.hour
        
        # Временные фильтры
        if hour_utc not in active_hours_utc:
            continue
            
        # Кулдаун
        if last_signal_time and (now - last_signal_time).total_seconds() < signal_cooldown_minutes * 60:
            continue
            
        # Базовые фильтры
        if last['spread_pct'] > max_spread:
            continue
            
        if last['adx'] < min_adx:
            continue
            
        # ИСПРАВЛЕНО: RSI диапазон НЕ блокирует сигналы!
        # Перепроданные/перекупленные состояния должны генерировать сигналы, а не блокироваться
            
        # ИСПРАВЛЕНО: RSI экстремальные значения НЕ блокируют сигналы!
        # Экстремальные RSI генерируют СИЛЬНЫЕ сигналы в триггерах ниже
            
        # Объем - теперь в USDT - СИНХРОНИЗАЦИЯ с ботом
        volume = last.get('volume_usdt', 1_000_000)
        # КРИТИЧНО: Приводим к миллионам USDT для сравнения с min_volume
        volume_millions = volume / 1_000_000  # Переводим в миллионы USDT
        if volume_millions < min_volume:
            continue
            
        # BB width
        if 'bollinger_high' in last and 'bollinger_low' in last:
            bb_width = (last['bollinger_high'] - last['bollinger_low']) / last['close']
            if bb_width < min_bb_width:
                continue
                
        # Candle body
        candle_body = abs(last['close'] - last['open'])
        candle_range = last['high'] - last['low']
        if candle_range > 0:
            body_pct = candle_body / candle_range
            if body_pct < min_candle_body_pct:
                continue
                
        # Wick ratio фильтр
        if candle_body > 0:
            wick_ratio = candle_range / candle_body
            if wick_ratio > max_wick_to_body_ratio:
                continue
                
        # Volume MA ratio фильтр (теперь в USDT)
        if 'volume_usdt' in current_df.columns and i > 0:
            volume_ma = current_df['volume_usdt'].iloc[i-20:i].mean() if i >= 20 else current_df['volume_usdt'].iloc[:i].mean()
            if volume_ma > 0:
                volume_ratio = last['volume_usdt'] / volume_ma
                if volume_ratio < min_volume_ma_ratio:
                    continue
                    
        # Volume consistency фильтр (теперь в USDT)
        if i >= 5:
            recent_volumes = current_df['volume_usdt'].iloc[i-5:i]
            volume_std = recent_volumes.std()
            volume_mean = recent_volumes.mean()
            if volume_mean > 0:
                volume_cv = volume_std / volume_mean
                if volume_cv > (1 - min_volume_consistency):
                    continue
                    
        # RSI volatility фильтр
        if i > 0 and 'rsi' in current_df.columns:
            rsi_change = abs(last['rsi'] - current_df['rsi'].iloc[i-1])
            if rsi_change > max_rsi_volatility:
                continue
                
        # Триггеры
        buy_triggers = 0
        sell_triggers = 0
        
        # КРИТИЧНО: RSI экстремальные значения дают СИЛЬНЫЕ триггеры (как в боте)
        if last['rsi'] <= rsi_extreme_oversold:
            buy_triggers += 2.0  # Очень сильный сигнал покупки
        elif last['rsi'] < rsi_min:
            buy_triggers += 1.0  # Сильный сигнал покупки
            
        if last['rsi'] >= rsi_extreme_overbought:
            sell_triggers += 2.0  # Очень сильный сигнал продажи
        elif last['rsi'] > rsi_max:
            sell_triggers += 1.0  # Сильный сигнал продажи
        
        # EMA кроссовер
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            buy_triggers += 1
        elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
            buy_triggers += 0.5
            
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            sell_triggers += 1
        elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
            sell_triggers += 0.5
            
        # MACD
        if 'macd' in current_df.columns:
            if last['macd'] > last['macd_signal']:
                buy_triggers += 0.5
            if last['macd'] < last['macd_signal']:
                sell_triggers += 0.5
                
        # MACD Histogram фильтр будет применен позже после определения signal_type
                
        # Bollinger Bands
        if 'bollinger_low' in current_df.columns:
            bb_position = (last['close'] - last['bollinger_low']) / (last['bollinger_high'] - last['bollinger_low'])
            if bb_position <= 0.3:
                buy_triggers += 0.5
            if bb_position >= 0.7:
                sell_triggers += 0.5
                
        # VWAP
        if USE_VWAP and 'vwap' in current_df.columns:
            vwap_dev = last.get('vwap_deviation', 0)
            if vwap_dev <= 0 and vwap_dev >= -VWAP_DEVIATION_THRESHOLD * 2:
                buy_triggers += 0.3
            if vwap_dev >= 0 and vwap_dev <= VWAP_DEVIATION_THRESHOLD * 2:
                sell_triggers += 0.3
                
        # Минимальные триггеры
        min_triggers = min_triggers_active_hours if hour_utc in active_hours_utc else min_triggers_inactive_hours
        
        signal_type = None
        # ИСПРАВЛЕНО: Учитываем экстремальные RSI как валидные для сигналов (как в боте)
        if buy_triggers >= min_triggers and (last['rsi'] <= rsi_max or last['rsi'] <= rsi_extreme_oversold):
            signal_type = 'BUY'
        elif sell_triggers >= min_triggers and (last['rsi'] >= rsi_min or last['rsi'] >= rsi_extreme_overbought):
            signal_type = 'SELL'
            
        # MACD Histogram фильтр (если включен)
        if signal_type and require_macd_histogram and 'macd_hist' in current_df.columns and i > 0:
            current_hist = last['macd_hist']
            prev_hist = current_df['macd_hist'].iloc[i-1]
            if signal_type == 'BUY' and not (current_hist > 0 and prev_hist <= 0):
                continue
            elif signal_type == 'SELL' and not (current_hist < 0 and prev_hist >= 0):
                continue
            
        # Дополнительные условия для short/long
        if signal_type == 'SELL' and last['adx'] < short_min_adx:
            continue
        if signal_type == 'SELL' and last['rsi'] < short_min_rsi:
            continue
        if signal_type == 'BUY' and last['rsi'] > long_max_rsi:
            continue
            
        if signal_type:
            try:
                score, pattern = evaluate_signal_strength(current_df, symbol, signal_type)
                if score >= min_score:
                    entry_price = last['close']
                    entry_time = now
                    future_data = df_analyzed.iloc[i+1:i+385]  # 4 суток = 384 свечи
                    
                    if len(future_data) >= 10:
                        atr = last['atr']
                        tp_distance = atr * tp_mult
                        sl_distance = atr * sl_mult
                        
                        if signal_type == 'BUY':
                            tp_price = entry_price + tp_distance
                            sl_price = entry_price - sl_distance
                        else:
                            tp_price = entry_price - tp_distance
                            sl_price = entry_price + sl_distance
                        
                        # Проверка минимального расстояния между TP и SL
                        tp_sl_distance = abs(tp_price - sl_price) / entry_price
                        if tp_sl_distance < min_tp_sl_distance:
                            continue
                            
                        result = None
                        for idx, candle in future_data.iterrows():
                            if signal_type == 'BUY':
                                if candle['high'] >= tp_price:
                                    result = 'tp'
                                    break
                                elif candle['low'] <= sl_price:
                                    result = 'sl'
                                    break
                            else:
                                if candle['low'] <= tp_price:
                                    result = 'tp'
                                    break
                                elif candle['high'] >= sl_price:
                                    result = 'sl'
                                    break
                                    
                        if not result:
                            result = 'timeout'
                            
                        signals.append({
                            'symbol': symbol,
                            'type': signal_type,
                            'entry_time': entry_time,
                            'score': score,
                            'result': result,
                            'tp_pct': ((tp_price - entry_price) / entry_price * 100) if signal_type == 'BUY' else ((entry_price - tp_price) / entry_price * 100),
                            'sl_pct': ((entry_price - sl_price) / entry_price * 100) if signal_type == 'BUY' else ((sl_price - entry_price) / entry_price * 100)
                        })
                        last_signal_time = now
                        
            except Exception as e:
                logging.warning(f"Ошибка оценки сигнала {symbol} в {now}: {e}")
                continue
                
    return signals

def test_single_params(params, hours_back=None, active_hours_utc=None):
    """Тестирует один набор параметров"""
    if hours_back is None:
        hours_back = GLOBAL_HOURS_BACK
    if active_hours_utc is None:
        active_hours_utc = GLOBAL_ACTIVE_HOURS_UTC
        
    all_signals = []
    mon_stats = {}
    
    for symbol in GLOBAL_ALL_SYMBOLS:
        df = get_historical_data(symbol, hours_back)
        if df.empty:
            continue
        
        signals = simulate_signals(df, symbol, params, active_hours_utc)
        all_signals.extend(signals)
        
        tp_signals = [s for s in signals if s['result'] == 'tp']
        sl_signals = [s for s in signals if s['result'] == 'sl']
        winrate = len(tp_signals) / (len(tp_signals) + len(sl_signals)) * 100 if (len(tp_signals) + len(sl_signals)) > 0 else 0
        
        mon_stats[symbol] = {
            'signals': len(signals),
            'winrate': winrate,
            'tp': len(tp_signals),
            'sl': len(sl_signals)
        }
    
    # Исключаем монеты с winrate < 30%
    good_symbols = [s for s, stat in mon_stats.items() if stat['winrate'] >= 20 and stat['signals'] > 0]
    filtered_signals = [s for s in all_signals if s['symbol'] in good_symbols]
    
    tp_signals = [s for s in filtered_signals if s['result'] == 'tp']
    sl_signals = [s for s in filtered_signals if s['result'] == 'sl']
    tp_count = len(tp_signals)
    sl_count = len(sl_signals)
    winrate = tp_count / (tp_count + sl_count) * 100 if (tp_count + sl_count) > 0 else 0
    tp_sum = sum([s['tp_pct'] for s in tp_signals])
    sl_sum = sum([s['sl_pct'] for s in sl_signals])
    tp_sl_count_ratio = tp_count / sl_count if sl_count > 0 else 0
    tp_sl_profit_ratio = tp_sum / abs(sl_sum) if abs(sl_sum) > 0 else 0
    signals_per_day = len(filtered_signals) / (hours_back / 24)
    
    return {
        'params': params,
        'signals': len(filtered_signals),
        'signals_per_day': signals_per_day,
        'winrate': winrate,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'tp_sl_count_ratio': tp_sl_count_ratio,
        'tp_sum': tp_sum,
        'sl_sum': sl_sum,
        'tp_sl_profit_ratio': tp_sl_profit_ratio,
        'good_symbols': good_symbols
    }

def objective(trial: optuna.Trial) -> float:
    """Целевая функция для Optuna - оптимизируем комплексную метрику"""
    try:
        # Получаем параметры от Optuna
        params = suggest_parameters(trial)
        
        # Тестируем параметры
        result = test_single_params(params)
        
        # Если нет сигналов - возвращаем плохую оценку
        if result['signals'] == 0 or result['signals_per_day'] < GLOBAL_MIN_SIGNALS_PER_DAY:
            return 0.0
            
        # Комплексная метрика учитывающая:
        # - Winrate (60% от общей оценки)
        # - TP/SL count ratio (20% от общей оценки) 
        # - TP/SL profit ratio (20% от общей оценки)
        # Плюс штрафы за слишком много/мало сигналов
        
        winrate_score = min(result['winrate'] / 100.0, 1.0)  # 0-1
        count_ratio_score = min(result['tp_sl_count_ratio'] / 2.0, 1.0)  # 0-1 (цель 2.0)
        profit_ratio_score = min(result['tp_sl_profit_ratio'] / 2.0, 1.0)  # 0-1 (цель 2.0)
        
        # Штраф за слишком много сигналов (более 150/день)
        signal_penalty = 1.0
        if result['signals_per_day'] > 150:
            signal_penalty = 0.5
        elif result['signals_per_day'] > 100:
            signal_penalty = 0.8
            
        # Комплексная оценка
        score = (winrate_score * 0.6 + count_ratio_score * 0.2 + profit_ratio_score * 0.2) * signal_penalty
        
        # Дополнительная проверка на минимальные требования
        if (result['winrate'] >= 60 and 
            result['tp_sl_count_ratio'] >= 1.4 and 
            result['tp_sl_profit_ratio'] >= 1.4 and
            result['signals_per_day'] <= 150):
            score *= 1.2  # Бонус за выполнение всех условий
            
        return score
        
    except Exception as e:
        logging.error(f"Ошибка в objective: {e}")
        return 0.0

def save_optuna_results(study: optuna.Study, filename: str = 'optuna_results.json'):
    """Сохраняет результаты оптимизации Optuna"""
    try:
        best_trial = study.best_trial
        best_params_result = test_single_params(best_trial.params)
        
        results = {
            'best_trial': {
                'params': best_trial.params,
                'value': best_trial.value,
                'number': best_trial.number
            },
            'best_result': best_params_result,
            'study_stats': {
                'n_trials': len(study.trials),
                'best_value': study.best_value,
                'direction': study.direction.name
            },
            'top_trials': []
        }
        
        # Топ-10 лучших попыток
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
        for trial in sorted_trials[:10]:
            if trial.value:
                trial_result = test_single_params(trial.params)
                results['top_trials'].append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'result': trial_result
                })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
        print(f"✅ Результаты Optuna сохранены в {filename}")
        
    except Exception as e:
        logging.error(f"Ошибка сохранения результатов Optuna: {e}")

def create_optuna_visualizations(study: optuna.Study):
    """Создает визуализации результатов Optuna"""
    try:
        import optuna.visualization as vis
        
        # История оптимизации
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html("optuna_history.html")
        
        # Важность параметров
        fig2 = vis.plot_param_importances(study)
        fig2.write_html("optuna_param_importance.html")
        
        # Срезы параметров
        fig3 = vis.plot_slice(study)
        fig3.write_html("optuna_slice.html")
        
        # Параллельные координаты
        fig4 = vis.plot_parallel_coordinate(study)
        fig4.write_html("optuna_parallel_coordinate.html")
        
        print("📊 Визуализации Optuna созданы:")
        print("  - optuna_history.html")
        print("  - optuna_param_importance.html") 
        print("  - optuna_slice.html")
        print("  - optuna_parallel_coordinate.html")
        
    except Exception as e:
        logging.error(f"Ошибка создания визуализации: {e}")
        print("⚠️  Визуализации не созданы из-за ошибки")

def optimize_filters():
    """НОВАЯ ОПТИМИЗАЦИЯ С OPTUNA - умный поиск параметров"""
    global GLOBAL_ALL_SYMBOLS, GLOBAL_HOURS_BACK, GLOBAL_ACTIVE_HOURS_UTC, GLOBAL_MIN_SIGNALS_PER_DAY
    
    # Настройки оптимизации
    GLOBAL_HOURS_BACK = 200  # Загружаем ВСЕ доступные данные из CSV файлов
    GLOBAL_ACTIVE_HOURS_UTC = list(range(6, 24))  # 6:00 до 23:59 UTC
    GLOBAL_MIN_SIGNALS_PER_DAY = 5  # Снижено для тестирования
    N_TRIALS = 100  # Уменьшено для быстрого тестирования исправленных параметров
    
    # Загружаем символы
    GLOBAL_ALL_SYMBOLS = get_all_symbols_from_data()
    
    print("🚀 ЗАПУСК УМНОЙ ОПТИМИЗАЦИИ С OPTUNA")
    print("="*60)
    print(f"🎯 Алгоритм: TPE (Tree-structured Parzen Estimator)")
    print(f"📊 Количество попыток: {N_TRIALS}")
    print(f"⏰ Временной период: {GLOBAL_HOURS_BACK} часов назад")
    print(f"🕐 Активные часы UTC: {GLOBAL_ACTIVE_HOURS_UTC[0]}:00 - {GLOBAL_ACTIVE_HOURS_UTC[-1]}:59")
    print(f"📈 Минимум сигналов/день: {GLOBAL_MIN_SIGNALS_PER_DAY}")
    print(f"💱 Количество торговых пар: {len(GLOBAL_ALL_SYMBOLS)}")
    print("\n🔧 ОПТИМИЗИРУЕМЫЕ ПАРАМЕТРЫ:")
    print("  ✅ Основные фильтры (min_score, ADX, RSI)")
    print("  ✅ TP/SL мультипликаторы") 
    print("  ✅ Объемные фильтры (volume, spread, BB width)")
    print("  ✅ RSI экстремальные значения")
    print("  ✅ Candle фильтры (body, wick)")
    print("  ✅ Временные фильтры (cooldown, triggers)")
    print("  ✅ Веса системы оценок")
    print("  ✅ Индикаторные параметры")
    print("  ✅ MIN_TP_SL_DISTANCE")
    print("  ✅ BB Squeeze, MACD Signal Window")
    print("  ✅ Stochastic RSI параметры")
    print("\n📊 ЦЕЛЕВАЯ ФУНКЦИЯ:")
    print("  🎯 60% - Winrate")
    print("  📈 20% - TP/SL Count Ratio")
    print("  💰 20% - TP/SL Profit Ratio")
    print("  ⚡ Штрафы за слишком много сигналов")
    print("  🏆 Бонус за выполнение всех условий")
    print("="*60)
    
    # Создаем study Optuna
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=50, n_warmup_steps=10)
    )
    
    print("🔥 НАЧИНАЕМ ОПТИМИЗАЦИЮ...")
    try:
        # Запускаем оптимизацию
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
        
        print("\n🏁 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print("="*60)
        
        # Анализируем результаты
        print(f"📊 СТАТИСТИКА OPTUNA:")
        print(f"  🔢 Всего попыток: {len(study.trials)}")
        print(f"  🏆 Лучшая оценка: {study.best_value:.4f}")
        print(f"  📈 Лучшая попытка: #{study.best_trial.number}")
        
        # Получаем подробные результаты лучших параметров
        best_result = test_single_params(study.best_trial.params)
        
        print(f"\n🥇 ЛУЧШИЕ НАЙДЕННЫЕ ПАРАМЕТРЫ:")
        print(f"  📊 Winrate: {best_result['winrate']:.1f}%")
        print(f"  📈 TP/SL Count Ratio: {best_result['tp_sl_count_ratio']:.2f}")
        print(f"  💰 TP/SL Profit Ratio: {best_result['tp_sl_profit_ratio']:.2f}")
        print(f"  ⚡ Сигналов/день: {best_result['signals_per_day']:.1f}")
        print(f"  🎯 TP: {best_result['tp_count']}, SL: {best_result['sl_count']}")
        print(f"  💱 Хороших монет: {len(best_result['good_symbols'])}")
        
        # Проверяем идеальные условия
        is_perfect = (
            best_result['winrate'] >= 60 and
            best_result['tp_sl_count_ratio'] >= 1.4 and
            best_result['tp_sl_profit_ratio'] >= 1.4 and
            best_result['signals_per_day'] >= GLOBAL_MIN_SIGNALS_PER_DAY and
            best_result['signals_per_day'] <= 150
        )
        
        if is_perfect:
            print("\n🌟 НАЙДЕНЫ ИДЕАЛЬНЫЕ ПАРАМЕТРЫ! ✨")
        else:
            print(f"\n💡 УСЛОВИЯ НЕ ПОЛНОСТЬЮ ВЫПОЛНЕНЫ:")
            if best_result['winrate'] < 60:
                print(f"  ❌ Winrate {best_result['winrate']:.1f}% < 60%")
            if best_result['tp_sl_count_ratio'] < 1.4:
                print(f"  ❌ TP/SL Count Ratio {best_result['tp_sl_count_ratio']:.2f} < 1.4")
            if best_result['tp_sl_profit_ratio'] < 1.4:
                print(f"  ❌ TP/SL Profit Ratio {best_result['tp_sl_profit_ratio']:.2f} < 1.4")
            if best_result['signals_per_day'] < GLOBAL_MIN_SIGNALS_PER_DAY:
                print(f"  ❌ Сигналов/день {best_result['signals_per_day']:.1f} < {GLOBAL_MIN_SIGNALS_PER_DAY}")
            if best_result['signals_per_day'] > 150:
                print(f"  ❌ Сигналов/день {best_result['signals_per_day']:.1f} > 150")
        
        # Сохраняем результаты
        save_optuna_results(study, 'optuna_results.json')
        
        # Создаем визуализации
        print("\n📊 Создаем визуализации...")
        create_optuna_visualizations(study)
        
        # Выводим лучшие параметры
        print(f"\n🔧 ЛУЧШИЕ ПАРАМЕТРЫ:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
            
        # Анализ важности параметров
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\n🎯 ТОП-10 САМЫХ ВАЖНЫХ ПАРАМЕТРОВ:")
            for i, (param, imp) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]):
                print(f"  {i+1:2d}. {param}: {imp:.4f}")
        except:
            print("⚠️  Анализ важности параметров недоступен")
            
        print(f"\n🎉 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print(f"📁 Файлы созданы:")
        print(f"  - optuna_results.json (результаты)")
        print(f"  - optuna_*.html (визуализации)")
        
        # Рекомендации по улучшению
        if not is_perfect:
            print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ:")
            print(f"  - Увеличьте N_TRIALS до 1000-2000")
            print(f"  - Попробуйте другие samplers (CmaEsSampler, RandomSampler)")
            print(f"  - Настройте pruner для более агрессивной обрезки")
            print(f"  - Измените веса в целевой функции")
            
    except Exception as e:
        logging.error(f"Ошибка оптимизации: {e}")
        print(f"❌ Ошибка во время оптимизации: {e}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Оптимизация прервана пользователем")
        if len(study.trials) > 0:
            print(f"💾 Сохраняем результаты {len(study.trials)} попыток...")
            save_optuna_results(study, 'optuna_results_interrupted.json')

if __name__ == '__main__':
    optimize_filters() 