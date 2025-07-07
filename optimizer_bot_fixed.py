#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ИСПРАВЛЕННАЯ версия оптимизатора без проблемных фильтров
Исключены: Volatility Filter, EMA Separation Filter, Momentum Filter
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import *
from crypto_signal_bot import analyze, evaluate_signal_strength, SYMBOLS
import logging
import random
import multiprocessing as mp
from functools import partial
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# --- УЛУЧШЕННОЕ ПОИСКОВОЕ ПРОСТРАНСТВО (БЕЗ ПРОБЛЕМНЫХ ФИЛЬТРОВ) ---
fixed_search_space = {
    # === ОСНОВНЫЕ ФИЛЬТРЫ ===
    'min_score': [2.5, 3.0, 3.5, 4.0, 4.5],
    'min_adx': [8, 10, 12, 14, 16, 18],
    'short_min_adx': [8, 10, 12, 14, 16],
    'short_min_rsi': [35, 40, 45, 50, 55],
    'long_max_rsi': [60, 65, 70, 75, 80],
    'rsi_min': [10, 15, 20, 25, 30],
    'rsi_max': [70, 75, 80, 85, 90],
    
    # === TP/SL МУЛЬТИПЛИКАТОРЫ ===
    'tp_mult': [1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
    'sl_mult': [1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
    
    # === ОБЪЕМНЫЕ ФИЛЬТРЫ ===
    'min_volume': [700_000, 900_000],
    'max_spread': [0.008, 0.01, 0.012, 0.015, 0.018],
    'min_bb_width': [0.005, 0.008, 0.01, 0.012, 0.015],
    
    # === RSI ФИЛЬТРЫ ===
    'rsi_extreme_oversold': [8, 10, 12, 15, 18],
    'rsi_extreme_overbought': [82, 85, 88, 90, 92],
    
    # === CANDLE ФИЛЬТРЫ (ОСЛАБЛЕННЫЕ) ===
    'min_candle_body_pct': [0.40, 0.50, 0.60, 0.70, 0.80],
    'max_wick_to_body_ratio': [2.0, 2.5, 3.0, 3.5, 4.0],
    
    # === ВРЕМЕННЫЕ ФИЛЬТРЫ ===
    'signal_cooldown_minutes': [15, 20, 25, 30, 35],
    'min_triggers_active_hours': [0.8, 1.0, 1.2, 1.5, 1.8],
    'min_triggers_inactive_hours': [1.2, 1.5, 1.8, 2.0, 2.2],
    
    # === ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ ===
    'bb_squeeze_threshold': [0.03, 0.05, 0.07, 0.10],
    'macd_signal_window': [7, 9, 11, 13],
    'stoch_rsi_k': [3, 5, 7],
    'stoch_rsi_d': [3, 5, 7],
    'stoch_rsi_length': [10, 12, 14, 16],
    'stoch_rsi_smooth': [2, 3, 4],
    
    # === НОВЫЕ ФИЛЬТРЫ ИЗ CONFIG.PY ===
    'min_volume_ma_ratio': [0.6, 0.8, 1.0, 1.2, 1.5],
    'min_volume_consistency': [0.5, 0.6, 0.7, 0.8, 0.9],
    'max_rsi_volatility': [8, 10, 12, 15, 18],
    'require_macd_histogram': [False, True],  # True только для очень строгих настроек
    
    # === ВЕСА СИСТЕМЫ ===
    'weight_rsi': [1.0, 1.2, 1.5, 1.8, 2.0],
    'weight_macd': [1.2, 1.5, 1.8, 2.0, 2.2],
    'weight_bb': [0.8, 1.0, 1.2, 1.4, 1.6],
    'weight_vwap': [1.0, 1.2, 1.4, 1.6, 1.8],
    'weight_volume': [1.5, 1.8, 2.0, 2.2, 2.5],
    'weight_adx': [2.0, 2.5, 3.0, 3.5, 4.0],
    
    # === SHORT/LONG НАСТРОЙКИ ===
    'short_boost_multiplier': [1.1, 1.2, 1.3, 1.4, 1.5],
    'long_penalty_in_downtrend': [0.10, 0.12, 0.15, 0.18, 0.20],
    
    # === МУЛЬТИТАЙМФРЕЙМ ===
    'mtf_confluence_weight': [1.2, 1.5, 1.8, 2.0, 2.2],
    # === ДОБАВЛЕННЫЕ КЛЮЧЕВЫЕ ПАРАМЕТРЫ ===
    'RSI_WINDOW': [10, 12, 14, 16, 18],
    'MA_FAST': [10, 16, 22, 28],
    'MA_SLOW': [34, 42, 52, 62],
    'ATR_WINDOW': [10, 14, 18, 22],
    'TRAIL_ATR_MULT': [1.5, 2.0, 2.5, 3.0],
    'TP_MIN': [0.006, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020],
    'SL_MIN': [0.010, 0.015, 0.020, 0.025, 0.030, 0.035],
    'BB_WINDOW': [14, 20, 24],
    'BB_STD_DEV': [1.5, 2.0, 2.5],
    'MACD_FAST': [8, 12, 16],
    'MACD_SLOW': [20, 26, 34],
    'MACD_SIGNAL': [7, 9, 11],
    'STOCH_RSI_K': [3, 5, 7],
    'STOCH_RSI_D': [3, 5, 7],
    'STOCH_RSI_LENGTH': [10, 14, 16],
    'STOCH_RSI_SMOOTH': [2, 3, 4],
    'MIN_TP_SL_DISTANCE': [0.008, 0.010, 0.012, 0.015],
}

def get_historical_data(symbol, hours_back=72):
    candles_needed = int(hours_back * 60 / 15) + 100
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=candles_needed)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    except Exception as e:
        logging.warning(f"Ошибка загрузки {symbol}: {e}")
        return pd.DataFrame()

# --- ИСПРАВЛЕННАЯ ФУНКЦИЯ simulate_signals (БЕЗ ПРОБЛЕМНЫХ ФИЛЬТРОВ) ---
def simulate_signals_fixed(df, symbol, params, active_hours_utc):
    if df.empty or len(df) < MIN_15M_CANDLES + 50:
        return []
    
    df_analyzed = analyze(df.copy())
    if df_analyzed.empty:
        return []
    
    signals = []
    last_signal_time = None
    
    # Извлекаем параметры
    min_score = params['min_score']
    min_adx = params['min_adx']
    short_min_adx = params['short_min_adx']
    short_min_rsi = params['short_min_rsi']
    long_max_rsi = params['long_max_rsi']
    rsi_min = params['rsi_min']
    rsi_max = params['rsi_max']
    tp_mult = params['tp_mult']
    sl_mult = params['sl_mult']
    min_volume = params['min_volume']
    max_spread = params['max_spread']
    min_bb_width = params['min_bb_width']
    rsi_extreme_oversold = params['rsi_extreme_oversold']
    rsi_extreme_overbought = params['rsi_extreme_overbought']
    min_candle_body_pct = params['min_candle_body_pct']
    max_wick_to_body_ratio = params['max_wick_to_body_ratio']
    signal_cooldown_minutes = params['signal_cooldown_minutes']
    min_triggers_active_hours = params['min_triggers_active_hours']
    min_triggers_inactive_hours = params['min_triggers_inactive_hours']
    
    # Новые фильтры
    min_volume_ma_ratio = params['min_volume_ma_ratio']
    min_volume_consistency = params['min_volume_consistency']
    max_rsi_volatility = params['max_rsi_volatility']
    require_macd_histogram = params['require_macd_histogram']
    
    # Добавляем параметры для индикаторов
    RSI_WINDOW = params['RSI_WINDOW']
    MA_FAST = params['MA_FAST']
    MA_SLOW = params['MA_SLOW']
    ATR_WINDOW = params['ATR_WINDOW']
    TRAIL_ATR_MULT = params['TRAIL_ATR_MULT']
    TP_MIN = params['TP_MIN']
    SL_MIN = params['SL_MIN']
    BB_WINDOW = params['BB_WINDOW']
    BB_STD_DEV = params['BB_STD_DEV']
    MACD_FAST = params['MACD_FAST']
    MACD_SLOW = params['MACD_SLOW']
    MACD_SIGNAL = params['MACD_SIGNAL']
    STOCH_RSI_K = params['STOCH_RSI_K']
    STOCH_RSI_D = params['STOCH_RSI_D']
    STOCH_RSI_LENGTH = params['STOCH_RSI_LENGTH']
    STOCH_RSI_SMOOTH = params['STOCH_RSI_SMOOTH']
    MIN_TP_SL_DISTANCE = params['MIN_TP_SL_DISTANCE']
    
    for i in range(MIN_15M_CANDLES, len(df_analyzed) - 20):
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
            
        # Базовые фильтры (ИСКЛЮЧЕНЫ ПРОБЛЕМНЫЕ)
        if last['spread_pct'] > max_spread:
            continue
            
        if last['adx'] < min_adx:
            continue
            
        if last['rsi'] < rsi_min or last['rsi'] > rsi_max:
            continue
            
        # RSI экстремальные значения
        if last['rsi'] < rsi_extreme_oversold or last['rsi'] > rsi_extreme_overbought:
            continue
            
        # Объем
        volume = last.get('volume', 1_000_000)
        if volume < min_volume:
            continue
            
        # BB width
        if 'bollinger_high' in last and 'bollinger_low' in last:
            bb_width = (last['bollinger_high'] - last['bollinger_low']) / last['close']
            if bb_width < min_bb_width:
                continue
                
        # Candle body (ОСЛАБЛЕННЫЙ)
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
                
        # Volume MA ratio фильтр
        if 'volume_ma' in current_df.columns and i > 0:
            volume_ma = current_df['volume'].iloc[i-20:i].mean() if i >= 20 else current_df['volume'].iloc[:i].mean()
            if volume_ma > 0:
                volume_ratio = last['volume'] / volume_ma
                if volume_ratio < min_volume_ma_ratio:
                    continue
                    
        # Volume consistency фильтр
        if i >= 5:
            recent_volumes = current_df['volume'].iloc[i-5:i]
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
                
        # ИСКЛЮЧЕНЫ: Volatility Filter, EMA Separation Filter, Momentum Filter
                
        # Триггеры
        buy_triggers = 0
        sell_triggers = 0
        
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
                
        # MACD Histogram фильтр (если включен)
        if require_macd_histogram and 'macd_hist' in current_df.columns and i > 0:
            current_hist = last['macd_hist']
            prev_hist = current_df['macd_hist'].iloc[i-1]
            if signal_type == 'BUY' and not (current_hist > 0 and prev_hist <= 0):
                continue
            elif signal_type == 'SELL' and not (current_hist < 0 and prev_hist >= 0):
                continue
                
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
        if buy_triggers >= min_triggers and last['rsi'] <= rsi_max and last['rsi'] >= rsi_min:
            signal_type = 'BUY'
        elif sell_triggers >= min_triggers and last['rsi'] >= rsi_min and last['rsi'] <= rsi_max:
            signal_type = 'SELL'
            
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
                    future_data = df_analyzed.iloc[i+1:i+21]
                    
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

def test_single_params_fixed(params, hours_back, max_symbols, active_hours_utc):
    """Тестирует один набор параметров (исправленная версия)"""
    all_signals = []
    mon_stats = {}
    
    for symbol in SYMBOLS[:max_symbols]:
        df = get_historical_data(symbol, hours_back)
        if df.empty:
            continue
            
        signals = simulate_signals_fixed(df, symbol, params, active_hours_utc)
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
    good_symbols = [s for s, stat in mon_stats.items() if stat['winrate'] >= 30 and stat['signals'] > 0]
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

def optimize_filters_fixed():
    """Улучшенная оптимизация без проблемных фильтров"""
    hours_back = 96
    max_symbols = 20
    active_hours_utc = [8,9,10,13,15,16,17,19]
    min_signals_per_day = 10  # Увеличено с 5 до 10
    N_TRIALS = 500
    
    print(f"🚀 ЗАПУСК УЛУЧШЕННОЙ ОПТИМИЗАЦИИ")
    print(f"Количество попыток: {N_TRIALS}")
    print(f"Оптимизируем {len(fixed_search_space)} параметров")
    print(f"ДОБАВЛЕНЫ новые фильтры из config.py:")
    print(f"  ✅ Volume MA Ratio")
    print(f"  ✅ Volume Consistency")
    print(f"  ✅ RSI Volatility")
    print(f"  ✅ Wick to Body Ratio")
    print(f"  ✅ MACD Histogram (опционально)")
    print(f"ИСКЛЮЧЕНЫ проблемные фильтры:")
    print(f"  ❌ Volatility Filter (1.4% прохождения)")
    print(f"  ❌ EMA Separation Filter (12.7% прохождения)")
    print(f"  ❌ Momentum Filter (0.8% прохождения)")
    print(f"Минимум сигналов/день: {min_signals_per_day}")
    
    cpu_count = mp.cpu_count()
    processes_to_use = max(1, (cpu_count * 7) // 10)

    print(f"Используем {processes_to_use} из {cpu_count} ядер")
    print("="*60)
    
    all_params = []
    for _ in range(N_TRIALS):
        params = {k: random.choice(v) for k, v in fixed_search_space.items()}
        all_params.append(params)
        
    with mp.Pool(processes=processes_to_use) as pool:
        test_func = partial(test_single_params_fixed, hours_back=hours_back, max_symbols=max_symbols, active_hours_utc=active_hours_utc)
        results = pool.map(test_func, all_params)
    
    # Анализируем результаты
    all_results = []
    perfect_results = []
    
    for result in results:
        if result['signals'] > 0:
            all_results.append(result)
            
            # Улучшенные условия для вашей задачи
            conditions_met = (
                result['winrate'] >= 60 and  # Увеличено с 45 до 60
                result['tp_sl_count_ratio'] >= 1.5 and  # Увеличено с 1.2 до 1.5
                result['tp_sl_profit_ratio'] >= 1.5 and  # Увеличено с 1.2 до 1.5
                result['signals_per_day'] >= min_signals_per_day and
                result['signals_per_day'] <= 130  # Максимум 130 сигналов в день
            )
            
            if conditions_met:
                perfect_results.append(result)
    
    print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print(f"Всего протестировано: {len(results)}")
    print(f"Результатов с сигналами: {len(all_results)}")
    print(f"Идеальных результатов: {len(perfect_results)}")
    
    if all_results:
        best_by_winrate = max(all_results, key=lambda x: x['winrate'])
        best_by_signals = max(all_results, key=lambda x: x['signals_per_day'])
        
        print(f"\n🏆 ЛУЧШИЕ НАЙДЕННЫЕ РЕЗУЛЬТАТЫ:")
        
        print(f"\n🥇 ЛУЧШИЙ ПО WINRATE ({best_by_winrate['winrate']:.1f}%):")
        print(f"  Параметры: {best_by_winrate['params']}")
        print(f"  TP: {best_by_winrate['tp_count']}, SL: {best_by_winrate['sl_count']}")
        print(f"  TP/SL (кол-во): {best_by_winrate['tp_sl_count_ratio']:.2f}")
        print(f"  TP/SL (прибыль): {best_by_winrate['tp_sl_profit_ratio']:.2f}")
        print(f"  Сигналов/день: {best_by_winrate['signals_per_day']:.1f}")
        
        print(f"\n🥈 ЛУЧШИЙ ПО КОЛИЧЕСТВУ СИГНАЛОВ ({best_by_signals['signals_per_day']:.1f}/день):")
        print(f"  Параметры: {best_by_signals['params']}")
        print(f"  TP: {best_by_signals['tp_count']}, SL: {best_by_signals['sl_count']}")
        print(f"  Winrate: {best_by_signals['winrate']:.1f}%")
        print(f"  TP/SL (кол-во): {best_by_signals['tp_sl_count_ratio']:.2f}")
        print(f"  TP/SL (прибыль): {best_by_signals['tp_sl_profit_ratio']:.2f}")
        
        # Сохраняем лучшие результаты
        best_results = {
            'perfect_results': perfect_results,
            'best_by_winrate': best_by_winrate,
            'best_by_signals': best_by_signals,
            'all_results_count': len(all_results),
            'perfect_results_count': len(perfect_results)
        }
        
        with open('best_params_fixed.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(best_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены в best_params_fixed.json")
        
        if perfect_results:
            print(f"\n✅ НАЙДЕНО {len(perfect_results)} ИДЕАЛЬНЫХ КОМБИНАЦИЙ!")
            best_perfect = max(perfect_results, key=lambda x: x['winrate'])
            print(f"🏆 ЛУЧШАЯ ИДЕАЛЬНАЯ КОМБИНАЦИЯ:")
            print(f"  Параметры: {best_perfect['params']}")
            print(f"  Winrate: {best_perfect['winrate']:.1f}%, Сигналов/день: {best_perfect['signals_per_day']:.1f}")
        else:
            print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ:")
            print(f"  - Увеличьте N_TRIALS до 1000-2000")
            print(f"  - Или еще больше ослабьте условия")
            print(f"  - Или добавьте больше символов")
    else:
        print("❌ Не найдено ни одного результата с сигналами!")
        print("💡 Возможно, нужно еще больше ослабить параметры")

if __name__ == '__main__':
    optimize_filters_fixed() 