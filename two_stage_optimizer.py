#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-ЭТАПНЫЙ ОПТИМИЗАТОР ДЛЯ РЕАЛЬНОГО БОТА
Этап 1: Оптимизация качества сигналов (правильное направление)
Этап 2: Оптимизация TP/SL для максимальной прибыли
"""

import pandas as pd
import numpy as np
import logging
import asyncio
import aiohttp
import optuna
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Импортируем функции из основного скрипта
from test_current_vs_new_filters import (
    CURRENT_PARAMS, SYMBOLS,
    calculate_ema, calculate_rsi, calculate_macd, calculate_adx, 
    calculate_atr, 
    evaluate_signal_strength, calculate_tp_sl, get_historical_data
)

def analyze_with_all_filters(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Анализ данных со всеми фильтрами"""
    try:
        if df.empty or len(df) < params['MA_SLOW']:
            return pd.DataFrame()
        
        df = df.copy()
        
        # EMA с оптимизируемыми периодами
        df['ema_fast'] = calculate_ema(df['close'], params['MA_FAST'])
        df['ema_slow'] = calculate_ema(df['close'], params['MA_SLOW'])
        
        # MACD с оптимизируемыми параметрами
        macd_line, macd_signal, macd_hist = calculate_macd(
            df['close'], 
            params['MACD_FAST'], 
            params['MACD_SLOW'], 
            params['MACD_SIGNAL']
        )
        df['macd_line'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd'] = macd_hist
        df['macd_hist'] = macd_hist
        
        # RSI с оптимизируемым окном
        df['rsi'] = calculate_rsi(df['close'], params['RSI_WINDOW'])
        
        # ADX с оптимизируемым окном
        df['adx'] = calculate_adx(df['high'], df['low'], df['close'], params['ADX_WINDOW'])
        
        # ATR с оптимизируемым окном
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], params['ATR_WINDOW'])
        
        # Bollinger Bands и VWAP убраны - не используются в скоринге реального бота
        # Volume ratio остается только для фильтрации
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Очистка данных
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < 2:
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        logging.error(f"Ошибка в анализе данных: {e}")
        return pd.DataFrame()

# Кэш данных для оптимизации
DATA_CACHE = {}

async def load_data_for_optimization():
    """Загружаем данные для оптимизации"""
    global DATA_CACHE
    
    logging.info("📊 Загружаем данные для оптимизации...")
    
    for symbol in SYMBOLS:
        logging.info(f"📈 Загружаем {symbol}...")
        df = await get_historical_data(symbol, hours_back=720)  # 30 дней
        if not df.empty:
            DATA_CACHE[symbol] = df
            logging.info(f"✅ {symbol}: {len(df)} свечей")
        else:
            logging.warning(f"⚠️ Нет данных для {symbol}")
    
    logging.info(f"📊 Загружено {len(DATA_CACHE)} символов")

def check_signal_direction_quality(df: pd.DataFrame, symbol: str, params: Dict) -> Dict:
    """Проверка качества сигналов - правильное ли направление"""
    try:
        if df.empty or len(df) < params['MIN_15M_CANDLES']:
            return {'correct_signals': 0, 'total_signals': 0, 'accuracy': 0}
        
        df_analyzed = analyze_with_all_filters(df.copy(), params)
        if df_analyzed.empty:
            return {'correct_signals': 0, 'total_signals': 0, 'accuracy': 0}
        
        correct_signals = 0
        total_signals = 0
        total_direction_score = 0.0
        last_signal_time = {}
        
        # Проходим по всем свечам
        for i in range(params['MIN_15M_CANDLES'], len(df_analyzed) - 50):  # Нужно 50 свечей для проверки направления (12.5 часов)
            current_df = df_analyzed.iloc[:i+1].copy()
            last = current_df.iloc[-1]
            prev = current_df.iloc[-2]
            
            current_time = last['timestamp']
            
            # Проверяем cooldown
            if symbol in last_signal_time:
                time_diff = current_time - last_signal_time[symbol]
                if time_diff < timedelta(minutes=params['SIGNAL_COOLDOWN_MINUTES']):
                    continue
            
            # Проверяем базовые фильтры
            if last['adx'] < params['MIN_ADX']:
                continue
            
            # Volume фильтр
            if 'volume_ratio' in current_df.columns:
                volume_ratio = last.get('volume_ratio', 1.0)
                if volume_ratio < params['MIN_VOLUME_MA_RATIO']:
                    continue
            
            # Триггеры
            buy_triggers = 0
            sell_triggers = 0
            
            # RSI триггеры
            if last['rsi'] <= params['RSI_EXTREME_OVERSOLD']:
                buy_triggers += 2.0
            elif last['rsi'] < params['RSI_MIN']:
                buy_triggers += 1.0
                
            if last['rsi'] >= params['RSI_EXTREME_OVERBOUGHT']:
                sell_triggers += 2.0
            elif last['rsi'] > params['RSI_MAX']:
                sell_triggers += 1.0
            
            # EMA триггеры
            if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
                buy_triggers += 1.5
            elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
                buy_triggers += 0.5
                
            if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
                sell_triggers += 1.5
            elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
                sell_triggers += 0.5
            
            # MACD триггеры
            if 'macd_line' in current_df.columns and 'macd_signal' in current_df.columns:
                if last['macd_line'] > last['macd_signal']:
                    buy_triggers += 0.5
                if last['macd_line'] < last['macd_signal']:
                    sell_triggers += 0.5
            
            # Bollinger Bands убраны - не используются в скоринге реального бота
            
            min_triggers = params['MIN_TRIGGERS_ACTIVE_HOURS']
            
            # Определяем тип сигнала
            signal_type = None
            if buy_triggers >= min_triggers and last['rsi'] <= params['LONG_MAX_RSI']:
                signal_type = 'BUY'
            elif sell_triggers >= min_triggers and last['rsi'] >= params['SHORT_MIN_RSI']:
                signal_type = 'SELL'
            
            if signal_type:
                # Дополнительные фильтры
                if signal_type == 'SELL' and last['adx'] < params['SHORT_MIN_ADX']:
                    continue
                
                # Проверяем score
                try:
                    score, pattern = evaluate_signal_strength(current_df, symbol, signal_type, params)
                    if score >= params['MIN_COMPOSITE_SCORE']:
                        total_signals += 1
                        
                        # Проверяем движение в течение периода (более реалистично)
                        entry_price = last['close']
                        max_candles_to_check = min(48, len(df_analyzed) - i - 1)  # 12 часов максимум
                        direction_score = 0.0
                        
                        # Проверяем движение в течение периода
                        if max_candles_to_check >= 24:  # Минимум 24 свечи (6 часов)
                            # Ищем максимальное движение в нужном направлении за весь период
                            max_favorable_move = 0.0
                            max_drawdown = 0.0
                            
                            for j in range(i + 1, i + 1 + max_candles_to_check):
                                future_candle = df_analyzed.iloc[j]
                                
                                if signal_type == 'BUY':
                                    # Реалистичное движение вверх за период (по close ценам)
                                    price_change = (future_candle['close'] - entry_price) / entry_price
                                    max_favorable_move = max(max_favorable_move, price_change)
                                    
                                    # Реалистичная просадка (по close ценам)
                                    drawdown = (entry_price - future_candle['close']) / entry_price
                                    max_drawdown = max(max_drawdown, drawdown)
                                    
                                else:  # SELL
                                    # Реалистичное движение вниз за период (по close ценам)
                                    price_change = (entry_price - future_candle['close']) / entry_price
                                    max_favorable_move = max(max_favorable_move, price_change)
                                    
                                    # Реалистичная просадка (по close ценам)
                                    drawdown = (future_candle['close'] - entry_price) / entry_price
                                    max_drawdown = max(max_drawdown, drawdown)
                            
                            # Сигнал правильный если:
                            # 1. Реалистичное движение >= 0.5% (по close ценам)
                            # 2. Реалистичная просадка <= 1.0% (по close ценам)
                            if max_favorable_move >= 0.005 and max_drawdown <= 0.01:
                                # Бонус за стабильность движения
                                stability_bonus = 1.0 if max_drawdown <= 0.005 else 0.5
                                direction_score = max_favorable_move * stability_bonus
                                correct_signals += 1
                        
                        total_direction_score += direction_score
                        
                        last_signal_time[symbol] = current_time
                        
                except Exception as e:
                    continue
        
        accuracy = correct_signals / total_signals if total_signals > 0 else 0
        
        return {
            'correct_signals': correct_signals,
            'total_signals': total_signals,
            'accuracy': accuracy,
            'avg_direction_score': total_direction_score / total_signals if total_signals > 0 else 0
        }
        
    except Exception as e:
        return {'correct_signals': 0, 'total_signals': 0, 'accuracy': 0}

def simulate_trading_with_tp_sl(df: pd.DataFrame, symbol: str, params: Dict) -> Dict:
    """Симуляция торговли с TP/SL"""
    try:
        if df.empty or len(df) < params['MIN_15M_CANDLES']:
            return {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0, 'avg_rr': 0, 'wins': 0, 'losses': 0}
        
        df_analyzed = analyze_with_all_filters(df.copy(), params)
        if df_analyzed.empty:
            return {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0, 'avg_rr': 0, 'wins': 0, 'losses': 0}
        
        trades = []
        last_signal_time = {}
        total_pnl = 0
        
        # Проходим по всем свечам
        for i in range(params['MIN_15M_CANDLES'], len(df_analyzed) - 100):  # Нужно 100 свечей для проверки TP/SL (25 часов)
            current_df = df_analyzed.iloc[:i+1].copy()
            last = current_df.iloc[-1]
            prev = current_df.iloc[-2]
            
            current_time = last['timestamp']
            
            # Проверяем cooldown
            if symbol in last_signal_time:
                time_diff = current_time - last_signal_time[symbol]
                if time_diff < timedelta(minutes=params['SIGNAL_COOLDOWN_MINUTES']):
                    continue
            
            # Проверяем базовые фильтры
            if last['adx'] < params['MIN_ADX']:
                continue
            
            # Volume фильтр
            if 'volume_ratio' in current_df.columns:
                volume_ratio = last.get('volume_ratio', 1.0)
                if volume_ratio < params['MIN_VOLUME_MA_RATIO']:
                    continue
            
            # Триггеры (та же логика что и в check_signal_direction_quality)
            buy_triggers = 0
            sell_triggers = 0
            
            # RSI триггеры
            if last['rsi'] <= params['RSI_EXTREME_OVERSOLD']:
                buy_triggers += 2.0
            elif last['rsi'] < params['RSI_MIN']:
                buy_triggers += 1.0
                
            if last['rsi'] >= params['RSI_EXTREME_OVERBOUGHT']:
                sell_triggers += 2.0
            elif last['rsi'] > params['RSI_MAX']:
                sell_triggers += 1.0
            
            # EMA триггеры
            if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
                buy_triggers += 1.5
            elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
                buy_triggers += 0.5
                
            if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
                sell_triggers += 1.5
            elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
                sell_triggers += 0.5
            
            # MACD триггеры
            if 'macd_line' in current_df.columns and 'macd_signal' in current_df.columns:
                if last['macd_line'] > last['macd_signal']:
                    buy_triggers += 0.5
                if last['macd_line'] < last['macd_signal']:
                    sell_triggers += 0.5
            
            # Bollinger Bands убраны - не используются в скоринге реального бота
            
            min_triggers = params['MIN_TRIGGERS_ACTIVE_HOURS']
            
            # Определяем тип сигнала
            signal_type = None
            if buy_triggers >= min_triggers and last['rsi'] <= params['LONG_MAX_RSI']:
                signal_type = 'BUY'
            elif sell_triggers >= min_triggers and last['rsi'] >= params['SHORT_MIN_RSI']:
                signal_type = 'SELL'
            
            if signal_type:
                # Дополнительные фильтры
                if signal_type == 'SELL' and last['adx'] < params['SHORT_MIN_ADX']:
                    continue
                
                # Проверяем score
                try:
                    score, pattern = evaluate_signal_strength(current_df, symbol, signal_type, params)
                    if score >= params['MIN_COMPOSITE_SCORE']:
                        # Рассчитываем TP/SL
                        direction = 'SHORT' if signal_type == 'SELL' else 'LONG'
                        tp_price, sl_price = calculate_tp_sl(current_df, last['close'], last['atr'], direction, params)
                        
                        if tp_price is None or sl_price is None:
                            continue
                        
                        # Симулируем результат сделки
                        entry_price = last['close']
                        trade_result = None
                        
                        # Ищем следующую свечу для закрытия (максимум 80 свечей вперед = 20 часов)
                        for j in range(i + 1, min(i + 80, len(df_analyzed))):
                            future_candle = df_analyzed.iloc[j]
                            
                            # Проверяем TP/SL
                            if direction == 'LONG':
                                if future_candle['high'] >= tp_price:
                                    # TP достигнут
                                    pnl_pct = (tp_price - entry_price) / entry_price
                                    trade_result = {'type': 'WIN', 'pnl': pnl_pct, 'rr': abs(pnl_pct / ((entry_price - sl_price) / entry_price))}
                                    break
                                elif future_candle['low'] <= sl_price:
                                    # SL достигнут
                                    pnl_pct = (sl_price - entry_price) / entry_price
                                    trade_result = {'type': 'LOSS', 'pnl': pnl_pct, 'rr': 0}
                                    break
                            else:  # SHORT
                                if future_candle['low'] <= tp_price:
                                    # TP достигнут
                                    pnl_pct = (entry_price - tp_price) / entry_price
                                    trade_result = {'type': 'WIN', 'pnl': pnl_pct, 'rr': abs(pnl_pct / ((sl_price - entry_price) / entry_price))}
                                    break
                                elif future_candle['high'] >= sl_price:
                                    # SL достигнут
                                    pnl_pct = (entry_price - sl_price) / entry_price
                                    trade_result = {'type': 'LOSS', 'pnl': pnl_pct, 'rr': 0}
                                    break
                        
                        # Если сделка не закрылась за 80 свечей, считаем убыток
                        if trade_result is None:
                            if direction == 'LONG':
                                exit_price = df_analyzed.iloc[min(i + 80, len(df_analyzed) - 1)]['close']
                                pnl_pct = (exit_price - entry_price) / entry_price
                            else:
                                exit_price = df_analyzed.iloc[min(i + 80, len(df_analyzed) - 1)]['close']
                                pnl_pct = (entry_price - exit_price) / entry_price
                            
                            trade_result = {'type': 'TIMEOUT', 'pnl': pnl_pct, 'rr': 0}
                        
                        trades.append(trade_result)
                        total_pnl += trade_result['pnl']
                        last_signal_time[symbol] = current_time
                        
                except Exception as e:
                    continue
        
        # Рассчитываем статистику
        if trades:
            wins = [t for t in trades if t['type'] == 'WIN']
            losses = [t for t in trades if t['type'] in ['LOSS', 'TIMEOUT']]
            win_rate = len(wins) / len(trades) if trades else 0
            avg_rr = sum(t['rr'] for t in trades) / len(trades) if trades else 0
            
            return {
                'total_pnl': total_pnl * 100,  # В процентах
                'win_rate': win_rate,
                'total_trades': len(trades),
                'avg_rr': avg_rr,
                'wins': len(wins),
                'losses': len(losses)
            }
        else:
            return {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0, 'avg_rr': 0, 'wins': 0, 'losses': 0}
            
    except Exception as e:
        return {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0, 'avg_rr': 0, 'wins': 0, 'losses': 0}

def test_signal_quality(params: Dict) -> Optional[Dict]:
    """Тестирует качество сигналов"""
    try:
        total_correct = 0
        total_signals = 0
        total_direction_score = 0.0
        
        for symbol in SYMBOLS:
            if symbol not in DATA_CACHE:
                continue
            
            df = DATA_CACHE[symbol]
            result = check_signal_direction_quality(df, symbol, params)
            
            total_correct += result['correct_signals']
            total_signals += result['total_signals']
            total_direction_score += result.get('avg_direction_score', 0) * result['total_signals']
        
        if total_signals == 0:
            return None
        
        accuracy = total_correct / total_signals
        avg_direction_score = total_direction_score / total_signals if total_signals > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total_signals': total_signals,
            'correct_signals': total_correct,
            'avg_direction_score': avg_direction_score
        }
        
    except Exception as e:
        logging.error(f"Ошибка тестирования качества сигналов: {e}")
        return None

def test_trading_performance(params: Dict) -> Optional[Dict]:
    """Тестирует торговую производительность"""
    try:
        symbol_results = []
        total_trades = 0
        total_wins = 0
        total_losses = 0
        
        for symbol in SYMBOLS:
            if symbol not in DATA_CACHE:
                continue
            
            df = DATA_CACHE[symbol]
            result = simulate_trading_with_tp_sl(df, symbol, params)
            
            # Сохраняем результаты по символам для правильного расчета
            symbol_results.append(result)
            total_trades += result['total_trades']
            total_wins += result['wins']
            total_losses += result['losses']
        
        # Правильный расчет общего P&L (средний по символам)
        if symbol_results:
            total_pnl = sum(r['total_pnl'] for r in symbol_results) / len(symbol_results)
        else:
            total_pnl = 0
        
        if total_trades == 0:
            return None
        
        win_rate = total_wins / total_trades
        avg_pnl_per_trade = total_pnl / total_trades
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'wins': total_wins,
            'losses': total_losses
        }
        
    except Exception as e:
        logging.error(f"Ошибка тестирования торговой производительности: {e}")
        return None

# Глобальная переменная для хранения лучших фильтров из этапа 1
BEST_FILTERS_STAGE1 = None

# Минимальное количество сигналов для Stage 1
MIN_REQUIRED_SIGNALS = 30  # Минимум 30 сигналов за период тестирования (более мягкое требование)

def stage1_objective(trial: optuna.Trial) -> float:
    """ЭТАП 1: Оптимизация качества сигналов"""
    try:
        # Создаем параметры на основе текущих
        params = CURRENT_PARAMS.copy()
        
        # Оптимизируем только фильтры сигналов (НЕ TP/SL) с ограниченной точностью
        params['MIN_COMPOSITE_SCORE'] = trial.suggest_float('MIN_COMPOSITE_SCORE', 0.0, 1.0, step=0.5)
        params['MIN_ADX'] = trial.suggest_int('MIN_ADX', 3, 25)
        params['SHORT_MIN_ADX'] = trial.suggest_int('SHORT_MIN_ADX', 15, 60)
        params['RSI_MIN'] = trial.suggest_int('RSI_MIN', 15, 50)
        params['RSI_MAX'] = trial.suggest_int('RSI_MAX', 50, 95)
        params['LONG_MAX_RSI'] = trial.suggest_int('LONG_MAX_RSI', 25, 90)
        params['SHORT_MIN_RSI'] = trial.suggest_int('SHORT_MIN_RSI', 10, 95)
        params['SIGNAL_COOLDOWN_MINUTES'] = trial.suggest_int('SIGNAL_COOLDOWN_MINUTES', 30, 90)
        params['MIN_TRIGGERS_ACTIVE_HOURS'] = trial.suggest_float('MIN_TRIGGERS_ACTIVE_HOURS', 0.5, 2.0, step=0.1)
        params['MIN_VOLUME_MA_RATIO'] = trial.suggest_float('MIN_VOLUME_MA_RATIO', 0.1, 2.0, step=0.1)
        params['RSI_WINDOW'] = trial.suggest_int('RSI_WINDOW', 5, 21)
        params['RSI_EXTREME_OVERSOLD'] = trial.suggest_int('RSI_EXTREME_OVERSOLD', 10, 30)
        params['RSI_EXTREME_OVERBOUGHT'] = trial.suggest_int('RSI_EXTREME_OVERBOUGHT', 70, 90)
        params['ATR_WINDOW'] = trial.suggest_int('ATR_WINDOW', 10, 20)
        params['ADX_WINDOW'] = trial.suggest_int('ADX_WINDOW', 10, 20)
        # BB и VWAP параметры убраны - не используются в скоринге реального бота
        params['MACD_FAST'] = trial.suggest_int('MACD_FAST', 5, 15)
        params['MACD_SLOW'] = trial.suggest_int('MACD_SLOW', 15, 30)
        params['MACD_SIGNAL'] = trial.suggest_int('MACD_SIGNAL', 3, 12)
        params['MA_FAST'] = trial.suggest_int('MA_FAST', 5, 25)
        params['MA_SLOW'] = trial.suggest_int('MA_SLOW', 15, 35)
        params['WEIGHT_RSI'] = trial.suggest_float('WEIGHT_RSI', 1.0, 15.0, step=0.5)
        params['WEIGHT_MACD'] = trial.suggest_float('WEIGHT_MACD', 1.0, 10.0, step=0.5)
        params['WEIGHT_ADX'] = trial.suggest_float('WEIGHT_ADX', 1.0, 15.0, step=0.5)
        # WEIGHT_BB, WEIGHT_VWAP, WEIGHT_VOLUME убраны - не используются в скоринге
        params['SHORT_BOOST_MULTIPLIER'] = trial.suggest_float('SHORT_BOOST_MULTIPLIER', 0.5, 5.0, step=0.5)
        params['LONG_PENALTY_IN_DOWNTREND'] = trial.suggest_float('LONG_PENALTY_IN_DOWNTREND', 0.1, 1.0, step=0.1)
        params['REQUIRE_MACD_HISTOGRAM_CONFIRMATION'] = trial.suggest_categorical('REQUIRE_MACD_HISTOGRAM_CONFIRMATION', [True, False])
        

        # Тестируем качество сигналов
        result = test_signal_quality(params)
        
        if result is None:
            return 0.0
        
        # Целевая функция: комбинированная оценка качества сигналов
        accuracy = result['accuracy']
        avg_direction_score = result.get('avg_direction_score', 0)
        total_signals = result['total_signals']
        correct_signals_count = result['correct_signals']
        
        # НОВАЯ ЛОГИКА: Приоритет надежности и количества сигналов
        
        # 1. Базовая оценка: точность направления (более мягкая)
        direction_score = accuracy * (1 + avg_direction_score * 0.5)  # Уменьшаем вес качества движения
        
        # 2. Бонус за количество выигрышных сигналов (главный приоритет)
        winning_signals_bonus = min(correct_signals_count / 200, 2.0)  # 200 выигрышных = бонус 2.0 (максимум)
        
        # 3. Бонус за общее количество сигналов (второй приоритет)
        total_signals_bonus = min(total_signals / 500, 1.5)  # 500 сигналов = бонус 1.5 (максимум)
        
        # 4. Комбинированный score с приоритетом количества
        base_score = direction_score + winning_signals_bonus * 0.6 + total_signals_bonus * 0.4
        
        # 5. Мягкий штраф за очень низкую точность (только если < 75%)
        if accuracy < 0.75:
            accuracy_penalty = (0.75 - accuracy) * 1.0  # Мягкий штраф
            score = base_score - accuracy_penalty
        else:
            score = base_score
        
        # КРИТИЧНО: Минимальное требование - не менее 30 сигналов
        if total_signals < MIN_REQUIRED_SIGNALS:
            logging.info(f"Stage 1 Trial {trial.number}: ОТКЛОНЕН - недостаточно сигналов: {total_signals} < {MIN_REQUIRED_SIGNALS}")
            return 0.0  # Полностью отклоняем параметры с малым количеством сигналов
        
        # Логирование лучших результатов
        if score > 2.0 and trial.number % 20 == 0:
            avg_score = result.get('avg_direction_score', 0)
            logging.info(f"Stage 1 Trial {trial.number}: Score={score:.3f}, Accuracy={accuracy:.1%}, WinningSignals={correct_signals_count}, TotalSignals={total_signals}, DirectionScore={avg_score:.3f}")
        
        return score
        
    except Exception as e:
        logging.error(f"Ошибка в stage1_objective: {e}")
        return 0.0

def stage2_objective_max_profit(trial: optuna.Trial) -> float:
    """ЭТАП 2: Оптимизация TP/SL для максимальной прибыли"""
    try:
        # Используем лучшие фильтры из этапа 1
        if BEST_FILTERS_STAGE1 is None:
            logging.error("Этап 1 не завершен!")
            return 0.0
        
        params = BEST_FILTERS_STAGE1.copy()
        
        # Оптимизируем только TP/SL параметры с ограниченной точностью
        params['TP_ATR_MULT'] = trial.suggest_float('TP_ATR_MULT', 1.0, 8.0, step=0.1)
        params['SL_ATR_MULT'] = trial.suggest_float('SL_ATR_MULT', 1.0, 4.0, step=0.1)
        params['TP_MIN'] = trial.suggest_float('TP_MIN', 0.005, 0.04, step=0.002)
        params['SL_MIN'] = trial.suggest_float('SL_MIN', 0.005, 0.04, step=0.002)
        
        # Тестируем торговую производительность
        result = test_trading_performance(params)
        
        if result is None:
            return 0.0
        
        # Целевая функция: максимальная прибыльность
        score = result['total_pnl']
        
        # Логирование лучших результатов
        if score > 20 and trial.number % 20 == 0:
            logging.info(f"Stage 2 MaxProfit Trial {trial.number}: P&L={score:.1f}%, Winrate={result['win_rate']:.1%}, Trades={result['total_trades']}")
        
        return score
        
    except Exception as e:
        logging.error(f"Ошибка в stage2_objective_max_profit: {e}")
        return 0.0

def stage2_objective_max_winrate(trial: optuna.Trial) -> float:
    """ЭТАП 2: Оптимизация TP/SL для максимального винрейте с учетом прибыли"""
    try:
        # Используем лучшие фильтры из этапа 1
        if BEST_FILTERS_STAGE1 is None:
            logging.error("Этап 1 не завершен!")
            return 0.0
        
        params = BEST_FILTERS_STAGE1.copy()
        
        # Оптимизируем только TP/SL параметры с ограниченной точностью
        params['TP_ATR_MULT'] = trial.suggest_float('TP_ATR_MULT', 1.0, 8.0, step=0.1)
        params['SL_ATR_MULT'] = trial.suggest_float('SL_ATR_MULT', 1.0, 4.0, step=0.1)
        params['TP_MIN'] = trial.suggest_float('TP_MIN', 0.005, 0.04, step=0.002)
        params['SL_MIN'] = trial.suggest_float('SL_MIN', 0.005, 0.04, step=0.002)
        
        # Тестируем торговую производительность
        result = test_trading_performance(params)
        
        if result is None:
            return 0.0
        
        # Целевая функция: винрейт + прибыль (приоритет винрейте)
        winrate_score = result['win_rate'] * 100  # Винрейт в процентах
        profit_bonus = result['total_pnl'] * 0.3  # Бонус за прибыль (30% от P&L)
        
        # Комбинированный score: винрейт + бонус за прибыль
        score = winrate_score + profit_bonus
        
        # Логирование лучших результатов
        if score > 30 and trial.number % 20 == 0:
            logging.info(f"Stage 2 MaxWinrate Trial {trial.number}: Score={score:.1f}, Winrate={result['win_rate']:.1%}, P&L={result['total_pnl']:.1f}%, Trades={result['total_trades']}")
        
        return score
        
    except Exception as e:
        logging.error(f"Ошибка в stage2_objective_max_winrate: {e}")
        return 0.0

def stage2_objective_balanced(trial: optuna.Trial) -> float:
    """ЭТАП 2: Оптимизация TP/SL для сбалансированного результата"""
    try:
        # Используем лучшие фильтры из этапа 1
        if BEST_FILTERS_STAGE1 is None:
            logging.error("Этап 1 не завершен!")
            return 0.0
        
        params = BEST_FILTERS_STAGE1.copy()
        
        # Оптимизируем только TP/SL параметры с ограниченной точностью
        params['TP_ATR_MULT'] = trial.suggest_float('TP_ATR_MULT', 1.0, 8.0, step=0.1)
        params['SL_ATR_MULT'] = trial.suggest_float('SL_ATR_MULT', 1.0, 4.0, step=0.1)
        params['TP_MIN'] = trial.suggest_float('TP_MIN', 0.005, 0.04, step=0.002)
        params['SL_MIN'] = trial.suggest_float('SL_MIN', 0.005, 0.04, step=0.002)
        
        # Тестируем торговую производительность
        result = test_trading_performance(params)
        
        if result is None:
            return 0.0
        
        # Целевая функция: сбалансированный подход
        winrate_score = result['win_rate'] * 50  # Винрейт * 50
        profit_score = result['total_pnl'] * 0.5  # Прибыль * 0.5
        
        # Комбинированный score: равный вес винрейте и прибыли
        score = winrate_score + profit_score
        
        # Логирование лучших результатов
        if score > 25 and trial.number % 20 == 0:
            logging.info(f"Stage 2 Balanced Trial {trial.number}: Score={score:.1f}, Winrate={result['win_rate']:.1%}, P&L={result['total_pnl']:.1f}%, Trades={result['total_trades']}")
        
        return score
        
    except Exception as e:
        logging.error(f"Ошибка в stage2_objective_balanced: {e}")
        return 0.0

async def run_two_stage_optimization():
    """Запуск 2-этапной оптимизации"""
    global BEST_FILTERS_STAGE1
    
    logging.info("🚀 Запускаем 2-этапную оптимизацию...")
    
    # Загружаем данные
    await load_data_for_optimization()
    
    if not DATA_CACHE:
        logging.error("❌ Нет данных для оптимизации!")
        return
    
    # =============================================================================
    # ЭТАП 1: Оптимизация качества сигналов
    # =============================================================================
    logging.info("\n" + "="*80)
    logging.info("🎯 ЭТАП 1: ОПТИМИЗАЦИЯ КАЧЕСТВА СИГНАЛОВ")
    logging.info("="*80)
    
    study1 = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    logging.info("🔍 Ищем лучшие фильтры для качества сигналов...")
    study1.optimize(stage1_objective, n_trials=2000)
    
    best_filters = study1.best_params
    best_accuracy = study1.best_value
    
    logging.info(f"\n🏆 ЛУЧШИЕ ФИЛЬТРЫ ЭТАПА 1:")
    logging.info(f"📊 Комбинированный score: {best_accuracy:.3f}")
    logging.info(f"📈 Параметры фильтров:")
    
    for key, value in best_filters.items():
        logging.info(f"  {key}: {value}")
    
    # Сохраняем лучшие фильтры для этапа 2
    BEST_FILTERS_STAGE1 = CURRENT_PARAMS.copy()
    BEST_FILTERS_STAGE1.update(best_filters)
    
    # =============================================================================
    # ЭТАП 2: Оптимизация TP/SL (3 режима)
    # =============================================================================
    logging.info("\n" + "="*80)
    logging.info("💰 ЭТАП 2: ОПТИМИЗАЦИЯ TP/SL (3 РЕЖИМА)")
    logging.info("="*80)
    
    # Режим 1: Максимальная прибыль
    logging.info("\n🎯 РЕЖИМ 1: МАКСИМАЛЬНАЯ ПРИБЫЛЬ")
    logging.info("-" * 50)
    
    study_max_profit = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    logging.info("🔍 Ищем TP/SL для максимальной прибыли...")
    study_max_profit.optimize(stage2_objective_max_profit, n_trials=400)
    
    best_max_profit = study_max_profit.best_params
    best_max_profit_value = study_max_profit.best_value
    
    logging.info(f"💰 Лучший P&L: {best_max_profit_value:.1f}%")
    logging.info(f"📊 TP/SL параметры (максимальная прибыль):")
    for key, value in best_max_profit.items():
        logging.info(f"  {key}: {value}")
    
    # Режим 2: Максимальный винрейт
    logging.info("\n🎯 РЕЖИМ 2: МАКСИМАЛЬНЫЙ ВИНРЕЙТ")
    logging.info("-" * 50)
    
    study_max_winrate = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=43)
    )
    
    logging.info("🔍 Ищем TP/SL для максимального винрейте...")
    study_max_winrate.optimize(stage2_objective_max_winrate, n_trials=400)
    
    best_max_winrate = study_max_winrate.best_params
    best_max_winrate_value = study_max_winrate.best_value
    
    logging.info(f"🎯 Лучший Score: {best_max_winrate_value:.1f}")
    logging.info(f"📊 TP/SL параметры (максимальный винрейт):")
    for key, value in best_max_winrate.items():
        logging.info(f"  {key}: {value}")
    
    # Режим 3: Сбалансированный
    logging.info("\n🎯 РЕЖИМ 3: СБАЛАНСИРОВАННЫЙ")
    logging.info("-" * 50)
    
    study_balanced = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=44)
    )
    
    logging.info("🔍 Ищем TP/SL для сбалансированного результата...")
    study_balanced.optimize(stage2_objective_balanced, n_trials=400)
    
    best_balanced = study_balanced.best_params
    best_balanced_value = study_balanced.best_value
    
    logging.info(f"⚖️ Лучший Score: {best_balanced_value:.1f}")
    logging.info(f"📊 TP/SL параметры (сбалансированный):")
    for key, value in best_balanced.items():
        logging.info(f"  {key}: {value}")
    
    # =============================================================================
    # ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ И СРАВНЕНИЕ
    # =============================================================================
    logging.info("\n" + "="*80)
    logging.info("🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ 2-ЭТАПНОЙ ОПТИМИЗАЦИИ")
    logging.info("="*80)
    
    # Тестируем все три режима
    logging.info("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    logging.info("-" * 50)
    
    # Режим 1: Максимальная прибыль
    params_max_profit = BEST_FILTERS_STAGE1.copy()
    params_max_profit.update(best_max_profit)
    result_max_profit = test_trading_performance(params_max_profit)
    
    # Режим 2: Максимальный винрейт
    params_max_winrate = BEST_FILTERS_STAGE1.copy()
    params_max_winrate.update(best_max_winrate)
    result_max_winrate = test_trading_performance(params_max_winrate)
    
    # Режим 3: Сбалансированный
    params_balanced = BEST_FILTERS_STAGE1.copy()
    params_balanced.update(best_balanced)
    result_balanced = test_trading_performance(params_balanced)
    
    # Выводим сравнение
    logging.info(f"💰 РЕЖИМ 1 (Максимальная прибыль):")
    if result_max_profit:
        logging.info(f"   P&L: {result_max_profit['total_pnl']:.1f}%")
        logging.info(f"   Винрейт: {result_max_profit['win_rate']:.1%}")
        logging.info(f"   Сделок: {result_max_profit['total_trades']}")
        logging.info(f"   R:R: {result_max_profit.get('avg_rr', 0):.2f}")
    
    logging.info(f"\n🎯 РЕЖИМ 2 (Максимальный винрейт):")
    if result_max_winrate:
        logging.info(f"   P&L: {result_max_winrate['total_pnl']:.1f}%")
        logging.info(f"   Винрейт: {result_max_winrate['win_rate']:.1%}")
        logging.info(f"   Сделок: {result_max_winrate['total_trades']}")
        logging.info(f"   R:R: {result_max_winrate.get('avg_rr', 0):.2f}")
    
    logging.info(f"\n⚖️ РЕЖИМ 3 (Сбалансированный):")
    if result_balanced:
        logging.info(f"   P&L: {result_balanced['total_pnl']:.1f}%")
        logging.info(f"   Винрейт: {result_balanced['win_rate']:.1%}")
        logging.info(f"   Сделок: {result_balanced['total_trades']}")
        logging.info(f"   R:R: {result_balanced.get('avg_rr', 0):.2f}")
    
    # Рекомендация
    logging.info(f"\n💡 РЕКОМЕНДАЦИИ:")
    if result_max_profit and result_max_winrate and result_balanced:
        max_profit_pnl = result_max_profit['total_pnl']
        max_winrate_pnl = result_max_winrate['total_pnl']
        balanced_pnl = result_balanced['total_pnl']
        
        max_profit_wr = result_max_profit['win_rate']
        max_winrate_wr = result_max_winrate['win_rate']
        balanced_wr = result_balanced['win_rate']
        
        logging.info(f"   🏆 Лучший P&L: Режим 1 ({max_profit_pnl:.1f}%)")
        logging.info(f"   🎯 Лучший винрейт: Режим 2 ({max_winrate_wr:.1%})")
        
        # Выбираем сбалансированный как рекомендуемый
        logging.info(f"   ⚖️ Рекомендуемый: Режим 3 (P&L: {balanced_pnl:.1f}%, WR: {balanced_wr:.1%})")
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"two_stage_optimization_results_{timestamp}.json"
    
    results = {
        'timestamp': timestamp,
        'stage1_accuracy': best_accuracy,
        'stage1_best_filters': best_filters,
        'max_profit_params': params_max_profit,
        'max_profit_performance': result_max_profit,
        'max_winrate_params': params_max_winrate,
        'max_winrate_performance': result_max_winrate,
        'balanced_params': params_balanced,
        'balanced_performance': result_balanced,
        'symbols_tested': SYMBOLS,
        'data_period_days': 30
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n💾 Результаты сохранены в {filename}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_two_stage_optimization())
