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
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Создаем отдельный логгер для оптимизатора
opt_logger = logging.getLogger('optimizer')
opt_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
opt_logger.addHandler(handler)

# Отключаем логи Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Импортируем из РЕАЛЬНОГО БОТА
from crypto_signal_bot import (
    analyze,
    evaluate_signal_strength, 
    calculate_tp_sl,
    check_signals,
    SYMBOLS,
    EXCHANGE
)

# Импортируем параметры из config.py
from config import *

# Собираем текущие параметры из config.py
CURRENT_PARAMS = {
    'MIN_COMPOSITE_SCORE': MIN_COMPOSITE_SCORE,
    'MIN_ADX': MIN_ADX,
    'RSI_MIN': RSI_MIN,
    'RSI_MAX': RSI_MAX,
    'SHORT_MIN_ADX': SHORT_MIN_ADX,
    'SHORT_MIN_RSI': SHORT_MIN_RSI,
    'LONG_MAX_RSI': LONG_MAX_RSI,
    'TP_ATR_MULT': TP_ATR_MULT,
    'SL_ATR_MULT': SL_ATR_MULT,
    'TP_MIN': TP_MIN,
    'SL_MIN': SL_MIN,
    'SIGNAL_COOLDOWN_MINUTES': SIGNAL_COOLDOWN_MINUTES,
    'MIN_TRIGGERS_ACTIVE_HOURS': MIN_TRIGGERS_ACTIVE_HOURS,
    'MIN_VOLUME_MA_RATIO': MIN_VOLUME_MA_RATIO,
    'RSI_WINDOW': RSI_WINDOW,
    'RSI_EXTREME_OVERSOLD': RSI_EXTREME_OVERSOLD,
    'RSI_EXTREME_OVERBOUGHT': RSI_EXTREME_OVERBOUGHT,
    'ATR_WINDOW': ATR_WINDOW,
    'ADX_WINDOW': ADX_WINDOW,
    'MACD_FAST': MACD_FAST,
    'MACD_SLOW': MACD_SLOW,
    'MACD_SIGNAL': MACD_SIGNAL,
    'WEIGHT_RSI': WEIGHT_RSI,
    'WEIGHT_MACD': WEIGHT_MACD,
    'WEIGHT_ADX': WEIGHT_ADX,
    'SHORT_BOOST_MULTIPLIER': SHORT_BOOST_MULTIPLIER,
    'LONG_PENALTY_IN_DOWNTREND': LONG_PENALTY_IN_DOWNTREND,
    'MA_FAST': MA_FAST,
    'MA_SLOW': MA_SLOW,
    'LIMIT': LIMIT,
    'MIN_15M_CANDLES': MIN_15M_CANDLES,
    'FEE_RATE': FEE_RATE
}

def apply_params_to_config(params: Dict):
    """Применяет параметры к config.py"""
    import config
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)

def analyze_with_params(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Анализ данных с оптимизируемыми параметрами
    ВАЖНО: Использует analyze() из реального бота
    """
    try:
        if df.empty:
            return pd.DataFrame()
        
        # Применяем параметры к config
        apply_params_to_config(params)
        
        # Используем реальную функцию analyze() из бота
        df_analyzed = analyze(df.copy())
        
        return df_analyzed
        
    except Exception as e:
        logging.error(f"Ошибка в анализе данных: {e}")
        return pd.DataFrame()

def evaluate_signal_strength_with_params(df: pd.DataFrame, symbol: str, action: str, params: Dict):
    """Обертка для evaluate_signal_strength с параметрами"""
    apply_params_to_config(params)
    return evaluate_signal_strength(df, symbol, action)

def calculate_tp_sl_with_params(df: pd.DataFrame, price: float, atr: float, direction: str, params: Dict):
    """Обертка для calculate_tp_sl с параметрами"""
    apply_params_to_config(params)
    return calculate_tp_sl(df, price, atr, direction)

def check_signals_with_params(df: pd.DataFrame, symbol: str, params: Dict):
    """Обертка для check_signals с параметрами"""
    apply_params_to_config(params)
    return check_signals(df, symbol)

# Кэш данных для оптимизации
DATA_CACHE = {}

def get_historical_data_sync(symbol: str, days: int = 30) -> pd.DataFrame:
    """Загрузка исторических данных через EXCHANGE (синхронно)"""
    try:
        import time
        
        # 15m: 96 свечей в день
        limit = 1000
        all_data = []
        candles_needed = days * 96
        requests_needed = (candles_needed // limit) + 1
        
        since = None
        
        for i in range(min(requests_needed, 3)):  # Максимум 3 запроса
            try:
                if since:
                    ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe='15m', limit=limit, since=since)
                else:
                    ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe='15m', limit=limit)
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Следующий запрос
                since = ohlcv[-1][0] + 1
                
                # Пауза между запросами
                time.sleep(0.2)
                
                if len(all_data) >= candles_needed:
                    break
                    
            except Exception as e:
                logging.error(f"Ошибка загрузки {symbol}: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        # Создаем DataFrame как в боте
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Moscow')
        df['volume_usdt'] = df['volume'] * df['close']
        
        # Удаляем дубликаты
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        logging.error(f"Ошибка get_historical_data_sync для {symbol}: {e}")
        return pd.DataFrame()

async def load_data_for_optimization():
    """Загружаем данные для оптимизации"""
    global DATA_CACHE
    
    opt_logger.info("📊 Загружаем данные для оптимизации...")
    
    for symbol in SYMBOLS:
        opt_logger.info(f"📈 Загружаем {symbol}...")
        df = get_historical_data_sync(symbol, days=30)  # 30 дней
        if not df.empty:
            DATA_CACHE[symbol] = df
            opt_logger.info(f"✅ {symbol}: {len(df)} свечей (~{len(df)/96:.1f} дней)")
        else:
            opt_logger.info(f"⚠️ Нет данных для {symbol}")
    
    opt_logger.info(f"📊 Загружено {len(DATA_CACHE)} символов")

def check_signal_direction_quality(df: pd.DataFrame, symbol: str, params: Dict) -> Dict:
    """Проверка качества сигналов - правильное ли направление"""
    try:
        if df.empty or len(df) < 100:
            return {'correct_signals': 0, 'total_signals': 0, 'accuracy': 0}
        
        df_analyzed = analyze_with_params(df.copy(), params)
        if df_analyzed.empty or len(df_analyzed) < 100:
            return {'correct_signals': 0, 'total_signals': 0, 'accuracy': 0}
        
        correct_signals = 0
        total_signals = 0
        total_direction_score = 0.0
        
        # Проходим по всем свечам
        for i in range(100, len(df_analyzed) - 50, 10):  # Шаг 10 свечей для ускорения
            current_df = df_analyzed.iloc[:i+1].copy()
            last = current_df.iloc[-1]
            current_time = last['timestamp']
            
            # УПРОЩЕНО: Используем check_signals из бота напрямую!
            try:
                signals = check_signals_with_params(current_df, symbol, params)
                
                if signals:
                    total_signals += 1
                    
                    # Определяем тип сигнала из текста
                    signal_text = signals[0]
                    if 'LONG' in signal_text or '🟢' in signal_text:
                        signal_type = 'BUY'
                    else:
                        signal_type = 'SELL'
                    
                    entry_price = last['close']  # Цена входа
                    
                    # Проверяем движение в следующих 48 свечах (12 часов)
                    max_candles = min(48, len(df_analyzed) - i - 1)
                    direction_score = 0.0
                    
                    if max_candles >= 24:  # Минимум 6 часов
                        max_favorable_move = 0.0
                        
                        for j in range(i + 1, i + 1 + max_candles):
                            future_price = df_analyzed.iloc[j]['close']
                            
                            if signal_type == 'BUY':
                                price_change = (future_price - entry_price) / entry_price
                            else:  # SELL
                                price_change = (entry_price - future_price) / entry_price
                            
                            max_favorable_move = max(max_favorable_move, price_change)
                        
                        # Сигнал правильный если движение >= 0.5%
                        if max_favorable_move >= 0.005:
                            direction_score = max_favorable_move
                            correct_signals += 1
                    
                    total_direction_score += direction_score
                    
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
    """Симуляция торговли с TP/SL используя check_signals из бота"""
    try:
        if df.empty or len(df) < 100:
            return {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0, 'avg_rr': 0, 'wins': 0, 'losses': 0}
        
        df_analyzed = analyze_with_params(df.copy(), params)
        if df_analyzed.empty or len(df_analyzed) < 100:
            return {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0, 'avg_rr': 0, 'wins': 0, 'losses': 0}
        
        trades = []
        total_pnl = 0
        
        # Проходим по всем свечам с шагом 10 для ускорения
        for i in range(100, len(df_analyzed) - 100, 10):  # Шаг 10 свечей
            current_df = df_analyzed.iloc[:i+1].copy()
            last = current_df.iloc[-1]
            
            # УПРОЩЕНО: Используем check_signals из бота!
            try:
                signals = check_signals_with_params(current_df, symbol, params)
                
                if signals:
                    # Определяем тип
                    signal_text = signals[0]
                    if 'LONG' in signal_text or '🟢' in signal_text:
                        direction = 'LONG'
                    else:
                        direction = 'SHORT'
                    
                    entry_price = last['close']
                    atr = last['atr']
                    
                    # Рассчитываем TP/SL
                    tp_price, sl_price = calculate_tp_sl_with_params(current_df, entry_price, atr, direction, params)
                    
                    if tp_price is None or sl_price is None:
                        continue
                    
                    # Симулируем результат сделки
                    trade_result = None
                    
                    # Ищем выход в следующих 80 свечах (20 часов)
                    for j in range(i + 1, min(i + 80, len(df_analyzed))):
                        future_candle = df_analyzed.iloc[j]
                        
                        # Проверяем TP/SL
                        if direction == 'LONG':
                            if future_candle['high'] >= tp_price:
                                pnl_pct = (tp_price - entry_price) / entry_price
                                trade_type = 'WIN' if pnl_pct > 0 else 'LOSS'
                                trade_result = {'type': trade_type, 'pnl': pnl_pct, 'rr': abs(pnl_pct / ((entry_price - sl_price) / entry_price)) if pnl_pct > 0 else 0}
                                break
                            elif future_candle['low'] <= sl_price:
                                pnl_pct = (sl_price - entry_price) / entry_price
                                trade_type = 'WIN' if pnl_pct > 0 else 'LOSS'
                                trade_result = {'type': trade_type, 'pnl': pnl_pct, 'rr': 0}
                                break
                        else:  # SHORT
                            if future_candle['low'] <= tp_price:
                                pnl_pct = (entry_price - tp_price) / entry_price
                                trade_type = 'WIN' if pnl_pct > 0 else 'LOSS'
                                trade_result = {'type': trade_type, 'pnl': pnl_pct, 'rr': abs(pnl_pct / ((sl_price - entry_price) / entry_price)) if pnl_pct > 0 else 0}
                                break
                            elif future_candle['high'] >= sl_price:
                                pnl_pct = (entry_price - sl_price) / entry_price
                                trade_type = 'WIN' if pnl_pct > 0 else 'LOSS'
                                trade_result = {'type': trade_type, 'pnl': pnl_pct, 'rr': 0}
                                break
                    
                    # Если не закрылась за 80 свечей, закрываем по текущей цене
                    if trade_result is None:
                        if direction == 'LONG':
                            exit_price = df_analyzed.iloc[min(i + 80, len(df_analyzed) - 1)]['close']
                            pnl_pct = (exit_price - entry_price) / entry_price
                        else:
                            exit_price = df_analyzed.iloc[min(i + 80, len(df_analyzed) - 1)]['close']
                            pnl_pct = (entry_price - exit_price) / entry_price
                        
                        trade_type = 'WIN' if pnl_pct > 0 else 'LOSS'
                        trade_result = {'type': trade_type, 'pnl': pnl_pct, 'rr': 0}
                    
                    trades.append(trade_result)
                    total_pnl += trade_result['pnl']
                    
            except Exception as e:
                continue
        
        # Рассчитываем статистику
        if trades:
            wins = [t for t in trades if t['type'] == 'WIN']
            losses = [t for t in trades if t['type'] == 'LOSS']
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
        
        # ИСПРАВЛЕНО: Правильный расчет общего P&L (СУММА, а не среднее!)
        if symbol_results:
            # Общий P&L = сумма P&L по всем символам
            total_pnl = sum(r['total_pnl'] for r in symbol_results)
        else:
            total_pnl = 0
        
        if total_trades == 0:
            return None
        
        win_rate = total_wins / total_trades
        # Средний P&L на сделку = общий P&L / количество сделок
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

# Минимальное количество сделок для Stage 1
MIN_REQUIRED_TRADES = 5  # Минимум 5 сделок (ищем баланс: МНОГО прибыльных сигналов)

def stage1_objective(trial: optuna.Trial) -> float:
    """ЭТАП 1: Оптимизация качества сигналов"""
    try:
        # Создаем параметры на основе текущих
        params = CURRENT_PARAMS.copy()
        
        # ОЧЕНЬ МЯГКИЕ диапазоны для максимума возможностей
        params['MIN_COMPOSITE_SCORE'] = trial.suggest_float('MIN_COMPOSITE_SCORE', 0.0, 2.0, step=0.2)  # Начинаем с 0!
        params['MIN_ADX'] = trial.suggest_int('MIN_ADX', 5, 20)  # Еще мягче
        params['SHORT_MIN_ADX'] = trial.suggest_int('SHORT_MIN_ADX', 10, 30)  # Еще мягче
        params['RSI_MIN'] = trial.suggest_int('RSI_MIN', 10, 40)  # Очень широкий
        params['RSI_MAX'] = trial.suggest_int('RSI_MAX', 60, 90)  # Очень широкий
        params['LONG_MAX_RSI'] = trial.suggest_int('LONG_MAX_RSI', 25, 50)  # Еще шире!
        params['SHORT_MIN_RSI'] = trial.suggest_int('SHORT_MIN_RSI', 10, 30)  # Шире
        params['SIGNAL_COOLDOWN_MINUTES'] = trial.suggest_int('SIGNAL_COOLDOWN_MINUTES', 15, 60)  # Короче!
        params['MIN_TRIGGERS_ACTIVE_HOURS'] = trial.suggest_float('MIN_TRIGGERS_ACTIVE_HOURS', 0.3, 1.5, step=0.1)  # Еще мягче!
        params['MIN_VOLUME_MA_RATIO'] = trial.suggest_float('MIN_VOLUME_MA_RATIO', 0.1, 2.0, step=0.1)  # Очень мягкий
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
        
        # ПРАВИЛЬНО: Stage 1 проверяет НАПРАВЛЕНИЕ сигнала (не P&L!)
        # Тестируем что сигналы идут в правильную сторону
        result = test_signal_quality(params)
        
        if result is None or result['total_signals'] == 0:
            return 0.0
        
        accuracy = result['accuracy']  # % сигналов что пошли в правильном направлении
        total_signals = result['total_signals']
        correct_signals = result['correct_signals']
        avg_direction_score = result.get('avg_direction_score', 0)
        
        # 1. Требуем минимальную точность направления
        if accuracy < 0.60:  # Минимум 60% правильных направлений
            return 0.0
        
        # 2. Требуем минимум сигналов
        if total_signals < 5:
            return 0.0
        
        # 3. Score: accuracy + качество движения + бонус за количество
        base_score = accuracy * 100  # Главное - правильность направления!
        
        # Бонус за сильное движение в правильном направлении
        direction_bonus = avg_direction_score * 200  # Чем сильнее движение, тем лучше
        
        # Бонус за МНОГО правильных сигналов
        if correct_signals >= 30:
            quantity_bonus = 50.0
        elif correct_signals >= 20:
            quantity_bonus = 30.0
        elif correct_signals >= 10:
            quantity_bonus = 20.0
        elif correct_signals >= 5:
            quantity_bonus = 10.0
        else:
            quantity_bonus = 0
        
        score = base_score + direction_bonus + quantity_bonus
        
        # Логирование хороших результатов
        if score > 80:
            opt_logger.info(f"✅ Trial {trial.number}: Score={score:.1f}, Accuracy={accuracy:.1%}, CorrectSignals={correct_signals}/{total_signals}, AvgMove={avg_direction_score:.3f}")
        
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
        
        # Оптимизируем TP/SL для МАКСИМАЛЬНОЙ ПРИБЫЛИ (агрессивные TP)
        params['TP_ATR_MULT'] = trial.suggest_float('TP_ATR_MULT', 1.0, 8.0, step=0.3)  # Более высокие TP
        params['SL_ATR_MULT'] = trial.suggest_float('SL_ATR_MULT', 1.0, 8.0, step=0.3)  # Умеренные SL
        params['TP_MIN'] = trial.suggest_float('TP_MIN', 0.005, 0.06, step=0.003)  # Выше минимум (2.5-6%)
        params['SL_MIN'] = trial.suggest_float('SL_MIN', 0.005, 0.06, step=0.003)  # 1.5-3.5%
        
        # Тестируем торговую производительность
        result = test_trading_performance(params)
        
        if result is None:
            return 0.0
        
        # Целевая функция: максимальная прибыльность
        score = result['total_pnl']
        
        # Логирование лучших результатов
        if score > 10:
            opt_logger.info(f"✅ Stage 2 MaxProfit Trial {trial.number}: P&L={score:.1f}%, WR={result['win_rate']:.1%}, Trades={result['total_trades']}")
        
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
        if score > 40:
            opt_logger.info(f"✅ Stage 2 MaxWinrate Trial {trial.number}: Score={score:.1f}, WR={result['win_rate']:.1%}, P&L={result['total_pnl']:.1f}%")
        
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
        if score > 30:
            opt_logger.info(f"✅ Stage 2 Balanced Trial {trial.number}: Score={score:.1f}, WR={result['win_rate']:.1%}, P&L={result['total_pnl']:.1f}%")
        
        return score
        
    except Exception as e:
        logging.error(f"Ошибка в stage2_objective_balanced: {e}")
        return 0.0

async def run_two_stage_optimization():
    """Запуск 2-этапной оптимизации"""
    global BEST_FILTERS_STAGE1
    
    opt_logger.info("🚀 Запускаем 2-этапную оптимизацию...")
    
    # Загружаем данные
    await load_data_for_optimization()
    
    if not DATA_CACHE:
        logging.error("❌ Нет данных для оптимизации!")
        return
    
    # =============================================================================
    # ЭТАП 1: Оптимизация качества сигналов
    # =============================================================================
    opt_logger.info("\n" + "="*80)
    opt_logger.info("🎯 ЭТАП 1: ОПТИМИЗАЦИЯ КАЧЕСТВА СИГНАЛОВ")
    opt_logger.info("="*80)
    
    study1 = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    opt_logger.info("🔍 Ищем лучшие фильтры для качества сигналов...")
    study1.optimize(stage1_objective, n_trials=2000)
    
    best_filters = study1.best_params
    best_accuracy = study1.best_value
    
    opt_logger.info(f"\n🏆 ЛУЧШИЕ ФИЛЬТРЫ ЭТАПА 1:")
    opt_logger.info(f"📊 Комбинированный score: {best_accuracy:.3f}")
    opt_logger.info(f"📈 Параметры фильтров:")
    
    for key, value in best_filters.items():
        opt_logger.info(f"  {key}: {value}")
    
    # Сохраняем лучшие фильтры для этапа 2
    BEST_FILTERS_STAGE1 = CURRENT_PARAMS.copy()
    BEST_FILTERS_STAGE1.update(best_filters)
    
    # =============================================================================
    # ЭТАП 2: Оптимизация TP/SL (3 режима)
    # =============================================================================
    opt_logger.info("\n" + "="*80)
    opt_logger.info("💰 ЭТАП 2: ОПТИМИЗАЦИЯ TP/SL (3 РЕЖИМА)")
    opt_logger.info("="*80)
    
    # Режим 1: Максимальная прибыль
    opt_logger.info("\n🎯 РЕЖИМ 1: МАКСИМАЛЬНАЯ ПРИБЫЛЬ")
    opt_logger.info("-" * 50)
    
    study_max_profit = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    opt_logger.info("🔍 Ищем TP/SL для максимальной прибыли...")
    study_max_profit.optimize(stage2_objective_max_profit, n_trials=400)
    
    best_max_profit = study_max_profit.best_params
    best_max_profit_value = study_max_profit.best_value
    
    opt_logger.info(f"💰 Лучший P&L: {best_max_profit_value:.1f}%")
    opt_logger.info(f"📊 TP/SL параметры (максимальная прибыль):")
    for key, value in best_max_profit.items():
        opt_logger.info(f"  {key}: {value}")
    
    # Режим 2: Максимальный винрейт
    opt_logger.info("\n🎯 РЕЖИМ 2: МАКСИМАЛЬНЫЙ ВИНРЕЙТ")
    opt_logger.info("-" * 50)
    
    study_max_winrate = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=43)
    )
    
    opt_logger.info("🔍 Ищем TP/SL для максимального винрейте...")
    study_max_winrate.optimize(stage2_objective_max_winrate, n_trials=400)
    
    best_max_winrate = study_max_winrate.best_params
    best_max_winrate_value = study_max_winrate.best_value
    
    opt_logger.info(f"🎯 Лучший Score: {best_max_winrate_value:.1f}")
    opt_logger.info(f"📊 TP/SL параметры (максимальный винрейт):")
    for key, value in best_max_winrate.items():
        opt_logger.info(f"  {key}: {value}")
    
    # Режим 3: Сбалансированный
    opt_logger.info("\n🎯 РЕЖИМ 3: СБАЛАНСИРОВАННЫЙ")
    opt_logger.info("-" * 50)
    
    study_balanced = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=44)
    )
    
    opt_logger.info("🔍 Ищем TP/SL для сбалансированного результата...")
    study_balanced.optimize(stage2_objective_balanced, n_trials=400)
    
    best_balanced = study_balanced.best_params
    best_balanced_value = study_balanced.best_value
    
    opt_logger.info(f"⚖️ Лучший Score: {best_balanced_value:.1f}")
    opt_logger.info(f"📊 TP/SL параметры (сбалансированный):")
    for key, value in best_balanced.items():
        opt_logger.info(f"  {key}: {value}")
    
    # =============================================================================
    # ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ И СРАВНЕНИЕ
    # =============================================================================
    opt_logger.info("\n" + "="*80)
    opt_logger.info("🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ 2-ЭТАПНОЙ ОПТИМИЗАЦИИ")
    opt_logger.info("="*80)
    
    # Тестируем все три режима
    opt_logger.info("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    opt_logger.info("-" * 50)
    
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
    opt_logger.info(f"💰 РЕЖИМ 1 (Максимальная прибыль):")
    if result_max_profit:
        opt_logger.info(f"   P&L: {result_max_profit['total_pnl']:.1f}%")
        opt_logger.info(f"   Винрейт: {result_max_profit['win_rate']:.1%}")
        opt_logger.info(f"   Сделок: {result_max_profit['total_trades']}")
        opt_logger.info(f"   R:R: {result_max_profit.get('avg_rr', 0):.2f}")
    
    opt_logger.info(f"\n🎯 РЕЖИМ 2 (Максимальный винрейт):")
    if result_max_winrate:
        opt_logger.info(f"   P&L: {result_max_winrate['total_pnl']:.1f}%")
        opt_logger.info(f"   Винрейт: {result_max_winrate['win_rate']:.1%}")
        opt_logger.info(f"   Сделок: {result_max_winrate['total_trades']}")
        opt_logger.info(f"   R:R: {result_max_winrate.get('avg_rr', 0):.2f}")
    
    opt_logger.info(f"\n⚖️ РЕЖИМ 3 (Сбалансированный):")
    if result_balanced:
        opt_logger.info(f"   P&L: {result_balanced['total_pnl']:.1f}%")
        opt_logger.info(f"   Винрейт: {result_balanced['win_rate']:.1%}")
        opt_logger.info(f"   Сделок: {result_balanced['total_trades']}")
        opt_logger.info(f"   R:R: {result_balanced.get('avg_rr', 0):.2f}")
    
    # Рекомендация
    opt_logger.info(f"\n💡 РЕКОМЕНДАЦИИ:")
    if result_max_profit and result_max_winrate and result_balanced:
        max_profit_pnl = result_max_profit['total_pnl']
        max_winrate_pnl = result_max_winrate['total_pnl']
        balanced_pnl = result_balanced['total_pnl']
        
        max_profit_wr = result_max_profit['win_rate']
        max_winrate_wr = result_max_winrate['win_rate']
        balanced_wr = result_balanced['win_rate']
        
        opt_logger.info(f"   🏆 Лучший P&L: Режим 1 ({max_profit_pnl:.1f}%)")
        opt_logger.info(f"   🎯 Лучший винрейт: Режим 2 ({max_winrate_wr:.1%})")
        
        # Выбираем сбалансированный как рекомендуемый
        opt_logger.info(f"   ⚖️ Рекомендуемый: Режим 3 (P&L: {balanced_pnl:.1f}%, WR: {balanced_wr:.1%})")
    
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
    
    opt_logger.info(f"\n💾 Результаты сохранены в {filename}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_two_stage_optimization())
