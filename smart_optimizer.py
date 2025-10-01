"""
🎯 УМНЫЙ 2-ЭТАПНЫЙ ОПТИМИЗАТОР
================================

ЭТАП 1: Качество сигналов
- Ищем фильтры где сигнал идет в ПРАВИЛЬНОМ направлении
- Чем БОЛЬШЕ движение в нужную сторону, тем ЛУЧШЕ
- НЕ учитываем TP/SL - только направление и силу движения

ЭТАП 2: Оптимизация TP/SL
- Берем лучшие фильтры из Этапа 1
- Ищем оптимальные точки закрытия (TP/SL)
- 3 режима: Максимальная прибыль / Максимальный винрейт / Баланс
"""

import sys
import logging
import pandas as pd
import numpy as np
import optuna
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# ========================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ========================================
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('SmartOptimizer')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
optuna.logging.set_verbosity(optuna.logging.ERROR)

# ========================================
# ИМПОРТ ИЗ РЕАЛЬНОГО БОТА
# ========================================
from crypto_signal_bot import EXCHANGE, SYMBOLS, analyze
from config import *

# ========================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ========================================
DATA_CACHE = {}  # Кэш исторических данных
BEST_STAGE1_PARAMS = {}  # Лучшие параметры из Stage 1

# ========================================
# ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ
# ========================================
def load_historical_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Загрузка исторических данных с Binance"""
    try:
        timeframe = '15m'
        candles_needed = days * 96  # 96 свечей 15m в день
        
        logger.info(f"   Загружаем {symbol} за {days} дней...")
        
        all_data = []
        requests_needed = (candles_needed + 999) // 1000
        
        for i in range(min(requests_needed, 30)):
            if i == 0:
                data = EXCHANGE.fetch_ohlcv(symbol, timeframe, limit=1000)
            else:
                since = all_data[0][0] - (1000 * 15 * 60 * 1000)
                data = EXCHANGE.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            if not data:
                break
            
            all_data = data + all_data
            
            if len(all_data) >= candles_needed:
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        logger.info(f"   ✅ {symbol}: {len(df)} свечей (~{len(df)/96:.1f} дней)")
        return df
        
    except Exception as e:
        logger.error(f"   ❌ Ошибка загрузки {symbol}: {e}")
        return pd.DataFrame()

async def load_all_data():
    """Загрузка данных для всех символов"""
    global DATA_CACHE
    
    logger.info("\n📊 ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ")
    logger.info("="*60)
    
    for symbol in SYMBOLS:
        df = load_historical_data(symbol, days=30)
        if not df.empty:
            DATA_CACHE[symbol] = df
    
    logger.info(f"\n✅ Загружено символов: {len(DATA_CACHE)}/{len(SYMBOLS)}")

# ========================================
# ПРИМЕНЕНИЕ ПАРАМЕТРОВ
# ========================================
def apply_params_to_config(params: Dict):
    """Временно применяем параметры к config.py"""
    for key, value in params.items():
        if key in globals() or hasattr(sys.modules['config'], key):
            setattr(sys.modules['config'], key, value)
            globals()[key] = value

def check_signals_with_params(df, symbol, params):
    """Wrapper для check_signals с применением параметров"""
    apply_params_to_config(params)
    from crypto_signal_bot import check_signals
    return check_signals(df, symbol)

# ========================================
# ЭТАП 1: ТЕСТИРОВАНИЕ КАЧЕСТВА СИГНАЛОВ
# ========================================
def test_signal_quality(params: Dict) -> Optional[Dict]:
    """
    Тестируем КАЧЕСТВО сигналов - идут ли они в правильном направлении
    
    Возвращает:
    - total_signals: количество сигналов
    - correct_signals: сигналы что пошли в правильную сторону
    - accuracy: % правильных
    - avg_movement: среднее движение в правильном направлении (%)
    - max_movement: максимальное движение в правильном направлении (%)
    """
    try:
        apply_params_to_config(params)
        
        all_signals = []
        
        for symbol, df in DATA_CACHE.items():
            if df.empty or len(df) < 200:
                continue
            
            # ОПТИМИЗАЦИЯ: Анализируем данные ОДИН РАЗ!
            df_analyzed = analyze(df)
            
            if df_analyzed is None or df_analyzed.empty:
                continue
            
            # Проходим по проанализированному DF
            last_signal_time = None
            
            for i in range(100, len(df_analyzed) - 20, 20):  # Каждая 20-я свеча
                # Берём окно с индикаторами
                current_df = df_analyzed.iloc[:i+1].copy()
                
                # Используем check_signals с параметрами
                signals = check_signals_with_params(current_df, symbol, params)
                
                # Если сигнала нет - пропускаем
                if not signals or len(signals) == 0:
                    continue
                
                # Определяем тип из текста сигнала
                signal_text = signals[0] if isinstance(signals[0], str) else signals[0].get('text', '')
                if 'LONG' in signal_text or '🟢' in signal_text:
                    signal_type = 'BUY'
                elif 'SHORT' in signal_text or '🔴' in signal_text:
                    signal_type = 'SELL'
                else:
                    continue
                
                current_time = current_df.iloc[-1]['timestamp']
                current_price = current_df.iloc[-1]['close']
                
                # Смотрим на следующие 20 свечей (5 часов)
                future_df = df_analyzed.iloc[i+1:i+21]
                
                if len(future_df) < 5:
                    continue
                
                future_highs = future_df['high'].values
                future_lows = future_df['low'].values
                
                # Для BUY: ищем максимум
                if signal_type == 'BUY':
                    max_price = max(future_highs)
                    movement_pct = ((max_price - current_price) / current_price) * 100
                    
                    # Правильный если цена выросла хотя бы на 0.3%
                    is_correct = movement_pct > 0.3
                    
                # Для SELL: ищем минимум  
                else:  # SELL
                    min_price = min(future_lows)
                    movement_pct = ((current_price - min_price) / current_price) * 100
                    
                    # Правильный если цена упала хотя бы на 0.3%
                    is_correct = movement_pct > 0.3
                
                all_signals.append({
                    'symbol': symbol,
                    'type': signal_type,
                    'movement': movement_pct,
                    'correct': is_correct,
                    'score': last.get('composite_score', 0)
                })
        
        if not all_signals:
            return None
        
        total = len(all_signals)
        correct = sum(1 for s in all_signals if s['correct'])
        accuracy = correct / total if total > 0 else 0
        
        # Среднее движение для ПРАВИЛЬНЫХ сигналов
        correct_movements = [s['movement'] for s in all_signals if s['correct']]
        avg_movement = np.mean(correct_movements) if correct_movements else 0
        max_movement = max(correct_movements) if correct_movements else 0
        
        return {
            'total_signals': total,
            'correct_signals': correct,
            'accuracy': accuracy,
            'avg_movement': avg_movement,
            'max_movement': max_movement,
            'signals': all_signals
        }
        
    except Exception as e:
        logger.error(f"Ошибка в test_signal_quality: {e}")
        return None

# ========================================
# ЭТАП 2: ТЕСТИРОВАНИЕ С TP/SL
# ========================================
def test_with_tp_sl(params: Dict) -> Optional[Dict]:
    """
    Тестируем торговлю с TP/SL используя фильтры из Stage 1
    
    Возвращает:
    - total_trades: количество сделок
    - wins: выигрышные сделки
    - losses: проигрышные сделки
    - win_rate: винрейт
    - total_pnl: общий P&L (%)
    - avg_win: средний выигрыш (%)
    - avg_loss: средний проигрыш (%)
    """
    try:
        apply_params_to_config(params)
        
        all_trades = []
        
        for symbol, df in DATA_CACHE.items():
            if df.empty or len(df) < 200:
                continue
            
            # ОПТИМИЗАЦИЯ: Анализируем данные ОДИН РАЗ!
            df_analyzed = analyze(df)
            
            if df_analyzed is None or df_analyzed.empty:
                continue
            
            # Проходим по проанализированному DF
            for i in range(100, len(df_analyzed) - 50, 20):  # Каждая 20-я свеча
                # Берём окно с индикаторами
                current_df = df_analyzed.iloc[:i+1].copy()
                
                # Используем check_signals с параметрами
                signals = check_signals_with_params(current_df, symbol, params)
                
                # Если сигнала нет - пропускаем
                if not signals or len(signals) == 0:
                    continue
                
                # Определяем тип из текста сигнала
                signal_text = signals[0] if isinstance(signals[0], str) else signals[0].get('text', '')
                if 'LONG' in signal_text or '🟢' in signal_text:
                    signal_type = 'BUY'
                elif 'SHORT' in signal_text or '🔴' in signal_text:
                    signal_type = 'SELL'
                else:
                    continue
                
                entry_price = current_df.iloc[-1]['close']
                atr = current_df.iloc[-1].get('atr', entry_price * 0.02)
                
                # Расчет TP/SL
                tp_distance = max(atr * params.get('TP_ATR_MULT', 2.0), entry_price * params.get('TP_MIN', 0.015))
                sl_distance = max(atr * params.get('SL_ATR_MULT', 1.0), entry_price * params.get('SL_MIN', 0.01))
                
                if signal_type == 'BUY':
                    tp_price = entry_price + tp_distance
                    sl_price = entry_price - sl_distance
                else:  # SELL
                    tp_price = entry_price - tp_distance
                    sl_price = entry_price + sl_distance
                
                # Проверяем следующие свечи
                future_df = df_analyzed.iloc[i+1:i+51]
                
                exit_reason = 'TIMEOUT'
                exit_price = future_df.iloc[-1]['close'] if len(future_df) > 0 else entry_price
                
                for j, row in future_df.iterrows():
                    if signal_type == 'BUY':
                        if row['high'] >= tp_price:
                            exit_price = tp_price
                            exit_reason = 'TP'
                            break
                        elif row['low'] <= sl_price:
                            exit_price = sl_price
                            exit_reason = 'SL'
                            break
                    else:  # SELL
                        if row['low'] <= tp_price:
                            exit_price = tp_price
                            exit_reason = 'TP'
                            break
                        elif row['high'] >= sl_price:
                            exit_price = sl_price
                            exit_reason = 'SL'
                            break
                
                # Расчет P&L
                if signal_type == 'BUY':
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:  # SELL
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                
                # Минус комиссия
                pnl_pct -= 0.12  # 0.06% вход + 0.06% выход
                
                all_trades.append({
                    'symbol': symbol,
                    'type': signal_type,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct
                })
        
        if not all_trades:
            return None
        
        total = len(all_trades)
        wins = [t for t in all_trades if t['pnl_pct'] > 0]
        losses = [t for t in all_trades if t['pnl_pct'] <= 0]
        
        win_rate = len(wins) / total if total > 0 else 0
        total_pnl = sum(t['pnl_pct'] for t in all_trades)
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        
        return {
            'total_trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': all_trades
        }
        
    except Exception as e:
        logger.error(f"Ошибка в test_with_tp_sl: {e}")
        return None

# ========================================
# STAGE 1: ОПТИМИЗАЦИЯ ФИЛЬТРОВ
# ========================================
def stage1_objective(trial: optuna.Trial) -> float:
    """
    ЭТАП 1: Оптимизация фильтров для качества сигналов
    
    Цель: Найти параметры где сигналы:
    1. Идут в ПРАВИЛЬНОМ направлении (accuracy)
    2. Двигаются СИЛЬНО в нужную сторону (movement)
    3. Генерируются в ДОСТАТОЧНОМ количестве
    """
    try:
        params = {
            # Базовые параметры из config
            'TP_ATR_MULT': TP_ATR_MULT,
            'SL_ATR_MULT': SL_ATR_MULT,
            'TP_MIN': TP_MIN,
            'SL_MIN': SL_MIN,
        }
        
        # Оптимизируем ФИЛЬТРЫ (только нужные!)
        params['MIN_COMPOSITE_SCORE'] = trial.suggest_float('MIN_COMPOSITE_SCORE', 0.1, 0.6, step=0.05)
        params['MIN_ADX'] = trial.suggest_int('MIN_ADX', 10, 25)
        params['RSI_MIN'] = trial.suggest_int('RSI_MIN', 10, 35)
        params['RSI_MAX'] = trial.suggest_int('RSI_MAX', 50, 80)
        params['SIGNAL_COOLDOWN_MINUTES'] = trial.suggest_int('SIGNAL_COOLDOWN_MINUTES', 30, 120)
        
        # EMA периоды
        params['MA_FAST'] = trial.suggest_int('MA_FAST', 5, 15)
        params['MA_SLOW'] = trial.suggest_int('MA_SLOW', 15, 40)
        
        # Веса индикаторов
        params['WEIGHT_RSI'] = trial.suggest_float('WEIGHT_RSI', 0.5, 10.0)
        params['WEIGHT_MACD'] = trial.suggest_float('WEIGHT_MACD', 1.0, 15.0)
        params['WEIGHT_ADX'] = trial.suggest_float('WEIGHT_ADX', 1.0, 15.0)
        
        # Модификаторы сигналов
        params['SHORT_BOOST_MULTIPLIER'] = trial.suggest_float('SHORT_BOOST_MULTIPLIER', 1.0, 5.0)
        
        # Тестируем качество сигналов
        result = test_signal_quality(params)
        
        if result is None or result['total_signals'] < 10:
            return 0.0
        
        accuracy = result['accuracy']
        total_signals = result['total_signals']
        avg_movement = result['avg_movement']
        
        # Минимальные требования
        if accuracy < 0.55:  # Минимум 55% правильных
            return 0.0
        
        if avg_movement < 0.5:  # Минимум 0.5% среднее движение
            return 0.0
        
        # SCORING: ГЛАВНОЕ - ТОЧНОСТЬ И ДВИЖЕНИЕ!
        import math
        
        # 1️⃣ ACCURACY: ГЛАВНОЕ! Каждый 1% = +10 баллов (550-1000)
        accuracy_score = accuracy * 1000
        
        # 2️⃣ ДВИЖЕНИЕ: ВАЖНОЕ! Каждый 0.1% = +20 баллов (100-500)
        movement_score = avg_movement * 200
        
        # 3️⃣ КОЛИЧЕСТВО: небольшой бонус (√signals = +3-32)
        # 10→3, 50→7, 100→10, 329→18, 500→22, 687→26, 1000→32
        quantity_bonus = math.sqrt(total_signals)
        
        # ИТОГО: приоритет на качество!
        score = accuracy_score + movement_score + quantity_bonus
        
        # Логирование каждого trial с параметрами
        if trial.number % 10 == 0 or score > 1000:
            logger.info(f"Trial #{trial.number}: Score={score:.0f} | Acc={accuracy:.1%} | Move={avg_movement:.2f}% | Sig={total_signals}")
            logger.info(f"  → MinScore={params['MIN_COMPOSITE_SCORE']:.2f}, ADX={params['MIN_ADX']}, RSI=[{params['RSI_MIN']}-{params['RSI_MAX']}], Cooldown={params['SIGNAL_COOLDOWN_MINUTES']}m")
        
        return score
        
    except Exception as e:
        logger.error(f"Ошибка в stage1_objective: {e}")
        return 0.0

# ========================================
# STAGE 2: ОПТИМИЗАЦИЯ TP/SL
# ========================================
def stage2_max_profit(trial: optuna.Trial) -> float:
    """ЭТАП 2 (режим 1): Максимальная прибыль"""
    try:
        params = BEST_STAGE1_PARAMS.copy()
        
        # Оптимизируем только TP/SL
        params['TP_ATR_MULT'] = trial.suggest_float('TP_ATR_MULT', 1.5, 4.0, step=0.1)
        params['SL_ATR_MULT'] = trial.suggest_float('SL_ATR_MULT', 0.8, 2.5, step=0.1)
        params['TP_MIN'] = trial.suggest_float('TP_MIN', 0.010, 0.030, step=0.001)
        params['SL_MIN'] = trial.suggest_float('SL_MIN', 0.008, 0.025, step=0.001)
        
        result = test_with_tp_sl(params)
        
        if result is None or result['total_trades'] < 10:
            return 0.0
        
        # Минимальные требования
        if result['win_rate'] < 0.40:
            return 0.0
        
        score = result['total_pnl']
        
        # Логирование каждого trial с параметрами
        if trial.number % 10 == 0 or score > 10:
            logger.info(f"Trial #{trial.number} [MaxProfit]: P&L={score:.1f}% | WR={result['win_rate']:.1%} | Trades={result['total_trades']}")
            logger.info(f"  → TP: {params['TP_ATR_MULT']:.1f}×ATR (min {params['TP_MIN']*100:.1f}%), SL: {params['SL_ATR_MULT']:.1f}×ATR (min {params['SL_MIN']*100:.1f}%)")
        
        return score
        
    except Exception as e:
        logger.error(f"Ошибка в stage2_max_profit: {e}")
        return 0.0

def stage2_max_winrate(trial: optuna.Trial) -> float:
    """ЭТАП 2 (режим 2): Максимальный винрейт"""
    try:
        params = BEST_STAGE1_PARAMS.copy()
        
        params['TP_ATR_MULT'] = trial.suggest_float('TP_ATR_MULT', 1.0, 2.5, step=0.1)
        params['SL_ATR_MULT'] = trial.suggest_float('SL_ATR_MULT', 1.5, 3.5, step=0.1)
        params['TP_MIN'] = trial.suggest_float('TP_MIN', 0.008, 0.020, step=0.001)
        params['SL_MIN'] = trial.suggest_float('SL_MIN', 0.015, 0.035, step=0.001)
        
        result = test_with_tp_sl(params)
        
        if result is None or result['total_trades'] < 10:
            return 0.0
        
        if result['total_pnl'] < 0:
            return 0.0
        
        # Score: винрейт важнее, но прибыль тоже учитываем
        winrate_score = result['win_rate'] * 100
        profit_bonus = result['total_pnl'] * 0.5
        score = winrate_score + profit_bonus
        
        # Логирование каждого trial с параметрами
        if trial.number % 10 == 0 or score > 40:
            logger.info(f"Trial #{trial.number} [MaxWinrate]: Score={score:.1f} | WR={result['win_rate']:.1%} | P&L={result['total_pnl']:.1f}%")
            logger.info(f"  → TP: {params['TP_ATR_MULT']:.1f}×ATR (min {params['TP_MIN']*100:.1f}%), SL: {params['SL_ATR_MULT']:.1f}×ATR (min {params['SL_MIN']*100:.1f}%)")
        
        return score
        
    except Exception as e:
        logger.error(f"Ошибка в stage2_max_winrate: {e}")
        return 0.0

def stage2_balanced(trial: optuna.Trial) -> float:
    """ЭТАП 2 (режим 3): Сбалансированный"""
    try:
        params = BEST_STAGE1_PARAMS.copy()
        
        params['TP_ATR_MULT'] = trial.suggest_float('TP_ATR_MULT', 1.2, 3.0, step=0.1)
        params['SL_ATR_MULT'] = trial.suggest_float('SL_ATR_MULT', 1.0, 3.0, step=0.1)
        params['TP_MIN'] = trial.suggest_float('TP_MIN', 0.010, 0.025, step=0.001)
        params['SL_MIN'] = trial.suggest_float('SL_MIN', 0.010, 0.030, step=0.001)
        
        result = test_with_tp_sl(params)
        
        if result is None or result['total_trades'] < 10:
            return 0.0
        
        if result['win_rate'] < 0.45 or result['total_pnl'] < 5:
            return 0.0
        
        # Score: равный баланс
        winrate_score = result['win_rate'] * 50
        profit_score = result['total_pnl'] * 0.5
        score = winrate_score + profit_score
        
        # Логирование каждого trial с параметрами
        if trial.number % 10 == 0 or score > 30:
            logger.info(f"Trial #{trial.number} [Balanced]: Score={score:.1f} | WR={result['win_rate']:.1%} | P&L={result['total_pnl']:.1f}%")
            logger.info(f"  → TP: {params['TP_ATR_MULT']:.1f}×ATR (min {params['TP_MIN']*100:.1f}%), SL: {params['SL_ATR_MULT']:.1f}×ATR (min {params['SL_MIN']*100:.1f}%)")
        
        return score
        
    except Exception as e:
        logger.error(f"Ошибка в stage2_balanced: {e}")
        return 0.0

# ========================================
# ГЛАВНАЯ ФУНКЦИЯ
# ========================================
async def run_smart_optimization():
    """Запуск умной 2-этапной оптимизации"""
    global BEST_STAGE1_PARAMS
    
    logger.info("\n" + "="*60)
    logger.info("🚀 УМНЫЙ 2-ЭТАПНЫЙ ОПТИМИЗАТОР")
    logger.info("="*60)
    
    # Загружаем данные
    await load_all_data()
    
    if not DATA_CACHE:
        logger.error("❌ Нет данных для оптимизации!")
        return
    
    # ========================================
    # ЭТАП 1: ОПТИМИЗАЦИЯ ФИЛЬТРОВ
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("🎯 ЭТАП 1: ПОИСК ЛУЧШИХ ФИЛЬТРОВ")
    logger.info("="*60)
    logger.info("Цель: Сигналы идут в правильном направлении + сильное движение")
    logger.info("")
    
    study1 = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study1.optimize(stage1_objective, n_trials=1000, show_progress_bar=True)
    
    best_filters = study1.best_params
    best_score = study1.best_value
    
    logger.info("\n" + "-"*60)
    logger.info(f"🏆 ЛУЧШИЕ ФИЛЬТРЫ: Score={best_score:.1f}")
    logger.info("-"*60)
    
    # Сохраняем лучшие параметры
    BEST_STAGE1_PARAMS = {
        'TP_ATR_MULT': TP_ATR_MULT,
        'SL_ATR_MULT': SL_ATR_MULT,
        'TP_MIN': TP_MIN,
        'SL_MIN': SL_MIN,
    }
    BEST_STAGE1_PARAMS.update(best_filters)
    
    for key, value in best_filters.items():
        logger.info(f"  {key}: {value}")
    
    # Тестируем лучшие фильтры
    logger.info("\n📊 Тестирование лучших фильтров:")
    test_result = test_signal_quality(BEST_STAGE1_PARAMS)
    if test_result:
        logger.info(f"  Сигналов: {test_result['total_signals']}")
        logger.info(f"  Правильных: {test_result['correct_signals']} ({test_result['accuracy']:.1%})")
        logger.info(f"  Среднее движение: {test_result['avg_movement']:.2f}%")
        logger.info(f"  Макс движение: {test_result['max_movement']:.2f}%")
    
    # ========================================
    # ЭТАП 2: ОПТИМИЗАЦИЯ TP/SL (3 РЕЖИМА)
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("💰 ЭТАП 2: ОПТИМИЗАЦИЯ TP/SL")
    logger.info("="*60)
    
    results = {}
    
    # Режим 1: Максимальная прибыль
    logger.info("\n🎯 РЕЖИМ 1: МАКСИМАЛЬНАЯ ПРИБЫЛЬ")
    logger.info("-"*60)
    
    study_profit = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_profit.optimize(stage2_max_profit, n_trials=300, show_progress_bar=True)
    
    params_profit = BEST_STAGE1_PARAMS.copy()
    params_profit.update(study_profit.best_params)
    results['max_profit'] = {
        'params': study_profit.best_params,
        'test': test_with_tp_sl(params_profit)
    }
    
    logger.info(f"\n💰 Лучший результат: P&L={study_profit.best_value:.1f}%")
    for k, v in study_profit.best_params.items():
        logger.info(f"  {k}: {v}")
    
    # Режим 2: Максимальный винрейт
    logger.info("\n🎯 РЕЖИМ 2: МАКСИМАЛЬНЫЙ ВИНРЕЙТ")
    logger.info("-"*60)
    
    study_winrate = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=43))
    study_winrate.optimize(stage2_max_winrate, n_trials=300, show_progress_bar=True)
    
    params_winrate = BEST_STAGE1_PARAMS.copy()
    params_winrate.update(study_winrate.best_params)
    results['max_winrate'] = {
        'params': study_winrate.best_params,
        'test': test_with_tp_sl(params_winrate)
    }
    
    logger.info(f"\n🎯 Лучший результат: Score={study_winrate.best_value:.1f}")
    for k, v in study_winrate.best_params.items():
        logger.info(f"  {k}: {v}")
    
    # Режим 3: Сбалансированный
    logger.info("\n🎯 РЕЖИМ 3: СБАЛАНСИРОВАННЫЙ")
    logger.info("-"*60)
    
    study_balanced = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=44))
    study_balanced.optimize(stage2_balanced, n_trials=300, show_progress_bar=True)
    
    params_balanced = BEST_STAGE1_PARAMS.copy()
    params_balanced.update(study_balanced.best_params)
    results['balanced'] = {
        'params': study_balanced.best_params,
        'test': test_with_tp_sl(params_balanced)
    }
    
    logger.info(f"\n⚖️ Лучший результат: Score={study_balanced.best_value:.1f}")
    for k, v in study_balanced.best_params.items():
        logger.info(f"  {k}: {v}")
    
    # ========================================
    # ФИНАЛЬНОЕ СРАВНЕНИЕ
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    logger.info("="*60)
    
    for mode_name, mode_data in results.items():
        test = mode_data['test']
        if test:
            logger.info(f"\n📊 {mode_name.upper().replace('_', ' ')}:")
            logger.info(f"  P&L: {test['total_pnl']:.2f}%")
            logger.info(f"  Винрейт: {test['win_rate']:.1%}")
            logger.info(f"  Сделок: {test['total_trades']}")
            logger.info(f"  Средний WIN: {test['avg_win']:.2f}%")
            logger.info(f"  Средний LOSS: {test['avg_loss']:.2f}%")
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': timestamp,
        'stage1_filters': BEST_STAGE1_PARAMS,
        'stage1_score': best_score,
        'stage2_results': {
            mode: {
                'params': data['params'],
                'performance': {
                    'total_pnl': data['test']['total_pnl'],
                    'win_rate': data['test']['win_rate'],
                    'total_trades': data['test']['total_trades'],
                    'avg_win': data['test']['avg_win'],
                    'avg_loss': data['test']['avg_loss']
                } if data['test'] else None
            }
            for mode, data in results.items()
        }
    }
    
    filename = f'smart_optimization_{timestamp}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 Результаты сохранены: {filename}")
    logger.info("\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")

# ========================================
# ЗАПУСК
# ========================================
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_smart_optimization())

