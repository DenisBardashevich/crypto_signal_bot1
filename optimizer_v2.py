"""
ДВУХЭТАПНЫЙ ОПТИМИЗАТОР V2
Дата создания: 01.10.2025

ЭТАП 1: Находит параметры фильтров для надежных сигналов
        - Оптимизирует: RSI_MIN, RSI_MAX, MIN_ADX, веса, окна индикаторов
        - Метрика: точность направления × количество сигналов
        - Цель: сигналы после которых цена идет в нужную сторону

ЭТАП 2: Подбирает оптимальные TP/SL
        - Оптимизирует: TP_ATR_MULT, SL_ATR_MULT, TP_MIN, SL_MIN
        - Метрика: общий P&L
        - Цель: максимальная прибыль
"""

import ccxt
import pandas as pd
import ta
import optuna
import json
from datetime import datetime, timezone
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ========== КОНФИГУРАЦИЯ ==========
EXCHANGE = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

# Символы для оптимизации
SYMBOLS = [
    'BNB/USDT:USDT',
    'LTC/USDT:USDT',
    'IMX/USDT:USDT',
    'SUI/USDT:USDT',
    'ORDI/USDT:USDT'
]

# Параметры загрузки данных
TIMEFRAME = '15m'
LIMIT = 1500  # Больше данных для более точной оптимизации (было 1000)
DATA_DIR = Path('optimization_data')
DATA_DIR.mkdir(exist_ok=True)

# ВАЖНО: По умолчанию всегда загружаются СВЕЖИЕ данные с биржи!
# Кэш используется только для ускорения повторных запусков с одинаковыми данными

# Параметры оптимизации
STAGE1_TRIALS = 800  # Количество попыток для этапа 1 (было 100)
STAGE2_TRIALS = 300  # Количество попыток для этапа 2 (было 50)

# Параметры анализа
LOOKAHEAD_CANDLES = 15  # Сколько свечей анализировать после сигнала (Этап 1) - УМЕНЬШЕНО с 50!
MAX_TRADE_DURATION = 100  # Максимальная длительность сделки в свечах (Этап 2)
WARMUP_CANDLES = 50  # Отступ от начала для прогрева индикаторов
RESERVE_CANDLES = 20  # Резерв свечей в конце для lookahead - уменьшен под новый lookahead

# ========== ЗАГРУЗКА ДАННЫХ ==========
def load_data(symbol, force_reload=False):
    """Загрузить или обновить данные с биржи"""
    filename = DATA_DIR / f"{symbol.replace('/', '_').replace(':', '_')}.json"
    
    if filename.exists() and not force_reload:
        logging.info(f"📂 Загрузка данных {symbol} из кэша")
        with open(filename, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    logging.info(f"📥 Загрузка данных {symbol} с биржи...")
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Сохраняем в кэш
        df.to_json(filename, orient='records', date_format='iso')
        logging.info(f"✅ Загружено {len(df)} свечей для {symbol}")
        return df
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки {symbol}: {e}")
        return pd.DataFrame()

def load_all_data(force_reload=True):
    """Загрузить данные по всем символам (по умолчанию всегда свежие)"""
    data = {}
    for symbol in SYMBOLS:
        df = load_data(symbol, force_reload=force_reload)
        if not df.empty:
            data[symbol] = df
    return data

# ========== РАСЧЕТ ИНДИКАТОРОВ ==========
def calculate_indicators(df, params):
    """Рассчитать индикаторы с заданными параметрами"""
    try:
        df = df.copy()
        
        # EMA
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=params['ma_fast'])
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=params['ma_slow'])
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=params['rsi_window'])
        
        # MACD
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=params['macd_slow'],
            window_fast=params['macd_fast'],
            window_sign=params['macd_signal']
        )
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # ADX
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=params['adx_window'])
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=params['atr_window'])
        
        df = df.dropna().reset_index(drop=True)
        return df
        
    except Exception as e:
        logging.error(f"Ошибка расчета индикаторов: {e}")
        return pd.DataFrame()

# ========== ГЕНЕРАЦИЯ СИГНАЛОВ ==========
def check_signal(df, idx, params):
    """
    Проверить сигнал на индексе idx
    Возвращает: 'LONG', 'SHORT' или None
    
    СМЯГЧЕННАЯ ЛОГИКА: достаточно 3 из 4 условий (ADX обязателен)
    """
    if idx < 1 or idx >= len(df):
        return None
    
    row = df.iloc[idx]
    
    # Базовая проверка ADX (обязательное условие)
    if row['adx'] < params['min_adx']:
        return None
    
    # RSI должен быть ОБЯЗАТЕЛЬНЫМ! Проверяем отдельно
    rsi_long = row['rsi'] <= params['rsi_min']
    rsi_short = row['rsi'] >= params['rsi_max']
    
    # Дополнительные условия (нужно минимум 1 из 2)
    ema_bullish = row['ema_fast'] > row['ema_slow']
    ema_bearish = row['ema_fast'] < row['ema_slow']
    macd_bullish = row['macd_line'] > row['macd_signal']
    macd_bearish = row['macd_line'] < row['macd_signal']
    
    # ЛОГИКА: RSI обязателен + минимум 1 подтверждение (EMA или MACD)
    if rsi_long and (ema_bullish or macd_bullish):
        return 'LONG'
    elif rsi_short and (ema_bearish or macd_bearish):
        return 'SHORT'
    
    return None

def calculate_signal_strength(df, idx, signal_type, params):
    """Рассчитать силу сигнала"""
    row = df.iloc[idx]
    score = 0
    
    # RSI компонент
    if signal_type == 'LONG':
        rsi_normalized = 1 - (row['rsi'] / 100)
    else:
        rsi_normalized = row['rsi'] / 100
    
    score += rsi_normalized * params['weight_rsi']
    
    # MACD компонент
    macd_diff = abs(row['macd_line'] - row['macd_signal'])
    score += macd_diff * params['weight_macd']
    
    # ADX компонент
    adx_normalized = row['adx'] / 100
    score += adx_normalized * params['weight_adx']
    
    return score

# ========== ЭТАП 1: ОПТИМИЗАЦИЯ ФИЛЬТРОВ ==========
def check_direction_correctness(df, signal_idx, signal_type, lookahead=None):
    """
    УПРОЩЕННАЯ проверка: просто считаем % движения в правильную сторону
    
    Возвращает:
    - correctness: 0-1, процент времени когда цена двигается правильно
    - max_favorable: максимальное благоприятное движение %
    """
    if lookahead is None:
        lookahead = LOOKAHEAD_CANDLES
    
    if signal_idx + lookahead >= len(df):
        return 0, 0
    
    entry_price = df.iloc[signal_idx]['close']
    future_prices = df.iloc[signal_idx+1:signal_idx+1+lookahead]['close'].values
    
    if len(future_prices) == 0:
        return 0, 0
    
    if signal_type == 'LONG':
        # Для LONG: ищем максимум и процент времени выше входа
        max_price = max(future_prices)
        max_favorable = ((max_price - entry_price) / entry_price) * 100
        
        # Процент времени когда цена выше входа
        correctness = sum(p > entry_price for p in future_prices) / len(future_prices)
        
    else:  # SHORT
        # Для SHORT: ищем минимум и процент времени ниже входа
        min_price = min(future_prices)
        max_favorable = ((entry_price - min_price) / entry_price) * 100
        
        # Процент времени когда цена ниже входа
        correctness = sum(p < entry_price for p in future_prices) / len(future_prices)
    
    return correctness, max_favorable

def evaluate_stage1(params, data):
    """
    ЭТАП 1: Оценка качества фильтров
    
    Метрика: точность направления × количество сигналов × средняя амплитуда
    """
    total_signals = 0
    correct_directions = 0
    total_amplitude = 0
    
    # Счетчики для диагностики
    total_candles_checked = 0
    symbols_processed = 0
    
    for symbol, df in data.items():
        # Рассчитываем индикаторы
        df_calc = calculate_indicators(df, params)
        if df_calc.empty:
            logging.warning(f"⚠️ {symbol}: пустой датафрейм после расчета индикаторов")
            continue
        
        symbols_processed += 1
        last_signal_idx = -999999  # Последний сигнал для cooldown
        cooldown_candles = int(params['signal_cooldown'] / 15)  # Cooldown в свечах (15m)
        
        # Ищем сигналы
        for idx in range(WARMUP_CANDLES, len(df_calc) - RESERVE_CANDLES):
            total_candles_checked += 1
            
            # Проверяем cooldown
            if idx - last_signal_idx < cooldown_candles:
                continue  # Слишком рано для нового сигнала
            
            signal = check_signal(df_calc, idx, params)
            
            if signal:
                # Проверяем правильность направления
                correctness, amplitude = check_direction_correctness(df_calc, idx, signal)
                
                total_signals += 1
                correct_directions += correctness  # correctness от 0 до 1
                total_amplitude += amplitude
                last_signal_idx = idx  # Обновляем время последнего сигнала
    
    if total_signals == 0:
        # Диагностика
        if symbols_processed == 0:
            logging.warning(f"❌ Не обработан ни один символ!")
        elif total_candles_checked == 0:
            logging.warning(f"❌ Не проверено ни одной свечи!")
        else:
            logging.warning(f"⚠️ Проверено {total_candles_checked} свечей, но сигналов нет. RSI_MIN={params['rsi_min']}, RSI_MAX={params['rsi_max']}, MIN_ADX={params['min_adx']}")
        return 0  # Нет сигналов - плохой результат
    
    # Средняя точность направления
    avg_correctness = correct_directions / total_signals
    
    # Средняя амплитуда движения
    avg_amplitude = total_amplitude / total_signals
    
    # ФИЛЬТР: Отсекаем слабые сигналы (лучше случайности)
    if avg_correctness < 0.52:  # Меньше 52% точности - отклоняем
        logging.warning(f"⚠️ Низкая точность: {avg_correctness:.2%} < 52% - отклонено")
        return 0
    
    # НОРМАЛИЗАЦИЯ АМПЛИТУДЫ: оптимум 1-2%, больше не нужно
    if avg_amplitude < 1.0:
        # Меньше 1% = слабое движение, штраф
        amplitude_factor = avg_amplitude * 0.5  # Снижаем сильно
    elif avg_amplitude <= 2.0:
        # 1-2% = ИДЕАЛЬНО! Линейный рост награды
        amplitude_factor = 1.0 + (avg_amplitude - 1.0)  # От 1.0 до 2.0
    else:
        # Больше 2% = не даем дополнительных бонусов (риск высокий)
        amplitude_factor = 2.0  # Ограничиваем 2.0
    
    # УЛУЧШЕННАЯ метрика: точность^2 × log(сигналов) × нормализованная_амплитуда
    import math
    accuracy_factor = avg_correctness ** 2
    signal_factor = math.log(total_signals + 1)
    
    score = accuracy_factor * signal_factor * amplitude_factor
    
    logging.info(f"📊 Сигналов: {total_signals} | Точность: {avg_correctness:.2%} | Амплитуда: {avg_amplitude:.2f}% | Фактор: {amplitude_factor:.2f} | Score: {score:.2f}")
    
    return score

def optimize_stage1(data):
    """ЭТАП 1: Оптимизация параметров фильтров"""
    logging.info("=" * 60)
    logging.info("🎯 ЭТАП 1: Оптимизация фильтров для надежных сигналов")
    logging.info("=" * 60)
    
    def objective(trial):
        # ОПТИМИЗАЦИЯ ТОЛЬКО КЛЮЧЕВЫХ параметров
        # Остальные - СТАНДАРТНЫЕ проверенные значения
        
        # === СТАНДАРТНЫЕ значения окон (проверенные временем) ===
        # RSI: классика = 14
        # MACD: классика = 12, 26, 9
        # ADX: классика = 14
        # ATR: классика = 14
        
        # === ОПТИМИЗИРУЕМ ТОЛЬКО КРИТИЧНЫЕ ===
        
        # 1. EMA: оптимизируем (сильно влияют на тренд)
        ma_slow = trial.suggest_int('ma_slow', 20, 50)
        ma_fast = trial.suggest_int('ma_fast', 8, ma_slow - 1)
        
        # 2. RSI фильтры: ОПТИМИЗИРУЕМ (ключевые для входа)
        rsi_min = trial.suggest_int('rsi_min', 10, 25)
        rsi_max = trial.suggest_int('rsi_max', rsi_min + 25, 80)
        
        params = {
            # === ОПТИМИЗИРУЕМЫЕ параметры ===
            
            # EMA (влияют на определение тренда)
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            
            # Фильтры входа (КЛЮЧЕВЫЕ!)
            'rsi_min': rsi_min,
            'rsi_max': rsi_max,
            'min_adx': trial.suggest_int('min_adx', 15, 30),
            
            # Веса индикаторов (определяют важность каждого)
            'weight_rsi': trial.suggest_float('weight_rsi', 0, 6.0, step=0.3),
            'weight_macd': trial.suggest_float('weight_macd', 0, 6.0, step=0.3),
            'weight_adx': trial.suggest_float('weight_adx', 0, 6.0, step=0.3),
            
            # Cooldown (частота сигналов)
            'signal_cooldown': trial.suggest_int('signal_cooldown', 30, 90, step=15),
            
            # === ВЫБОР ИЗ ПОПУЛЯРНЫХ значений (проверенные варианты) ===
            
            # RSI: 3 популярных окна
            'rsi_window': trial.suggest_categorical('rsi_window', [9, 14, 21]),
            # 9 = быстрый, 14 = классика, 21 = медленный
            
            # ADX: 3 популярных окна
            'adx_window': trial.suggest_categorical('adx_window', [10, 14, 20]),
            # 10 = быстрый, 14 = классика, 20 = медленный
            
            # ATR: 3 популярных окна
            'atr_window': trial.suggest_categorical('atr_window', [10, 14, 20]),
            # 10 = быстрый, 14 = классика, 20 = медленный
        }
        
        # MACD: 3 популярных пресета (добавляем после создания словаря)
        macd_preset = trial.suggest_categorical('macd_preset', ['classic', 'fast', 'slow'])
        if macd_preset == 'classic':
            params['macd_fast'] = 12
            params['macd_slow'] = 26
            params['macd_signal'] = 9
        elif macd_preset == 'fast':
            params['macd_fast'] = 8
            params['macd_slow'] = 17
            params['macd_signal'] = 9
        else:  # slow
            params['macd_fast'] = 5
            params['macd_slow'] = 35
            params['macd_signal'] = 5
        
        score = evaluate_stage1(params, data)
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=STAGE1_TRIALS, show_progress_bar=True)
    
    logging.info("=" * 60)
    logging.info(f"✅ ЭТАП 1 завершен. Лучший score: {study.best_value:.2f}")
    logging.info(f"🏆 Лучшие параметры: {study.best_params}")
    
    # Воссоздаем полные параметры с MACD
    best_full_params = dict(study.best_params)
    macd_preset = best_full_params.get('macd_preset', 'classic')
    if macd_preset == 'classic':
        best_full_params['macd_fast'] = 12
        best_full_params['macd_slow'] = 26
        best_full_params['macd_signal'] = 9
    elif macd_preset == 'fast':
        best_full_params['macd_fast'] = 8
        best_full_params['macd_slow'] = 17
        best_full_params['macd_signal'] = 9
    else:  # slow
        best_full_params['macd_fast'] = 5
        best_full_params['macd_slow'] = 35
        best_full_params['macd_signal'] = 5
    
    return best_full_params, study.best_value

# ========== ЭТАП 2: ОПТИМИЗАЦИЯ TP/SL ==========
def calculate_tp_sl(price, atr, signal_type, params):
    """Рассчитать TP/SL"""
    if signal_type == 'LONG':
        tp_price = price + (atr * params['tp_atr_mult'])
        sl_price = price - (atr * params['sl_atr_mult'])
        
        tp_pct = (tp_price - price) / price
        sl_pct = (price - sl_price) / price
        
        if tp_pct < params['tp_min']:
            tp_price = price * (1 + params['tp_min'])
        if sl_pct < params['sl_min']:
            sl_price = price * (1 - params['sl_min'])
    else:  # SHORT
        tp_price = price - (atr * params['tp_atr_mult'])
        sl_price = price + (atr * params['sl_atr_mult'])
        
        tp_pct = (price - tp_price) / price
        sl_pct = (sl_price - price) / price
        
        if tp_pct < params['tp_min']:
            tp_price = price * (1 - params['tp_min'])
        if sl_pct < params['sl_min']:
            sl_price = price * (1 + params['sl_min'])
    
    return tp_price, sl_price

def backtest_with_tp_sl(df, signal_idx, signal_type, params):
    """
    Симуляция сделки с TP/SL
    
    Возвращает: P&L в %
    """
    entry_price = df.iloc[signal_idx]['close']
    atr = df.iloc[signal_idx]['atr']
    
    # Рассчитываем TP/SL
    tp_price, sl_price = calculate_tp_sl(entry_price, atr, signal_type, params)
    
    # Смотрим что произойдет дальше
    for idx in range(signal_idx + 1, min(signal_idx + MAX_TRADE_DURATION, len(df))):
        row = df.iloc[idx]
        
        if signal_type == 'LONG':
            # Проверяем TP
            if row['high'] >= tp_price:
                pnl = ((tp_price - entry_price) / entry_price) * 100
                return pnl
            # Проверяем SL
            if row['low'] <= sl_price:
                pnl = ((sl_price - entry_price) / entry_price) * 100
                return pnl
        else:  # SHORT
            # Проверяем TP
            if row['low'] <= tp_price:
                pnl = ((entry_price - tp_price) / entry_price) * 100
                return pnl
            # Проверяем SL
            if row['high'] >= sl_price:
                pnl = ((entry_price - sl_price) / entry_price) * 100
                return pnl
    
    # Если не закрылись, считаем текущий P&L
    current_price = df.iloc[-1]['close']
    if signal_type == 'LONG':
        pnl = ((current_price - entry_price) / entry_price) * 100
    else:
        pnl = ((entry_price - current_price) / entry_price) * 100
    
    return pnl

def evaluate_stage2(stage1_params, tp_sl_params, data):
    """
    ЭТАП 2: Оценка TP/SL параметров
    
    Метрика: общий P&L
    """
    # Объединяем параметры
    params = {**stage1_params, **tp_sl_params}
    
    total_pnl = 0
    total_trades = 0
    winning_trades = 0
    
    for symbol, df in data.items():
        df_calc = calculate_indicators(df, params)
        if df_calc.empty:
            continue
        
        last_signal_idx = -999999  # Последний сигнал для cooldown
        cooldown_candles = int(params['signal_cooldown'] / 15)  # Cooldown в свечах (15m)
        
        # Ищем сигналы и торгуем
        for idx in range(WARMUP_CANDLES, len(df_calc) - MAX_TRADE_DURATION):
            # Проверяем cooldown
            if idx - last_signal_idx < cooldown_candles:
                continue
            
            signal = check_signal(df_calc, idx, params)
            
            if signal:
                pnl = backtest_with_tp_sl(df_calc, idx, signal, params)
                total_pnl += pnl
                total_trades += 1
                if pnl > 0:
                    winning_trades += 1
                last_signal_idx = idx  # Обновляем время последнего сигнала
    
    if total_trades == 0:
        return 0
    
    winrate = (winning_trades / total_trades) * 100
    avg_pnl = total_pnl / total_trades
    
    logging.info(f"💰 Сделок: {total_trades} | WR: {winrate:.1f}% | Avg: {avg_pnl:+.2f}% | Total: {total_pnl:+.2f}%")
    
    return total_pnl

def optimize_stage2(stage1_params, data):
    """ЭТАП 2: Оптимизация TP/SL"""
    logging.info("=" * 60)
    logging.info("🎯 ЭТАП 2: Оптимизация TP/SL для максимальной прибыли")
    logging.info("=" * 60)
    
    def objective(trial):
        # ПРАВИЛЬНАЯ ГЕНЕРАЦИЯ: гарантируем TP > SL
        
        # 1. Сначала генерируем SL (меньшие значения)
        sl_atr_mult = trial.suggest_float('sl_atr_mult', 1.0, 4.0)
        sl_min = trial.suggest_float('sl_min', 0.01, 0.04)  # 1%-4%
        
        # 2. Потом генерируем TP (должны быть больше SL)
        tp_atr_mult = trial.suggest_float('tp_atr_mult', sl_atr_mult + 0.5, 10.0)  # Гарантированно > SL
        tp_min = trial.suggest_float('tp_min', sl_min * 1.5, 0.08)  # Гарантированно > SL × 1.5
        
        tp_sl_params = {
            'tp_atr_mult': tp_atr_mult,
            'sl_atr_mult': sl_atr_mult,
            'tp_min': tp_min,
            'sl_min': sl_min,
        }
        
        total_pnl = evaluate_stage2(stage1_params, tp_sl_params, data)
        return total_pnl
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=STAGE2_TRIALS, show_progress_bar=True)
    
    logging.info("=" * 60)
    logging.info(f"✅ ЭТАП 2 завершен. Лучший P&L: {study.best_value:.2f}%")
    logging.info(f"🏆 Лучшие параметры: {study.best_params}")
    
    return study.best_params, study.best_value

# ========== ГЛАВНАЯ ФУНКЦИЯ ==========
def main():
    """Запуск двухэтапной оптимизации"""
    logging.info("🚀 Запуск двухэтапного оптимизатора V2")
    
    # Загружаем данные
    logging.info("📥 Загрузка исторических данных...")
    data = load_all_data()
    
    if not data:
        logging.error("❌ Не удалось загрузить данные!")
        return
    
    logging.info(f"✅ Загружено {len(data)} символов")
    
    # ЭТАП 1: Оптимизация фильтров
    stage1_params, stage1_score = optimize_stage1(data)
    
    # ЭТАП 2: Оптимизация TP/SL
    stage2_params, stage2_pnl = optimize_stage2(stage1_params, data)
    
    # Объединяем результаты
    final_params = {**stage1_params, **stage2_params}
    
    # Сохраняем результаты
    results = {
        'timestamp': datetime.now().isoformat(),
        'stage1': {
            'params': stage1_params,
            'score': stage1_score
        },
        'stage2': {
            'params': stage2_params,
            'pnl': stage2_pnl
        },
        'final_params': final_params
    }
    
    filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("=" * 60)
    logging.info("🎉 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    logging.info("=" * 60)
    logging.info(f"📁 Результаты сохранены в: {filename}")
    logging.info("")
    logging.info("📊 ИТОГОВЫЕ ПАРАМЕТРЫ:")
    for key, value in final_params.items():
        logging.info(f"  {key}: {value}")
    logging.info("")
    logging.info(f"🎯 Этап 1 (качество сигналов): {stage1_score:.2f}")
    logging.info(f"💰 Этап 2 (прибыльность): {stage2_pnl:+.2f}%")

if __name__ == '__main__':
    main()

