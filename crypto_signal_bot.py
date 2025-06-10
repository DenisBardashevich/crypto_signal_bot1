import ccxt
import pandas as pd
import ta
import asyncio
from telegram import Bot
import os
import json
from datetime import datetime, timedelta, timezone
import time
import math
from telegram.ext import Application, CommandHandler, ContextTypes
import threading
import logging
from collections import defaultdict
from config import *
import numpy as np

# ========== НАСТРОЙКИ ==========
# Удаляю старые параметры, заменяю на импорт из config.py
# Было:
# TIMEFRAME = '5m'
# LIMIT = 400
# TAKE_PROFIT = 0.02
# STOP_LOSS = -0.02
# TELEGRAM_TOKEN = ...
# TELEGRAM_CHAT_ID = ...
# ...
# Теперь всё берётся из config.py
# ... existing code ...

EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'  # Используем фьючерсный рынок (USDT perpetual)
    }
})

# Белый список топ-50 популярных монет + перспективные альткойны и волатильные монеты (фьючерсы)
TOP_SYMBOLS = [
    # Топовые ликвидные (основа)
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'MATIC/USDT:USDT',
    'TRX/USDT:USDT', 'DOT/USDT:USDT', 'LTC/USDT:USDT',
    # Новые трендовые альты 2024–2025
    'JASMY/USDT:USDT', 'ARKM/USDT:USDT', 'STRK/USDT:USDT', 'ACE/USDT:USDT',
    'WLD/USDT:USDT', 'ORDI/USDT:USDT', 'ENA/USDT:USDT', 'TNSR/USDT:USDT',
    'NOT/USDT:USDT', 'MAVIA/USDT:USDT', 'ZRO/USDT:USDT', 'BB/USDT:USDT', 'OMNI/USDT:USDT',
    # Мемкоины и волатильные
    'PEPE/USDT:USDT', '1000PEPE/USDT:USDT', 'FLOKI/USDT:USDT', 'BONK/USDT:USDT', 'SHIB/USDT:USDT', 'WIF/USDT:USDT',
    # Перспективные альткойны
    'PYTH/USDT:USDT', 'JUP/USDT:USDT', 'TIA/USDT:USDT', 'SEI/USDT:USDT',
    # Ещё ликвидные и трендовые (добавляем до 50)
    'OP/USDT:USDT', 'ARB/USDT:USDT', 'FIL/USDT:USDT', 'APT/USDT:USDT', 'RNDR/USDT:USDT',
    'INJ/USDT:USDT', 'NEAR/USDT:USDT', 'SUI/USDT:USDT', 'STX/USDT:USDT', 'DYDX/USDT:USDT',
    'LDO/USDT:USDT', 'UNI/USDT:USDT', 'AAVE/USDT:USDT', 'MKR/USDT:USDT', 'ATOM/USDT:USDT',
]
markets = EXCHANGE.load_markets()
# Фильтруем только те пары, которые есть на фьючерсах (swap) и активны
SYMBOLS = [symbol for symbol in TOP_SYMBOLS if symbol in markets and markets[symbol]['active'] and markets[symbol]['type'] == 'swap']
print(f"FUTURES SYMBOLS: {SYMBOLS}")  # Для отладки

# ========== ВИРТУАЛЬНЫЙ ПОРТФЕЛЬ ========== 
PORTFOLIO_FILE = 'virtual_portfolio.json'

# Загрузка портфеля
if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r') as f:
        virtual_portfolio = json.load(f)
else:
    virtual_portfolio = {}

# Открытые сделки (символ: {'buy_price': ..., 'time': ...})
open_trades = {}
if 'open_trades' in virtual_portfolio:
    open_trades = virtual_portfolio['open_trades']
else:
    virtual_portfolio['open_trades'] = open_trades

# Сохраняем портфель
def save_portfolio():
    virtual_portfolio['open_trades'] = open_trades
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(virtual_portfolio, f)

# Фиксация сделки
def record_trade(symbol, action, price, time, side, score=None):
    """
    Записывает сделку в виртуальный портфель
    
    action: 'OPEN' или 'CLOSE'
    side: 'long' или 'short'
    """
    if symbol not in virtual_portfolio:
        virtual_portfolio[symbol] = []
    
    # Определяем действие для записи (BUY/SELL)
    trade_action = None
    if action == 'OPEN':
        trade_action = 'BUY' if side == 'long' else 'SELL'
    elif action == 'CLOSE':
        trade_action = 'SELL' if side == 'long' else 'BUY'
    
    # Создаем запись о сделке
    trade = {
        'action': trade_action,
        'side': side,
        'price': price,
        'time': time.strftime('%Y-%m-%d %H:%M')
    }
    
    # Добавляем оценку силы сигнала, если есть
    if score is not None:
        trade['score'] = score
    
    # Добавляем информацию о типе операции (открытие/закрытие)
    trade['operation'] = action
    
    # Добавляем сделку в портфель
    virtual_portfolio[symbol].append(trade)
    save_portfolio()
    
    # Логируем информацию о сделке
    logging.info(f"Записана сделка: {symbol} {action} {side} по цене {price} в {time}")

# Открытие сделки
def open_trade(symbol, price, time, side, atr=None, score=None, position_size=0.03):
    open_trades[symbol] = {
        'side': side,  # 'long' или 'short'
        'entry_price': price,
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'atr': atr if atr is not None else 0,
        'trail_pct': TRAIL_ATR_MULT,
        'last_peak': price,
        'score': score,
        'position_size': position_size
    }
    save_portfolio()

# Закрытие сделки
def close_trade(symbol):
    if symbol in open_trades:
        del open_trades[symbol]
        save_portfolio()

# Подсчёт прибыли
def calculate_profit():
    """
    Подсчёт прибыли по виртуальному портфелю с учётом:
    1. Комиссий биржи (FEE_RATE)
    2. Финансирования (funding)
    3. Рекомендованного плеча
    
    Возвращает:
    - отчёт о прибыли/убытках в строковом виде
    - количество прибыльных сделок
    - количество убыточных сделок
    - общую расчетную P&L в USDT
    """
    report = []
    total_profit = 0
    win, loss = 0, 0
    total_pnl_usdt = 0
    
    for symbol, trades in virtual_portfolio.items():
        if symbol == 'open_trades':
            continue
            
        symbol_win = 0
        symbol_loss = 0
        symbol_pnl = 0
        last_buy = None
        last_side = None
        last_score = None
        
        for trade in trades:
            if 'score' in trade:
                last_score = trade['score']
                
            if trade['action'] == 'BUY':
                last_buy = float(trade['price'])
                last_side = trade['side']
                
            elif trade['action'] == 'SELL' and last_buy is not None:
                exit_price = float(trade['price'])
                entry_price = last_buy
                side = last_side
                
                # Для LONG позиций: (exit - entry) / entry
                # Для SHORT позиций: (entry - exit) / entry
                pnl_pct = (exit_price - entry_price) / entry_price if side == 'long' else (entry_price - exit_price) / entry_price
                
                # Базовый размер позиции
                size = 1
                
                # Рекомендуемое плечо на основе силы сигнала
                leverage = 1
                if last_score is not None:
                    label, strength = signal_strength_label(last_score)
                    if strength >= 0.85:
                        leverage = 10
                    elif strength >= 0.7:
                        leverage = 5
                    elif strength >= 0.5:
                        leverage = 3
                    else:
                        leverage = 2
                
                # Комиссия за открытие и закрытие позиции
                fee = (entry_price + exit_price) * size * FEE_RATE
                
                # Получаем funding rate
                try:
                    ticker = EXCHANGE.fetch_ticker(symbol)
                    funding = ticker.get('fundingRate', 0) * size * entry_price
                except Exception:
                    funding = 0
                
                # Расчет P&L с учетом плеча, комиссий и funding
                pnl_pct = pnl_pct - (fee / (entry_price * size)) - (funding / (entry_price * size))
                pnl_leverage = pnl_pct * leverage
                pnl_usdt = pnl_leverage * entry_price * size
                
                symbol_pnl += pnl_usdt
                total_pnl_usdt += pnl_usdt
                
                if pnl_usdt > 0:
                    symbol_win += 1
                    win += 1
                else:
                    symbol_loss += 1
                    loss += 1
                
                last_buy = None
                last_side = None
                last_score = None
        
        if symbol_win > 0 or symbol_loss > 0:
            winrate = (symbol_win / (symbol_win + symbol_loss)) * 100 if (symbol_win + symbol_loss) > 0 else 0
            report.append(f"{symbol}: прибыльных {symbol_win}, убыточных {symbol_loss}, WR {winrate:.1f}%, P&L {symbol_pnl:.2f} USDT")
    
    # Сортируем отчет по общей прибыли
    report.sort(key=lambda x: float(x.split("P&L ")[-1].split(" USDT")[0]), reverse=True)
    
    # Добавляем общую статистику
    total_trades = win + loss
    if total_trades > 0:
        total_winrate = (win / total_trades) * 100
        report.append(f"\nИтого: {total_trades} сделок, WR {total_winrate:.1f}%, P&L {total_pnl_usdt:.2f} USDT")
    
    return report, win, loss, total_pnl_usdt

# ========== ФУНКЦИИ АНАЛИЗА ==========
def get_ohlcv(symbol):
    """Получить исторические данные по монете."""
    for attempt in range(3):  # Добавляем повторные попытки
        try:
            ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
            if not ohlcv or len(ohlcv) < MA_SLOW:  # Проверяем достаточность данных
                logging.warning(f"{symbol}: недостаточно данных для анализа")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Moscow')
            return df
        except ccxt.RateLimitExceeded as e:
            wait_time = getattr(e, 'retry_after', 1)
            logging.warning(f"Rate limit exceeded for {symbol}, жду {wait_time} сек.")
            time.sleep(wait_time)
        except ccxt.NetworkError as e:
            logging.error(f"Network error for {symbol}: {e}")
            time.sleep(5)  # Ждём подольше при сетевой ошибке
        except Exception as e:
            logging.error(f"Ошибка получения OHLCV по {symbol}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()  # Возвращаем пустой DataFrame после всех попыток

def analyze(df):
    """Анализ по индикаторам: EMA, MACD, ATR (15m), RSI, ADX, Bollinger Bands, Volume, VWAP, Stochastic."""
    try:
        if df.empty or len(df) < MA_SLOW:
            return pd.DataFrame()
            
        # Базовые индикаторы
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=MA_FAST)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=MA_SLOW)
        
        # Дополнительные EMA для подтверждения тренда на 15-минутках
        df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
        df['ema_13'] = ta.trend.ema_indicator(df['close'], window=13)
        
        # MACD с сигнальной линией
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_line'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        # RSI и его EMA для фильтрации ложных сигналов
        df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_WINDOW)
        df['rsi_ema'] = ta.trend.ema_indicator(df['rsi'], window=5)
        
        # Индикаторы волатильности
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=ATR_WINDOW)
        df['bollinger_mid'] = ta.volatility.bollinger_mavg(df['close'], window=20)
        df['bollinger_high'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['bollinger_low'] = ta.volatility.bollinger_lband(df['close'], window=20)
        # Ширина полос Боллинджера для оценки волатильности на 15м
        df['bb_width'] = (df['bollinger_high'] - df['bollinger_low']) / df['bollinger_mid']
        
        # Индикаторы тренда
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['pdi'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
        df['mdi'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
        
        # Объём - расширенный анализ для 15-минутного таймфрейма
        if USE_VOLUME_FILTER:
            df['volume_ema'] = ta.trend.ema_indicator(df['volume'], window=20)
            df['volume_ratio'] = df['volume'] / df['volume_ema']
            # Добавляем обнаружение аномальных объемов для 15-минутного таймфрейма
            df['volume_std'] = df['volume'].rolling(20).std()
            df['volume_z_score'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume_std']
            # Кумулятивный дельта-объем (разница между объемами при росте и падении)
            df['vol_delta'] = df['volume'] * ((df['close'] - df['open']) / abs(df['close'] - df['open'] + 0.000001))
            df['cum_vol_delta'] = df['vol_delta'].rolling(10).sum()
        
        # Вычисление спреда для каждой свечи
        df['spread_pct'] = (df['high'] - df['low']) / df['low']
        
        # Детектор импульса для 15-минутного таймфрейма
        df['price_change'] = df['close'].pct_change()
        df['momentum'] = df['price_change'].rolling(5).mean() * 100  # 5 свечей = ~1 час
        
        # Убираем строки с NaN, чтобы не ловить фантомные кресты
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < 2:  # Проверяем, что осталось достаточно данных
            return pd.DataFrame()
            
        # === VWAP ===
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_vol_price'] = (df['close'] * df['volume']).cumsum()
        df['vwap'] = df['cum_vol_price'] / df['cum_vol']

        # === Stochastic Oscillator ===
        stoch_k = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        stoch_d = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        return df
    except Exception as e:
        logging.error(f"Ошибка в анализе данных: {e}")
        return pd.DataFrame()

# === ПРОСТЫЕ ГРАФИЧЕСКИЕ ПАТТЕРНЫ ===
def detect_double_bottom(df, window=20):
    # Двойное дно: два минимума примерно на одном уровне, между ними локальный максимум
    lows = df['low'].iloc[-window:]
    if len(lows) < window: return False
    idx_min1 = lows.idxmin()
    min1 = lows.min()
    # Ищем второй минимум после первого
    lows2 = lows.iloc[lows.index.get_loc(idx_min1)+1:] if idx_min1+1 < len(df.index) else None
    if lows2 is not None and not lows2.empty:
        min2 = lows2.min()
        idx_min2 = lows2.idxmin()
        # Минимумы должны быть близки по значению
        if abs(min1 - min2) / min1 < 0.01:
            # Между минимумами должен быть локальный максимум
            between = df['high'].loc[idx_min1+1:idx_min2] if idx_min2 > idx_min1+1 else None
            if between is not None and not between.empty:
                if between.max() > min1 * 1.01:
                    return True
    return False

def detect_double_top(df, window=20):
    # Двойная вершина: два максимума примерно на одном уровне, между ними локальный минимум
    highs = df['high'].iloc[-window:]
    if len(highs) < window: return False
    idx_max1 = highs.idxmax()
    max1 = highs.max()
    highs2 = highs.iloc[highs.index.get_loc(idx_max1)+1:] if idx_max1+1 < len(df.index) else None
    if highs2 is not None and not highs2.empty:
        max2 = highs2.max()
        idx_max2 = highs2.idxmax()
        if abs(max1 - max2) / max1 < 0.01:
            between = df['low'].loc[idx_max1+1:idx_max2] if idx_max2 > idx_max1+1 else None
            if between is not None and not between.empty:
                if between.min() < max1 * 0.99:
                    return True
    return False

def detect_triangle(df, window=20):
    # Простейшее определение: сужающийся диапазон high и low
    highs = df['high'].iloc[-window:]
    lows = df['low'].iloc[-window:]
    if len(highs) < window or len(lows) < window: return False
    if highs.max() > highs.min() * 1.01 and lows.max() > lows.min() * 1.01:
        # Проверяем, что high понижаются, а low повышаются
        highs_trend = highs.diff().mean() < 0
        lows_trend = lows.diff().mean() > 0
        if highs_trend and lows_trend:
            return True
    return False

def detect_chart_pattern(df):
    if detect_double_bottom(df):
        return "Двойное дно"
    if detect_double_top(df):
        return "Двойная вершина"
    if detect_triangle(df):
        return "Треугольник"
    return None

# ========== ОЦЕНКА СИЛЫ СИГНАЛА ПО ГРАФИКУ ==========
def evaluate_signal_strength(df, symbol, action):
    """Оценивает силу сигнала в баллах (score) для 15-минутного таймфрейма, теперь с учётом VWAP и Stochastic."""
    score = 0
    pattern_name = None
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 1. Сила тренда по ADX - более гибкие критерии для 15м
    if last['adx'] > 25:
        score += 1.5
        logging.info(f"{symbol}: +1.5 балла за adx > 25 (сильный тренд)")
    elif last['adx'] > 18:
        score += 1
        logging.info(f"{symbol}: +1 балл за adx > 18 (хороший тренд)")

    # 2. Положение RSI - адаптировано для 15м таймфрейма
    if action == 'BUY':
        if last['rsi'] < 70 and last['rsi'] > 45:
            score += 1
            logging.info(f"{symbol}: +1 балл за оптимальный rsi: {last['rsi']:.1f}")
        elif last['rsi'] < 45 and last['rsi'] > 30:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за rsi: {last['rsi']:.1f} (возможен отскок)")
    elif action == 'SELL':
        if last['rsi'] > 30 and last['rsi'] < 55:
            score += 1
            logging.info(f"{symbol}: +1 балл за оптимальный rsi: {last['rsi']:.1f}")
        elif last['rsi'] > 55 and last['rsi'] < 70:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за rsi: {last['rsi']:.1f} (возможен отскок)")

    # 3. Положение цены относительно Bollinger Bands - улучшено для 15м
    if 'bollinger_mid' in df.columns:
        if action == 'BUY':
            if last['close'] > last['bollinger_mid'] and last['close'] < last['bollinger_high']:
                score += 1
                logging.info(f"{symbol}: +1 балл за положение цены между серединой и верхней BB")
            elif last['close'] < last['bollinger_mid'] and last['close'] > last['bollinger_low']:
                score += 0.5
                logging.info(f"{symbol}: +0.5 балла за положение цены между серединой и нижней BB (возможен отскок)")
        elif action == 'SELL':
            if last['close'] < last['bollinger_mid'] and last['close'] > last['bollinger_low']:
                score += 1
                logging.info(f"{symbol}: +1 балл за положение цены между серединой и нижней BB")
            elif last['close'] > last['bollinger_mid'] and last['close'] < last['bollinger_high']:
                score += 0.5
                logging.info(f"{symbol}: +0.5 балла за положение цены между серединой и верхней BB (возможен отскок)")
    
    # 4. Положение цены относительно уровней поддержки/сопротивления
    try:
        support, resistance = find_support_resistance(df, window=20)
        if support is not None and resistance is not None:
            price = last['close']
            if action == 'BUY' and price < (support * 1.02):
                score += 1
                logging.info(f"{symbol}: +1 балл за цену у поддержки {support:.4f}")
            elif action == 'SELL' and price > (resistance * 0.98):
                score += 1
                logging.info(f"{symbol}: +1 балл за цену у сопротивления {resistance:.4f}")
    except Exception as e:
        logging.warning(f"{symbol}: ошибка проверки S/R уровней: {e}")

    # 5. Price Action (паттерны свечей)
    if action == 'BUY' and (is_bullish_pinbar(last) or is_bullish_engulfing(prev, last)):
        score += 1
        logging.info(f"{symbol}: +1 балл за бычий price action")
    if action == 'SELL' and (is_bearish_pinbar(last) or is_bearish_engulfing(prev, last)):
        score += 1
        logging.info(f"{symbol}: +1 балл за медвежий price action")

    # 6. Графические паттерны (Двойное дно/вершина, треугольник)
    pattern_name = detect_chart_pattern(df)
    if pattern_name:
        if (action == 'BUY' and "Дно" in pattern_name) or \
           (action == 'SELL' and "Вершина" in pattern_name) or \
           ("Треугольник" in pattern_name):
            score += 1
            logging.info(f"{symbol}: +1 балл за графический паттерн: {pattern_name}")
    
    # 7. Проверка на дополнительное подтверждение 15-минутного сигнала через EMA 5/13
    if 'ema_5' in last and 'ema_13' in last:
        if action == 'BUY' and last['ema_5'] > last['ema_13']:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за ema_5 > ema_13 (краткосрочный восходящий тренд)")
        elif action == 'SELL' and last['ema_5'] < last['ema_13']:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за ema_5 < ema_13 (краткосрочный нисходящий тренд)")
    
    # 8. Анализ объема для 15-минутного таймфрейма
    if 'volume_z_score' in last and 'cum_vol_delta' in last:
        # Проверка аномальных объемов
        if action == 'BUY' and last['volume_z_score'] > 2 and last['cum_vol_delta'] > 0:
            score += 1
            logging.info(f"{symbol}: +1 балл за положительный всплеск объема (z-score: {last['volume_z_score']:.2f})")
        elif action == 'SELL' and last['volume_z_score'] > 2 and last['cum_vol_delta'] < 0:
            score += 1
            logging.info(f"{symbol}: +1 балл за отрицательный всплеск объема (z-score: {last['volume_z_score']:.2f})")
    
    # 9. Импульс для 15-минутного таймфрейма
    if 'momentum' in last:
        if action == 'BUY' and last['momentum'] > 0.5:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за положительный импульс {last['momentum']:.2f}%")
        elif action == 'SELL' and last['momentum'] < -0.5:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за отрицательный импульс {last['momentum']:.2f}%")

    # 10. VWAP (цена относительно VWAP)
    if 'vwap' in last:
        if action == 'BUY' and last['close'] > last['vwap']:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за close > VWAP")
        elif action == 'SELL' and last['close'] < last['vwap']:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за close < VWAP")
    # 11. Стохастик (Stochastic Oscillator)
    if 'stoch_k' in last and 'stoch_d' in last:
        if action == 'BUY' and last['stoch_k'] < 30 and last['stoch_k'] > last['stoch_d']:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за стохастик (k < 30 и k > d)")
        elif action == 'SELL' and last['stoch_k'] > 70 and last['stoch_k'] < last['stoch_d']:
            score += 0.5
            logging.info(f"{symbol}: +0.5 балла за стохастик (k > 70 и k < d)")

    return score, pattern_name

def signal_strength_label(score):
    """
    Преобразует числовую оценку силы сигнала (0-9) в текстовую метку
    и процентную вероятность успеха.
    
    Возвращает кортеж (метка, вероятность)
    """
    if score >= 7:
        return 'Очень сильный', 0.95
    elif score >= 6:
        return 'Сильный', 0.85
    elif score >= 5:
        return 'Средний', 0.75
    elif score >= 4:
        return 'Умеренный', 0.65
    elif score >= 3:
        return 'Слабый', 0.55
    elif score >= 2:
        return 'Очень слабый', 0.45  # Увеличил с 0.40 до 0.45
    elif score >= 1:
        return 'Ненадёжный', 0.35  # Увеличил с 0.30 до 0.35
    else:
        return 'Крайне ненадёжный', 0.25  # Добавил новую категорию

# ========== СТАТИСТИКА ПО ИСТОРИИ ==========
def get_signal_stats(symbol, action):
    """Возвращает процент успешных сигналов по монете и действию ('BUY'/'SELL')."""
    if symbol not in virtual_portfolio:
        return 0, 0
    trades = virtual_portfolio[symbol]
    total = 0
    success = 0
    last_buy = None
    for trade in trades:
        if trade['action'] == 'BUY':
            last_buy = float(trade['price'])
        elif trade['action'] == 'SELL' and last_buy is not None:
            total += 1
            if float(trade['price']) > last_buy and action == 'BUY':
                success += 1
            if float(trade['price']) < last_buy and action == 'SELL':
                success += 1
            last_buy = None
    percent = (success / total * 100) if total > 0 else 0
    return percent, total

# ========== РЕКОМЕНДАЦИЯ ПО ПЛЕЧУ ==========
def recommend_leverage(strength_score, history_percent):
    # Усредняем силу по графику и по истории
    avg = (strength_score + (history_percent / 100 * 3)) / 2
    if avg >= 2.5:
        return 'x10'
    elif avg >= 1.5:
        return 'x5'
    elif avg >= 1.0:
        return 'x3'
    else:
        return 'x2'

# ========== ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ ОБЪЁМА ==========
def get_24h_volume(symbol):
    try:
        ticker = EXCHANGE.fetch_ticker(symbol)
        volume = ticker.get('quoteVolume', 0)
        return volume
    except ccxt.RateLimitExceeded as e:
        logging.warning(f"Rate limit exceeded for {symbol}, жду {getattr(e, 'retry_after', 1)} сек.")
        time.sleep(getattr(e, 'retry_after', 1))
        return 0
    except Exception as e:
        print(f"Ошибка получения объёма по {symbol}: {e}")
        return 0

last_signal_time = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))

def is_bullish_pinbar(row):
    body = abs(row['close'] - row['open'])
    candle_range = row['high'] - row['low']
    lower_shadow = min(row['open'], row['close']) - row['low']
    upper_shadow = row['high'] - max(row['open'], row['close'])
    return (
        body < candle_range * 0.3 and
        lower_shadow > body * 2 and
        upper_shadow < body
    )

def is_bearish_pinbar(row):
    body = abs(row['close'] - row['open'])
    candle_range = row['high'] - row['low']
    lower_shadow = min(row['open'], row['close']) - row['low']
    upper_shadow = row['high'] - max(row['open'], row['close'])
    return (
        body < candle_range * 0.3 and
        upper_shadow > body * 2 and
        lower_shadow < body
    )

def is_bullish_engulfing(prev, last):
    return (
        prev['close'] < prev['open'] and
        last['close'] > last['open'] and
        last['close'] > prev['open'] and
        last['open'] < prev['close']
    )

def is_bearish_engulfing(prev, last):
    return (
        prev['close'] > prev['open'] and
        last['close'] < last['open'] and
        last['open'] > prev['close'] and
        last['close'] < prev['open']
    )

def get_btc_adx():
    try:
        ohlcv = EXCHANGE.fetch_ohlcv('BTC/USDT:USDT', timeframe=TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        return df['adx'].iloc[-1]
    except Exception as e:
        logging.error(f"Ошибка получения ADX BTC: {e}")
        return 99

def is_global_uptrend(symbol: str) -> int:
    """
    Возвращает количество совпавших условий мультифреймового тренда (0-4):
    1. EMA21 > EMA50 на дневке (1d)
    2. Цена выше EMA21 на дневке (1d)
    3. RSI(14) > 50 на 4h
    4. Цена на 4h выше своей SMA20
    
    Оптимизированная версия с более гибкими условиями для большего числа сигналов.
    """
    try:
        # Получаем данные дневного таймфрейма
        ohlcv_daily = EXCHANGE.fetch_ohlcv(symbol, '1d', limit=50)
        df_daily = pd.DataFrame(ohlcv_daily, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df_daily['ema21'] = ta.trend.ema_indicator(df_daily['c'], 21)
        df_daily['ema50'] = ta.trend.ema_indicator(df_daily['c'], 50)
        last_daily = df_daily.iloc[-1]
        
        # Получаем данные 4-часового таймфрейма
        ohlcv_4h = EXCHANGE.fetch_ohlcv(symbol, '4h', limit=100)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df_4h['rsi'] = ta.momentum.rsi(df_4h['c'], 14)
        df_4h['sma20'] = df_4h['c'].rolling(20).mean()
        last_4h = df_4h.iloc[-1]
        
        # Добавляем данные 1-часового таймфрейма для дополнительного анализа
        ohlcv_1h = EXCHANGE.fetch_ohlcv(symbol, '1h', limit=50)
        df_1h = pd.DataFrame(ohlcv_1h, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df_1h['ema9'] = ta.trend.ema_indicator(df_1h['c'], 9)
        df_1h['ema21'] = ta.trend.ema_indicator(df_1h['c'], 21)
        last_1h = df_1h.iloc[-1]
        
        # Основные условия (как раньше)
        conditions = [
            last_daily['c'] > last_daily['ema21'],  # Цена выше EMA21 на дневке
            last_daily['ema21'] > last_daily['ema50'],  # EMA21 > EMA50 на дневке
            last_4h['rsi'] > 50,  # RSI > 50 на 4h
            last_4h['c'] > last_4h['sma20']  # Цена выше SMA20 на 4h
        ]
        
        # Дополнительные условия для более точной оценки тренда
        # Учитываем только если основные условия дают неоднозначный результат (2 из 4)
        if sum(conditions) == 2:
            # Проверяем тренд на 1h
            if last_1h['ema9'] > last_1h['ema21']:
                # Если на 1h есть восходящий тренд, добавляем 0.5 балла
                return 2.5
            else:
                # Если на 1h нет восходящего тренда, вычитаем 0.5 балла
                return 1.5
        
        return sum(conditions)
    except Exception as e:
        logging.error(f"Ошибка при определении глобального тренда для {symbol}: {e}")
        return 0

def check_signals(df, symbol):
    try:
        if df.empty or len(df) < 2:
            return []
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Проверка на минимальное количество свечей для анализа 15м таймфрейма
        if len(df) < MIN_15M_CANDLES:
            logging.info(f"{symbol}: недостаточно данных для анализа на 15м (требуется {MIN_15M_CANDLES} свечей)")
            return []
            
        signals = []
        score_penalty = 0
        vwap_penalty = 0
        stoch_penalty = 0
        
        # === Фильтр по BTC ADX для альтов ===
        # Смягчаем фильтр по BTC ADX для увеличения числа сигналов
        if symbol != 'BTC/USDT:USDT':
            btc_adx = get_btc_adx()
            if btc_adx < 12:  # Уменьшил с 20 до 12
                logging.info(f"{symbol}: BTC ADX {btc_adx:.2f} < 12, сигналы по альтам не формируются")
                return []
                
        # === БАЗОВЫЕ ФИЛЬТРЫ ===
        if last['adx'] < MIN_ADX:
            logging.info(f"{symbol}: ADX {last['adx']:.2f} < {MIN_ADX}, сигнал не формируется (слабый тренд)")
            return []
            
        volume = get_24h_volume(symbol)
        volume_mln = volume / 1_000_000
        if volume < MIN_VOLUME_USDT:
            logging.info(f"{symbol}: объём {volume_mln:.2f} млн < {MIN_VOLUME_USDT/1_000_000:.0f} млн, сигнал не формируется")
            return []
            
        # Штраф за экстремальный RSI УДАЛЕН (обрабатывается в evaluate_signal_strength)
            
        if last['spread_pct'] > MAX_SPREAD_PCT:
            logging.info(f"{symbol}: большой спред {last['spread_pct']*100:.2f}% > {MAX_SPREAD_PCT*100:.2f}%, сигнал не формируется")
            return []
            
        # Фильтр по объему для 15м
        if USE_VOLUME_FILTER:
            if last['volume_ratio'] < VOLUME_SPIKE_MULT:
                logging.info(f"{symbol}: низкий относительный объём {last['volume_ratio']:.2f} < {VOLUME_SPIKE_MULT}, сигнал не формируется")
                return []
            
            # Дополнительная проверка для 15м на основе z-score объема (смягчена)
            if 'volume_z_score' in last and abs(last['volume_z_score']) < 0.5:  # Снижено с 0.8
                logging.info(f"{symbol}: недостаточный z-score объема {last['volume_z_score']:.2f}, сигнал не формируется")
                return []
        
        # Проверка импульса для 15м
        if 'momentum' in last and abs(last['momentum']) < MIN_MOMENTUM:
            logging.info(f"{symbol}: слабый импульс {last['momentum']:.2f} < {MIN_MOMENTUM}, сигнал не формируется")
            return []
            
        # Проверка ширины полос Боллинджера для контроля волатильности на 15м
        if 'bb_width' in last and last['bb_width'] > MAX_BB_WIDTH:
            logging.info(f"{symbol}: чрезмерная волатильность BB {last['bb_width']:.3f} > {MAX_BB_WIDTH}, сигнал не формируется")
            return []
            
        # Проверка на тренд по 5 свечам
        price_trend = sum(1 if df['close'].iloc[i] > df['close'].iloc[i-1] else -1 for i in range(-5, 0))
            
        # === СИГНАЛЫ НА ПОКУПКУ ===
        if prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            if last['macd'] > 0 or last['macd'] > last['macd_signal']:  # Изменил И на ИЛИ для большего числа сигналов
                # Смягчаем проверку на динамику RSI
                if last['rsi'] < prev['rsi'] - 7: # Только если RSI значительно падает
                    score_penalty -= 0.5
                    logging.info(f"{symbol}: штраф -0.5 к score за падение RSI для BUY")
                
                # Проверка тренда на 1h - делаем опциональной
                try:
                    ohlcv_1h = EXCHANGE.fetch_ohlcv(symbol, '1h', limit=5)
                    df_1h = pd.DataFrame(ohlcv_1h, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                    hourly_trend = 1 if df_1h['c'].iloc[-1] > df_1h['c'].iloc[-2] else -1
                    if hourly_trend < 0:
                        score_penalty -= 0.5
                        logging.info(f"{symbol}: штраф -0.5 к score за отсутствие роста на 1h для BUY")
                except Exception as e:
                    logging.error(f"Ошибка при проверке 1h тренда: {e}")
                
                # Price Action: делаем опциональным, т.к. слишком сильно фильтрует
                # if not (is_bullish_pinbar(last) or is_bullish_engulfing(prev, last)):
                #     score_penalty -= 0.5
                #     logging.info(f"{symbol}: штраф -0.5 к score за отсутствие price action для BUY")
                
                # Мультифреймовый фильтр: смягчаем
                trend_score = is_global_uptrend(symbol)
                if trend_score < 1:  # Было < 2
                    score_penalty -= 1
                    logging.info(f"{symbol}: штраф -1 к score за отсутствие подтверждения тренда (совпало {trend_score} из 4)")
                
                # Проверка динамики MACD - делаем опциональной
                if len(df) >= 3 and last['macd'] < df.iloc[-2]['macd'] * 0.8:  # Только если MACD значительно снижается
                    score_penalty -= 0.5 # Было -1
                    logging.info(f"{symbol}: штраф -0.5 к score за значительное снижение MACD")
                
                # Проверка объема - делаем опциональной
                vol_avg_5 = df['volume'].iloc[-5:].mean()
                if last['volume'] < vol_avg_5 * 0.7:  # Только если объем значительно ниже
                    score_penalty -= 0.5 # Было -1
                    logging.info(f"{symbol}: штраф -0.5 к score за объем значительно ниже среднего за 5 свечей")
                
                # Учет краткосрочного тренда - делаем опциональным
                if price_trend < -2:  # Только если явно нисходящий тренд
                    score_penalty -= 1
                    logging.info(f"{symbol}: штраф -1 к score за явно нисходящий тренд по 5 свечам")
                
                # Штраф за возможную дивергенцию MACD УДАЛЕН (обрабатывается в evaluate_signal_strength)
                
                action = 'BUY'
                
                # Проверка на рост объема - слишком специфичный фильтр, убираем
                # if df['volume'].iloc[-1] < df['volume'].iloc[-2] * 0.6 and df['volume'].iloc[-2] < df['volume'].iloc[-3] * 0.6:
                #     score_penalty -= 1
                #     logging.info(f"{symbol}: штраф -1 к score за значительное падение объёма для BUY")
                
                # Проверка на бычью свечу - делаем опциональной
                if df['close'].iloc[-1] < df['open'].iloc[-1] * 0.99:  # Только если явно медвежья
                    score_penalty -= 0.5 # Было -1
                    logging.info(f"{symbol}: штраф -0.5 к score за явно медвежью свечу для BUY")
                
                # === Гибкий фильтр по VWAP ===
                if 'vwap' in last:
                    # Для BUY
                    if prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
                        if last['close'] < last['vwap'] * 1.002:
                            vwap_penalty -= 0.5
                            logging.info(f"{symbol}: close ниже VWAP, штраф -0.5 к score (VWAP фильтр)")
                
                # === Гибкий фильтр по стохастику ===
                if 'stoch_k' in last and 'stoch_d' in last:
                    # Для BUY
                    if prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
                        if last['stoch_k'] > 60:
                            stoch_penalty -= 0.5
                            logging.info(f"{symbol}: stoch_k > 60, штраф -0.5 к score (Stochastic фильтр)")
                
                # Рассчитываем финальный score
                score, pattern_name = evaluate_signal_strength(df, symbol, action)
                score += score_penalty + vwap_penalty + stoch_penalty
                
                # Снижаем минимальный порог для формирования сигнала до 2.5 (было 3)
                if score < 2.5:
                    logging.info(f"{symbol}: score {score} < 2.5, сигнал не формируется")
                    return []
                
                label, strength_chance = signal_strength_label(score)
                history_percent, total = get_signal_stats(symbol, action)
                winrate = get_score_winrate(score, action)
                
                msg = f'\U0001F4C8 Сигнал (ФЬЮЧЕРСЫ BYBIT): КУПИТЬ!\nСила сигнала: {label}\nОценка по графику: {strength_chance*100:.2f}%\nРекомендуемое плечо: {recommend_leverage(score, history_percent)}\nОбъём торгов: {volume_mln:.2f} млн USDT/сутки\nADX: {last["adx"]:.1f} (сила тренда)\nTP/SL указываются ниже, выставлять их на бирже!\nПричина: EMA_fast пересёк EMA_slow вверх, MACD бычий.'
                if pattern_name:
                    msg += f"\nОбнаружен паттерн: {pattern_name}"
                msg += f"\nWinrate: {winrate if winrate is not None else 'нет данных'}"
                signals.append(msg)
                logging.info(f"{symbol}: BUY сигнал сформирован (фьючерсы)")
        
        # === СИГНАЛЫ НА ПРОДАЖУ ===
        if prev['ema_fast'] > prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            if last['macd'] < 0 or last['macd'] < last['macd_signal']:  # Изменил И на ИЛИ для большего числа сигналов
                # Смягчаем проверку на динамику RSI
                if last['rsi'] > prev['rsi'] + 7:  # Только если RSI значительно растет
                    score_penalty -= 0.5
                    logging.info(f"{symbol}: штраф -0.5 к score за рост RSI для SELL")
                
                # Проверка тренда на 1h - делаем опциональной
                try:
                    ohlcv_1h = EXCHANGE.fetch_ohlcv(symbol, '1h', limit=5)
                    df_1h = pd.DataFrame(ohlcv_1h, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                    hourly_trend = 1 if df_1h['c'].iloc[-1] > df_1h['c'].iloc[-2] else -1
                    if hourly_trend > 0:
                        score_penalty -= 0.5
                        logging.info(f"{symbol}: штраф -0.5 к score за отсутствие падения на 1h для SELL")
                except Exception as e:
                    logging.error(f"Ошибка при проверке 1h тренда: {e}")
                
                # Price Action: делаем опциональным, т.к. слишком сильно фильтрует
                # if not (is_bearish_pinbar(last) or is_bearish_engulfing(prev, last)):
                #     score_penalty -= 0.5
                #     logging.info(f"{symbol}: штраф -0.5 к score за отсутствие price action для SELL")
                
                # Мультифреймовый фильтр: смягчаем
                trend_score = is_global_uptrend(symbol)
                if trend_score > 3:  # Было > 2
                    score_penalty -= 1
                    logging.info(f"{symbol}: штраф -1 к score за отсутствие подтверждения нисходящего тренда (совпало {trend_score} из 4)")
                
                # Проверка динамики MACD - исправлена логика для SELL
                if len(df) >= 3 and last['macd'] > df.iloc[-2]['macd'] * 1.2:  # Штраф если MACD растет вместо падения
                    score_penalty -= 0.5 # Было -1
                    logging.info(f"{symbol}: штраф -0.5 к score за рост MACD для SELL вместо ожидаемого падения")

                # Проверка объема - делаем опциональной
                vol_avg_5 = df['volume'].iloc[-5:].mean()
                if last['volume'] < vol_avg_5 * 0.7:  # Только если объем значительно ниже
                    score_penalty -= 0.5 # Было -1
                    logging.info(f"{symbol}: штраф -0.5 к score за объем значительно ниже среднего за 5 свечей")
                
                # Учет краткосрочного тренда - делаем опциональным
                if price_trend > 2:  # Только если явно восходящий тренд
                    score_penalty -= 1
                    logging.info(f"{symbol}: штраф -1 к score за явно восходящий тренд по 5 свечам")
                
                # Штраф за возможную дивергенцию MACD УДАЛЕН (обрабатывается в evaluate_signal_strength)
                
                action = 'SELL'
                
                # Проверка на медвежью свечу - делаем опциональной
                if df['close'].iloc[-1] > df['open'].iloc[-1] * 1.01:  # Только если явно бычья
                    score_penalty -= 0.5 # Было -1
                    logging.info(f"{symbol}: штраф -0.5 к score за явно бычью свечу для SELL")
                
                # === Гибкий фильтр по VWAP ===
                if 'vwap' in last:
                    # Для SELL
                    if prev['ema_fast'] > prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
                        if last['close'] > last['vwap'] * 0.998:
                            vwap_penalty -= 0.5
                            logging.info(f"{symbol}: close выше VWAP, штраф -0.5 к score (VWAP фильтр)")
                
                # === Гибкий фильтр по стохастику ===
                if 'stoch_k' in last and 'stoch_d' in last:
                    # Для SELL
                    if prev['ema_fast'] > prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
                        if last['stoch_k'] < 40:
                            stoch_penalty -= 0.5
                            logging.info(f"{symbol}: stoch_k < 40, штраф -0.5 к score (Stochastic фильтр)")
                
                # Рассчитываем финальный score
                score, pattern_name = evaluate_signal_strength(df, symbol, action)
                score += score_penalty + vwap_penalty + stoch_penalty
                
                # Снижаем минимальный порог для формирования сигнала до 2.5 (было 3)
                if score < 2.5:
                    logging.info(f"{symbol}: score {score} < 2.5, сигнал не формируется")
                    return []
                
                label, strength_chance = signal_strength_label(score)
                history_percent, total = get_signal_stats(symbol, action)
                winrate = get_score_winrate(score, action)
                
                msg = f'\U0001F4C9 Сигнал (ФЬЮЧЕРСЫ BYBIT): ПРОДАТЬ!\nСила сигнала: {label}\nОценка по графику: {strength_chance*100:.2f}%\nРекомендуемое плечо: {recommend_leverage(score, history_percent)}\nОбъём торгов: {volume_mln:.2f} млн USDT/сутки\nADX: {last["adx"]:.1f} (сила тренда)\nTP/SL указываются ниже, выставлять их на бирже!\nПричина: EMA_fast пересёк EMA_slow вниз, MACD медвежий.'
                if pattern_name:
                    msg += f"\nОбнаружен паттерн: {pattern_name}"
                msg += f"\nWinrate: {winrate if winrate is not None else 'нет данных'}"
                signals.append(msg)
                logging.info(f"{symbol}: SELL сигнал сформирован (фьючерсы)")
        
        if last_signal_time[symbol].tzinfo is None:
            last_signal_time[symbol] = last_signal_time[symbol].replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        if now - last_signal_time[symbol] < timedelta(minutes=SIGNAL_COOLDOWN_MINUTES):
            return []
        if signals:
            last_signal_time[symbol] = now
        return signals
    except Exception as e:
        logging.error(f"Ошибка при проверке сигналов для {symbol}: {e}")
        return []

def analyze_long(df):
    """Долгосрочный анализ: EMA50/200, MACD, RSI на дневках."""
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=200)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Moscow')
    return df

def check_signals_long(df):
    """Сигналы для долгосрока: Golden/Death Cross + MACD + RSI на дневках."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    # Golden Cross (EMA50 пересёк EMA200 вверх) + MACD бычий + RSI < 65
    if prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow'] and last['macd'] > 0 and last['rsi'] < 65:
        signals.append('Сигнал: КУПИТЬ НА ДОЛГОСРОК!\nПричина: EMA50 пересёк EMA200 вверх (Golden Cross), MACD бычий, RSI < 65.')
    # Death Cross (EMA50 пересёк EMA200 вниз) + MACD медвежий + RSI > 35
    if prev['ema_fast'] > prev['ema_slow'] and last['ema_fast'] < last['ema_slow'] and last['macd'] < 0 and last['rsi'] > 35:
        signals.append('Сигнал: ПРОДАТЬ НА ДОЛГОСРОК!\nПричина: EMA50 пересёк EMA200 вниз (Death Cross), MACD медвежий, RSI > 35.')
    return signals

# ========== ОТПРАВКА В TELEGRAM ==========
async def send_telegram_message(text):
    bot = Bot(token=TELEGRAM_TOKEN)
    for attempt in range(3):
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
            break
        except Exception as e:
            logging.error(f"Ошибка отправки сообщения в Telegram: {e}")
            await asyncio.sleep(2)

# ========== ОТПРАВКА ОТЧЁТА ==========
async def send_daily_report():
    report, win, loss = simple_stats()
    text = '📊 Отчёт по виртуальным сделкам за сутки:\n'
    if report:
        text += '\n'.join(report)
    else:
        text += 'Нет завершённых сделок.'
    await send_telegram_message(text)

# ========== ОБРАБОТЧИК КОМАНДЫ /stats ==========
async def stats_command(update, context):
    report, win, loss = simple_stats()
    text = '📊 Статистика по виртуальным сделкам:\n'
    if report:
        text += '\n'.join(report)
    else:
        text += 'Нет завершённых сделок.'
    await update.message.reply_text(text)

# ========== ОСНОВНОЙ ЦИКЛ ==========
TIME_SHIFT_HOURS = 3  # Сдвиг времени для локального времени пользователя
async def telegram_bot():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("stats", stats_command))
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    await asyncio.Event().wait()  # чтобы задача не завершалась

async def monitor_open_positions():
    """
    Отдельная асинхронная функция для мониторинга открытых позиций и проверки TP/SL.
    Работает параллельно с основным циклом анализа, проверяя позиции каждые 30 секунд.
    """
    while True:
        try:
            for symbol in list(open_trades.keys()):
                # Получаем актуальные данные
                df = get_ohlcv(symbol)
                if df.empty:
                    continue
                
                df = analyze(df)
                if df.empty:
                    continue
                
                price = df['close'].iloc[-1]
                time = df['timestamp'].iloc[-1]
                
                # Проверка достижения TP/SL
                if check_tp_sl(symbol, price, time, df):
                    logging.info(f"Монитор позиций: {symbol} закрыт по TP/SL")
            
            # Проверяем каждые 3 минуты для баланса между точностью и нагрузкой
            await asyncio.sleep(60 * 3)
        except Exception as e:
            logging.error(f"Ошибка в мониторе позиций: {e}")
            await asyncio.sleep(60)  # В случае ошибки ждем минуту перед повторной попыткой

async def process_symbol(symbol):
    """Обработка одного символа для асинхронного анализа"""
    try:
        df = get_ohlcv(symbol)
        if df.empty:
            return None, symbol
        
        df = analyze(df)
        if df.empty:
            return None, symbol
        
        signals = check_signals(df, symbol)
        price = df['close'].iloc[-1]
        time = df['timestamp'].iloc[-1]
        
        # Расчёт адаптивных целей по ATR и волатильности
        atr = df['atr'].iloc[-1]
        if not pd.isna(atr) and price > 0:
            tp, sl = calculate_tp_sl(df, price, atr)
            adaptive_targets[symbol] = {'tp': tp, 'sl': sl}
        else:
            tp, sl = TP_MIN, SL_MIN
            adaptive_targets[symbol] = {'tp': tp, 'sl': sl}
        
        # Проверка на открытые сделки (перенесено в monitor_open_positions)
        
        return signals, symbol, price, time, df, atr
    except Exception as e:
        logging.error(f"Ошибка обработки {symbol}: {e}")
        return None, symbol

async def main():
    global adaptive_targets
    tz_msk = timezone(timedelta(hours=3))
    last_alive = datetime.now(tz_msk) - timedelta(hours=6)  # timezone-aware
    last_report_hours = set()  # Часы, когда уже был отправлен отчёт (например, {9, 22})
    last_long_signal = datetime.now(tz_msk) - timedelta(days=1)  # timezone-aware
    adaptive_targets = {}  # symbol: {'tp': ..., 'sl': ...}

    # Запускаем Telegram-бота как асинхронную задачу
    asyncio.create_task(telegram_bot())
    
    # Запускаем отдельную задачу для мониторинга открытых позиций
    asyncio.create_task(monitor_open_positions())

    MAX_DD_PCT = 0.03  # 3% дневной просадки
    trading_enabled = True
    last_dd_check = None

    def get_daily_drawdown():
        # Считаем просадку за последние сутки
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)
        profit = 0
        for symbol, trades in virtual_portfolio.items():
            if symbol == 'open_trades':
                continue
            last_buy = None
            for trade in trades:
                t = datetime.strptime(trade['time'], '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
                if t < day_ago:
                    continue
                if trade['action'] == 'BUY':
                    last_buy = float(trade['price'])
                elif trade['action'] == 'SELL' and last_buy is not None:
                    profit += float(trade['price']) - last_buy
                    last_buy = None
        return profit

    MAX_LOSSES = 4
    consecutive_losses = 0

    def update_consecutive_losses(pnl):
        global consecutive_losses
        if pnl < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

    while True:
        # Проверка наличия монет
        if not SYMBOLS:
            error_msg = "❗️ Ошибка: список монет для анализа пуст. Проверь подключение к бирже или фильтры."
            print(error_msg)
            await send_telegram_message(error_msg)
            await asyncio.sleep(60 * 10)  # Ждать 10 минут перед повтором
            continue
        signals_sent = False
        processed_symbols = []
        
        # Асинхронная обработка всех монет параллельно
        tasks = [process_symbol(symbol) for symbol in SYMBOLS]
        results = await asyncio.gather(*tasks)
        
        # Обработка результатов анализа
        for result in results:
            if result is None or len(result) < 2:
                continue
                
            if len(result) >= 6:
                signals, symbol, price, time, df, atr = result
                processed_symbols.append(symbol)
                
                # Если сигналов нет, переходим к следующей монете
                if not signals:
                    continue
                
                # Сигналы на вход/выход
                tp = adaptive_targets[symbol]['tp'] if symbol in adaptive_targets else 0.02
                sl = adaptive_targets[symbol]['sl'] if symbol in adaptive_targets else 0.02
                tp_price = round(price * (1 + tp), 6)
                sl_price = round(price * (1 - sl), 6)
                msg = f"\n\U0001F4B0 Сигналы для {symbol} на {time.strftime('%d.%m.%Y %H:%M')}:\n" + '\n\n'.join(signals)
                msg += f"\nАдаптивный тейк-профит: +{tp*100:.2f}% ({tp_price}), стоп-лосс: -{sl*100:.2f}% ({sl_price})"
                await send_telegram_message(msg)
                logging.info(f"{symbol}: сигнал отправлен в Telegram")
                signals_sent = True
                
                # Открытие позиций по сигналам
                for s in signals:
                    if 'КУПИТЬ' in s and (symbol not in open_trades or open_trades[symbol]['side'] != 'long'):
                        score = evaluate_signal_strength(df, symbol, 'BUY')[0]  # Получаем только score, без pattern_name
                        record_trade(symbol, 'OPEN', price, time, 'long', score=score)
                        open_trade(symbol, price, time, 'long', atr=atr, score=score)
                        logging.info(f"{symbol}: LONG открыт по цене {price}")
                    if 'ПРОДАТЬ' in s and (symbol not in open_trades or open_trades[symbol]['side'] != 'short'):
                        score = evaluate_signal_strength(df, symbol, 'SELL')[0]  # Получаем только score, без pattern_name
                        record_trade(symbol, 'OPEN', price, time, 'short', score=score)
                        open_trade(symbol, price, time, 'short', atr=atr, score=score)
                        logging.info(f"{symbol}: SHORT открыт по цене {price}")
            else:
                _, symbol = result
                logging.warning(f"Неполный результат для {symbol}, пропускаем")
        # Долгосрочный анализ раз в сутки
        now_utc = datetime.now(timezone.utc)
        now_msk = now_utc.astimezone(tz_msk)
        now = datetime.now(tz_msk)  # timezone-aware now для сравнения с last_long_signal
        if (now - last_long_signal) > timedelta(hours=23):
            for symbol in SYMBOLS:
                try:
                    ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe='1d', limit=400)
                    df_long = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Moscow')
                    df_long = analyze_long(df_long)
                    signals_long = check_signals_long(df_long)
                    if signals_long:
                        msg = f"\n\U0001F4BC Сигнал (долгосрок) для {symbol} на {df_long['timestamp'].iloc[-1].strftime('%d.%m.%Y')}:\n" + '\n\n'.join(signals_long)
                        await send_telegram_message(msg)
                except Exception as e:
                    print(f"Ошибка долгосрок по {symbol}: {e}")
            last_long_signal = now
        # Alive-отчёт раз в 6 часов + список обработанных монет
        if (now_msk - last_alive) > timedelta(hours=6):
            msg = f"⏳ Бот работает, обновил данные на {now_msk.strftime('%d.%m.%Y %H:%M')}\n"
            msg += f"Обработано монет: {len(processed_symbols)}\n"
            msg += ', '.join(processed_symbols) if processed_symbols else 'Монеты не обработаны.'
            if not signals_sent:
                msg += "\nСигналов нет."
            await send_telegram_message(msg)
            last_alive = now_msk
        # Ежедневный отчёт в 9:00 и 22:00 по Москве
        report_hours = [9, 22]
        current_hour = now_msk.hour
        if current_hour in report_hours and current_hour not in last_report_hours:
            await send_daily_report()
            last_report_hours = {current_hour}  # Сбросить, чтобы не было дублирования в этом часу
        if current_hour not in report_hours:
            last_report_hours = set()  # Обнуляем, чтобы в следующий раз снова отправить
        await asyncio.sleep(60 * 5)  # Проверять каждые 5 минут как раньше

# Функция для расчёта winrate по score на истории
score_history_stats = {}
def get_score_winrate(score, action):
    key = f'{score}_{action}'
    if key in score_history_stats:
        return score_history_stats[key]
    # Считаем по виртуальному портфелю
    total, success = 0, 0
    for symbol, trades in virtual_portfolio.items():
        if symbol == 'open_trades':
            continue
        last_buy = None
        last_score = None
        for trade in trades:
            if 'score' in trade:
                last_score = trade['score']
            if trade['action'] == 'BUY':
                last_buy = float(trade['price'])
                last_score = trade.get('score', None)
            elif trade['action'] == 'SELL' and last_buy is not None and last_score == score:
                total += 1
                if (float(trade['price']) > last_buy and action == 'BUY') or (float(trade['price']) < last_buy and action == 'SELL'):
                    success += 1
                last_buy = None
                last_score = None
    percent = (success / total * 100) if total > 0 else None
    score_history_stats[key] = percent
    return percent

def calculate_risk_params():
    """Анализ общей волатильности рынка (BTC/USDT) и динамическая настройка TP/SL и размера позиции"""
    try:
        btc_ohlcv = EXCHANGE.fetch_ohlcv('BTC/USDT:USDT', TIMEFRAME, limit=100)
        btc_df = pd.DataFrame(btc_ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        btc_df['returns'] = btc_df['c'].pct_change()
        market_volatility = btc_df['returns'].std() * math.sqrt(365)
        if market_volatility > 0.8:  # Высокая волатильность
            return {
                'tp_mult': 1.8,
                'sl_mult': 1.2,
                'position_size': 0.02  # 2% депозита
            }
        elif market_volatility < 0.4:  # Низкая волатильность
            return {
                'tp_mult': 3.0,
                'sl_mult': 2.0,
                'position_size': 0.05  # 5% депозита
            }
        else:  # Средняя волатильность
            return {
                'tp_mult': 2.5,
                'sl_mult': 1.8,
                'position_size': 0.03  # 3% депозита
            }
    except Exception as e:
        logging.error(f"Ошибка при расчёте рыночной волатильности: {e}")
        return {
            'tp_mult': 2.5,
            'sl_mult': 1.8,
            'position_size': 0.03
        }

def find_support_resistance(df, window=20):
    """
    Находит ближайшие уровни поддержки и сопротивления по локальным экстремумам за window свечей.
    Возвращает (support, resistance)
    """
    closes = df['close']
    lows = df['low']
    highs = df['high']
    last_close = closes.iloc[-1]
    # Поддержка — ближайший минимум ниже текущей цены
    support = lows.iloc[-window:].min()
    if support >= last_close:
        support = lows.iloc[:-1][lows.iloc[:-1] < last_close].max() if (lows.iloc[:-1] < last_close).any() else None
    # Сопротивление — ближайший максимум выше текущей цены
    resistance = highs.iloc[-window:].max()
    if resistance <= last_close:
        resistance = highs.iloc[:-1][highs.iloc[:-1] > last_close].min() if (highs.iloc[:-1] > last_close).any() else None
    return support, resistance

def calculate_tp_sl(df, price, atr):
    """
    Усовершенствованный расчет TP/SL для 15-минутного таймфрейма:
    - Множители зависят от ADX и волатильности (ширины BB)
    - RR адаптируется к текущей волатильности
    - Учитывается импульс цены
    - Минимальный SL 0.008, TP 0.01
    """
    last = df.iloc[-1]
    adx = last['adx']
    
    # Учитываем волатильность на 15-минутном таймфрейме через ширину BB
    bb_width = last['bb_width'] if 'bb_width' in last else 0.02
    momentum = abs(last['momentum']) if 'momentum' in last else 0
    
    # Базовые множители на основе ADX
    if adx > 30:
        tp_mult = 2.2
        sl_mult = 1.1
    elif adx > 22:
        tp_mult = 1.8
        sl_mult = 1.0
    else:
        tp_mult = 1.5
        sl_mult = 1.0
    
    # Корректируем множители на основе волатильности BB и импульса
    # Для высокой волатильности увеличиваем TP и SL
    if bb_width > 0.05:  # Широкие полосы = высокая волатильность
        tp_mult *= 1.2
        sl_mult *= 1.1
    elif bb_width < 0.02:  # Узкие полосы = низкая волатильность
        tp_mult *= 0.9
        sl_mult *= 0.9
    
    # Учитываем силу импульса для TP
    if momentum > 1.0:  # Сильный импульс
        tp_mult *= 1.1  # Увеличиваем TP при сильном импульсе
    
    # Расчет TP и SL с учетом всех факторов
    tp = max(round((atr * tp_mult) / price, 4), 0.01)
    sl = max(round((atr * sl_mult) / price, 4), 0.008)
    
    # Обеспечиваем минимальное соотношение риск/доходность
    min_rr = 1.5
    if momentum > 1.5:
        min_rr = 2.0  # При сильном импульсе требуем лучшего R:R
    
    if tp / sl < min_rr:
        tp = sl * min_rr
    
    return tp, sl

def check_tp_sl(symbol, price, time, df):
    global adaptive_targets
    if symbol not in open_trades:
        return False
    
    trade = open_trades[symbol]
    side = trade['side']
    entry = trade['entry_price']
    score = trade.get('score', None)
    
    # Получаем или рассчитываем TP/SL
    if symbol in adaptive_targets:
        tp = adaptive_targets[symbol]['tp'] 
        sl = adaptive_targets[symbol]['sl']
    else:
        # Рассчитываем ATR
        if 'atr' in trade and trade['atr'] > 0:
            atr = trade['atr']
        else:
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else price * 0.01
            
        # Рассчитываем TP/SL с учетом score
        tp, sl = calculate_tp_sl(df, price, atr)
        adaptive_targets[symbol] = {'tp': tp, 'sl': sl}
    
    # Для long
    if side == 'long':
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)
        
        # Проверка достижения TP или SL
        if price >= tp_price or price <= sl_price:
            reason = 'TP' if price >= tp_price else 'SL'
            result = 'УДАЧНО' if reason == 'TP' else 'НЕУДАЧНО'
            pnl_pct = ((price - entry) / entry) * 100
            
            msg = f"{symbol} {side.upper()} закрыт по {reason}: вход {entry}, выход {price}, P&L: {pnl_pct:.2f}%, результат: {result}"
            asyncio.create_task(send_telegram_message(msg))
            
            # Записываем результат в портфель
            record_trade(symbol, 'CLOSE', price, time, side, score)
            close_trade(symbol)
            return True
    
    # Для short
    elif side == 'short':
        tp_price = entry * (1 - tp)  # Для SHORT TP ниже входа
        sl_price = entry * (1 + sl)  # Для SHORT SL выше входа
        
        # Проверка достижения TP или SL
        if price <= tp_price or price >= sl_price:
            reason = 'TP' if price <= tp_price else 'SL'
            result = 'УДАЧНО' if reason == 'TP' else 'НЕУДАЧНО'
            pnl_pct = ((entry - price) / entry) * 100
            
            msg = f"{symbol} {side.upper()} закрыт по {reason}: вход {entry}, выход {price}, P&L: {pnl_pct:.2f}%, результат: {result}"
            asyncio.create_task(send_telegram_message(msg))
            
            # Записываем результат в портфель
            record_trade(symbol, 'CLOSE', price, time, side, score)
            close_trade(symbol)
            return True
    
    return False

def simple_stats():
    """
    Формирует простую статистику: для каждой монеты — список завершённых сделок с результатом (УДАЧНО/НЕУДАЧНО),
    внизу — общий итог по удачным и неудачным сделкам.
    """
    report = []
    total_win = 0
    total_loss = 0
    
    for symbol, trades in virtual_portfolio.items():
        if symbol == 'open_trades':
            continue
            
        # Группируем сделки по парам открытие-закрытие
        symbol_trades = []
        open_trade = None
        
        for trade in trades:
            # Проверяем, есть ли информация о типе операции
            operation = trade.get('operation', None)
            
            # Если нет явного указания операции, определяем по действию (старый формат)
            if operation is None:
                if trade['action'] == 'BUY' and (open_trade is None or open_trade['action'] == 'SELL'):
                    open_trade = trade
                elif trade['action'] == 'SELL' and open_trade is not None and open_trade['action'] == 'BUY':
                    # Закрытие long позиции
                    symbol_trades.append((open_trade, trade))
                    open_trade = None
                elif trade['action'] == 'SELL' and open_trade is None:
                    open_trade = trade
                elif trade['action'] == 'BUY' and open_trade is not None and open_trade['action'] == 'SELL':
                    # Закрытие short позиции
                    symbol_trades.append((open_trade, trade))
                    open_trade = None
            else:
                # Новый формат с явным указанием операции
                if operation == 'OPEN':
                    open_trade = trade
                elif operation == 'CLOSE' and open_trade is not None:
                    symbol_trades.append((open_trade, trade))
                    open_trade = None
        
        # Анализируем завершенные сделки
        for open_trade, close_trade in symbol_trades:
            entry = float(open_trade['price'])
            exit = float(close_trade['price'])
            side = open_trade['side'].upper()
            
            # Определяем результат сделки
            if side == 'LONG':
                pnl = (exit - entry) / entry
                result = 'УДАЧНО' if exit > entry else 'НЕУДАЧНО'
            else:  # SHORT
                pnl = (entry - exit) / entry
                result = 'УДАЧНО' if exit < entry else 'НЕУДАЧНО'
            
            if result == 'УДАЧНО':
                total_win += 1
            else:
                total_loss += 1
                
            pnl_pct = pnl * 100
            report.append(f"{symbol}: {side}, вход {entry}, выход {exit}, P&L: {pnl_pct:.2f}%, результат: {result}")
    
    # Добавляем общую статистику
    if total_win + total_loss > 0:
        winrate = (total_win / (total_win + total_loss)) * 100
        report.append(f"\nВсего удачных: {total_win}")
        report.append(f"Всего неудачных: {total_loss}")
        report.append(f"Винрейт: {winrate:.1f}%")
    else:
        report.append("\nНет завершённых сделок.")
    
    return report, total_win, total_loss

logging.basicConfig(level=logging.ERROR,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ])
error_handler = logging.FileHandler('bot_error.log', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
logging.getLogger().addHandler(error_handler)

def analyze_long(df):
    """Долгосрочный анализ: EMA50/200, MACD, RSI на дневках."""
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=200)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Moscow')
    return df

if __name__ == '__main__':
    asyncio.run(main()) 