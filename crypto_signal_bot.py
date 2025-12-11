import ccxt
import pandas as pd
import ta
import asyncio
from telegram import Bot
import os
import json
from datetime import datetime, timedelta, timezone
import time
from telegram.ext import Application, CommandHandler
import logging
from collections import defaultdict
from config import *
import warnings

# Подавляем RuntimeWarnings от библиотеки TA (деление на ноль в некоторых индикаторах)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')

# ========== НАСТРОЙКИ ==========

EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'  # Используем фьючерсный рынок (USDT perpetual)
    }
})

MARKET_SNAPSHOT_CACHE = {}
MARKET_SNAPSHOT_TTL = MARKET_SNAPSHOT_TTL_SECONDS
GLOBAL_MARKET_STATE = {'ts': 0.0, 'adx': None, 'funding': 0.0}
HIGHER_TF_CACHE = {}

# РЕКОМЕНДОВАННЫЙ СПИСОК МОНЕТ (обновлён 2025-11-08 по ликвидности Bybit)
TOP_SYMBOLS = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'SOL/USDT:USDT',
    'XRP/USDT:USDT',
    'FIL/USDT:USDT',
    'DOGE/USDT:USDT',
    'NEAR/USDT:USDT',
    'ICP/USDT:USDT',
    'ZEC/USDT:USDT',
    'LTC/USDT:USDT',
    'SUI/USDT:USDT',
    'ADA/USDT:USDT',
    'BNB/USDT:USDT',
    'DOT/USDT:USDT',
    'LINK/USDT:USDT',
    'ENA/USDT:USDT',
    'STRK/USDT:USDT',
    'WIF/USDT:USDT',
    'ETC/USDT:USDT',
    'AVAX/USDT:USDT',
    'ORDI/USDT:USDT',
    'TAO/USDT:USDT',
    'AIA/USDT:USDT',
    'MMT/USDT:USDT',
    '1000PEPE/USDT:USDT',
    'SEI/USDT:USDT',
    'TIA/USDT:USDT',
    'ARB/USDT:USDT',
    'INJ/USDT:USDT',
    'APT/USDT:USDT',
    'MNT/USDT:USDT',
    'WLD/USDT:USDT',
    'AAVE/USDT:USDT'
]
markets = EXCHANGE.load_markets()
# Фильтруем только те пары, которые есть на фьючерсах (swap) и активны
SYMBOLS = [symbol for symbol in TOP_SYMBOLS if symbol in markets and markets[symbol]['active'] and markets[symbol]['type'] == 'swap']
logging.info(f"Загружено {len(SYMBOLS)} фьючерсных символов")

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
    # Валидация входных параметров
    if not symbol or action not in ['OPEN', 'CLOSE'] or side not in ['long', 'short']:
        logging.error(f"Неверные параметры для record_trade: {symbol}, {action}, {side}")
        return
    
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
        'price': float(price),  # Убеждаемся что price - число
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'operation': action  # Добавляем информацию о типе операции
    }
    
    # Добавляем оценку силы сигнала, если есть
    if score is not None:
        trade['score'] = float(score)
    
    # Добавляем сделку в портфель
    virtual_portfolio[symbol].append(trade)
    save_portfolio()
    
    # Логируем информацию о сделке
    logging.info(f"Записана сделка: {symbol} {action} {side} по цене {price} в {time} (score: {score})")

# Открытие сделки
def open_trade(symbol, price, time, side, atr=None, score=None, position_size=0.03):
    open_trades[symbol] = {
        'side': side,  # 'long' или 'short'
        'entry_price': price,
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'atr': atr if atr is not None else 0,
        'trail_pct': 7.3,  # Не используется в оптимизированной логике
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
                
                # Расчет P&L в зависимости от направления позиции
                if side == 'long':
                    # Для LONG: прибыль когда цена выхода выше входа
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:  # short
                    # Для SHORT: прибыль когда цена выхода ниже входа
                    pnl_pct = (entry_price - exit_price) / entry_price
                
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
    for attempt in range(3):
        try:
            ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
            if not ohlcv or len(ohlcv) < MA_SLOW:
                logging.warning(f"{symbol}: недостаточно данных для анализа")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Moscow')
            # Новый столбец: объём в USDT
            df['volume_usdt'] = df['volume'] * df['close']
            return df
        except ccxt.RateLimitExceeded as e:
            wait_time = getattr(e, 'retry_after', 1)
            logging.warning(f"Rate limit exceeded for {symbol}, жду {wait_time} сек.")
            time.sleep(wait_time)
        except ccxt.NetworkError as e:
            logging.error(f"Network error for {symbol}: {e}")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Ошибка получения OHLCV по {symbol}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def get_higher_tf_context(symbol):
    """
    Получает контекст старшего таймфрейма для подтверждения направления сделки.
    Возвращает словарь с bias: long/short/neutral.
    """
    now = time.time()
    cached = HIGHER_TF_CACHE.get(symbol)
    if cached and (now - cached['ts']) < HIGHER_TF_CACHE_SECONDS:
        return cached['data']

    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=HIGHER_TIMEFRAME, limit=HIGHER_TF_MIN_BARS)
        if not ohlcv or len(ohlcv) < HIGHER_TF_MIN_BARS:
            logging.info(f"{symbol}: недостаточно данных старшего ТФ ({HIGHER_TIMEFRAME})")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Moscow')
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=HIGHER_TF_EMA_FAST)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=HIGHER_TF_EMA_SLOW)
        df['rsi'] = ta.momentum.rsi(df['close'], window=HIGHER_TF_RSI_WINDOW)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=ADX_WINDOW)
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            return None

        last = df.iloc[-1]
        bias = 'neutral'
        if last['ema_fast'] > last['ema_slow'] and last['rsi'] >= HIGHER_TF_RSI_BULL and last['adx'] >= HIGHER_TF_ADX_MIN:
            bias = 'long'
        elif last['ema_fast'] < last['ema_slow'] and last['rsi'] <= HIGHER_TF_RSI_BEAR and last['adx'] >= HIGHER_TF_ADX_MIN:
            bias = 'short'

        context = {
            'bias': bias,
            'timestamp': last['timestamp'],
            'adx': float(last['adx']),
            'rsi': float(last['rsi'])
        }
        HIGHER_TF_CACHE[symbol] = {'ts': now, 'data': context}
        return context

    except ccxt.RateLimitExceeded as e:
        wait_time = getattr(e, 'retry_after', 1)
        logging.warning(f"Rate limit higher TF {symbol}, жду {wait_time} сек.")
        time.sleep(wait_time)
        return cached['data'] if cached else None
    except Exception as e:
        logging.error(f"Ошибка получения старшего ТФ {symbol}: {e}")
        return cached['data'] if cached else None

def analyze(df):
    """ОПТИМИЗИРОВАННЫЙ анализ для 15-минутных фьючерсов с современными настройками 2025."""
    try:
        if df.empty or len(df) < MA_SLOW:
            return pd.DataFrame()
        
        # EMA с обновленными периодами
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=MA_FAST)  # 9
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=MA_SLOW)  # 21
        
        # MACD через класс ta.trend.MACD (используем правильные параметры как в оптимизаторе)
        macd_obj = ta.trend.MACD(
            close=df['close'],
            window_slow=MACD_SLOW,
            window_fast=MACD_FAST,
            window_sign=MACD_SIGNAL
        )
        df['macd_line'] = macd_obj.macd()
        df['macd_signal'] = macd_obj.macd_signal()
        df['macd'] = macd_obj.macd_diff()  # гистограмма
        df['macd_hist'] = macd_obj.macd_diff()  # для совместимости с оптимизатором
        
        # RSI с оптимизированным окном
        df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_WINDOW)  # 9
        
        # Stochastic RSI для дополнительного подтверждения
        stoch_rsi = ta.momentum.stochrsi(df['close'], window=STOCH_RSI_LENGTH, smooth1=STOCH_RSI_K, smooth2=STOCH_RSI_D)
        df['stoch_rsi_k'] = stoch_rsi * 100  # Приводим к шкале 0-100
        
        # ADX для определения силы тренда
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        # ATR для расчёта TP/SL
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=ATR_WINDOW)
        
        # Bollinger Bands с новыми настройками
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=BB_WINDOW, window_dev=BB_STD_DEV)
        df['bollinger_mid'] = bb_indicator.bollinger_mavg()
        df['bollinger_high'] = bb_indicator.bollinger_hband()
        df['bollinger_low'] = bb_indicator.bollinger_lband()
        df['bb_width'] = (df['bollinger_high'] - df['bollinger_low']) / df['bollinger_mid']
        
        # VWAP (критически важен для 15м)
        if USE_VWAP:
            # Простой расчет VWAP
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['vwap_numerator'] = (df['typical_price'] * df['volume']).cumsum()
            df['vwap_denominator'] = df['volume'].cumsum()
            df['vwap'] = df['vwap_numerator'] / df['vwap_denominator']
            df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Объём с улучшенной фильтрацией (как в оптимизаторе)
        if USE_VOLUME_FILTER:
            df['volume_ma_usdt'] = df['volume_usdt'].rolling(window=20).mean()
            df['volume_ratio_usdt'] = df['volume_usdt'] / df['volume_ma_usdt']
        
        # Волатильность за последние периоды
        df['volatility'] = df['close'].rolling(window=VOLATILITY_LOOKBACK).std() / df['close'].rolling(window=VOLATILITY_LOOKBACK).mean()
        
        # Спред и импульс
        df['spread_pct'] = (df['high'] - df['low']) / df['low']
        df['momentum'] = df['close'].pct_change(5) * 100  # 5 свечей назад
        
        # Дополнительные индикаторы для адаптивной системы
        # Trending vs Ranging market detection
        df['ema_slope'] = df['ema_slow'].pct_change(3) * 100  # Наклон EMA
        
        # Williams %R для дополнительного подтверждения
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        
        # Очистка данных
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < 2:
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        logging.error(f"Ошибка в анализе данных: {e}")
        return pd.DataFrame()

# ========== ОЦЕНКА СИЛЫ СИГНАЛА ПО ГРАФИКУ ==========
def evaluate_signal_strength(df, symbol, action):
    """СИНХРОНИЗИРОВАННАЯ с оптимизатором оценка силы сигнала - используем те же параметры."""
    try:
        if df.empty or len(df) < 5:
            return 0, None
            
        score = 0
        last = df.iloc[-1]
        prev = df.iloc[-2]
        # Время последней свечи (UTC) для синхронизации с оптимизатором
        try:
            last_time = last.get('timestamp') if isinstance(last, pd.Series) else None
        except Exception:
            last_time = None
        if last_time is not None:
            try:
                current_time_utc = last_time.tz_convert(timezone.utc)
            except Exception:
                try:
                    current_time_utc = last_time.tz_localize(timezone.utc)
                except Exception:
                    current_time_utc = datetime.now(timezone.utc)
        else:
            current_time_utc = datetime.now(timezone.utc)
        prev2 = df.iloc[-3] if len(df) > 3 else prev
        
        # Определяем текущую волатильность для адаптации
        current_volatility = last.get('volatility', 0.02)
        is_high_vol = current_volatility > HIGH_VOLATILITY_THRESHOLD
        is_low_vol = current_volatility < LOW_VOLATILITY_THRESHOLD
        
        # Адаптируем пороги в зависимости от времени
        # Используем время последней свечи, если доступно (важно для backtest/оптимизатора)
        try:
            last_time = last.get('timestamp') if isinstance(last, pd.Series) else None
        except Exception:
            last_time = None
        if last_time is not None:
            try:
                now_utc = last_time.tz_convert(timezone.utc)
            except Exception:
                try:
                    now_utc = last_time.tz_localize(timezone.utc)
                except Exception:
                    now_utc = datetime.now(timezone.utc)
        else:
            now_utc = datetime.now(timezone.utc)
        is_active_hour = now_utc.hour in ACTIVE_HOURS_UTC
        
        # СИНХРОНИЗАЦИЯ: Менее строгие условия как в оптимизаторе
        
        # 1. RSI анализ (более мягкие условия как в оптимизаторе)
        rsi_score = 0
        rsi_momentum = last['rsi'] - prev['rsi']
        
        if action == 'BUY':
            # Более мягкие условия для BUY как в оптимизаторе
            if last['rsi'] < RSI_EXTREME_OVERSOLD:  # Убираем требование momentum
                rsi_score = 3.0  # Возвращаем высокий балл
            elif last['rsi'] < RSI_MIN:  # Упрощаем условие
                rsi_score = 2.5  # Высокий балл за oversold
            elif RSI_MIN < last['rsi'] < 50:  # Расширяем диапазон
                rsi_score = 1.5  # Средний балл за умеренные значения
            elif last['rsi'] > RSI_MAX:  # Уменьшаем штраф
                rsi_score = -0.5  # Небольшой штраф
                
        elif action == 'SELL':
            # Более мягкие условия для SELL как в оптимизаторе
            if last['rsi'] > RSI_EXTREME_OVERBOUGHT:  # Убираем требование momentum
                rsi_score = 3.0  # Возвращаем высокий балл
            elif last['rsi'] > RSI_MAX:  # Упрощаем условие
                rsi_score = 2.5  # Высокий балл за overbought
            elif 50 < last['rsi'] < RSI_MAX:  # Расширяем диапазон
                rsi_score = 1.5  # Средний балл за умеренные значения
            elif last['rsi'] < RSI_MIN:  # Уменьшаем штраф
                rsi_score = -0.5  # Небольшой штраф
                
        score += rsi_score * WEIGHT_RSI
        
        # 2. MACD анализ (более мягкие условия как в оптимизаторе)
        macd_score = 0
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_cross = last['macd'] - last['macd_signal']
            prev_macd_cross = prev['macd'] - prev['macd_signal']
            macd_momentum = last['macd'] - prev['macd']
            
            if action == 'BUY':
                # Упрощенные условия как в оптимизаторе
                if macd_cross > 0 and prev_macd_cross <= 0:  # Простой кроссовер
                    macd_score = 4.0  # Высокий балл за кроссовер
                elif macd_cross > 0:  # Просто выше сигнала
                    macd_score = 2.0  # Средний балл
                elif macd_cross > prev_macd_cross:  # Растет к сигналу
                    macd_score = 1.0  # Небольшой балл
                else:
                    macd_score = 0  # Нет штрафа
                    
            elif action == 'SELL':
                # Упрощенные условия как в оптимизаторе
                if macd_cross < 0 and prev_macd_cross >= 0:  # Простой кроссовер
                    macd_score = 4.0  # Высокий балл за кроссовер
                elif macd_cross < 0:  # Просто ниже сигнала
                    macd_score = 2.0  # Средний балл
                elif macd_cross < prev_macd_cross:  # Падает к сигналу
                    macd_score = 1.0  # Небольшой балл
                else:
                    macd_score = 0  # Нет штрафа
        score += macd_score * WEIGHT_MACD
        
        # 3. Bollinger Bands (СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ)
        bb_score = 0
        if 'bollinger_low' in df.columns and 'bollinger_high' in df.columns:
            close = last['close']
            bb_position = (close - last['bollinger_low']) / (last['bollinger_high'] - last['bollinger_low'])
            
            if action == 'BUY':
                # СИНХРОНИЗИРОВАНО: используем те же пороги что и в оптимизаторе
                if bb_position <= 0.05:  # как в оптимизаторе
                    bb_score = 1.5
                elif bb_position <= 0.15:  # как в оптимизаторе
                    bb_score = 1.0
                elif bb_position <= 0.3:  # как в оптимизаторе
                    bb_score = 0.5
            elif action == 'SELL':
                # СИНХРОНИЗИРОВАНО: используем те же пороги что и в оптимизаторе
                if bb_position >= 0.95:  # как в оптимизаторе
                    bb_score = 1.5
                elif bb_position >= 0.85:  # как в оптимизаторе
                    bb_score = 1.0
                elif bb_position >= 0.7:  # как в оптимизаторе
                    bb_score = 0.5
        score += bb_score * WEIGHT_BB
        
        # 4. VWAP анализ (СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ)
        vwap_score = 0
        if USE_VWAP and 'vwap' in df.columns:
            vwap_dev = last.get('vwap_deviation', 0)
            if action == 'BUY':
                # СИНХРОНИЗИРОВАНО: используем те же пороги что и в оптимизаторе
                if vwap_dev <= -VWAP_DEVIATION_THRESHOLD * 1.5:  # как в оптимизаторе
                    vwap_score = 1.5
                elif vwap_dev <= -VWAP_DEVIATION_THRESHOLD:  # как в оптимизаторе
                    vwap_score = 1.0
                elif vwap_dev <= 0:  # как в оптимизаторе
                    vwap_score = 0.3
            elif action == 'SELL':
                # СИНХРОНИЗИРОВАНО: используем те же пороги что и в оптимизаторе
                if vwap_dev >= VWAP_DEVIATION_THRESHOLD * 1.5:  # как в оптимизаторе
                    vwap_score = 1.5
                elif vwap_dev >= VWAP_DEVIATION_THRESHOLD:  # как в оптимизаторе
                    vwap_score = 1.0
                elif vwap_dev >= 0:  # как в оптимизаторе
                    vwap_score = 0.3
        score += vwap_score * WEIGHT_VWAP
        
        # 5. Объём анализ (СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ)
        volume_score = 0
        if USE_VOLUME_FILTER and 'volume_ratio_usdt' in df.columns:
            vol_ratio = last.get('volume_ratio_usdt', 1.0)
            # СИНХРОНИЗИРОВАНО: используем те же пороги что и в оптимизаторе
            if vol_ratio >= 2.0:  # как в оптимизаторе
                volume_score = 1.5
            elif vol_ratio >= 1.5:  # как в оптимизаторе
                volume_score = 1.0
            elif vol_ratio >= 1.2:  # как в оптимизаторе
                volume_score = 0.5
        score += volume_score * WEIGHT_VOLUME
        
        # 6. ADX анализ (СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ)
        adx_score = 0
        # СИНХРОНИЗИРОВАНО: используем упрощенные пороги как в оптимизаторе
        if is_high_vol:
            min_adx = max(HIGH_VOL_ADX_MIN, 5)
        elif is_low_vol:
            min_adx = max(LOW_VOL_ADX_MIN, 5)
        else:
            min_adx = max(MIN_ADX, 5)
        
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
            
        score += adx_score * WEIGHT_ADX
        
        # 7. Дополнительные бонусы (уменьшаем влияние)
        bonus_score = 0
        
        # Convergence/Divergence patterns
        if len(df) >= 10:
            price_trend = df['close'].iloc[-5:].pct_change().sum()
            rsi_trend = df['rsi'].iloc[-5:].pct_change().sum()
            
            # Уменьшаем бонус за дивергенции
            if action == 'BUY' and price_trend < -0.01 and rsi_trend > 0.02:  # Строже
                bonus_score += 0.5  # было 1.0, теперь 0.5
            elif action == 'SELL' and price_trend > 0.01 and rsi_trend < -0.02:  # Строже
                bonus_score += 0.5  # было 1.0, теперь 0.5
        
        # Stochastic RSI (СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ)
        if 'stoch_rsi_k' in df.columns:
            stoch_k = last.get('stoch_rsi_k', 50)
            # СИНХРОНИЗИРОВАНО: используем те же пороги что и в оптимизаторе
            if action == 'BUY' and stoch_k <= 15:  # как в оптимизаторе
                bonus_score += 0.3
            elif action == 'SELL' and stoch_k >= 85:  # как в оптимизаторе
                bonus_score += 0.3
        
        score += bonus_score
        
        # Применяем корректировки для SHORT/LONG из конфигурации (менее агрессивно)
        if action == 'SELL':
            score *= SHORT_BOOST_MULTIPLIER
        
        # Уменьшаем штраф для LONG в нисходящем тренде
        if action == 'BUY' and len(df) >= 10:
            price_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            if price_trend < -0.05:  # Делаем условие строже (было -0.03)
                score *= max(0.8, LONG_PENALTY_IN_DOWNTREND)  # Ограничиваем штраф
        
        # Бонус в активные часы (больше чем раньше)
        if is_active_hour:
            score *= 1.1  # Увеличиваем бонус
        
        # КРИТИЧНО: Убираем большинство штрафующих корректировок
        # Возвращаем более высокие скоры как в оптимизаторе
        final_score = max(0, score)  # Убираем верхний лимит - пусть будет как есть
        
        return final_score, None
        
    except Exception as e:
        logging.error(f"Ошибка в оценке силы сигнала: {e}")
        return 0, None

# ========== ОЦЕНКА СИЛЫ СИГНАЛА ПО ГРАФИКУ ==========
def signal_strength_label(score):
    """
    СИНХРОНИЗИРОВАННАЯ с оптимизатором функция: высокие вероятности для высоких скоров
    
    Возвращает кортеж (метка, вероятность)
    """
    if score >= 15:
        return 'Превосходный', 0.90  # Максимальное качество
    elif score >= 12:
        return 'Отличный', 0.85  # Очень высокое качество
    elif score >= 10:
        return 'Сильный', 0.75  # Высокое качество  
    elif score >= 8:
        return 'Хороший', 0.65  # Выше среднего
    elif score >= 6:
        return 'Умеренный', 0.55  # Средний
    elif score >= 4:
        return 'Слабый', 0.45  # Ниже среднего
    elif score >= 2:
        return 'Очень слабый', 0.35  # Низкое качество
    else:
        return 'Ненадёжный', 0.25  # Критически низкое качество

# ========== РЕКОМЕНДАЦИЯ ПО ПЛЕЧУ ==========
def recommend_leverage(strength_score, history_percent):
    """
    Рекомендует оптимальное плечо на основе:
    1. Силы сигнала по графику
    2. Исторической успешности сигналов по монете
    3. Общей волатильности рынка
    4. Текущего времени (часа)
    
    Возвращает строку с рекомендацией плеча
    """
    # Проверяем общую волатильность рынка через BTC
    try:
        ohlcv_btc = EXCHANGE.fetch_ohlcv('BTC/USDT:USDT', timeframe='15m', limit=20)
        df_btc = pd.DataFrame(ohlcv_btc, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        # Считаем средний размер свечи относительно цены за последние 20 свечей
        btc_volatility = ((df_btc['h'] - df_btc['l']) / df_btc['c']).mean() * 100  # в процентах
        
        # Снижаем плечо при высокой волатильности
        volatility_mult = 1.0
        if btc_volatility > 1.5:  # Очень высокая волатильность
            volatility_mult = 0.5  # Снижаем плечо в 2 раза
        elif btc_volatility > 1.0:  # Высокая волатильность
            volatility_mult = 0.7  # Снижаем плечо на 30%
        elif btc_volatility < 0.5:  # Низкая волатильность
            volatility_mult = 1.2  # Можно увеличить плечо на 20%
    except Exception:
        volatility_mult = 0.8  # При ошибке берем консервативный множитель
    
    # Учитываем время суток (риск выше ночью и в выходные)
    now = datetime.now(timezone.utc)
    hour_moscow = (now.hour + 3) % 24  # Московское время
    time_mult = 1.0
    
    # Снижаем плечо ночью (меньше ликвидности и больше волатильности)
    if 0 <= hour_moscow < 7:
        time_mult = 0.7  # Ночью снижаем плечо на 30%
    elif 22 <= hour_moscow <= 23:
        time_mult = 0.8  # Вечером снижаем плечо на 20%
    
    # Проверяем день недели (снижаем плечо в выходные)
    if now.weekday() >= 5:  # 5=суббота, 6=воскресенье
        time_mult *= 0.8  # В выходные снижаем плечо еще на 20%
    
    # Средняя оценка силы (усредняем score и исторический процент)
    avg_score = (strength_score + (history_percent / 100 * 3)) / 2
    
    # Базовое плечо на основе силы сигнала
    if avg_score >= 2.5:
        base_leverage = 10
    elif avg_score >= 2.0:
        base_leverage = 7
    elif avg_score >= 1.5:
        base_leverage = 5
    elif avg_score >= 1.0:
        base_leverage = 3
    else:
        base_leverage = 2
    
    # Применяем корректировки
    final_leverage = int(base_leverage * volatility_mult * time_mult)
    
    # Обеспечиваем минимальное/максимальное значение
    final_leverage = max(1, min(final_leverage, 10))
    
    return f'x{final_leverage}'

# ========== ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ ОБЪЁМА ==========
def get_market_snapshot(symbol):
    now = time.time()
    cached = MARKET_SNAPSHOT_CACHE.get(symbol)
    if cached and (now - cached['ts']) < MARKET_SNAPSHOT_TTL:
        return cached['data']
    try:
        ticker = EXCHANGE.fetch_ticker(symbol)
        MARKET_SNAPSHOT_CACHE[symbol] = {'ts': now, 'data': ticker}
        return ticker
    except ccxt.RateLimitExceeded as e:
        logging.warning(f"Rate limit snapshot {symbol}, жду {getattr(e, 'retry_after', 1)} сек.")
        time.sleep(getattr(e, 'retry_after', 1))
    except ccxt.BaseError as e:
        logging.error(f"Биржевой API snapshot ошибка {symbol}: {e}")
    except Exception as e:
        logging.error(f"Не удалось получить snapshot {symbol}: {e}")
    return {}


def get_24h_volume(symbol):
    try:
        ticker = get_market_snapshot(symbol)
        if not ticker:
            return 0
        volume = ticker.get('quoteVolume') or ticker.get('baseVolume') or 0
        return float(volume)
    except ccxt.RateLimitExceeded as e:
        logging.warning(f"Rate limit exceeded for {symbol}, жду {getattr(e, 'retry_after', 1)} сек.")
        time.sleep(getattr(e, 'retry_after', 1))
        return 0
    except Exception as e:
        logging.error(f"Ошибка получения объёма по {symbol}: {e}")
        return 0

last_signal_time = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))

def get_btc_adx(symbol=GLOBAL_TREND_SYMBOL):
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        return float(df['adx'].iloc[-1])
    except Exception as e:
        logging.error(f"Ошибка получения ADX {symbol}: {e}")
        return None


def get_global_market_state():
    now = time.time()
    if (now - GLOBAL_MARKET_STATE['ts']) < MARKET_SNAPSHOT_TTL:
        return GLOBAL_MARKET_STATE

    adx = get_btc_adx(GLOBAL_TREND_SYMBOL)
    funding = 0.0
    snapshot = get_market_snapshot(GLOBAL_TREND_SYMBOL) if USE_GLOBAL_TREND_FILTER else {}
    info = snapshot.get('info') if isinstance(snapshot, dict) else {}
    if isinstance(info, dict):
        for key in ('funding_rate', 'fundingRate'):
            if key in info:
                try:
                    funding = float(info[key])
                except (TypeError, ValueError):
                    funding = 0.0
                break

    GLOBAL_MARKET_STATE.update({
        'ts': now,
        'adx': adx if adx is not None else 0.0,
        'funding': funding
    })
    return GLOBAL_MARKET_STATE

def check_signals(df, symbol):
    """
    СИНХРОНИЗИРОВАННАЯ с оптимизатором система генерации сигналов.
    Использует точно такую же логику как в optimizer_bot_fixed.py
    """
    try:
        if df.empty or len(df) < MIN_15M_CANDLES:
            return []
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []

        # Синхронизация времени: используем метку времени последней свечи (UTC)
        try:
            last_time = last.get('timestamp') if isinstance(last, pd.Series) else None
        except Exception:
            last_time = None
        if last_time is not None:
            try:
                current_time_utc = last_time.tz_convert(timezone.utc)
            except Exception:
                try:
                    current_time_utc = last_time.tz_localize(timezone.utc)
                except Exception:
                    current_time_utc = datetime.now(timezone.utc)
        else:
            current_time_utc = datetime.now(timezone.utc)
        
        # === БАЗОВЫЕ ФИЛЬТРЫ С ДИАГНОСТИКОЙ ===
        
        # Диагностика: начальные значения
        logging.info(f"🔍 {symbol}: RSI={last['rsi']:.1f}, ADX={last['adx']:.1f}, час_UTC={current_time_utc.hour}")

        market_snapshot = {}
        if MIN_24H_VOLUME_USDT or MAX_FUNDING_RATE_ABS or MAX_SPREAD_PCT:
            market_snapshot = get_market_snapshot(symbol)

        if MIN_24H_VOLUME_USDT:
            volume_24h = 0.0
            if isinstance(market_snapshot, dict):
                raw_volume = market_snapshot.get('quoteVolume') or market_snapshot.get('baseVolume') or 0.0
                try:
                    volume_24h = float(raw_volume)
                except (TypeError, ValueError):
                    volume_24h = 0.0
            if volume_24h < MIN_24H_VOLUME_USDT:
                logging.info(f"🔍 {symbol}: ОТКЛОНЕН по 24h объёму ({volume_24h:,.0f} < {MIN_24H_VOLUME_USDT:,.0f})")
                return []

        if MIN_VOLUME_USDT and last.get('volume_usdt', 0) < MIN_VOLUME_USDT:
            logging.info(f"🔍 {symbol}: ОТКЛОНЕН по объёму свечи ({last.get('volume_usdt', 0):,.0f} < {MIN_VOLUME_USDT:,.0f})")
            return []

        if USE_GLOBAL_TREND_FILTER:
            global_state = get_global_market_state()
            if global_state['adx'] < GLOBAL_MIN_ADX:
                logging.info(f"🔍 {symbol}: ОТКЛОНЕН по глобальному ADX ({global_state['adx']:.1f} < {GLOBAL_MIN_ADX})")
                return []
            if GLOBAL_MAX_ABS_FUNDING and abs(global_state['funding']) > GLOBAL_MAX_ABS_FUNDING:
                logging.info(f"🔍 {symbol}: ОТКЛОНЕН из-за funding BTC ({global_state['funding']:.5f})")
                return []

        if MAX_FUNDING_RATE_ABS and isinstance(market_snapshot, dict):
            funding = 0.0
            info = market_snapshot.get('info', {})
            if isinstance(info, dict):
                for key in ('funding_rate', 'fundingRate'):
                    if key in info:
                        try:
                            funding = float(info[key])
                        except (TypeError, ValueError):
                            funding = 0.0
                        break
            if abs(funding) > MAX_FUNDING_RATE_ABS:
                logging.info(f"🔍 {symbol}: ОТКЛОНЕН по funding ({funding:.5f} > {MAX_FUNDING_RATE_ABS:.5f})")
                return []

        if MAX_SPREAD_PCT:
            spread_pct = None
            if isinstance(market_snapshot, dict):
                bid = market_snapshot.get('bid')
                ask = market_snapshot.get('ask')
                try:
                    bid = float(bid)
                    ask = float(ask)
                except (TypeError, ValueError):
                    bid = ask = None
                if bid and ask:
                    mid = (bid + ask) / 2
                    if mid:
                        spread_pct = (ask - bid) / mid
            if spread_pct is None:
                spread_pct = last.get('spread_pct', 0)
            if spread_pct and spread_pct > MAX_SPREAD_PCT:
                logging.info(f"🔍 {symbol}: ОТКЛОНЕН по спреду ({spread_pct:.4f} > {MAX_SPREAD_PCT:.4f})")
                return []

        if MIN_ATR_PCT:
            atr = last.get('atr')
            if atr and last['close'] > 0:
                atr_pct = atr / last['close']
                if atr_pct < MIN_ATR_PCT:
                    logging.info(f"🔍 {symbol}: ОТКЛОНЕН по ATR ({atr_pct:.4f} < {MIN_ATR_PCT:.4f})")
                    return []
        
        # 3. Проверка Cooldown
        if symbol not in last_signal_time:
            last_signal_time[symbol] = current_time_utc - timedelta(minutes=SIGNAL_COOLDOWN_MINUTES + 1)
        
        if last_signal_time[symbol].tzinfo is None:
            last_signal_time[symbol] = last_signal_time[symbol].replace(tzinfo=timezone.utc)
        
        if current_time_utc - last_signal_time[symbol] < timedelta(minutes=SIGNAL_COOLDOWN_MINUTES):
            logging.info(f"🔍 {symbol}: ОТКЛОНЕН по cooldown")
            return []
        
        # 4. Проверяем открытые позиции
        if symbol in open_trades:
            logging.info(f"🔍 {symbol}: ОТКЛОНЕН - есть открытая позиция")
            return []
        
        # 5. Временные фильтры (как в оптимизаторе)
        hour_utc = current_time_utc.hour
        if hour_utc not in ACTIVE_HOURS_UTC:
            logging.info(f"🔍 {symbol}: ОТКЛОНЕН - неактивные часы UTC {hour_utc}")
            return []
        
        # 6. Базовые фильтры ADX и RSI (как в оптимизаторе)
        if last['adx'] < MIN_ADX:  # 21 из config.py (как в оптимизаторе)
            logging.info(f"🔍 {symbol}: ОТКЛОНЕН по ADX ({last['adx']:.1f} < {MIN_ADX})")
            return []
        
        # ИСПРАВЛЕНО: RSI фильтр не должен блокировать перепроданные/перекупленные состояния
        # Они должны генерировать сигналы, а не отфильтровываться
        # Убираем жёсткий RSI фильтр - логика обрабатывается в триггерах
        
        # 7. RSI экстремальные значения - ИСПРАВЛЕНО! 
        # Экстремальные значения должны генерировать СИЛЬНЫЕ сигналы, а не отфильтровываться
        # Убираем неправильный фильтр - экстремальные RSI обрабатываются в триггерах
        
        # 8-10. Убираем жёсткие фильтры по BB width, телу свечи и wick ratio — оставим их влияние через скоринг
        
        # 11. Volume MA ratio фильтр (теперь в USDT) - ИСПРАВЛЕНО
        volume_ratio = 1.0  # По умолчанию
        if 'volume_ma_usdt' in df.columns:
            volume_ma = last.get('volume_ma_usdt', 0)
            if volume_ma > 0:
                volume_ratio = last['volume_usdt'] / volume_ma
                if volume_ratio < MIN_VOLUME_MA_RATIO:
                    logging.info(f"🔍 {symbol}: ОТКЛОНЕН по объему MA ({volume_ratio:.2f} < {MIN_VOLUME_MA_RATIO})")
                    return []
        elif 'volume_ratio_usdt' in df.columns:
            # Альтернативный способ проверки volume ratio если колонка есть
            volume_ratio = last.get('volume_ratio_usdt', 1.0)
            if volume_ratio < MIN_VOLUME_MA_RATIO:
                logging.info(f"🔍 {symbol}: ОТКЛОНЕН по объему ratio ({volume_ratio:.2f} < {MIN_VOLUME_MA_RATIO})")
                return []
        
        logging.info(f"🔍 {symbol}: Прошел фильтры. Объем_ratio={volume_ratio:.2f}")
        
        # 12-13. Удалены лишние фильтры (консистентность объёма, волатильность RSI)
        
        # === СТАРШИЙ ТАЙМФРЕЙМ ДЛЯ ПОДТВЕРЖДЕНИЯ ===
        higher_tf_bias = 'neutral'
        long_allowed = True
        short_allowed = True
        if REQUIRE_HIGHER_TF:
            higher_context = get_higher_tf_context(symbol)
            if not higher_context:
                logging.info(f"🔍 {symbol}: ОТКЛОНЕН - нет данных {HIGHER_TIMEFRAME} для подтверждения")
                return []
            higher_tf_bias = higher_context.get('bias', 'neutral')
            long_allowed = higher_tf_bias == 'long'
            short_allowed = higher_tf_bias == 'short'
            if higher_tf_bias == 'neutral':
                logging.info(f"🔍 {symbol}: ОТКЛОНЕН - нейтральный тренд на {HIGHER_TIMEFRAME}")
                return []
        
        # === ТРИГГЕРЫ (точно как в оптимизаторе) ===
        buy_triggers = 0
        sell_triggers = 0
        
        # КРИТИЧНО: RSI экстремальные значения дают СИЛЬНЫЕ триггеры (СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ)
        if last['rsi'] <= RSI_EXTREME_OVERSOLD:  # 12 из config.py
            buy_triggers += 2.0  # Очень сильный сигнал покупки
        elif last['rsi'] < RSI_MIN:  # 15 из config.py (как в оптимизаторе)
            buy_triggers += 1.0  # Сильный сигнал покупки
            
        if last['rsi'] >= RSI_EXTREME_OVERBOUGHT:  # 89 из config.py
            sell_triggers += 2.0  # Очень сильный сигнал продажи
        elif last['rsi'] > RSI_MAX:  # 77 из config.py (как в оптимизаторе)
            sell_triggers += 1.0  # Сильный сигнал продажи
        
        # EMA кроссовер (СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ)
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            buy_triggers += 1.5  # Основной триггер для 15м (как в оптимизаторе)
        elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
            buy_triggers += 0.5
            
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            sell_triggers += 1.5  # Основной триггер для 15м (как в оптимизаторе)
        elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
            sell_triggers += 0.5
            
        # MACD (СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ)
        if hasattr(last, 'macd') and hasattr(last, 'macd_signal'):
            if last['macd'] > last['macd_signal']:
                buy_triggers += 0.5
            if last['macd'] < last['macd_signal']:
                sell_triggers += 0.5
                
        # Bollinger Bands (СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ)
        if 'bollinger_low' in df.columns:
            denom = max((last['bollinger_high'] - last['bollinger_low']), 1e-12)
            bb_position = (last['close'] - last['bollinger_low']) / denom
            if bb_position <= 0.25:  # Более строго для 15м (как в оптимизаторе)
                buy_triggers += 0.5
            if bb_position >= 0.75:  # Более строго для 15м (как в оптимизаторе)
                sell_triggers += 0.5
                
        # VWAP триггеры отключены (упрощение и снижение шума)
                
        # Минимальные триггеры (как в оптимизаторе)
        min_triggers = MIN_TRIGGERS_ACTIVE_HOURS if hour_utc in ACTIVE_HOURS_UTC else MIN_TRIGGERS_INACTIVE_HOURS
        
        # === ДИАГНОСТИКА ТРИГГЕРОВ ===
        logging.info(f"🔍 {symbol}: Триггеры - BUY:{buy_triggers:.1f}, SELL:{sell_triggers:.1f}, мин_требуется:{min_triggers:.1f}")
        logging.info(f"🔍 {symbol}: RSI_пороги - LONG_MAX_RSI:{LONG_MAX_RSI}, SHORT_MIN_RSI:{SHORT_MIN_RSI}")
        
        # === ОПРЕДЕЛЕНИЕ ТИПА СИГНАЛА (ПОЛНОСТЬЮ СИНХРОНИЗИРОВАНО С ОПТИМИЗАТОРОМ) ===
        signal_type = None
        # КРИТИЧНО: RSI фильтры применяются ПРИ определении типа сигнала (как в оптимизаторе)
        # СИНХРОНИЗИРОВАНО: используем параметры из config.py как в оптимизаторе
        if buy_triggers >= min_triggers and last['rsi'] <= LONG_MAX_RSI:  # 38 из config.py
            signal_type = 'BUY'
            logging.info(f"🔍 {symbol}: ✅ НАЙДЕН BUY сигнал! RSI={last['rsi']:.1f} <= {LONG_MAX_RSI}")
        elif sell_triggers >= min_triggers and last['rsi'] >= SHORT_MIN_RSI:  # 32 из config.py
            signal_type = 'SELL'
            logging.info(f"🔍 {symbol}: ✅ НАЙДЕН SELL сигнал! RSI={last['rsi']:.1f} >= {SHORT_MIN_RSI}")
        else:
            # Диагностика почему сигнал не найден
            if buy_triggers >= min_triggers:
                logging.info(f"🔍 {symbol}: ❌ BUY отклонен: RSI={last['rsi']:.1f} > {LONG_MAX_RSI}")
            if sell_triggers >= min_triggers:
                logging.info(f"🔍 {symbol}: ❌ SELL отклонен: RSI={last['rsi']:.1f} < {SHORT_MIN_RSI}")
            if buy_triggers < min_triggers and sell_triggers < min_triggers:
                logging.info(f"🔍 {symbol}: ❌ Недостаточно триггеров для любого сигнала")
        
        # Подтверждение направления по старшему таймфрейму
        if signal_type == 'BUY' and not long_allowed:
            logging.info(f"🔍 {symbol}: ❌ BUY отклонен: старший ТФ {HIGHER_TIMEFRAME} в сторону {higher_tf_bias}")
            signal_type = None
        if signal_type == 'SELL' and not short_allowed:
            logging.info(f"🔍 {symbol}: ❌ SELL отклонен: старший ТФ {HIGHER_TIMEFRAME} в сторону {higher_tf_bias}")
            signal_type = None
        
        # MACD Histogram фильтр (как в оптимизаторе)
        if signal_type and REQUIRE_MACD_HISTOGRAM_CONFIRMATION and 'macd_hist' in df.columns and len(df) > 1:
            current_hist = last['macd_hist']
            prev_hist = df['macd_hist'].iloc[-2]
            if signal_type == 'BUY' and not (current_hist > 0 and prev_hist <= 0):
                logging.info(f"🔍 {symbol}: ❌ BUY отклонен по MACD Histogram")
                return []
            elif signal_type == 'SELL' and not (current_hist < 0 and prev_hist >= 0):
                logging.info(f"🔍 {symbol}: ❌ SELL отклонен по MACD Histogram")
                return []
        
        # Дополнительные условия для short (как в оптимизаторе)
        # RSI проверки уже применены при определении типа сигнала
        if signal_type == 'SELL' and last['adx'] < SHORT_MIN_ADX:  # 23 из config.py (как в оптимизаторе)
            logging.info(f"🔍 {symbol}: ❌ SELL отклонен по SHORT_MIN_ADX ({last['adx']:.1f} < {SHORT_MIN_ADX})")
            return []
        
        # === ГЕНЕРАЦИЯ СИГНАЛА ===
        if signal_type:
            try:
                score, pattern = evaluate_signal_strength(df, symbol, signal_type)
                logging.info(f"🔍 {symbol}: Оценка сигнала {signal_type}: score={score:.2f}, требуется >= {MIN_COMPOSITE_SCORE}")
                if score < MIN_SIGNAL_SCORE_TO_SEND:
                    logging.info(f"🔍 {symbol}: ❌ Сигнал {signal_type} отклонен: score {score:.2f} < {MIN_SIGNAL_SCORE_TO_SEND}")
                    return []
                if score >= MIN_COMPOSITE_SCORE:
                    # Получаем метку силы
                    strength_label, win_prob = signal_strength_label(score)
                    
                    # Рассчитываем TP/SL
                    direction = 'SHORT' if signal_type == 'SELL' else 'LONG'
                    tp_price, sl_price = calculate_tp_sl(df, last['close'], last['atr'], direction)
                    
                    # Удаляем проверку минимальной дистанции TP/SL — минимальные TP/SL уже заданы
                    
                    # Рекомендуем плечо
                    leverage = recommend_leverage(score, win_prob * 100)
                    
                    # Рассчитываем проценты для TP/SL
                    if signal_type == 'BUY':
                        tp_pct = ((tp_price - last['close']) / last['close']) * 100
                        sl_pct = ((last['close'] - sl_price) / last['close']) * 100
                        side = 'long'
                        signal_emoji = "🟢 LONG"
                    else:
                        tp_pct = ((last['close'] - tp_price) / last['close']) * 100
                        sl_pct = ((sl_price - last['close']) / last['close']) * 100
                        side = 'short'
                        signal_emoji = "🔴 SHORT"
                    
                    # Рассчитываем реальное соотношение R:R
                    real_rr = tp_pct / sl_pct if sl_pct > 0 else 0
                    if real_rr < RISK_REWARD_MIN:
                        logging.info(f"🔍 {symbol}: ❌ Сигнал {signal_type} отклонен: R:R {real_rr:.2f} < {RISK_REWARD_MIN}")
                        return []
                    
                    # Составляем сообщение
                    signal = f"{signal_emoji} {symbol}\n"
                    signal += f"Цена: {last['close']:.6f}\n"
                    signal += f"Сила: {strength_label} ({score:.1f})\n"
                    signal += f"Вероятность: {win_prob:.0%}\n"
                    signal += f"TP: +{tp_pct:.2f}% | SL: -{sl_pct:.2f}%\n"
                    signal += f"R:R = {real_rr:.2f}:1\n"
                    signal += f"RSI: {last['rsi']:.1f} | ADX: {last['adx']:.1f}\n"
                    signal += f"Старший ТФ ({HIGHER_TIMEFRAME}): {higher_tf_bias.upper()}\n"
                    
                    # Добавляем детали триггеров
                    triggers = buy_triggers if signal_type == 'BUY' else sell_triggers
                    signal += f"Триггеры: {triggers:.1f}"
                    if USE_VWAP and 'vwap' in df.columns:
                        signal += f" | VWAP: {last.get('vwap_deviation', 0)*100:.1f}%"
                    if 'bb_width' in df.columns:
                        bb_width = (last['bollinger_high'] - last['bollinger_low']) / last['close']
                        signal += f" | BB: {bb_width*100:.1f}%"
                    
                    signals.append(signal)
                    
                    # Открываем виртуальную сделку
                    open_trade(symbol, last['close'], current_time_utc, side, last['atr'], score)
                    record_trade(symbol, 'OPEN', last['close'], current_time_utc, side, score)
                    
                    last_signal_time[symbol] = current_time_utc
                else:
                    logging.info(f"🔍 {symbol}: ❌ Сигнал {signal_type} отклонен по низкому score ({score:.2f} < {MIN_COMPOSITE_SCORE})")
                    
            except Exception as e:
                logging.error(f"Ошибка оценки сигнала {symbol}: {e}")
                return []
        
        return signals
        
    except Exception as e:
        logging.error(f"Ошибка в check_signals для {symbol}: {e}")
        return []

# Функция calculate_rr_ratio удалена - теперь используется реальное соотношение TP/SL

# ========== ОТПРАВКА В TELEGRAM ==========
def ensure_telegram_config():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Не заданы TELEGRAM_TOKEN и TELEGRAM_CHAT_ID в переменных окружения")

async def send_telegram_message(text):
    ensure_telegram_config()
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

# ========== ОБРАБОТЧИКИ КОМАНД ТЕЛЕГРАМ БОТА ==========
async def stats_command(update, context):
    """Показать статистику портфеля"""
    report, win, loss = simple_stats()
    text = '📊 Статистика по виртуальным сделкам:\n'
    if report:
        text += '\n'.join(report)
    else:
        text += 'Нет завершённых сделок.'
    # Разбиваем длинное сообщение на части по 4000 символов (лимит Telegram)
    max_len = 4000
    parts = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    for part in parts:
        await update.message.reply_text(part)

async def del_command(update, context):
    """Очистить весь портфель (сброс к начальному состоянию)"""
    global virtual_portfolio, open_trades, adaptive_targets
    
    # Подсчитываем статистику перед удалением
    report, win, loss = simple_stats()
    total_trades = win + loss
    
    # Очищаем портфель
    virtual_portfolio.clear()
    open_trades.clear()
    adaptive_targets = {}
    virtual_portfolio['open_trades'] = {}
    
    # Сохраняем пустой портфель
    save_portfolio()
    
    text = f"🗑 Портфель полностью очищен!\n\n"
    text += f"📊 Последняя статистика была:\n"
    text += f"• Завершённых сделок: {total_trades}\n"
    text += f"• Удачных: {win}\n"
    text += f"• Неудачных: {loss}\n"
    if total_trades > 0:
        winrate = (win / total_trades) * 100
        text += f"• Винрейт: {winrate:.1f}%"
    
    await update.message.reply_text(text)

async def open_positions_command(update, context):
    """Показать открытые позиции"""
    if not open_trades:
        await update.message.reply_text("📭 Нет открытых позиций")
        return
    
    text = "📈 Открытые позиции:\n\n"
    for symbol, trade in open_trades.items():
        side = trade['side'].upper()
        entry_price = trade['entry_price']
        time_str = trade['time']
        score = trade.get('score', 'N/A')
        
        # Получаем текущую цену
        try:
            df = get_ohlcv(symbol)
            if not df.empty:
                current_price = df['close'].iloc[-1]
                # Расчет текущего P&L
                if side == 'LONG':
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
                text += f"🔹 {symbol}\n"
                text += f"   Направление: {side}\n"
                text += f"   Вход: {entry_price:.6f}\n"
                text += f"   Текущая: {current_price:.6f}\n"
                text += f"   P&L: {pnl_pct:+.2f}%\n"
                text += f"   Время входа: {time_str}\n"
                text += f"   Сила: {score}\n\n"
            else:
                text += f"🔹 {symbol} ({side}) - ошибка получения цены\n\n"
        except Exception as e:
            text += f"🔹 {symbol} ({side}) - ошибка: {str(e)[:50]}\n\n"
    
    # Разбиваем длинное сообщение
    max_len = 4000
    parts = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    for part in parts:
        await update.message.reply_text(part)

async def close_position_command(update, context):
    """Принудительно закрыть позицию по символу"""
    if not context.args:
        await update.message.reply_text("❗️ Укажите символ для закрытия: /close BTCUSDT")
        return
    
    symbol_arg = context.args[0].upper()
    # Ищем символ в открытых позициях
    found_symbol = None
    for symbol in open_trades.keys():
        if symbol_arg in symbol.replace('/', '').replace(':', ''):
            found_symbol = symbol
            break
    
    if not found_symbol:
        await update.message.reply_text(f"❗️ Позиция {symbol_arg} не найдена в открытых позициях")
        return
    
    try:
        # Получаем текущую цену
        df = get_ohlcv(found_symbol)
        if df.empty:
            await update.message.reply_text(f"❗️ Не удалось получить цену для {found_symbol}")
            return
        
        current_price = df['close'].iloc[-1]
        current_time = df['timestamp'].iloc[-1]
        
        trade = open_trades[found_symbol]
        side = trade['side']
        entry_price = trade['entry_price']
        score = trade.get('score')
        
        # Расчет P&L
        if side == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Записываем закрытие
        record_trade(found_symbol, 'CLOSE', current_price, current_time, side, score)
        close_trade(found_symbol)
        
        # Очистка данных
        if found_symbol in adaptive_targets:
            del adaptive_targets[found_symbol]
        
        result = "УДАЧНО" if pnl_pct > 0 else "НЕУДАЧНО"
        text = f"✅ Позиция принудительно закрыта:\n"
        text += f"🔹 {found_symbol} {side.upper()}\n"
        text += f"   Вход: {entry_price:.6f}\n"
        text += f"   Выход: {current_price:.6f}\n"
        text += f"   P&L: {pnl_pct:+.2f}%\n"
        text += f"   Результат: {result}"
        
        await update.message.reply_text(text)
        
    except Exception as e:
        await update.message.reply_text(f"❗️ Ошибка при закрытии позиции: {str(e)}")

async def help_command(update, context):
    """Показать список доступных команд"""
    text = "🤖 Доступные команды:\n\n"
    text += "/stats - 📊 Статистика портфеля\n"
    text += "/positions - 📈 Открытые позиции\n"
    text += "/close <символ> - ❌ Закрыть позицию\n"
    text += "/del - 🗑 Очистить весь портфель\n"
    text += "/status - ⚡️ Статус бота\n"
    text += "/help - ❓ Показать эту справку\n\n"
    text += "Примеры:\n"
    text += "• /close BTCUSDT - закрыть позицию по BTC\n"
    text += "• /close BTC - поиск по частичному совпадению"
    
    await update.message.reply_text(text)

async def status_command(update, context):
    """Показать текущий статус бота"""
    text = "⚡️ Статус крипто-бота:\n\n"
    text += f"🔍 Отслеживается монет: {len(SYMBOLS)}\n"
    text += f"📈 Открытых позиций: {len(open_trades)}\n"
    
    # Показываем последнюю активность
    if virtual_portfolio:
        total_trades = 0
        for symbol, trades in virtual_portfolio.items():
            if symbol != 'open_trades':
                total_trades += len(trades)
        text += f"📊 Всего записей сделок: {total_trades}\n"
    
    # Статус соединения с биржей
    try:
        test_symbol = SYMBOLS[0] if SYMBOLS else 'BTC/USDT:USDT'
        df = get_ohlcv(test_symbol)
        if not df.empty:
            last_update = df['timestamp'].iloc[-1].strftime('%H:%M:%S')
            text += f"🌐 Биржа: ✅ Подключено (обновлено {last_update})\n"
        else:
            text += f"🌐 Биржа: ❌ Проблемы с получением данных\n"
    except:
        text += f"🌐 Биржа: ❌ Ошибка подключения\n"
    
    text += f"💻 Время работы: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    await update.message.reply_text(text)

# ========== ОСНОВНОЙ ЦИКЛ ==========
async def telegram_bot():
    try:
        ensure_telegram_config()
    except Exception as e:
        logging.error(f"Telegram config error: {e}")
        return
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Добавляем обработчики команд
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("del", del_command))
    app.add_handler(CommandHandler("positions", open_positions_command))
    app.add_handler(CommandHandler("close", close_position_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("start", help_command))  # /start показывает справку
    
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
            # НЕ перезаписываем TP/SL для уже открытых позиций
            # calculate_tp_sl вызывается уже в check_tp_sl при необходимости
            pass
        else:
            # Не записываем цели до появления реального сигнала/позиции
            pass
        
        # Проверка на открытые сделки (перенесено в monitor_open_positions)
        
        return signals, symbol, price, time, df, atr
    except Exception as e:
        logging.error(f"Ошибка обработки {symbol}: {e}")
        return None, symbol

async def main():
    global adaptive_targets
    tz_msk = timezone(timedelta(hours=3))
    last_alive = datetime.now(tz_msk) - timedelta(hours=6)
    last_report_hours = set()
    adaptive_targets = {}  # symbol: {'tp': ..., 'sl': ...}

    # Запускаем Telegram-бота как асинхронную задачу
    asyncio.create_task(telegram_bot())
    
    # Запускаем отдельную задачу для мониторинга открытых позиций
    asyncio.create_task(monitor_open_positions())

    trading_enabled = True

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

    while True:
        # Проверка наличия монет
        if not SYMBOLS:
            error_msg = "❗️ Ошибка: список монет для анализа пуст. Проверь подключение к бирже или фильтры."
            logging.error(error_msg)
            await send_telegram_message(error_msg)
            await asyncio.sleep(60 * 10)  # Ждать 10 минут перед повтором
            continue
        signals_sent = False
        processed_symbols = []
        all_current_signals = []  # Собираем все потенциальные сигналы
        
        # Асинхронная обработка всех монет параллельно
        tasks = [process_symbol(symbol) for symbol in SYMBOLS]
        results = await asyncio.gather(*tasks)
        
        # Обработка результатов анализа - СНАЧАЛА СОБИРАЕМ, ПОТОМ ФИЛЬТРУЕМ
        for result in results:
            if result is None or len(result) < 2:
                continue
                
            if len(result) >= 6:
                signals, symbol, price, time, df, atr = result
                processed_symbols.append(symbol)
                
                # Если сигналов нет, переходим к следующей монете
                if not signals:
                    continue
                
                # Получаем правильные TP/SL значения
                direction = 'SHORT' if '🔴 SHORT' in signals[0] else 'LONG'
                if symbol in adaptive_targets:
                    tp_price = adaptive_targets[symbol]['tp']
                    sl_price = adaptive_targets[symbol]['sl']
                else:
                    # Рассчитываем TP/SL правильно
                    tp_price, sl_price = calculate_tp_sl(df, price, atr, direction)
                    adaptive_targets[symbol] = {'tp': tp_price, 'sl': sl_price}
                
                # Рассчитываем проценты для отображения
                if direction == 'LONG':
                    tp_pct = ((tp_price - price) / price) * 100
                    sl_pct = ((price - sl_price) / price) * 100
                else:  # SHORT
                    tp_pct = ((price - tp_price) / price) * 100
                    sl_pct = ((sl_price - price) / price) * 100
                
                # Извлекаем силу сигнала для сортировки
                signal_strength = 0
                try:
                    for signal in signals:
                        if 'Сила:' in signal:
                            strength_line = [line for line in signal.split('\n') if 'Сила:' in line][0]
                            signal_strength = float(strength_line.split('(')[1].split(')')[0])
                            break
                except:
                    signal_strength = 0
                
                # Собираем информацию о сигнале
                signal_info = {
                    'signals': signals,
                    'symbol': symbol,
                    'price': price,
                    'time': time,
                    'df': df,
                    'atr': atr,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'strength': signal_strength,
                    'direction': direction
                }
                all_current_signals.append(signal_info)
            else:
                _, symbol = result
                logging.warning(f"Неполный результат для {symbol}, пропускаем")
        
        # Отправляем все найденные надежные сигналы
        if all_current_signals and trading_enabled:
            # Сортируем по силе сигнала (берем самые сильные первыми)
            all_current_signals.sort(key=lambda x: x['strength'], reverse=True)
            if MAX_SIGNALS_PER_ROUND:
                all_current_signals = all_current_signals[:MAX_SIGNALS_PER_ROUND]
            logging.info(f"Найдено {len(all_current_signals)} надежных сигналов")
            
            # Отправляем в несколько сообщений по группам
            MAX_SIGNALS_PER_MESSAGE = 3  # Только для группировки по длине сообщения
            MAX_MESSAGE_LENGTH = 3500  # Максимальная длина сообщения Telegram
            
            # Разбиваем сигналы на группы только для удобства отправки
            signal_groups = []
            for i in range(0, len(all_current_signals), MAX_SIGNALS_PER_MESSAGE):
                signal_groups.append(all_current_signals[i:i+MAX_SIGNALS_PER_MESSAGE])
            
            # Отправляем ВСЕ группы (убираем ограничение на 3 группы)
            for group_idx, signal_group in enumerate(signal_groups):
                combined_msg = f"💰 Надежные сигналы на {signal_group[0]['time'].strftime('%d.%m.%Y %H:%M')}:\n\n"
                
                for signal_info in signal_group:
                    signals = signal_info['signals']
                    
                    # Добавляем каждый сигнал
                    signal_text = '\n'.join(signals) + "\n"
                    
                    # Проверяем длину сообщения
                    if len(combined_msg + signal_text) > MAX_MESSAGE_LENGTH:
                        # Если текущий сигнал не помещается, отправляем то что есть
                        if len(combined_msg) > 200:  # Если есть что отправить
                            combined_msg += f"\n📊 Всего найдено: {len(all_current_signals)} надежных сигналов"
                            
                            # Добавляем номер группы если групп больше одной
                            if len(signal_groups) > 1:
                                combined_msg = f"📋 Сигналы (часть {group_idx + 1}/{len(signal_groups)}):\n\n" + combined_msg[combined_msg.find('💰'):]
                            
                            # Отправляем накопленное сообщение
                            try:
                                await send_telegram_message(combined_msg)
                                signals_sent = True
                                await asyncio.sleep(1)  # Пауза между сообщениями
                            except Exception as e:
                                logging.error(f"Ошибка отправки группы сигналов {group_idx + 1}: {e}")
                            
                            # Начинаем новое сообщение с текущим сигналом
                            group_idx += 1
                            combined_msg = f"💰 Надежные сигналы (продолжение):\n\n" + signal_text
                        else:
                            break  # Если даже один сигнал не помещается
                    else:
                        combined_msg += signal_text
                    
                    # Позиции уже открыты в check_signals(), не дублируем здесь
                    symbol = signal_info['symbol']
                    direction = signal_info['direction']
                    
                    if symbol in open_trades:
                        logging.info(f"{symbol}: {direction} позиция уже открыта")
                
                # Отправляем последнее накопленное сообщение
                if len(combined_msg) > 200:
                    combined_msg += f"\n📊 Всего найдено: {len(all_current_signals)} надежных сигналов"
                    
                    # Добавляем номер группы если групп больше одной
                    if len(signal_groups) > 1:
                        combined_msg = f"📋 Сигналы (часть {group_idx + 1}/{len(signal_groups)}):\n\n" + combined_msg[combined_msg.find('💰'):]
                    
                    # Отправляем сообщение
                    try:
                        await send_telegram_message(combined_msg)
                        signals_sent = True
                        # Небольшая пауза между сообщениями
                        if group_idx < len(signal_groups) - 1:
                            await asyncio.sleep(1)
                    except Exception as e:
                        logging.error(f"Ошибка отправки группы сигналов {group_idx + 1}: {e}")
                        # Если сообщение все еще слишком длинное, отправляем укороченную версию
                        if "too long" in str(e).lower():
                            short_msg = f"⚡ {len(signal_group)} сигналов на {signal_group[0]['time'].strftime('%H:%M')}:\n"
                            for signal_info in signal_group:
                                symbol = signal_info['symbol']
                                direction = "🟢 LONG" if signal_info['direction'] == 'LONG' else "🔴 SHORT"
                                strength = signal_info['strength']
                                short_msg += f"{direction} {symbol} (сила: {strength:.1f})\n"
                            await send_telegram_message(short_msg)

        # Alive-отчёт раз в 6 часов + список обработанных монет  
        now_utc = datetime.now(timezone.utc)
        now_msk = now_utc.astimezone(tz_msk)
        if (now_msk - last_alive) > timedelta(hours=6):
            msg = f"⏳ Бот работает, обновил данные на {now_msk.strftime('%d.%m.%Y %H:%M')}\n"
            msg += f"Обработано монет: {len(processed_symbols)}\n"
            msg += f"📊 Минимальный порог сигналов: {MIN_COMPOSITE_SCORE} (строго фиксированный)\n"
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

def calculate_tp_sl(df, price, atr, direction='LONG'):
    """
    Расчет TP/SL в стиле оптимизатора: ATR-множители + минимальные проценты из конфигурации.
    Без адаптаций по ADX/волатильности и без ограничений TP_MAX/SL_MAX.
    Возвращает абсолютные цены TP и SL.
    """
    try:
        tp_mult = TP_ATR_MULT
        sl_mult = SL_ATR_MULT

        # Начальные уровни по ATR
        if direction.upper() == 'LONG':
            tp_price_raw = price + atr * tp_mult
            sl_price_raw = price - atr * sl_mult
        else:
            tp_price_raw = price - atr * tp_mult
            sl_price_raw = price + atr * sl_mult

        # Применяем минимальные проценты в терминах цены
        def enforce_min_levels(entry, tp_price, sl_price, side):
            if side == 'LONG':
                tp_eff = max((tp_price - entry) / entry, TP_MIN)
                sl_eff = max((entry - sl_price) / entry, SL_MIN)
                return entry * (1 + tp_eff), entry * (1 - sl_eff)
            else:
                tp_eff = max((entry - tp_price) / entry, TP_MIN)
                sl_eff = max((sl_price - entry) / entry, SL_MIN)
                return entry * (1 - tp_eff), entry * (1 + sl_eff)

        tp_price, sl_price = enforce_min_levels(price, tp_price_raw, sl_price_raw, direction.upper())
        return tp_price, sl_price

    except Exception as e:
        logging.error(f"Ошибка в calculate_tp_sl: {e}")
        # Возвращаем консервативные значения по конфигу
        if direction.upper() == 'LONG':
            return price * (1 + max(TP_MIN, 0.008)), price * (1 - max(SL_MIN, 0.025))
        else:
            return price * (1 - max(TP_MIN, 0.008)), price * (1 + max(SL_MIN, 0.025))

def check_tp_sl(symbol, price, time, df):
    global adaptive_targets
    if symbol not in open_trades:
        return False
    
    trade = open_trades[symbol]
    side = trade['side']
    entry = trade['entry_price']
    score = trade.get('score', None)
    
    # Получаем или рассчитываем TP/SL (теперь это абсолютные цены)
    if symbol in adaptive_targets:
        tp_price = adaptive_targets[symbol]['tp'] 
        sl_price = adaptive_targets[symbol]['sl']
    else:
        # Рассчитываем ATR
        if 'atr' in trade and trade['atr'] > 0:
            atr = trade['atr']
        else:
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else entry * 0.01
            
        # Рассчитываем TP/SL - возвращает абсолютные цены
        direction = 'LONG' if side == 'long' else 'SHORT'
        tp_price, sl_price = calculate_tp_sl(df, entry, atr, direction)
        adaptive_targets[symbol] = {'tp': tp_price, 'sl': sl_price}
    
    # Определяем логику закрытия на основе реального движения цены
    reason = None
    result = None
    
    # Для long позиций
    if side == 'long':
        # LONG: прибыль если цена выше входа, убыток если ниже
        if price >= tp_price:
            reason = 'TP'
            result = 'УДАЧНО'
        elif price <= sl_price:
            reason = 'SL'
            result = 'НЕУДАЧНО'
        else:
            return False  # Цена не достигла ни TP, ни SL
            
        pnl_pct = ((price - entry) / entry) * 100
    
    # Для short позиций
    elif side == 'short':
        # SHORT: прибыль если цена ниже входа, убыток если выше
        if price <= tp_price:
            reason = 'TP'
            result = 'УДАЧНО'
        elif price >= sl_price:
            reason = 'SL'
            result = 'НЕУДАЧНО'
        else:
            return False  # Цена не достигла ни TP, ни SL
            
        pnl_pct = ((entry - price) / entry) * 100
    
    # Если достигнут TP или SL, закрываем позицию
    if reason:
        # Дополнительная проверка корректности результата
        if side == 'long':
            # Для LONG: если цена выше входа - это должно быть успешно
            actual_result = 'УДАЧНО' if price > entry else 'НЕУДАЧНО'
        else:  # short
            # Для SHORT: если цена ниже входа - это должно быть успешно
            actual_result = 'УДАЧНО' if price < entry else 'НЕУДАЧНО'
        
        # Используем фактический результат для определения успешности
        final_result = actual_result
        
        # ИСПРАВЛЕНИЕ: Корректируем отображение результата - если закрыто по TP, то это всегда УДАЧНО
        display_result = final_result
        if reason == 'TP':
            display_result = 'УДАЧНО'  # TP всегда означает успех
        elif reason == 'SL':
            display_result = 'НЕУДАЧНО'  # SL всегда означает убыток
        
        # Добавляем силу сигнала в сообщение, если она есть
        signal_strength_msg = ''
        if score is not None:
            strength_label, win_prob = signal_strength_label(score)
            signal_strength_msg = f"\nСила сигнала: {strength_label} ({score:.1f})"
        
        msg = f"{symbol} {side.upper()} закрыт по {reason}: вход {entry:.6f}, выход {price:.6f}, P&L: {pnl_pct:+.2f}%, результат: {display_result}{signal_strength_msg}"
        asyncio.create_task(send_telegram_message(msg))
        
        # Записываем результат в портфель
        record_trade(symbol, 'CLOSE', price, time, side, score)
        close_trade(symbol)
        
        # Очистка данных после закрытия позиции
        if symbol in adaptive_targets:
            del adaptive_targets[symbol]
            
        return True
    
    return False

def simple_stats():
    """
    Формирует простую статистику: для каждой завершённой сделки — только монета и результат (УДАЧНО/НЕУДАЧНО),
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
            operation = trade.get('operation', None)
            if operation is None:
                if trade['action'] == 'BUY' and (open_trade is None or open_trade['action'] == 'SELL'):
                    open_trade = trade
                elif trade['action'] == 'SELL' and open_trade is not None and open_trade['action'] == 'BUY':
                    symbol_trades.append((open_trade, trade))
                    open_trade = None
                elif trade['action'] == 'SELL' and open_trade is None:
                    open_trade = trade
                elif trade['action'] == 'BUY' and open_trade is not None and open_trade['action'] == 'SELL':
                    symbol_trades.append((open_trade, trade))
                    open_trade = None
            else:
                if operation == 'OPEN':
                    open_trade = trade
                elif operation == 'CLOSE' and open_trade is not None:
                    symbol_trades.append((open_trade, trade))
                    open_trade = None
        # Анализируем завершенные сделки
        for open_trade, close_trade in symbol_trades:
            side = open_trade['side'].upper()
            entry = float(open_trade['price'])
            exit = float(close_trade['price'])
            
            # Расчет P&L в процентах и определение результата
            if side == 'LONG':
                pnl_pct = ((exit - entry) / entry) * 100
                # Для LONG: прибыль если цена выхода выше входа
                result = 'УДАЧНО' if pnl_pct > 0 else 'НЕУДАЧНО'
            else:  # SHORT
                pnl_pct = ((entry - exit) / entry) * 100
                # Для SHORT: прибыль если цена выхода ниже входа
                result = 'УДАЧНО' if pnl_pct > 0 else 'НЕУДАЧНО'
            
            if result == 'УДАЧНО':
                total_win += 1
            else:
                total_loss += 1
            
            # Монета, результат и процент прибыли/убытка
            report.append(f"{symbol}: {result} ({pnl_pct:+.2f}%)")
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

# ========== НАСТРОЙКИ ОПТИМИЗИРОВАНЫ ДЛЯ 70% TP ==========

if __name__ == '__main__':
    asyncio.run(main()) 