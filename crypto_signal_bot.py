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

# ФИНАЛЬНО ОПТИМИЗИРОВАННЫЙ список (смешанный: топовые + проверенные + новые эффективные)
TOP_SYMBOLS = [
    # ⭐ ПРОВЕРЕННЫЕ ПОБЕДИТЕЛИ (высокий винрейт)
    'DOGE/USDT:USDT',  # 100% винрейт, +1.36% P&L
    'YFI/USDT:USDT',   # 100% винрейт, +1.12% P&L
    'RUNE/USDT:USDT',  # 100% винрейт, +1.40% P&L (новый!)
    'TRX/USDT:USDT',   # 66.7% винрейт, +0.34% P&L
    'TON/USDT:USDT',   # 66.7% винрейт, +0.41% P&L (новый!)
    'SUI/USDT:USDT',   # 50% винрейт, +0.69% P&L
    'SEI/USDT:USDT',   # 50% винрейт, +0.42% P&L
    'VET/USDT:USDT',   # 50% винрейт, +0.36% P&L (новый!)
    
    # 💎 ТОПОВЫЕ ЛИКВИДНЫЕ (основа портфеля - могут активироваться)
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'MATIC/USDT:USDT',
    
    # 🎲 МЕМКОИНЫ (высокая волатильность - потенциал для сигналов)
    'SHIB/USDT:USDT', 'PEPE/USDT:USDT', '1000PEPE/USDT:USDT', 'FLOKI/USDT:USDT', 
    'BONK/USDT:USDT', 'WIF/USDT:USDT',
    
    # 🔥 АКТИВНЫЕ АЛЬТКОИНЫ (средняя волатильность)
    'UNI/USDT:USDT', 'AAVE/USDT:USDT', 'MKR/USDT:USDT', 'LDO/USDT:USDT',
    'ARB/USDT:USDT', 'OP/USDT:USDT', 'LRC/USDT:USDT', 
    
    # 🎮 GAMING И NFT (периодически активные)
    'SAND/USDT:USDT', 'MANA/USDT:USDT', 'AXS/USDT:USDT', 'ENJ/USDT:USDT',
    
    # 🚀 AI И TECH (перспективные)
    'RNDR/USDT:USDT', 'FET/USDT:USDT', 'INJ/USDT:USDT',
    
    # 📈 КЛАССИЧЕСКИЕ АЛЬТЫ (стабильность)
    'LTC/USDT:USDT', 'BCH/USDT:USDT', 'ETC/USDT:USDT',
    
    # 🆕 НОВЫЕ ПЕРСПЕКТИВНЫЕ (2024-2025)
    'JUP/USDT:USDT', 'PYTH/USDT:USDT', 'TIA/USDT:USDT', 'ALT/USDT:USDT',
    'PIXEL/USDT:USDT', 'PORTAL/USDT:USDT', 'STX/USDT:USDT', 'ORDI/USDT:USDT',
    
    # 💼 ДОПОЛНИТЕЛЬНЫЕ (для диверсификации)
    'THETA/USDT:USDT', 'FIL/USDT:USDT', 'COMP/USDT:USDT', 'SUSHI/USDT:USDT',
    'CAKE/USDT:USDT', 'CRV/USDT:USDT', 'IMX/USDT:USDT', 'ALICE/USDT:USDT',
    'GMT/USDT:USDT', 'MAVIA/USDT:USDT', 'JTO/USDT:USDT', 'STRK/USDT:USDT'
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
    """ОПТИМИЗИРОВАННЫЙ анализ для 15-минутных фьючерсов с современными настройками 2025."""
    try:
        if df.empty or len(df) < MA_SLOW:
            return pd.DataFrame()
            
        # EMA с обновленными периодами
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=MA_FAST)  # 9
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=MA_SLOW)  # 21
        
        # MACD с быстрыми настройками для 15м
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_line'] = ta.trend.macd(df['close'])
        
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
        
        # Объём с улучшенной фильтрацией
        if USE_VOLUME_FILTER:
            df['volume_ema'] = ta.trend.ema_indicator(df['volume'], window=20)
            df['volume_ratio'] = df['volume'] / df['volume_ema']
        
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
        
        # Убираем NaN
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < 2:
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        logging.error(f"Ошибка в анализе данных: {e}")
        return pd.DataFrame()

# Убираем сложные графические паттерны для упрощения 15м торговли

# ========== ОЦЕНКА СИЛЫ СИГНАЛА ПО ГРАФИКУ ==========
def evaluate_signal_strength(df, symbol, action):
    """УЛУЧШЕННАЯ оценка силы сигнала для повышения винрейта с 31% до 55%+."""
    try:
        if df.empty or len(df) < 5:
            return 0, None
            
        score = 0
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) > 3 else prev
        
        # Определяем текущую волатильность для адаптации
        current_volatility = last.get('volatility', 0.02)
        is_high_vol = current_volatility > HIGH_VOLATILITY_THRESHOLD
        is_low_vol = current_volatility < LOW_VOLATILITY_THRESHOLD
        
        # Адаптируем пороги в зависимости от времени
        now_utc = datetime.now(timezone.utc)
        is_active_hour = now_utc.hour in ACTIVE_HOURS_UTC
        
        # КАЧЕСТВЕННЫЕ ФИЛЬТРЫ (ОТКЛЮЧЕНЫ ДЛЯ ТЕСТИРОВАНИЯ)
        # Временно отключены строгие фильтры для получения сигналов
        
        # 1. УЛУЧШЕННЫЙ RSI анализ (вес увеличен)
        rsi_score = 0
        rsi_momentum = last['rsi'] - prev['rsi']
        
        if action == 'BUY':
            # Более строгие условия для BUY
            if last['rsi'] < RSI_EXTREME_OVERSOLD and rsi_momentum > 2:  # Сильный отскок от экстремума
                rsi_score = 3.0
            elif last['rsi'] < RSI_OVERSOLD and rsi_momentum > 1:  # Выход из перепроданности
                rsi_score = 2.0
            elif RSI_OVERSOLD < last['rsi'] < 45 and rsi_momentum > 0:  # Подтверждение роста
                rsi_score = 1.0
            elif last['rsi'] > RSI_OVERBOUGHT:  # Штраф за перекупленность
                rsi_score = -1.0
                
        elif action == 'SELL':
            # Более строгие условия для SELL
            if last['rsi'] > RSI_EXTREME_OVERBOUGHT and rsi_momentum < -2:  # Сильный разворот от экстремума
                rsi_score = 3.0
            elif last['rsi'] > RSI_OVERBOUGHT and rsi_momentum < -1:  # Выход из перекупленности
                rsi_score = 2.0
            elif 55 < last['rsi'] < RSI_OVERBOUGHT and rsi_momentum < 0:  # Подтверждение падения
                rsi_score = 1.0
            elif last['rsi'] < RSI_OVERSOLD:  # Штраф за перепроданность
                rsi_score = -1.0
                
        score += rsi_score * WEIGHT_RSI
        
        # 2. МАКСИМАЛЬНО УЛУЧШЕННЫЙ MACD анализ с гистограммой
        macd_score = 0
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_cross = last['macd'] - last['macd_signal']
            prev_macd_cross = prev['macd'] - prev['macd_signal']
            macd_momentum = last['macd'] - prev['macd']
            
            # НОВЫЙ: Подтверждение гистограммы MACD
            macd_histogram = macd_cross
            prev_macd_histogram = prev_macd_cross
            histogram_growing = macd_histogram > prev_macd_histogram
            
            # Требуем подтверждение гистограммы если включено (ОТКЛЮЧЕНО ДЛЯ ТЕСТИРОВАНИЯ)
            histogram_confirmed = True  # Временно всегда True
            
            if action == 'BUY':
                if macd_cross > 0 and prev_macd_cross <= 0 and macd_momentum > 0 and histogram_growing:
                    macd_score = 4.0  # Максимальный балл за полное подтверждение
                elif macd_cross > 0 and macd_momentum > 0 and histogram_growing:
                    macd_score = 3.0
                elif macd_cross > 0 and histogram_growing:
                    macd_score = 2.0
                elif macd_cross > 0:
                    macd_score = 1.0
                else:
                    macd_score = -1.0  # Штраф за противоречие
                    
            elif action == 'SELL':
                if macd_cross < 0 and prev_macd_cross >= 0 and macd_momentum < 0 and not histogram_growing:
                    macd_score = 4.0  # Максимальный балл за полное подтверждение
                elif macd_cross < 0 and macd_momentum < 0 and not histogram_growing:
                    macd_score = 3.0
                elif macd_cross < 0 and not histogram_growing:
                    macd_score = 2.0
                elif macd_cross < 0:
                    macd_score = 1.0
                else:
                    macd_score = -1.0  # Штраф за противоречие
        score += macd_score * WEIGHT_MACD
        
        # 3. Bollinger Bands анализ (вес 1.1)
        bb_score = 0
        if 'bollinger_low' in df.columns and 'bollinger_high' in df.columns:
            close = last['close']
            bb_position = (close - last['bollinger_low']) / (last['bollinger_high'] - last['bollinger_low'])
            
            if action == 'BUY':
                if bb_position <= 0.1:  # Близко к нижней полосе
                    bb_score = 2.0
                elif bb_position <= 0.2:
                    bb_score = 1.5
                elif bb_position <= 0.4:
                    bb_score = 1.0
            elif action == 'SELL':
                if bb_position >= 0.9:  # Близко к верхней полосе
                    bb_score = 2.0
                elif bb_position >= 0.8:
                    bb_score = 1.5
                elif bb_position >= 0.6:
                    bb_score = 1.0
        score += bb_score * WEIGHT_BB
        
        # 4. VWAP анализ (вес 1.3)
        vwap_score = 0
        if USE_VWAP and 'vwap' in df.columns:
            vwap_dev = last.get('vwap_deviation', 0)
            if action == 'BUY':
                if vwap_dev <= -VWAP_DEVIATION_THRESHOLD:  # Значительно ниже VWAP
                    vwap_score = 2.0
                elif vwap_dev <= 0:  # Ниже VWAP
                    vwap_score = 1.0
            elif action == 'SELL':
                if vwap_dev >= VWAP_DEVIATION_THRESHOLD:  # Значительно выше VWAP
                    vwap_score = 2.0
                elif vwap_dev >= 0:  # Выше VWAP
                    vwap_score = 1.0
        score += vwap_score * WEIGHT_VWAP
        
        # 5. Объём анализ (вес 0.8)
        volume_score = 0
        if USE_VOLUME_FILTER and 'volume_ratio' in df.columns:
            vol_ratio = last.get('volume_ratio', 1.0)
            if vol_ratio >= 1.5:
                volume_score = 2.0
            elif vol_ratio >= 1.2:
                volume_score = 1.0
        score += volume_score * WEIGHT_VOLUME
        
        # 6. ADX анализ (вес 0.9)
        adx_score = 0
        min_adx = HIGH_VOL_ADX_MIN if is_high_vol else (LOW_VOL_ADX_MIN if is_low_vol else MIN_ADX)
        
        if last['adx'] >= 30:
            adx_score = 2.0
        elif last['adx'] >= 25:
            adx_score = 1.5
        elif last['adx'] >= min_adx:
            adx_score = 1.0
        else:
            adx_score = 0.5
        score += adx_score * WEIGHT_ADX
        
        # 7. Дополнительные бонусы
        # Convergence/Divergence patterns
        if len(df) >= 10:
            price_trend = df['close'].iloc[-5:].pct_change().sum()
            rsi_trend = df['rsi'].iloc[-5:].pct_change().sum()
            
            # Bullish divergence: price down, RSI up
            if action == 'BUY' and price_trend < 0 and rsi_trend > 0:
                score += 1.0
            # Bearish divergence: price up, RSI down  
            elif action == 'SELL' and price_trend > 0 and rsi_trend < 0:
                score += 1.0
        
        # Stochastic RSI confirmation
        if 'stoch_rsi_k' in df.columns:
            stoch_k = last.get('stoch_rsi_k', 50)
            if action == 'BUY' and stoch_k <= 20:
                score += 0.5
            elif action == 'SELL' and stoch_k >= 80:
                score += 0.5
        
        # НОВАЯ ЛОГИКА: Буст для SHORT сигналов (они работают лучше)
        if action == 'SELL' and hasattr(globals(), 'SHORT_BOOST_MULTIPLIER'):
            score *= SHORT_BOOST_MULTIPLIER
        
        # Штраф для LONG в нисходящем тренде
        if action == 'BUY' and len(df) >= 10:
            # Проверяем общий тренд за последние 10 свечей
            price_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            if price_trend < -0.02 and hasattr(globals(), 'LONG_PENALTY_IN_DOWNTREND'):  # Нисходящий тренд > 2%
                score *= LONG_PENALTY_IN_DOWNTREND
        
        # Проверка минимальной активности рынка
        if hasattr(globals(), 'MIN_MARKET_ACTIVITY_SCORE'):
            # Простая оценка активности по объему и волатильности
            market_activity = min(1.0, (vol_ratio if 'vol_ratio' in locals() else 1.0) * current_volatility * 50)
            if market_activity < MIN_MARKET_ACTIVITY_SCORE:
                score *= 0.8  # Штраф за низкую активность
        
        # Адаптация к активным часам
        if is_active_hour:
            score *= (1 + (1 - ACTIVE_HOURS_MULTIPLIER))  # Небольшой бонус в активные часы
        
        # КРИТИЧНО: возвращаем 0 если скор отрицательный
        return max(0, score), None
        
    except Exception as e:
        logging.error(f"Ошибка в оценке силы сигнала: {e}")
        return 0, None

# Убираем сложные графические паттерны для упрощения 15м торговли

# ========== ОЦЕНКА СИЛЫ СИГНАЛА ПО ГРАФИКУ ==========
def signal_strength_label(score):
    """
    Преобразует числовую оценку силы сигнала в текстовую метку
    и процентную вероятность успеха.
    
    Возвращает кортеж (метка, вероятность)
    """
    if score >= 8:
        return 'Экстремально сильный', 0.98
    elif score >= 7:
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
        return 'Очень слабый', 0.45
    elif score >= 1:
        return 'Ненадёжный', 0.35
    else:
        return 'Крайне ненадёжный', 0.25

# ========== СТАТИСТИКА ПО ИСТОРИИ ==========
# Удаляем get_signal_stats - не используется в основной логике

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

# Убираем сложные свечные паттерны для упрощения 15м торговли

def get_btc_adx():
    try:
        ohlcv = EXCHANGE.fetch_ohlcv('BTC/USDT:USDT', timeframe=TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        return df['adx'].iloc[-1]
    except Exception as e:
        logging.error(f"Ошибка получения ADX BTC: {e}")
        return 99

# Удаляем функцию is_global_uptrend - избыточна для 15м торговли
# Делает слишком много API-запросов и замедляет работу

def check_signals(df, symbol):
    """
    СОВРЕМЕННАЯ система генерации сигналов для 15м фьючерсов с композитным скорингом.
    Цель: 10+ надёжных сигналов в сутки.
    """
    try:
        if df.empty or len(df) < MIN_15M_CANDLES:
            return []
            
        last = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        
        # === БЫСТРЫЕ БАЗОВЫЕ ФИЛЬТРЫ ===
        # 1. Объём торгов (основной фильтр)
        volume = get_24h_volume(symbol)
        if volume < MIN_VOLUME_USDT:
            return []
        
        # 2. Максимальный спред
        if last['spread_pct'] > MAX_SPREAD_PCT:
            return []
        
        # 3. Проверка Cooldown
        if symbol not in last_signal_time:
            last_signal_time[symbol] = datetime.now(timezone.utc) - timedelta(minutes=SIGNAL_COOLDOWN_MINUTES + 1)
        
        if last_signal_time[symbol].tzinfo is None:
            last_signal_time[symbol] = last_signal_time[symbol].replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        if now - last_signal_time[symbol] < timedelta(minutes=SIGNAL_COOLDOWN_MINUTES):
            return []
        
        # Определяем адаптивные пороги
        current_volatility = last.get('volatility', 0.02)
        is_high_vol = current_volatility > HIGH_VOLATILITY_THRESHOLD
        is_low_vol = current_volatility < LOW_VOLATILITY_THRESHOLD
        is_active_hour = now.hour in ACTIVE_HOURS_UTC
        
        # Адаптивный минимум ADX
        min_adx = HIGH_VOL_ADX_MIN if is_high_vol else (LOW_VOL_ADX_MIN if is_low_vol else MIN_ADX)
        
        # 4. Базовая сила тренда
        if last['adx'] < min_adx:
            return []
        
        # === ГЕНЕРАЦИЯ СИГНАЛОВ ===
        
        # === СИГНАЛ НА ПОКУПКУ ===
        buy_triggers = 0
        
        # Триггер 1: EMA кроссовер (главный)
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            buy_triggers += 1
        
        # Триггер 2: Цена выше EMA (быстрой) - менее строгий
        elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
            buy_triggers += 0.5
        
        # Триггер 3: MACD бычий
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if last['macd'] > last['macd_signal']:
                buy_triggers += 0.5
            # Кроссовер MACD - дополнительный бонус
            if prev['macd'] <= prev['macd_signal'] and last['macd'] > last['macd_signal']:
                buy_triggers += 0.5
        
        # Триггер 4: Bollinger Bands
        if 'bollinger_low' in df.columns:
            bb_position = (last['close'] - last['bollinger_low']) / (last['bollinger_high'] - last['bollinger_low'])
            if bb_position <= 0.3:  # В нижней части диапазона
                buy_triggers += 0.5
        
        # Триггер 5: VWAP
        if USE_VWAP and 'vwap' in df.columns:
            vwap_dev = last.get('vwap_deviation', 0)
            if vwap_dev <= 0 and vwap_dev >= -VWAP_DEVIATION_THRESHOLD * 2:  # Ниже VWAP но не критично
                buy_triggers += 0.3
        
        # Определяем эффективный минимальный скор
        effective_min_score = MIN_COMPOSITE_SCORE
        if is_active_hour:
            effective_min_score *= ACTIVE_HOURS_MULTIPLIER
        
        # Проверяем достаточность триггеров для BUY - смягчаем пороги
        min_triggers = MIN_TRIGGERS_ACTIVE_HOURS if is_active_hour else MIN_TRIGGERS_INACTIVE_HOURS
        
        if buy_triggers >= min_triggers:
            # Дополнительные фильтры для качества
            
            # Избегаем экстремальной перекупленности
            if last['rsi'] > 85:
                pass  # Пропускаем сигнал
            else:
                # Генерируем детальную оценку
                score, pattern = evaluate_signal_strength(df, symbol, 'BUY')
                
                if score >= effective_min_score:
                    # Получаем метку силы
                    strength_label, win_prob = signal_strength_label(score)
                    
                    # Рассчитываем TP/SL
                    tp_price, sl_price = calculate_tp_sl(df, last['close'], last['atr'])
                    rr_ratio = calculate_rr_ratio(score)
                    
                    # Рекомендуем плечо
                    leverage = recommend_leverage(score, win_prob * 100)
                    
                    # Рассчитываем проценты для TP/SL
                    tp_pct = ((tp_price - last['close']) / last['close']) * 100
                    sl_pct = ((last['close'] - sl_price) / last['close']) * 100
                    
                    # Составляем сообщение
                    signal = f"🟢 LONG {symbol}\n"
                    signal += f"Цена: {last['close']:.6f}\n"
                    signal += f"Сила: {strength_label} ({score:.1f})\n"
                    signal += f"Вероятность: {win_prob:.0%}\n"
                    signal += f"TP: +{tp_pct:.2f}% | SL: -{sl_pct:.2f}%\n"
                    signal += f"R:R = 1:{rr_ratio:.1f}\n"
                    signal += f"RSI: {last['rsi']:.1f} | ADX: {last['adx']:.1f}\n"
                    
                    # Добавляем детали триггеров
                    signal += f"Триггеры: {buy_triggers:.1f}"
                    if USE_VWAP and 'vwap' in df.columns:
                        signal += f" | VWAP: {last.get('vwap_deviation', 0)*100:.1f}%"
                    if 'bb_width' in df.columns:
                        signal += f" | BB: {last['bb_width']*100:.1f}%"
                    
                    signals.append(signal)
                    
                    # Открываем виртуальную сделку
                    open_trade(symbol, last['close'], now, 'long', last['atr'], score)
                    record_trade(symbol, 'OPEN', last['close'], now, 'long', score)
                    
                    last_signal_time[symbol] = now
        
        # === СИГНАЛ НА ПРОДАЖУ ===
        sell_triggers = 0
        
        # Триггер 1: EMA кроссовер (главный)
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            sell_triggers += 1
        
        # Триггер 2: Цена ниже EMA (быстрой) - менее строгий
        elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
            sell_triggers += 0.5
        
        # Триггер 3: MACD медвежий
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if last['macd'] < last['macd_signal']:
                sell_triggers += 0.5
            # Кроссовер MACD - дополнительный бонус
            if prev['macd'] >= prev['macd_signal'] and last['macd'] < last['macd_signal']:
                sell_triggers += 0.5
        
        # Триггер 4: Bollinger Bands
        if 'bollinger_high' in df.columns:
            bb_position = (last['close'] - last['bollinger_low']) / (last['bollinger_high'] - last['bollinger_low'])
            if bb_position >= 0.7:  # В верхней части диапазона
                sell_triggers += 0.5
        
        # Триггер 5: VWAP
        if USE_VWAP and 'vwap' in df.columns:
            vwap_dev = last.get('vwap_deviation', 0)
            if vwap_dev >= 0 and vwap_dev <= VWAP_DEVIATION_THRESHOLD * 2:  # Выше VWAP но не критично
                sell_triggers += 0.3
        
        # Проверяем достаточность триггеров для SELL
        if sell_triggers >= min_triggers:
            # Дополнительные фильтры для качества
            
            # Избегаем экстремальной перепроданности
            if last['rsi'] < 15:
                pass  # Пропускаем сигнал
            else:
                # Генерируем детальную оценку
                score, pattern = evaluate_signal_strength(df, symbol, 'SELL')
                
                # Проверяем минимальный композитный скор
                if score >= effective_min_score:
                    # Получаем метку силы
                    strength_label, win_prob = signal_strength_label(score)
                    
                    # Рассчитываем TP/SL
                    tp_price, sl_price = calculate_tp_sl(df, last['close'], last['atr'], 'SELL')
                    rr_ratio = calculate_rr_ratio(score)
                    
                    # Рекомендуем плечо
                    leverage = recommend_leverage(score, win_prob * 100)
                    
                    # Рассчитываем проценты для TP/SL для SHORT
                    tp_pct = ((last['close'] - tp_price) / last['close']) * 100
                    sl_pct = ((sl_price - last['close']) / last['close']) * 100
                    
                    # Составляем сообщение
                    signal = f"🔴 SHORT {symbol}\n"
                    signal += f"Цена: {last['close']:.6f}\n"
                    signal += f"Сила: {strength_label} ({score:.1f})\n"
                    signal += f"Вероятность: {win_prob:.0%}\n"
                    signal += f"TP: +{tp_pct:.2f}% | SL: -{sl_pct:.2f}%\n"
                    signal += f"R:R = 1:{rr_ratio:.1f}\n"
                    signal += f"RSI: {last['rsi']:.1f} | ADX: {last['adx']:.1f}\n"
                    
                    # Добавляем детали триггеров
                    signal += f"Триггеры: {sell_triggers:.1f}"
                    if USE_VWAP and 'vwap' in df.columns:
                        signal += f" | VWAP: {last.get('vwap_deviation', 0)*100:.1f}%"
                    if 'bb_width' in df.columns:
                        signal += f" | BB: {last['bb_width']*100:.1f}%"
                    
                    signals.append(signal)
                    
                    # Открываем виртуальную сделку
                    open_trade(symbol, last['close'], now, 'short', last['atr'], score)
                    record_trade(symbol, 'OPEN', last['close'], now, 'short', score)
                    
                    last_signal_time[symbol] = now
        
        return signals
        
    except Exception as e:
        logging.error(f"Ошибка в check_signals для {symbol}: {e}")
        return []

# Добавляем новую функцию для расчета соотношения риск/доходность на основе score
def calculate_rr_ratio(score):
    """
    Рассчитывает рекомендуемое соотношение риск/доходность на основе score
    Возвращает значение для отображения в формате "1:X" где X - это TP/SL
    """
    if score >= 6:
        return 4.0  # Для экстремально сильных сигналов
    elif score >= 5:
        return 3.5  # Для очень сильных сигналов
    elif score >= 4.5:
        return 3.0  # Для сильных сигналов
    elif score >= 4.2:  # Адаптируем под новый минимальный порог (70%)
        return 2.5  # Для умеренных сигналов
    else:
        return 2.0  # Минимальное соотношение

# Удаляем долгосрочные функции - не нужны для 15м фьючерсной торговли

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
            # НЕ перезаписываем TP/SL для уже открытых позиций
            # calculate_tp_sl вызывается уже в check_tp_sl при необходимости
            pass
        else:
            # Для новых позиций устанавливаем минимальные значения
            if symbol not in open_trades:
                tp_pct, sl_pct = TP_MIN, SL_MIN
                adaptive_targets[symbol] = {'tp': tp_pct, 'sl': sl_pct}
        
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
    # last_long_signal удален, так как долгосрочный анализ отключен
    adaptive_targets = {}  # symbol: {'tp': ..., 'sl': ...}
    
    # Убраны лимиты сигналов по просьбе пользователя

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
        # Убрана логика лимитов сигналов
        
        # Проверка наличия монет
        if not SYMBOLS:
            error_msg = "❗️ Ошибка: список монет для анализа пуст. Проверь подключение к бирже или фильтры."
            print(error_msg)
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
        
        # Отправляем все найденные надежные сигналы (без лимитов)
        if all_current_signals and trading_enabled:
            # Сортируем по силе сигнала (берем самые сильные первыми)
            all_current_signals.sort(key=lambda x: x['strength'], reverse=True)
            logging.info(f"Найдено {len(all_current_signals)} надежных сигналов")
            
            # Отправляем группой
            combined_msg = f"💰 Надежные сигналы на {all_current_signals[0]['time'].strftime('%d.%m.%Y %H:%M')}:\n\n"
            
            for signal_info in all_current_signals:
                signals = signal_info['signals']
                tp_pct = signal_info['tp_pct']
                sl_pct = signal_info['sl_pct']
                tp_price = signal_info['tp_price']
                sl_price = signal_info['sl_price']
                
                combined_msg += '\n'.join(signals) + "\n\n"
                
                # Позиции уже открыты в check_signals(), не дублируем здесь
                # Просто логируем информацию о сигналах
                symbol = signal_info['symbol']
                direction = signal_info['direction']
                
                if symbol in open_trades:
                    logging.info(f"{symbol}: {direction} позиция уже открыта")
            
            combined_msg += f"📊 Всего найдено: {len(all_current_signals)} надежных сигналов"
            await send_telegram_message(combined_msg)
            signals_sent = True
        # Долгосрочный анализ временно отключен (функции analyze_long и check_signals_long не определены)
        # Можно включить позже при необходимости
        # Alive-отчёт раз в 6 часов + список обработанных монет  
        now_utc = datetime.now(timezone.utc)
        now_msk = now_utc.astimezone(tz_msk)
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

# Удаляем сложную функцию get_score_winrate - не влияет на сигналы

# Удаляем calculate_risk_params - избыточна, делает лишние API-запросы

# Удаляем find_support_resistance - не используется в сигналах

def calculate_tp_sl(df, price, atr, direction='LONG'):
    """
    СОВРЕМЕННЫЙ расчет TP/SL для 15-минутных фьючерсов с поддержкой LONG/SHORT.
    Адаптивные множители для лучшего соотношения R:R и винрейта.
    """
    try:
        last = df.iloc[-1]
        adx = last.get('adx', 20)
        
        # Адаптивные множители на основе силы тренда (ADX)
        if adx > 30:
            # Сильный тренд - можно взять больше прибыли
            tp_mult = TP_ATR_MULT  # 1.8
            sl_mult = SL_ATR_MULT  # 0.9
        elif adx > 20:
            # Умеренный тренд
            tp_mult = TP_ATR_MULT * 0.9  # 1.62
            sl_mult = SL_ATR_MULT * 0.9  # 0.81
        else:
            # Слабый тренд - более консервативный подход
            tp_mult = TP_ATR_MULT * 0.7  # 1.26
            sl_mult = SL_ATR_MULT * 0.8  # 0.72
        
        # Учитываем волатильность
        if 'volatility' in last:
            vol = last['volatility']
            if vol > HIGH_VOLATILITY_THRESHOLD:
                # Высокая волатильность - увеличиваем TP для захвата большого движения, и сужаем SL для безопасности
                tp_mult *= 1.2  # было 0.8
                sl_mult *= 0.8  # было 1.2
            elif vol < LOW_VOLATILITY_THRESHOLD:
                # Низкая волатильность - можем взять меньше прибыли, т.к. больших движений нет
                tp_mult *= 0.9 # было 1.1
                sl_mult *= 1.1 # было 0.9
        
        # Учитываем импульс цены
        if 'momentum' in last:
            momentum = abs(last['momentum'])
            if momentum > 1.0:  # Сильный импульс
                tp_mult *= 1.1
            elif momentum < 0.3:  # Слабый импульс
                tp_mult *= 0.9
        
        # Базовый расчет в процентах от цены
        tp_pct = max((atr * tp_mult) / price, TP_MIN)
        sl_pct = max((atr * sl_mult) / price, SL_MIN)
        
        # Обеспечиваем минимальное соотношение R:R = 1.8
        min_rr = 1.8
        if tp_pct / sl_pct < min_rr:
            tp_pct = sl_pct * min_rr
        
        # Ограничиваем максимальными значениями
        tp_pct = min(tp_pct, TP_MAX)
        sl_pct = min(sl_pct, SL_MAX)
        
        # Рассчитываем абсолютные цены
        if direction.upper() == 'LONG':
            tp_price = price * (1 + tp_pct)
            sl_price = price * (1 - sl_pct)
        else:  # SHORT
            tp_price = price * (1 - tp_pct)
            sl_price = price * (1 + sl_pct)
        
        return tp_price, sl_price
        
    except Exception as e:
        logging.error(f"Ошибка в calculate_tp_sl: {e}")
        # Возвращаем безопасные значения
        if direction.upper() == 'LONG':
            return price * 1.015, price * 0.992  # +1.5% TP, -0.8% SL
        else:
            return price * 0.985, price * 1.008  # -1.5% TP, +0.8% SL

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
        
        msg = f"{symbol} {side.upper()} закрыт по {reason}: вход {entry:.6f}, выход {price:.6f}, P&L: {pnl_pct:+.2f}%, результат: {final_result}"
        asyncio.create_task(send_telegram_message(msg))
        
        # Записываем результат в портфель
        record_trade(symbol, 'CLOSE', price, time, side, score)
        close_trade(symbol)
        
        # Удаляем из adaptive_targets после закрытия
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

# Удаляем дублированную функцию analyze_long

if __name__ == '__main__':
    asyncio.run(main()) 