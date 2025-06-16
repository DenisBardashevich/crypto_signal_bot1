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
    """УПРОЩЁННЫЙ анализ для 15-минутных фьючерсов: только нужные индикаторы."""
    try:
        if df.empty or len(df) < MA_SLOW:
            return pd.DataFrame()
            
        # Основные индикаторы для 15м
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=MA_FAST)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=MA_SLOW)
        
        # MACD для подтверждения сигналов
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        # RSI для фильтрации экстремальных значений
        df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_WINDOW)
        
        # ADX для определения силы тренда
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        # ATR для расчёта TP/SL
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=ATR_WINDOW)
        
        # Bollinger Bands для определения перекупленности/перепроданности
        df['bollinger_mid'] = ta.volatility.bollinger_mavg(df['close'], window=20)
        df['bollinger_high'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['bollinger_low'] = ta.volatility.bollinger_lband(df['close'], window=20)
        
        # Объём - только если включен фильтр
        if USE_VOLUME_FILTER:
            df['volume_ema'] = ta.trend.ema_indicator(df['volume'], window=20)
            df['volume_ratio'] = df['volume'] / df['volume_ema']
        
        # Спред и импульс для дополнительной фильтрации
        df['spread_pct'] = (df['high'] - df['low']) / df['low']
        df['momentum'] = df['close'].pct_change(5) * 100  # 5 свечей назад
        
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
    """Простая оценка силы сигнала для 15-минутных фьючерсов."""
    score = 0
    last = df.iloc[-1]

    # Сила тренда по ADX
    if last['adx'] > 30:
        score += 2.0
    elif last['adx'] > 25:
        score += 1.5
    elif last['adx'] > 20:
        score += 1.0
    else:
        score += 0.5

    # RSI
    if action == 'BUY':
        if 30 <= last['rsi'] <= 65:
            score += 1.0
        elif last['rsi'] < 30:
            score += 1.5
    elif action == 'SELL':
        if 35 <= last['rsi'] <= 70:
            score += 1.0
        elif last['rsi'] > 70:
            score += 1.5

    # Bollinger Bands
    if 'bollinger_low' in df.columns and 'bollinger_high' in df.columns:
        close = last['close']
        if action == 'BUY' and close <= last['bollinger_low'] * 1.02:
            score += 1.0
        elif action == 'SELL' and close >= last['bollinger_high'] * 0.98:
            score += 1.0

    return score, None

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
    ОПТИМИЗИРОВАННАЯ функция проверки сигналов для 15-минутных фьючерсов
    Фокус на простоте, скорости и надёжности
    """
    try:
        if df.empty or len(df) < MIN_15M_CANDLES:
            return []
            
        last = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        
        # === БАЗОВЫЕ ОБЯЗАТЕЛЬНЫЕ ФИЛЬТРЫ ===
        # 1. Минимальная сила тренда
        if last['adx'] < MIN_ADX:
            return []
            
        # 2. Минимальный объём торгов
        volume = get_24h_volume(symbol)
        if volume < MIN_VOLUME_USDT:
            return []
            
        # 3. Максимальный спред
        if last['spread_pct'] > MAX_SPREAD_PCT:
            return []
        
        # === ПРОВЕРКА COOLDOWN ===
        if symbol not in last_signal_time:
            last_signal_time[symbol] = datetime.now(timezone.utc) - timedelta(minutes=10)
        
        if last_signal_time[symbol].tzinfo is None:
            last_signal_time[symbol] = last_signal_time[symbol].replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        if now - last_signal_time[symbol] < timedelta(minutes=SIGNAL_COOLDOWN_MINUTES):
            return []
        
        # === СИГНАЛ НА ПОКУПКУ ===
        if prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            # Простая проверка MACD
            if last['macd'] > last['macd_signal'] or last['macd'] > 0:
                
                # Начальный score
                score = 2.0  # Базовый score за EMA кросс
                
                # БОНУСЫ (максимум +3.0)
                # 1. ADX - сила тренда
                if last['adx'] > 25:
                    score += 1.0
                elif last['adx'] > 20:
                    score += 0.5
                
                # 2. RSI - хороший уровень для входа
                if 30 <= last['rsi'] <= 65:
                    score += 0.8
                elif last['rsi'] < 30:  # Перепроданность
                    score += 1.2
                
                # 3. Объём - всплеск активности
                if USE_VOLUME_FILTER and 'volume_ratio' in last:
                    if last['volume_ratio'] > 1.2:
                        score += 0.5
                
                # 4. Импульс цены
                if 'momentum' in last and last['momentum'] > 0.05:
                    score += 0.3
                
                # 5. Bollinger Bands - цена у нижней границы
                if 'bollinger_low' in last and last['close'] <= last['bollinger_low'] * 1.02:
                    score += 0.7
                
                # 6. Бычья свеча
                if last['close'] > last['open']:
                    score += 0.3
                
                # ШТРАФЫ (максимум -1.0)
                # 1. Слабый BTC для альтов
                if symbol != 'BTC/USDT:USDT':
                    btc_adx = get_btc_adx()
                    if btc_adx < 8:
                        score -= 0.3
                
                # 2. Очень высокий RSI
                if last['rsi'] > 75:
                    score -= 0.5
                
                # 3. Очень низкий объём
                if last['volume'] < df['volume'].rolling(20).mean().iloc[-1] * 0.4:
                    score -= 0.2
                
                # Минимальный порог для сигнала (только сигналы выше 65%)
                if score >= 4.0:  # Изменено с 1.8 на 4.0 для фильтрации сигналов ниже 65%
                    label, strength_chance = signal_strength_label(score)
                    leverage = recommend_leverage(score, 50)  # Фиксированный процент
                    rr_ratio = calculate_rr_ratio(score)
                    
                    msg = f'🚀 ФЬЮЧЕРСЫ BYBIT: ЛОНГ!\n💪 Сила: {label} ({strength_chance*100:.0f}%)\n⚡ Плечо: {leverage}x\n🎯 R:R = 1:{rr_ratio}\n📊 {symbol}\n💰 Объём: {volume/1_000_000:.1f}М USDT\n📈 ADX: {last["adx"]:.0f}\n⭐ Score: {score:.1f}'
                    
                    signals.append(msg)
                    logging.info(f"{symbol}: LONG сигнал сформирован, score: {score:.1f}")
        
        # === СИГНАЛ НА ПРОДАЖУ ===
        if prev['ema_fast'] > prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            # Простая проверка MACD
            if last['macd'] < last['macd_signal'] or last['macd'] < 0:
                
                # Начальный score
                score = 2.0  # Базовый score за EMA кросс
                
                # БОНУСЫ (максимум +3.0)
                # 1. ADX - сила тренда
                if last['adx'] > 25:
                    score += 1.0
                elif last['adx'] > 20:
                    score += 0.5
                
                # 2. RSI - хороший уровень для входа
                if 35 <= last['rsi'] <= 70:
                    score += 0.8
                elif last['rsi'] > 70:  # Перекупленность
                    score += 1.2
                
                # 3. Объём - всплеск активности
                if USE_VOLUME_FILTER and 'volume_ratio' in last:
                    if last['volume_ratio'] > 1.2:
                        score += 0.5
                
                # 4. Импульс цены
                if 'momentum' in last and last['momentum'] < -0.05:
                    score += 0.3
                
                # 5. Bollinger Bands - цена у верхней границы
                if 'bollinger_high' in last and last['close'] >= last['bollinger_high'] * 0.98:
                    score += 0.7
                
                # 6. Медвежья свеча
                if last['close'] < last['open']:
                    score += 0.3
                
                # ШТРАФЫ (максимум -1.0)
                # 1. Слабый BTC для альтов
                if symbol != 'BTC/USDT:USDT':
                    btc_adx = get_btc_adx()
                    if btc_adx < 8:
                        score -= 0.3
                
                # 2. Очень низкий RSI
                if last['rsi'] < 25:
                    score -= 0.5
                
                # 3. Очень низкий объём
                if last['volume'] < df['volume'].rolling(20).mean().iloc[-1] * 0.4:
                    score -= 0.2
                
                # Минимальный порог для сигнала (только сигналы выше 65%)
                if score >= 4.0:  # Изменено с 1.8 на 4.0 для фильтрации сигналов ниже 65%
                    label, strength_chance = signal_strength_label(score)
                    leverage = recommend_leverage(score, 50)  # Фиксированный процент
                    rr_ratio = calculate_rr_ratio(score)
                    
                    msg = f'📉 ФЬЮЧЕРСЫ BYBIT: ШОРТ!\n💪 Сила: {label} ({strength_chance*100:.0f}%)\n⚡ Плечо: {leverage}x\n🎯 R:R = 1:{rr_ratio}\n📊 {symbol}\n💰 Объём: {volume/1_000_000:.1f}М USDT\n📈 ADX: {last["adx"]:.0f}\n⭐ Score: {score:.1f}'
                    
                    signals.append(msg)
                    logging.info(f"{symbol}: SHORT сигнал сформирован, score: {score:.1f}")
        
        # Обновляем время последнего сигнала
        if signals:
            last_signal_time[symbol] = now
            
        return signals
        
    except Exception as e:
        logging.error(f"Ошибка при проверке сигналов для {symbol}: {e}")
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
    elif score >= 4.0:  # Адаптируем под новый минимальный порог (65%)
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
            tp, sl = calculate_tp_sl(df, price, atr, symbol)
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

# Удаляем сложную функцию get_score_winrate - не влияет на сигналы

# Удаляем calculate_risk_params - избыточна, делает лишние API-запросы

# Удаляем find_support_resistance - не используется в сигналах

def calculate_tp_sl(df, price, atr, symbol=None):
    """
    УПРОЩЁННЫЙ расчет TP/SL для 15-минутных фьючерсов
    Без API-запросов, только на основе ADX и ATR
    """
    last = df.iloc[-1]
    adx = last['adx']
    
    # Простые множители на основе ADX
    if adx > 25:
        tp_mult = 2.5
        sl_mult = 1.0
    elif adx > 15:
        tp_mult = 2.0
        sl_mult = 0.8
    else:
        tp_mult = 1.8
        sl_mult = 0.7
    
    # Корректировка на основе импульса
    if 'momentum' in last and abs(last['momentum']) > 0.8:
        tp_mult *= 1.2
    
    # Расчет TP и SL
    tp = max(round((atr * tp_mult) / price, 4), TP_MIN)
    sl = max(round((atr * sl_mult) / price, 4), SL_MIN)
    
    # Обеспечиваем минимальное соотношение R:R = 2.0
    if tp / sl < 2.0:
        tp = sl * 2.0
    
    # Ограничиваем максимальными значениями
    tp = min(tp, TP_MAX)
    sl = min(sl, SL_MAX)
    
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
        tp, sl = calculate_tp_sl(df, price, atr, symbol)
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
            if side == 'LONG':
                result = 'УДАЧНО' if exit > entry else 'НЕУДАЧНО'
            else:
                result = 'УДАЧНО' if exit < entry else 'НЕУДАЧНО'
            if result == 'УДАЧНО':
                total_win += 1
            else:
                total_loss += 1
            # Только монета и результат
            report.append(f"{symbol}: {result}")
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