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
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'MATIC/USDT:USDT',
    'TRX/USDT:USDT', 'DOT/USDT:USDT', 'LTC/USDT:USDT', 'BCH/USDT:USDT', 'UNI/USDT:USDT',
    'ATOM/USDT:USDT', 'XLM/USDT:USDT', 'FIL/USDT:USDT', 'APT/USDT:USDT', 'OP/USDT:USDT',
    'ARB/USDT:USDT', 'NEAR/USDT:USDT', 'ETC/USDT:USDT', 'HBAR/USDT:USDT', 'VET/USDT:USDT',
    'ICP/USDT:USDT', 'SUI/USDT:USDT', 'INJ/USDT:USDT', 'STX/USDT:USDT', 'RNDR/USDT:USDT',
    'MKR/USDT:USDT', 'AAVE/USDT:USDT', 'EGLD/USDT:USDT', 'ALGO/USDT:USDT', 'GRT/USDT:USDT',
    'MANA/USDT:USDT', 'SAND/USDT:USDT', 'AXS/USDT:USDT', 'FTM/USDT:USDT', 'LDO/USDT:USDT',
    'CRV/USDT:USDT', 'DYDX/USDT:USDT', 'PEPE/USDT:USDT', 'TWT/USDT:USDT', 'CAKE/USDT:USDT',
    'ENS/USDT:USDT', 'BLUR/USDT:USDT', 'GMT/USDT:USDT', '1INCH/USDT:USDT', 'COMP/USDT:USDT',
    # Перспективные альткойны
    'PYTH/USDT:USDT', 'JUP/USDT:USDT', 'TIA/USDT:USDT', 'SEI/USDT:USDT', 'WIF/USDT:USDT', 'RON/USDT:USDT', 'BEAMX/USDT:USDT',
    # Фьючерсные/волатильные
    '1000PEPE/USDT:USDT', 'FLOKI/USDT:USDT', 'BONK/USDT:USDT', 'SHIB/USDT:USDT'
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
def record_trade(symbol, action, price, time):
    if symbol not in virtual_portfolio:
        virtual_portfolio[symbol] = []
    virtual_portfolio[symbol].append({
        'action': action,
        'price': price,
        'time': time.strftime('%Y-%m-%d %H:%M')
    })
    save_portfolio()

# Открытие сделки
def open_trade(symbol, price, time, atr=None):
    open_trades[symbol] = {
        'buy_price': price,
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'atr': atr if atr is not None else 0,
        'trail_pct': TRAIL_ATR_MULT,
        'last_peak': price
    }
    save_portfolio()

# Закрытие сделки
def close_trade(symbol):
    if symbol in open_trades:
        del open_trades[symbol]
        save_portfolio()

# Подсчёт прибыли
def calculate_profit():
    report = []
    total_profit = 0
    win, loss = 0, 0
    for symbol, trades in virtual_portfolio.items():
        if symbol == 'open_trades':
            continue
        win_count = 0
        loss_count = 0
        last_buy = None
        for trade in trades:
            if trade['action'] == 'BUY':
                last_buy = float(trade['price'])
            elif trade['action'] == 'SELL' and last_buy is not None:
                p = float(trade['price']) - last_buy
                if p > 0:
                    win_count += 1
                else:
                    loss_count += 1
                last_buy = None
        if win_count > 0 or loss_count > 0:
            report.append(f"{symbol}: прибыльных {win_count}, убыточных {loss_count}")
        win += win_count
        loss += loss_count
    return report, win, loss

# ========== ФУНКЦИИ АНАЛИЗА ==========
def get_ohlcv(symbol):
    """Получить исторические данные по монете."""
    ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Moscow')
    return df

def analyze(df):
    """Анализ по индикаторам: EMA, MACD, ATR (5m), RSI."""
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=MA_FAST)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=MA_SLOW)
    macd = ta.trend.macd_diff(df['close'])
    df['macd'] = macd
    df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_WINDOW)
    df['atr5m'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=ATR_WINDOW)
    # Убираем строки с NaN, чтобы не ловить фантомные кресты
    df = df.dropna().reset_index(drop=True)
    return df

# ========== ОЦЕНКА СИЛЫ СИГНАЛА ПО ГРАФИКУ ==========
def evaluate_signal_strength(df):
    """Оценка силы сигнала по индикаторам (0-3 балла)."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    # SMA пересечение
    if (prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow']) or (prev['ema_fast'] > prev['ema_slow'] and last['ema_fast'] < last['ema_slow']):
        score += 1
    # MACD
    if (last['macd'] > 0 and prev['macd'] <= 0) or (last['macd'] < 0 and prev['macd'] >= 0):
        score += 1
    # RSI
    if 30 < last['rsi'] < 70:
        score += 1
    return score

def signal_strength_label(score):
    if score == 3:
        return 'Сильный', 0.85
    elif score == 2:
        return 'Средний', 0.65
    elif score == 1:
        return 'Слабый', 0.45
    else:
        return 'Очень слабый', 0.3

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
        # Bybit возвращает объём в baseVolume (количество монет) и quoteVolume (в валюте котировки)
        volume = ticker.get('quoteVolume', 0)
        return volume
    except Exception as e:
        print(f"Ошибка получения объёма по {symbol}: {e}")
        return 0

last_signal_time = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))

def check_signals(df, symbol):
    """Golden/Death Cross по EMA + MACD + фильтр RSI + фильтр по тренду + фильтр по объёму + глобальный тренд."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    # Получаем объём торгов за 24ч
    volume = get_24h_volume(symbol)
    volume_mln = volume / 1_000_000
    min_volume = MIN_VOLUME_USDT
    if volume < min_volume:
        logging.info(f"{symbol}: объём {volume_mln:.2f} млн < {min_volume/1_000_000:.0f} млн, сигнал не формируется")
        return []
    # Фильтр по тренду
    if last['close'] < last['ema_slow']:
        logging.info(f"{symbol}: цена ниже EMA_slow, сигнал на покупку не формируется")
        return []
    # Фильтр по RSI (нейтральная зона)
    if RSI_NEUTRAL_LOW <= last['rsi'] <= RSI_NEUTRAL_HIGH:
        logging.info(f"{symbol}: RSI {last['rsi']:.2f} в нейтральной зоне, сигнал не формируется")
        return []
    # Фильтр по глобальному тренду (только для BUY)
    if prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
        if not is_global_uptrend(symbol):
            logging.info(f"{symbol}: глобальный тренд вниз — BUY пропущен")
            return []
    # Golden Cross (EMA50 пересёк EMA100 вверх) + MACD бычий + RSI < 70
    if prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow'] and last['macd'] > 0 and last['rsi'] < 70:
        action = 'BUY'
        score = evaluate_signal_strength(df)
        label, strength_chance = signal_strength_label(score)
        history_percent, total = get_signal_stats(symbol, action)
        avg_chance = int((strength_chance * 100 + history_percent) / 2)
        leverage = recommend_leverage(score, history_percent)
        signals.append(f'\U0001F4C8 Сигнал (ФЬЮЧЕРСЫ BYBIT): КУПИТЬ!\nСила сигнала: {label}\nИсторический шанс: {history_percent:.0f}% (по {total} сделкам)\nОценка по графику: {int(strength_chance*100)}%\nИтоговый шанс: {avg_chance}%\nРекомендуемое плечо: {leverage}\nОбъём торгов: {volume_mln:.2f} млн USDT/сутки\nTP/SL указываются ниже, выставлять их на бирже!\nПричина: EMA50 пересёк EMA100 вверх (Golden Cross), MACD бычий, RSI < 70.')
        logging.info(f"{symbol}: BUY сигнал сформирован (фьючерсы)")
    # Death Cross (EMA50 пересёк EMA100 вниз) + MACD медвежий + RSI > 30
    if prev['ema_fast'] > prev['ema_slow'] and last['ema_fast'] < last['ema_slow'] and last['macd'] < 0 and last['rsi'] > 30:
        action = 'SELL'
        score = evaluate_signal_strength(df)
        label, strength_chance = signal_strength_label(score)
        history_percent, total = get_signal_stats(symbol, action)
        avg_chance = int((strength_chance * 100 + history_percent) / 2)
        leverage = recommend_leverage(score, history_percent)
        signals.append(f'\U0001F4C9 Сигнал (ФЬЮЧЕРСЫ BYBIT): ПРОДАТЬ!\nСила сигнала: {label}\nИсторический шанс: {history_percent:.0f}% (по {total} сделкам)\nОценка по графику: {int(strength_chance*100)}%\nИтоговый шанс: {avg_chance}%\nРекомендуемое плечо: {leverage}\nОбъём торгов: {volume_mln:.2f} млн USDT/сутки\nTP/SL указываются ниже, выставлять их на бирже!\nПричина: EMA50 пересёк EMA100 вниз (Death Cross), MACD медвежий, RSI > 30.')
        logging.info(f"{symbol}: SELL сигнал сформирован (фьючерсы)")
    # Защита от naive datetime
    if last_signal_time[symbol].tzinfo is None:
        last_signal_time[symbol] = last_signal_time[symbol].replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if now - last_signal_time[symbol] < timedelta(minutes=SIGNAL_COOLDOWN_MINUTES):
        return []
    # Если сигнал сформирован:
    if signals:
        last_signal_time[symbol] = now
    return signals

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
    report, win, loss = calculate_profit()
    text = '📊 Отчёт по виртуальным сделкам за сутки:\n'
    if report:
        text += '\n'.join(report)
    else:
        text += 'Нет завершённых сделок.'
    text += f"\n\nВсего прибыльных сделок: {win}\nВсего убыточных сделок: {loss}"
    await send_telegram_message(text)

# ========== ОБРАБОТЧИК КОМАНДЫ /stats ==========
async def stats_command(update, context):
    report, win, loss = calculate_profit()
    text = '📊 Статистика по виртуальным сделкам:\n'
    if report:
        text += '\n'.join(report)
    else:
        text += 'Нет завершённых сделок.'
    text += f"\n\nВсего прибыльных сделок: {win}\nВсего убыточных сделок: {loss}"
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

async def main():
    tz_msk = timezone(timedelta(hours=3))
    last_alive = datetime.now(tz_msk) - timedelta(hours=6)  # timezone-aware
    last_report_hours = set()  # Часы, когда уже был отправлен отчёт (например, {9, 22})
    last_long_signal = datetime.now(tz_msk) - timedelta(days=1)  # timezone-aware
    adaptive_targets = {}  # symbol: {'tp': ..., 'sl': ...}

    # Запускаем Telegram-бота как асинхронную задачу
    asyncio.create_task(telegram_bot())

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
        for symbol in SYMBOLS:
            try:
                df = get_ohlcv(symbol)
                df = analyze(df)
                signals = check_signals(df, symbol)
                price = df['close'].iloc[-1]
                time = df['timestamp'].iloc[-1]
                processed_symbols.append(symbol)
                # Расчёт адаптивных целей по ATR 5m
                atr5m = df['atr5m'].iloc[-1]
                if not pd.isna(atr5m) and price > 0:
                    tp = min(max(round((atr5m * 3.0) / price, 4), 0.015), 0.15)  # минимум 1.5%, максимум 15%
                    sl = min(max(round((atr5m * 2.0) / price, 4), 0.015), 0.15)
                    adaptive_targets[symbol] = {'tp': tp, 'sl': sl}
                else:
                    tp = 0.015
                    sl = 0.015
                    adaptive_targets[symbol] = {'tp': tp, 'sl': sl}
                # Проверка на открытые сделки
                if symbol in open_trades:
                    buy_price = open_trades[symbol]['buy_price']
                    atr = open_trades[symbol].get('atr', atr5m)
                    trail_pct = open_trades[symbol].get('trail_pct', TRAIL_ATR_MULT)
                    last_peak = open_trades[symbol].get('last_peak', buy_price)
                    # Trailing-ATR: обновляем last_peak если цена выросла
                    if price > last_peak:
                        open_trades[symbol]['last_peak'] = price
                        last_peak = price
                        save_portfolio()
                    dynamic_sl = last_peak - atr * trail_pct
                    # Trailing-ATR стоп
                    if price <= dynamic_sl:
                        msg = f"⚠️ {symbol} сработал trailing-ATR стоп (динамический SL):\nТочка входа: {buy_price}, текущая цена: {price:.4f}, SL: {dynamic_sl:.4f}\nРекомендуется ПРОДАТЬ для ограничения убытков или фиксации прибыли."
                        await send_telegram_message(msg)
                        record_trade(symbol, 'SELL', price, time)
                        close_trade(symbol)
                        logging.info(f"{symbol}: сделка закрыта по trailing-ATR SL")
                        signals_sent = True
                        continue
                # Сигналы на вход/выход
                if signals:
                    tp = adaptive_targets[symbol]['tp'] if symbol in adaptive_targets else 0.02
                    sl = adaptive_targets[symbol]['sl'] if symbol in adaptive_targets else 0.02
                    tp_price = round(price * (1 + tp), 6)
                    sl_price = round(price * (1 - sl), 6)
                    msg = f"\n\U0001F4B0 Сигналы для {symbol} на {time.strftime('%d.%m.%Y %H:%M')}:\n" + '\n\n'.join(signals)
                    msg += f"\nАдаптивный тейк-профит: +{tp*100:.2f}% ({tp_price}), стоп-лосс: -{sl*100:.2f}% ({sl_price})"
                    await send_telegram_message(msg)
                    logging.info(f"{symbol}: сигнал отправлен в Telegram")
                    signals_sent = True
                    for s in signals:
                        if 'КУПИТЬ' in s and symbol not in open_trades:
                            record_trade(symbol, 'BUY', price, time)
                            open_trade(symbol, price, time, atr=atr5m)
                            logging.info(f"{symbol}: сделка открыта по цене {price}")
                        if 'ПРОДАТЬ' in s and symbol in open_trades:
                            record_trade(symbol, 'SELL', price, time)
                            close_trade(symbol)
                            logging.info(f"{symbol}: сделка закрыта по сигналу ПРОДАТЬ")
            except Exception as e:
                error_text = f"Ошибка по {symbol}: {e}"
                print(error_text)
                logging.error(error_text)
                await send_telegram_message(f"❗️ {error_text}")
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
        await asyncio.sleep(60 * 3)  # Проверять каждые 3 минуты

def is_global_uptrend(symbol: str) -> bool:
    ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=BACKUP_TIMEFRAME, limit=MA_SLOW*3)
    tmp = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    tmp['ema_f'] = ta.trend.ema_indicator(tmp['c'], window=MA_FAST)
    tmp['ema_s'] = ta.trend.ema_indicator(tmp['c'], window=MA_SLOW)
    return bool(tmp['ema_f'].iloc[-1] > tmp['ema_s'].iloc[-1])

if __name__ == '__main__':
    asyncio.run(main()) 