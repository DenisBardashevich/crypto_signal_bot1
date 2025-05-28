import ccxt
import pandas as pd
import ta
import asyncio
from telegram import Bot
import os
import json
from datetime import datetime, timedelta, timezone
import time

# ========== НАСТРОЙКИ ==========
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'  # Токен Telegram-бота
TELEGRAM_CHAT_ID = 931346988  # chat_id пользователя

EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'  # Используем спотовый рынок
    }
})

# Белый список топ-50 популярных монет + перспективные альткойны и волатильные монеты
TOP_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT',
    'TRX/USDT', 'DOT/USDT', 'LTC/USDT', 'BCH/USDT', 'UNI/USDT',
    'ATOM/USDT', 'XLM/USDT', 'FIL/USDT', 'APT/USDT', 'OP/USDT',
    'ARB/USDT', 'NEAR/USDT', 'ETC/USDT', 'HBAR/USDT', 'VET/USDT',
    'ICP/USDT', 'SUI/USDT', 'INJ/USDT', 'STX/USDT', 'RNDR/USDT',
    'MKR/USDT', 'AAVE/USDT', 'EGLD/USDT', 'ALGO/USDT', 'GRT/USDT',
    'MANA/USDT', 'SAND/USDT', 'AXS/USDT', 'FTM/USDT', 'LDO/USDT',
    'CRV/USDT', 'DYDX/USDT', 'PEPE/USDT', 'TWT/USDT', 'CAKE/USDT',
    'ENS/USDT', 'BLUR/USDT', 'GMT/USDT', '1INCH/USDT', 'COMP/USDT',
    # Перспективные альткойны
    'PYTH/USDT', 'JUP/USDT', 'TIA/USDT', 'SEI/USDT', 'WIF/USDT', 'RON/USDT', 'BEAMX/USDT',
    # Фьючерсные/волатильные
    '1000PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'SHIB/USDT'
]
markets = EXCHANGE.load_markets()
SYMBOLS = [symbol for symbol in TOP_SYMBOLS if symbol in markets and markets[symbol]['active']]
print(f"SYMBOLS: {SYMBOLS}")  # Для отладки
TIMEFRAME = '5m'  # Интервал свечей теперь 5 минут
LIMIT = 400  # Количество свечей для анализа (с запасом для всех индикаторов)

TAKE_PROFIT = 0.02  # +2%
STOP_LOSS = -0.02   # -2%

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
def open_trade(symbol, price, time):
    open_trades[symbol] = {'buy_price': price, 'time': time.strftime('%Y-%m-%d %H:%M')}
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
        profit = 0
        last_buy = None
        for trade in trades:
            if trade['action'] == 'BUY':
                last_buy = float(trade['price'])
            elif trade['action'] == 'SELL' and last_buy is not None:
                p = float(trade['price']) - last_buy
                profit += p
                if p > 0:
                    win += 1
                else:
                    loss += 1
                last_buy = None
        if profit != 0:
            report.append(f"{symbol}: {profit:+.2f} USDT")
        total_profit += profit
    return report, total_profit, win, loss

# ========== ФУНКЦИИ АНАЛИЗА ==========
def get_ohlcv(symbol):
    """Получить исторические данные по монете."""
    ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def analyze(df):
    """Анализ по индикаторам: SMA, MACD, ATR (8ч и сутки), RSI (SMA50 и SMA100)."""
    df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma100'] = ta.trend.sma_indicator(df['close'], window=100)
    macd = ta.trend.macd_diff(df['close'])
    df['macd'] = macd
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr8h'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=100)
    df['atr1d'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=288)
    return df

def check_signals(df):
    """Golden/Death Cross по SMA50/100 + MACD + мягкий фильтр RSI."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    # Golden Cross (SMA50 пересёк SMA100 вверх) + MACD бычий + RSI < 70
    if prev['sma50'] < prev['sma100'] and last['sma50'] > last['sma100'] and last['macd'] > 0 and last['rsi'] < 70:
        signals.append('Сигнал: КУПИТЬ!\nПричина: SMA50 пересёк SMA100 вверх (Golden Cross), MACD бычий, RSI < 70.')
    # Death Cross (SMA50 пересёк SMA100 вниз) + MACD медвежий + RSI > 30
    if prev['sma50'] > prev['sma100'] and last['sma50'] < last['sma100'] and last['macd'] < 0 and last['rsi'] > 30:
        signals.append('Сигнал: ПРОДАТЬ!\nПричина: SMA50 пересёк SMA100 вниз (Death Cross), MACD медвежий, RSI > 30.')
    return signals

def analyze_long(df):
    """Долгосрочный анализ: SMA50/200, MACD, RSI на дневках."""
    df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    return df

def check_signals_long(df):
    """Сигналы для долгосрока: Golden/Death Cross + MACD + RSI на дневках."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    # Golden Cross (SMA50 пересёк SMA200 вверх) + MACD бычий + RSI < 65
    if prev['sma50'] < prev['sma200'] and last['sma50'] > last['sma200'] and last['macd'] > 0 and last['rsi'] < 65:
        signals.append('Сигнал: КУПИТЬ НА ДОЛГОСРОК!\nПричина: SMA50 пересёк SMA200 вверх (Golden Cross), MACD бычий, RSI < 65.')
    # Death Cross (SMA50 пересёк SMA200 вниз) + MACD медвежий + RSI > 35
    if prev['sma50'] > prev['sma200'] and last['sma50'] < last['sma200'] and last['macd'] < 0 and last['rsi'] > 35:
        signals.append('Сигнал: ПРОДАТЬ НА ДОЛГОСРОК!\nПричина: SMA50 пересёк SMA200 вниз (Death Cross), MACD медвежий, RSI > 35.')
    return signals

# ========== ОТПРАВКА В TELEGRAM ==========
async def send_telegram_message(text):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)

# ========== ОТПРАВКА ОТЧЁТА ==========
async def send_daily_report():
    report, total, win, loss = calculate_profit()
    text = '📊 Отчёт по виртуальным сделкам за сутки:\n'
    if report:
        text += '\n'.join(report)
    else:
        text += 'Нет завершённых сделок.'
    text += f"\n\nВсего по всем монетам: {total:+.2f} USDT\nПрибыльных сделок: {win}\nУбыточных сделок: {loss}"
    await send_telegram_message(text)

# ========== ОСНОВНОЙ ЦИКЛ ==========
TIME_SHIFT_HOURS = 3  # Сдвиг времени для локального времени пользователя
async def main():
    last_report = datetime.now()
    last_alive = datetime.now() - timedelta(hours=3)  # чтобы сразу отправить первое alive-сообщение
    last_long_signal = datetime.now() - timedelta(days=1)
    adaptive_targets = {}  # symbol: {'tp': ..., 'sl': ...}
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
                signals = check_signals(df)
                price = df['close'].iloc[-1]
                time = df['timestamp'].iloc[-1] + timedelta(hours=TIME_SHIFT_HOURS)
                processed_symbols.append(symbol)
                # Расчёт адаптивных целей
                atr8h = df['atr8h'].iloc[-1]
                atr1d = df['atr1d'].iloc[-1]
                if not pd.isna(atr8h) and not pd.isna(atr1d) and price > 0:
                    atr = max(atr8h, atr1d)
                    tp = min(max(round((atr * 3.0) / price, 4), 0.008), 0.2)  # минимум 0.8%, максимум 20%
                    sl = min(max(round((atr * 2.0) / price, 4), 0.008), 0.2)
                    adaptive_targets[symbol] = {'tp': tp, 'sl': sl}
                else:
                    tp = 0.008
                    sl = 0.008
                    adaptive_targets[symbol] = {'tp': tp, 'sl': sl}
                # Проверка на открытые сделки
                if symbol in open_trades:
                    buy_price = open_trades[symbol]['buy_price']
                    change = (price - buy_price) / buy_price
                    tp = adaptive_targets[symbol]['tp']
                    sl = adaptive_targets[symbol]['sl']
                    # Тейк-профит
                    if change >= tp:
                        msg = f"🎯 {symbol} достиг цели +{tp*100:.2f}% (адаптивный тейк-профит)\nРекомендуется ПРОДАТЬ для фиксации прибыли.\nТочка входа: {buy_price}, текущая цена: {price:.4f}"
                        await send_telegram_message(msg)
                        record_trade(symbol, 'SELL', price, time)
                        close_trade(symbol)
                        signals_sent = True
                        continue
                    # Стоп-лосс
                    if change <= -sl:
                        msg = f"⚠️ {symbol} снизился на -{sl*100:.2f}% (адаптивный стоп-лосс)\nРекомендуется ПРОДАТЬ для ограничения убытков.\nТочка входа: {buy_price}, текущая цена: {price:.4f}"
                        await send_telegram_message(msg)
                        record_trade(symbol, 'SELL', price, time)
                        close_trade(symbol)
                        signals_sent = True
                        continue
                # Сигналы на вход/выход
                if signals:
                    tp = adaptive_targets[symbol]['tp'] if symbol in adaptive_targets else 0.02
                    sl = adaptive_targets[symbol]['sl'] if symbol in adaptive_targets else 0.02
                    msg = f"\n\U0001F4B0 Сигналы для {symbol} на {time.strftime('%d.%m.%Y %H:%M')}:\n" + '\n\n'.join(signals)
                    msg += f"\nАдаптивный тейк-профит: +{tp*100:.2f}%, стоп-лосс: -{sl*100:.2f}%"
                    await send_telegram_message(msg)
                    signals_sent = True
                    for s in signals:
                        if 'КУПИТЬ' in s and symbol not in open_trades:
                            record_trade(symbol, 'BUY', price, time)
                            open_trade(symbol, price, time)
                        if 'ПРОДАТЬ' in s and symbol in open_trades:
                            record_trade(symbol, 'SELL', price, time)
                            close_trade(symbol)
            except Exception as e:
                error_text = f"Ошибка по {symbol}: {e}"
                print(error_text)
                await send_telegram_message(f"❗️ {error_text}")
        # Долгосрочный анализ раз в сутки
        now = datetime.now()
        if (now - last_long_signal) > timedelta(hours=23):
            for symbol in SYMBOLS:
                try:
                    ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe='1d', limit=400)
                    df_long = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], unit='ms')
                    df_long = analyze_long(df_long)
                    signals_long = check_signals_long(df_long)
                    if signals_long:
                        msg = f"\n\U0001F4BC Сигнал (долгосрок) для {symbol} на {df_long['timestamp'].iloc[-1].strftime('%d.%m.%Y')}:\n" + '\n\n'.join(signals_long)
                        await send_telegram_message(msg)
                except Exception as e:
                    print(f"Ошибка долгосрок по {symbol}: {e}")
            last_long_signal = now
        # Alive-отчёт раз в 3 часа + список обработанных монет
        now = datetime.now() + timedelta(hours=TIME_SHIFT_HOURS)
        if (now - last_alive) > timedelta(hours=3):
            msg = f"⏳ Бот работает, обновил данные на {now.strftime('%d.%m.%Y %H:%M')}\n"
            msg += f"Обработано монет: {len(processed_symbols)}\n"
            msg += ', '.join(processed_symbols) if processed_symbols else 'Монеты не обработаны.'
            if not signals_sent:
                msg += "\nСигналов нет."
            await send_telegram_message(msg)
            last_alive = now
        # Ежедневный отчёт (раз в сутки)
        if (now - last_report) > timedelta(hours=24):
            await send_daily_report()
            last_report = now
        await asyncio.sleep(60 * 3)  # Проверять каждые 3 минуты

if __name__ == '__main__':
    asyncio.run(main()) 