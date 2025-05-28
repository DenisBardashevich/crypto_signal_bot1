import ccxt
import pandas as pd
import ta
import asyncio
from telegram import Bot
import os
import json
from datetime import datetime, timedelta

# ========== НАСТРОЙКИ ==========
TELEGRAM_TOKEN = '8046529777:AAHV4BfC_cPz7AptR8k6MOKxGQA6FVMm6oM'  # Токен Telegram-бота
TELEGRAM_CHAT_ID = 931346988  # chat_id пользователя

EXCHANGE = ccxt.binance()
# Получаем все монеты с парой к USDT и фильтруем по объёму
markets = EXCHANGE.load_markets()
# Оставляем только монеты с объёмом > 1 500 000 USDT за сутки
SYMBOLS = [
    symbol for symbol in markets
    if symbol.endswith('/USDT')
    and markets[symbol]['active']
    and markets[symbol].get('quoteVolume', 0) is not None
    and markets[symbol].get('quoteVolume', 0) > 1_000_000
]
TIMEFRAME = '15m'  # Интервал свечей
LIMIT = 200  # Количество свечей для анализа

TAKE_PROFIT = 0.03  # +3%
STOP_LOSS = -0.03   # -3%

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
    """Анализ по индикаторам: SMA, RSI, MACD."""
    df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.macd_diff(df['close'])
    df['macd'] = macd
    return df

def check_signals(df):
    """Проверка на сигналы по стратегиям: теперь только Golden/Death Cross + MACD."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    # Golden Cross (SMA50 пересёк SMA200 вверх) + MACD бычий
    if prev['sma50'] < prev['sma200'] and last['sma50'] > last['sma200'] and last['macd'] > 0:
        signals.append('Сигнал: КУПИТЬ!\nПричина: SMA50 пересёк SMA200 вверх (Golden Cross), MACD бычий.')
    # Death Cross (SMA50 пересёк SMA200 вниз) + MACD медвежий
    if prev['sma50'] > prev['sma200'] and last['sma50'] < last['sma200'] and last['macd'] < 0:
        signals.append('Сигнал: ПРОДАТЬ!\nПричина: SMA50 пересёк SMA200 вниз (Death Cross), MACD медвежий.')
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
async def main():
    last_report = datetime.now()
    last_alive = datetime.now() - timedelta(hours=3)  # чтобы сразу отправить первое alive-сообщение
    while True:
        signals_sent = False
        for symbol in SYMBOLS:
            try:
                df = get_ohlcv(symbol)
                df = analyze(df)
                signals = check_signals(df)
                price = df['close'].iloc[-1]
                time = df['timestamp'].iloc[-1]
                # Проверка на открытые сделки
                if symbol in open_trades:
                    buy_price = open_trades[symbol]['buy_price']
                    change = (price - buy_price) / buy_price
                    # Тейк-профит
                    if change >= TAKE_PROFIT:
                        msg = f"🎯 {symbol} достиг цели +3%!\nРекомендуется ПРОДАТЬ для фиксации прибыли.\nТочка входа: {buy_price}, текущая цена: {price:.4f}"
                        await send_telegram_message(msg)
                        record_trade(symbol, 'SELL', price, time)
                        close_trade(symbol)
                        signals_sent = True
                        continue
                    # Стоп-лосс
                    if change <= STOP_LOSS:
                        msg = f"⚠️ {symbol} снизился на 3% от точки входа.\nРекомендуется ПРОДАТЬ для ограничения убытков.\nТочка входа: {buy_price}, текущая цена: {price:.4f}"
                        await send_telegram_message(msg)
                        record_trade(symbol, 'SELL', price, time)
                        close_trade(symbol)
                        signals_sent = True
                        continue
                # Сигналы на вход/выход
                if signals:
                    msg = f"\n\U0001F4B0 Сигналы для {symbol} на {time.strftime('%d.%m.%Y %H:%M')}:\n" + '\n\n'.join(signals)
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
                print(f"Ошибка по {symbol}: {e}")
        # Если не было сигналов, отправляем сообщение о работе раз в 3 часа
        now = datetime.now()
        if not signals_sent and (now - last_alive) > timedelta(hours=3):
            await send_telegram_message(f"⏳ Бот работает, обновил данные на {now.strftime('%d.%m.%Y %H:%M')}. Сигналов нет.")
            last_alive = now
        # Ежедневный отчёт (раз в сутки)
        if (now - last_report) > timedelta(hours=24):
            await send_daily_report()
            last_report = now
        await asyncio.sleep(60 * 5)  # Проверять каждые 5 минуты

if __name__ == '__main__':
    asyncio.run(main()) 