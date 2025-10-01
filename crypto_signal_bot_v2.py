"""
CRYPTO SIGNAL BOT V2 - СИНХРОНИЗИРОВАНО с optimizer_v2.py
Дата: 01.10.2025

БАЛАНС: 3 из 4 условий (2 из 3 + ADX обязателен)
- ADX ≥ MIN_ADX (обязательно)
- + Минимум 2 из 3: RSI, EMA тренд, MACD импульс
"""

import ccxt
import pandas as pd
import ta
import asyncio
from telegram import Bot
from telegram.ext import Application, CommandHandler
import os
import json
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import warnings

from config import *

warnings.filterwarnings('ignore', category=RuntimeWarning)

logging.basicConfig(
    level=logging.ERROR,  # Только ошибки (как в старом боте)
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('bot_v2.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# HTTP логи отключены через ERROR level

EXCHANGE = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

TOP_SYMBOLS = ['BNB/USDT:USDT', 'LTC/USDT:USDT', 'IMX/USDT:USDT', 'SUI/USDT:USDT', 'ORDI/USDT:USDT', 'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT', 'ADA/USDT:USDT']

markets = EXCHANGE.load_markets()
SYMBOLS = [s for s in TOP_SYMBOLS if s in markets and markets[s]['active'] and markets[s]['type'] == 'swap']
logging.info(f"✅ {len(SYMBOLS)} символов: {', '.join(SYMBOLS)}")

PORTFOLIO_FILE = 'virtual_portfolio_v2.json'

if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r') as f:
        portfolio = json.load(f)
else:
    portfolio = {'trades': [], 'open_positions': {}}

def save_portfolio():
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)

def open_position(symbol, side, price, timestamp, atr, score):
    portfolio['open_positions'][symbol] = {
        'side': side, 'entry_price': price, 'timestamp': timestamp.isoformat(),
        'atr': atr, 'score': score
    }
    save_portfolio()
    logging.info(f"📈 Открыта: {symbol} {side.upper()} @ {price:.6f}")

def close_position(symbol, price, timestamp, pnl_pct):
    if symbol not in portfolio['open_positions']:
        return
    pos = portfolio['open_positions'][symbol]
    trade = {
        'symbol': symbol, 'side': pos['side'], 'entry_price': pos['entry_price'],
        'exit_price': price, 'entry_time': pos['timestamp'], 'exit_time': timestamp.isoformat(),
        'pnl_pct': pnl_pct, 'score': pos.get('score', 0)
    }
    portfolio['trades'].append(trade)
    del portfolio['open_positions'][symbol]
    save_portfolio()
    result = "✅ ПРИБЫЛЬ" if pnl_pct > 0 else "❌ УБЫТОК"
    logging.info(f"📉 Закрыта: {symbol} {pos['side'].upper()} @ {price:.6f} | P&L: {pnl_pct:+.2f}% | {result}")

def get_ohlcv(symbol):
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
        if not ohlcv or len(ohlcv) < MA_SLOW:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    except Exception as e:
        logging.error(f"❌ {symbol}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    try:
        if df.empty or len(df) < MA_SLOW:
            return pd.DataFrame()
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=MA_FAST)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=MA_SLOW)
        df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_WINDOW)
        macd = ta.trend.MACD(close=df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=ADX_WINDOW)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=ATR_WINDOW)
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        logging.error(f"❌ Индикаторы: {e}")
        return pd.DataFrame()

def calculate_signal_strength(df, signal_type):
    """Расчет силы сигнала (синхронизировано с optimizer_v2.py)"""
    last = df.iloc[-1]
    score = 0
    rsi_norm = (1 - last['rsi'] / 100) if signal_type == 'LONG' else (last['rsi'] / 100)
    score += rsi_norm * WEIGHT_RSI
    score += abs(last['macd_line'] - last['macd_signal']) * WEIGHT_MACD
    score += (last['adx'] / 100) * WEIGHT_ADX
    return round(score, 2)

def calculate_tp_sl(price, atr, signal_type):
    """Расчет TP/SL (синхронизировано с optimizer_v2.py)"""
    if signal_type == 'LONG':
        tp = price + atr * TP_ATR_MULT
        sl = price - atr * SL_ATR_MULT
        if (tp - price) / price < TP_MIN: tp = price * (1 + TP_MIN)
        if (price - sl) / price < SL_MIN: sl = price * (1 - SL_MIN)
    else:
        tp = price - atr * TP_ATR_MULT
        sl = price + atr * SL_ATR_MULT
        if (price - tp) / price < TP_MIN: tp = price * (1 - TP_MIN)
        if (sl - price) / price < SL_MIN: sl = price * (1 + SL_MIN)
    return tp, sl

def check_signal(df, symbol):
    """RSI обязателен + подтверждение (EMA или MACD) - синхронизировано с optimizer_v2.py"""
    if df.empty or len(df) < 2:
        return None
    last = df.iloc[-1]
    if last['adx'] < MIN_ADX:
        return None
    
    # RSI ОБЯЗАТЕЛЕН
    rsi_long = last['rsi'] <= RSI_MIN
    rsi_short = last['rsi'] >= RSI_MAX
    
    # Подтверждения (нужно хотя бы 1)
    ema_bull = last['ema_fast'] > last['ema_slow']
    ema_bear = last['ema_fast'] < last['ema_slow']
    macd_bull = last['macd_line'] > last['macd_signal']
    macd_bear = last['macd_line'] < last['macd_signal']
    
    # RSI + (EMA или MACD)
    if rsi_long and (ema_bull or macd_bull):
        signal_type = 'LONG'
        logging.info(f"🟢 {symbol}: LONG (RSI+подтв) | RSI={last['rsi']:.1f} ADX={last['adx']:.1f}")
    elif rsi_short and (ema_bear or macd_bear):
        signal_type = 'SHORT'
        logging.info(f"🔴 {symbol}: SHORT (RSI+подтв) | RSI={last['rsi']:.1f} ADX={last['adx']:.1f}")
    else:
        return None
    score = calculate_signal_strength(df, signal_type)
    tp_price, sl_price = calculate_tp_sl(last['close'], last['atr'], signal_type)
    return {
        'symbol': symbol, 'type': signal_type, 'price': last['close'], 'timestamp': last['timestamp'],
        'score': score, 'tp_price': tp_price, 'sl_price': sl_price,
        'rsi': last['rsi'], 'adx': last['adx'], 'atr': last['atr']
    }

def check_tp_sl(symbol, current_price, timestamp):
    if symbol not in portfolio['open_positions']:
        return False
    pos = portfolio['open_positions'][symbol]
    tp, sl = calculate_tp_sl(pos['entry_price'], pos['atr'], pos['side'])
    reason = None
    if pos['side'] == 'LONG':
        if current_price >= tp: reason = "TP"
        elif current_price <= sl: reason = "SL"
        pnl = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
    else:
        if current_price <= tp: reason = "TP"
        elif current_price >= sl: reason = "SL"
        pnl = ((pos['entry_price'] - current_price) / pos['entry_price']) * 100
    if reason:
        close_position(symbol, current_price, timestamp, pnl)
        result = "✅ ПРИБЫЛЬ" if pnl > 0 else "❌ УБЫТОК"
        msg = f"🔔 {symbol} {pos['side'].upper()} по {reason}\nВход: {pos['entry_price']:.6f}\nВыход: {current_price:.6f}\nP&L: {pnl:+.2f}%\n{result}"
        asyncio.create_task(send_telegram(msg))
        return True
    return False

async def send_telegram(text):
    try:
        await Bot(token=TELEGRAM_TOKEN).send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        logging.error(f"❌ Telegram: {e}")

async def stats_command(update, context):
    t = portfolio['trades']
    if not t:
        await update.message.reply_text("📊 Нет сделок")
        return
    total = sum(x['pnl_pct'] for x in t)
    wins = sum(1 for x in t if x['pnl_pct'] > 0)
    msg = f"📊 Статистика:\nСделок: {len(t)}\nПрибыльных: {wins}\nУбыточных: {len(t)-wins}\nWR: {wins/len(t)*100:.1f}%\nP&L: {total:+.2f}%\nСредний: {total/len(t):+.2f}%"
    await update.message.reply_text(msg)

async def positions_command(update, context):
    if not portfolio['open_positions']:
        await update.message.reply_text("📭 Нет позиций")
        return
    msg = "📈 Позиции:\n\n"
    for sym, pos in portfolio['open_positions'].items():
        df = get_ohlcv(sym)
        if df.empty: continue
        cur = df['close'].iloc[-1]
        pnl = ((cur - pos['entry_price']) / pos['entry_price'] * 100) if pos['side'] == 'LONG' else ((pos['entry_price'] - cur) / pos['entry_price'] * 100)
        e = "🟢" if pnl > 0 else "🔴"
        msg += f"{e} {sym} {pos['side'].upper()}\nВход: {pos['entry_price']:.6f}\nТекущая: {cur:.6f}\nP&L: {pnl:+.2f}%\n\n"
    await update.message.reply_text(msg)

async def clear_command(update, context):
    portfolio['trades'] = []
    portfolio['open_positions'] = {}
    save_portfolio()
    await update.message.reply_text("🗑 Очищено")

async def help_command(update, context):
    await update.message.reply_text("🤖 Команды:\n/stats - Статистика\n/positions - Позиции\n/clear - Очистить\n/help - Справка")

last_signal_time = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))

async def monitor_positions():
    while True:
        try:
            for symbol in list(portfolio['open_positions'].keys()):
                df = get_ohlcv(symbol)
                if not df.empty:
                    df = calculate_indicators(df)
                    if not df.empty:
                        check_tp_sl(symbol, df['close'].iloc[-1], df['timestamp'].iloc[-1])
            await asyncio.sleep(180)
        except Exception as e:
            logging.error(f"❌ Мониторинг: {e}")
            await asyncio.sleep(60)

async def scan_markets():
    while True:
        try:
            signals = []
            for symbol in SYMBOLS:
                if symbol in last_signal_time:
                    if datetime.now(timezone.utc) - last_signal_time[symbol] < timedelta(minutes=SIGNAL_COOLDOWN_MINUTES):
                        continue
                if symbol in portfolio['open_positions']:
                    continue
                df = get_ohlcv(symbol)
                if not df.empty:
                    df = calculate_indicators(df)
                    if not df.empty:
                        signal = check_signal(df, symbol)
                        if signal:
                            signals.append(signal)
                            last_signal_time[symbol] = datetime.now(timezone.utc)
            if signals:
                signals.sort(key=lambda x: x['score'], reverse=True)
                msg = f"💰 Сигналы ({len(signals)}):\n\n"
                for sig in signals:
                    e = "🟢" if sig['type'] == 'LONG' else "🔴"
                    if sig['type'] == 'LONG':
                        tp_pct = ((sig['tp_price'] - sig['price']) / sig['price']) * 100
                        sl_pct = ((sig['price'] - sig['sl_price']) / sig['price']) * 100
                    else:
                        tp_pct = ((sig['price'] - sig['tp_price']) / sig['price']) * 100
                        sl_pct = ((sig['sl_price'] - sig['price']) / sig['price']) * 100
                    msg += f"{e} {sig['symbol']} {sig['type']}\nЦена: {sig['price']:.6f}\nСила: {sig['score']:.1f}\nTP: +{tp_pct:.2f}% | SL: -{sl_pct:.2f}%\nR:R = {tp_pct/sl_pct:.2f}:1\nRSI: {sig['rsi']:.1f} ADX: {sig['adx']:.1f}\n\n"
                    open_position(sig['symbol'], sig['type'], sig['price'], sig['timestamp'], sig['atr'], sig['score'])
                await send_telegram(msg)
            await asyncio.sleep(300)
        except Exception as e:
            logging.error(f"❌ Сканирование: {e}")
            await asyncio.sleep(60)

async def telegram_bot():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("positions", positions_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("start", help_command))
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    await asyncio.Event().wait()

async def main():
    logging.info("🚀 Crypto Signal Bot V2")
    logging.info(f"📊 Логика: RSI обязателен + подтверждение (EMA или MACD)")
    logging.info(f"⏱️ Cooldown: {SIGNAL_COOLDOWN_MINUTES} мин")
    
    # Отправляем уведомление о запуске
    startup_msg = f"🚀 Бот V2 запущен!\n\n"
    startup_msg += f"📊 Логика: RSI обязателен + подтверждение\n"
    startup_msg += f"⏱️ Cooldown: {SIGNAL_COOLDOWN_MINUTES} мин\n"
    startup_msg += f"🎯 Отслеживаем: {len(SYMBOLS)} монет\n"
    startup_msg += f"📈 Параметры:\n"
    startup_msg += f"  • RSI: {RSI_MIN}-{RSI_MAX} (окно {RSI_WINDOW})\n"
    startup_msg += f"  • ADX: ≥{MIN_ADX} (окно {ADX_WINDOW})\n"
    startup_msg += f"  • EMA: {MA_FAST}/{MA_SLOW}\n"
    startup_msg += f"  • MACD: {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}\n"
    startup_msg += f"  • Веса: RSI={WEIGHT_RSI} MACD={WEIGHT_MACD} ADX={WEIGHT_ADX}\n"
    startup_msg += f"\n✅ Готов к работе!"
    
    await send_telegram(startup_msg)
    
    await asyncio.gather(telegram_bot(), scan_markets(), monitor_positions())

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("⏸ Остановлен")
    except Exception as e:
        logging.error(f"❌ Ошибка: {e}")

