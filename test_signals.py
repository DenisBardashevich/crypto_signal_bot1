import ccxt
import pandas as pd
import ta
import logging
import time
from datetime import datetime
from config import *
from crypto_signal_bot import (
    analyze, check_signals, evaluate_signal_strength, 
    signal_strength_label, get_24h_volume, is_global_uptrend
)

# Настройка логирования
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()])

# Инициализация биржи
EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'  # Используем фьючерсный рынок (USDT perpetual)
    }
})

# Список пар для тестирования (топ-10)
TEST_SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 
    'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT', 
    'LINK/USDT:USDT', 'BNB/USDT:USDT', 'ADA/USDT:USDT', 
    'DOT/USDT:USDT'
]

def get_ohlcv(symbol):
    """Получить исторические данные по монете."""
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    except ccxt.RateLimitExceeded as e:
        logging.warning(f"Rate limit exceeded for {symbol}, жду {getattr(e, 'retry_after', 1)} сек.")
        time.sleep(getattr(e, 'retry_after', 1))
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Ошибка получения OHLCV по {symbol}: {e}")
        return pd.DataFrame()

def run_test():
    """Проверяет текущие сигналы для списка тестовых пар."""
    print(f"==== Тест сигналов {datetime.now().strftime('%d.%m.%Y %H:%M:%S')} ====")
    print(f"Таймфрейм: {TIMEFRAME}, EMA: {MA_FAST}/{MA_SLOW}\n")
    
    signals_found = False
    
    for symbol in TEST_SYMBOLS:
        print(f"Проверка {symbol}...")
        try:
            # Получаем данные
            df = get_ohlcv(symbol)
            if df.empty:
                print(f"  Нет данных для {symbol}")
                continue
                
            # Проводим анализ
            df = analyze(df)
            
            # Получаем объём
            volume = get_24h_volume(symbol)
            volume_mln = volume / 1_000_000
            
            # Проверяем глобальный тренд
            trend = is_global_uptrend(symbol)
            trend_str = "Восходящий" if trend else "Нисходящий"
            
            # Получаем последние индикаторы
            last = df.iloc[-1]
            
            # Выводим основные показатели
            print(f"  Объём: {volume_mln:.2f} млн USDT")
            print(f"  Глобальный тренд: {trend_str}")
            print(f"  RSI: {last['rsi']:.2f}")
            print(f"  ADX: {last['adx']:.2f}")
            print(f"  MACD: {last['macd']:.6f}")
            
            # Проверяем сигналы
            signals = check_signals(df, symbol)
            if signals:
                signals_found = True
                print("\n  ⚠️ НАЙДЕН СИГНАЛ:")
                for signal in signals:
                    print(f"  {signal.replace('нСила', 'н\nСила')}")
            else:
                print("  Сигналов нет")
                
            print("\n" + "-" * 50 + "\n")
            
        except Exception as e:
            logging.error(f"Ошибка при тестировании {symbol}: {e}")
    
    if not signals_found:
        print("Сигналов не найдено ни по одной паре.")

if __name__ == "__main__":
    run_test() 