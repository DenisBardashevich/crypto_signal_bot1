import ccxt
import pandas as pd
import ta
import logging
import time
from datetime import datetime
from config import *
from crypto_signal_bot import (
    analyze, check_signals, evaluate_signal_strength, 
    signal_strength_label, get_24h_volume, SYMBOLS
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

# Получаем фьючерсные пары автоматически
def get_futures_symbols():
    """Получает список популярных фьючерсных пар с достаточным объёмом."""
    try:
        # Используем тот же список что и в основном боте
        return SYMBOLS
    except Exception as e:
        logging.error(f"Ошибка получения списка пар: {e}")
        # Резервный список топовых монет для 15м торговли
        return [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 
            'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT', 
            'LINK/USDT:USDT', 'BNB/USDT:USDT', 'ADA/USDT:USDT', 
            'DOT/USDT:USDT', '1000PEPE/USDT:USDT', 'WIF/USDT:USDT',
            'TIA/USDT:USDT', 'SEI/USDT:USDT', 'OP/USDT:USDT'
        ]

# Получаем список пар для тестирования
TEST_SYMBOLS = get_futures_symbols()
print(f"ТЕСТИРУЕМ РЕАЛЬНЫЕ МОНЕТЫ ИЗ БОТА: {len(TEST_SYMBOLS)} пар")

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
    print(f"\n==== ТЕСТ ОПТИМИЗИРОВАННЫХ СИГНАЛОВ {datetime.now().strftime('%d.%m.%Y %H:%M:%S')} ====")
    print(f"Цель: 10+ надёжных сигналов в сутки")
    print(f"Таймфрейм: {TIMEFRAME}")
    print(f"EMA: {MA_FAST}/{MA_SLOW}, RSI: {RSI_WINDOW}, MACD: {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}")
    print(f"Мин. композитный скор: {MIN_COMPOSITE_SCORE}")
    print(f"Тестируем {len(TEST_SYMBOLS)} пар...\n")
    
    signals_found = 0
    total_tested = 0
    
    for symbol in TEST_SYMBOLS:
        print(f"Проверка {symbol}...", end=" ")
        try:
            # Получаем данные
            df = get_ohlcv(symbol)
            if df.empty:
                print("Нет данных")
                continue
                
            # Проводим анализ
            df = analyze(df)
            if df.empty:
                print("Ошибка анализа")
                continue
                
            total_tested += 1
            
            # Получаем объём
            volume = get_24h_volume(symbol)
            volume_mln = volume / 1_000_000
            
            # Получаем последние индикаторы
            last = df.iloc[-1]
            
            # Проверяем сигналы
            signals = check_signals(df, symbol)
            if signals:
                signals_found += len(signals)
                print(f"\n🎯 СИГНАЛ НАЙДЕН!")
                print(f"Объём: {volume_mln:.1f}M USDT")
                for signal in signals:
                    print(f"{signal}")
                print("-" * 60)
            else:
                # Краткий статус
                print(f"RSI:{last['rsi']:.1f}, ADX:{last['adx']:.1f}, Vol:{volume_mln:.1f}M - OK")
                
        except Exception as e:
            print(f"Ошибка: {e}")
    
    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТ ТЕСТИРОВАНИЯ:")
    print(f"📊 Протестировано пар: {total_tested}")
    print(f"🎯 Найдено сигналов: {signals_found}")
    
    if total_tested > 0:
        signal_rate = (signals_found / total_tested) * 100
        print(f"📈 Частота сигналов: {signal_rate:.1f}% от пар")
        
        # Правильный прогноз на день:
        # 15м таймфрейм = 96 проверок в день
        # Кулдаун 30 минут = 2 таймфрейма = проверка каждые 30 минут = 48 проверок в день
        checks_per_day = 24 * 60 / SIGNAL_COOLDOWN_MINUTES  # 48 проверок в день
        daily_projection = (signals_found / total_tested) * total_tested * checks_per_day / total_tested
        daily_projection = signals_found * checks_per_day / total_tested if total_tested > 0 else 0
        
        print(f"🔮 Прогноз сигналов в день: ~{daily_projection:.0f} сигналов")
        print(f"⏰ Проверок в день с кулдауном {SIGNAL_COOLDOWN_MINUTES}м: {checks_per_day:.0f}")
        
        if daily_projection >= 10:
            print("✅ ЦЕЛЬ ДОСТИГНУТА: 10+ сигналов в день")
        else:
            print("⚠️ Нужно ещё снизить пороги для достижения 10+ сигналов")
            print(f"💡 Для 10+ сигналов нужна частота: {10 / checks_per_day * 100:.1f}% от пар")
    
    if signals_found == 0:
        print("❌ Сигналов не найдено - возможно, пороги слишком строгие")
        print("💡 Рекомендации:")
        print("   - Снизить MIN_COMPOSITE_SCORE с 7.0 до 6.5")
        print("   - Снизить MIN_ADX с 25 до 20")
        print("   - Увеличить RSI диапазоны")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    run_test() 