#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для скачивания исторических данных по всем символам из crypto_signal_bot.py
Создает CSV файлы в папке data/ для использования в оптимизаторе
"""

import ccxt
import pandas as pd
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from crypto_signal_bot import SYMBOLS, TOP_SYMBOLS

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Инициализация биржи
EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'
    }
})

def create_data_folder():
    """Создает папку data если её нет"""
    if not os.path.exists('data'):
        os.makedirs('data')
        logging.info("Создана папка data/")

def get_symbol_filename(symbol):
    """Преобразует символ в имя файла"""
    # Убираем / и : из символа для создания имени файла
    filename = symbol.replace('/', '').replace(':', '')
    return f"{filename}_15m.csv"

def download_symbol_data(symbol, days_back=7):
    """
    Скачивает исторические данные для одного символа
    
    Args:
        symbol: Торговая пара (например, 'BTC/USDT:USDT')
        days_back: Количество дней назад для скачивания
    """
    try:
        filename = get_symbol_filename(symbol)
        filepath = os.path.join('data', filename)
        
        # Проверяем, существует ли файл и не устарел ли он
        if os.path.exists(filepath):
            file_time = os.path.getmtime(filepath)
            file_age_hours = (time.time() - file_time) / 3600
            
            # Если файл свежий (менее 6 часов), пропускаем
            if file_age_hours < 6:
                logging.info(f"Файл {filename} свежий ({file_age_hours:.1f} часов), пропускаем")
                return True
        
        logging.info(f"Скачиваем данные для {symbol}...")
        
        # Рассчитываем количество свечей (15-минутные свечи)
        candles_needed = int(days_back * 24 * 4) + 100  # 4 свечи в час + запас
        
        # Получаем данные
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe='15m', limit=candles_needed)
        
        if not ohlcv or len(ohlcv) < 100:
            logging.warning(f"Недостаточно данных для {symbol}: {len(ohlcv) if ohlcv else 0} свечей")
            return False
        
        # Создаем DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Сортируем по времени
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Сохраняем в CSV
        df.to_csv(filepath, index=False)
        
        logging.info(f"✅ {symbol}: сохранено {len(df)} свечей в {filename}")
        return True
        
    except ccxt.RateLimitExceeded as e:
        wait_time = getattr(e, 'retry_after', 1)
        logging.warning(f"Rate limit для {symbol}, жду {wait_time} сек.")
        time.sleep(wait_time)
        return False
        
    except Exception as e:
        logging.error(f"Ошибка скачивания {symbol}: {e}")
        return False

def download_all_symbols():
    """Скачивает данные по всем символам"""
    create_data_folder()
    
    print("🚀 ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ")
    print("="*50)
    print(f"📊 Символов для загрузки: {len(SYMBOLS)}")
    print(f"⏰ Период: 7 дней (15-минутные свечи)")
    print(f"📁 Папка: data/")
    print("="*50)
    
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{i}/{len(SYMBOLS)}] Обрабатываем {symbol}...")
        
        if download_symbol_data(symbol, days_back=7):
            successful += 1
        else:
            failed += 1
            
        # Небольшая пауза между запросами
        time.sleep(0.5)
    
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТ ЗАГРУЗКИ:")
    print(f"✅ Успешно: {successful}")
    print(f"❌ Ошибок: {failed}")
    print(f"📁 Файлы сохранены в папке data/")
    
    if successful > 0:
        print(f"\n🎯 Теперь можно запускать оптимизатор:")
        print(f"   py optimizer_bot_fixed.py")
    
    return successful, failed

def check_existing_files():
    """Проверяет существующие файлы в папке data"""
    if not os.path.exists('data'):
        print("❌ Папка data/ не существует")
        return []
    
    files = os.listdir('data')
    csv_files = [f for f in files if f.endswith('_15m.csv')]
    
    print(f"📁 Найдено {len(csv_files)} CSV файлов в папке data/")
    
    if csv_files:
        print("📋 Список файлов:")
        for i, filename in enumerate(csv_files[:10], 1):  # Показываем первые 10
            filepath = os.path.join('data', filename)
            file_size = os.path.getsize(filepath) / 1024  # KB
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"   {i:2d}. {filename} ({file_size:.1f} KB, {file_time.strftime('%d.%m %H:%M')})")
        
        if len(csv_files) > 10:
            print(f"   ... и еще {len(csv_files) - 10} файлов")
    
    return csv_files

def main():
    """Основная функция"""
    print("🔍 ПРОВЕРКА СУЩЕСТВУЮЩИХ ФАЙЛОВ")
    existing_files = check_existing_files()
    
    if existing_files:
        print(f"\n❓ Обновить существующие файлы? (y/n): ", end="")
        response = input().lower().strip()
        if response not in ['y', 'yes', 'да', 'д']:
            print("✅ Операция отменена")
            return
    
    print(f"\n🚀 НАЧИНАЕМ ЗАГРУЗКУ...")
    successful, failed = download_all_symbols()
    
    if successful > 0:
        print(f"\n✅ Загрузка завершена! Теперь можно запускать оптимизатор.")
    else:
        print(f"\n❌ Загрузка не удалась. Проверьте подключение к интернету и доступность биржи.")

if __name__ == '__main__':
    main() 