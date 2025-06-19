import ccxt
import pandas as pd
import ta
import logging
import time
from datetime import datetime, timedelta, timezone
from config import *
from crypto_signal_bot import (
    analyze, check_signals, evaluate_signal_strength, 
    signal_strength_label, get_24h_volume, SYMBOLS
)
import json
import os

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

# Файл для сохранения результатов тестирования
DAILY_TEST_FILE = 'daily_test_results.json'

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

# БЫСТРЫЙ РЕЖИМ - только проверенные чемпионы для тестирования фильтров
QUICK_TEST_SYMBOLS = [
    'DOGE/USDT:USDT',  # 100% винрейт
    'YFI/USDT:USDT',   # 100% винрейт
    'RUNE/USDT:USDT',  # 100% винрейт
    'TRX/USDT:USDT',   # 66.7% винрейт
    'TON/USDT:USDT',   # 66.7% винрейт
    'BTC/USDT:USDT',   # Всегда активная пара
    'BNB/USDT:USDT',   # 33% винрейт но стабильная
    'SUI/USDT:USDT'    # 50% винрейт
]

print(f"ТЕСТИРУЕМ РЕАЛЬНЫЕ МОНЕТЫ ИЗ БОТА: {len(TEST_SYMBOLS)} пар")
print(f"БЫСТРЫЙ РЕЖИМ ДОСТУПЕН: {len(QUICK_TEST_SYMBOLS)} топовых пар")

def get_ohlcv(symbol, hours_back=24):
    """Получить исторические данные за указанное количество часов."""
    try:
        # Рассчитываем лимит для получения данных за N часов
        candles_needed = int(hours_back * 60 / 15) + LIMIT  # 15м таймфрейм
        
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=candles_needed)
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

def check_signal_filters(df, symbol):
    """
    Проверяет все фильтры сигналов и возвращает детальную диагностику.
    Возвращает словарь с результатами всех проверок.
    """
    if df.empty or len(df) < MIN_15M_CANDLES:
        return {'error': 'Недостаточно данных', 'status': 'fail'}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Диагностика
    diag = {
        'status': 'pass',
        'filters': {},
        'metrics': {},
        'signal_strength': None,
        'triggers': {'buy': 0, 'sell': 0},
        'reasons_failed': []
    }
    
    # === ОСНОВНЫЕ МЕТРИКИ ===
    diag['metrics'] = {
        'rsi': last['rsi'],
        'adx': last['adx'],
        'ema_fast': last['ema_fast'],
        'ema_slow': last['ema_slow'],
        'macd': last.get('macd', 0),
        'macd_signal': last.get('macd_signal', 0),
        'close': last['close'],
        'atr': last['atr'],
        'spread_pct': last['spread_pct'],
        'volatility': last.get('volatility', 0)
    }
    
    # === ФИЛЬТРЫ ===
    
    # 1. Объём торгов
    volume = get_24h_volume(symbol)
    diag['metrics']['volume_24h'] = volume
    diag['filters']['volume'] = volume >= MIN_VOLUME_USDT
    if not diag['filters']['volume']:
        diag['reasons_failed'].append(f"Объём {volume/1_000_000:.1f}M < {MIN_VOLUME_USDT/1_000_000:.1f}M")
    
    # 2. Спред
    diag['filters']['spread'] = last['spread_pct'] <= MAX_SPREAD_PCT
    if not diag['filters']['spread']:
        diag['reasons_failed'].append(f"Спред {last['spread_pct']*100:.3f}% > {MAX_SPREAD_PCT*100:.3f}%")
    
    # 3. ADX (адаптивный)
    current_volatility = last.get('volatility', 0.02)
    is_high_vol = current_volatility > HIGH_VOLATILITY_THRESHOLD
    is_low_vol = current_volatility < LOW_VOLATILITY_THRESHOLD
    min_adx = HIGH_VOL_ADX_MIN if is_high_vol else (LOW_VOL_ADX_MIN if is_low_vol else MIN_ADX)
    
    diag['filters']['adx'] = last['adx'] >= min_adx
    diag['metrics']['min_adx_required'] = min_adx
    diag['metrics']['volatility_level'] = 'high' if is_high_vol else ('low' if is_low_vol else 'normal')
    if not diag['filters']['adx']:
        diag['reasons_failed'].append(f"ADX {last['adx']:.1f} < {min_adx:.1f} ({diag['metrics']['volatility_level']} vol)")
    
    # === АНАЛИЗ ТРИГГЕРОВ ===
    
    # Триггеры покупки
    buy_triggers = 0
    sell_triggers = 0
    
    # EMA кроссовер
    if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
        buy_triggers += 1
        diag['triggers']['ema_cross_up'] = True
    elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
        buy_triggers += 0.5
        diag['triggers']['price_above_ema'] = True
    
    if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
        sell_triggers += 1
        diag['triggers']['ema_cross_down'] = True
    elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
        sell_triggers += 0.5
        diag['triggers']['price_below_ema'] = True
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        if last['macd'] > last['macd_signal']:
            buy_triggers += 0.5
            diag['triggers']['macd_bullish'] = True
        if last['macd'] < last['macd_signal']:
            sell_triggers += 0.5
            diag['triggers']['macd_bearish'] = True
        
        # MACD кроссовер
        if prev['macd'] <= prev['macd_signal'] and last['macd'] > last['macd_signal']:
            buy_triggers += 0.5
            diag['triggers']['macd_cross_up'] = True
        if prev['macd'] >= prev['macd_signal'] and last['macd'] < last['macd_signal']:
            sell_triggers += 0.5
            diag['triggers']['macd_cross_down'] = True
    
    # Bollinger Bands
    if 'bollinger_low' in df.columns and 'bollinger_high' in df.columns:
        bb_position = (last['close'] - last['bollinger_low']) / (last['bollinger_high'] - last['bollinger_low'])
        diag['metrics']['bb_position'] = bb_position
        
        if bb_position <= 0.3:
            buy_triggers += 0.5
            diag['triggers']['bb_oversold'] = True
        if bb_position >= 0.7:
            sell_triggers += 0.5
            diag['triggers']['bb_overbought'] = True
    
    # VWAP
    if USE_VWAP and 'vwap' in df.columns:
        vwap_dev = last.get('vwap_deviation', 0)
        diag['metrics']['vwap_deviation'] = vwap_dev
        
        if vwap_dev <= 0 and vwap_dev >= -VWAP_DEVIATION_THRESHOLD * 2:
            buy_triggers += 0.3
            diag['triggers']['vwap_buy'] = True
        if vwap_dev >= 0 and vwap_dev <= VWAP_DEVIATION_THRESHOLD * 2:
            sell_triggers += 0.3
            diag['triggers']['vwap_sell'] = True
    
    diag['triggers']['buy'] = buy_triggers
    diag['triggers']['sell'] = sell_triggers
    
    # === ФИНАЛЬНАЯ ПРОВЕРКА СИГНАЛОВ ===
    now = datetime.now(timezone.utc)
    is_active_hour = now.hour in ACTIVE_HOURS_UTC
    effective_min_score = MIN_COMPOSITE_SCORE
    if is_active_hour:
        effective_min_score *= ACTIVE_HOURS_MULTIPLIER
    
    min_triggers = MIN_TRIGGERS_ACTIVE_HOURS if is_active_hour else MIN_TRIGGERS_INACTIVE_HOURS
    diag['metrics']['min_triggers_required'] = min_triggers
    diag['metrics']['effective_min_score'] = effective_min_score
    
    # Проверяем сигналы
    potential_buy = buy_triggers >= min_triggers and last['rsi'] <= 85
    potential_sell = sell_triggers >= min_triggers and last['rsi'] >= 15
    
    if potential_buy:
        try:
            score, pattern = evaluate_signal_strength(df, symbol, 'BUY')
            diag['signal_strength'] = {'type': 'BUY', 'score': score, 'pattern': pattern}
            diag['filters']['signal_strength'] = score >= effective_min_score
            if not diag['filters']['signal_strength']:
                diag['reasons_failed'].append(f"BUY score {score:.1f} < {effective_min_score:.1f}")
        except Exception as e:
            diag['reasons_failed'].append(f"Ошибка расчета BUY score: {e}")
    
    if potential_sell:
        try:
            score, pattern = evaluate_signal_strength(df, symbol, 'SELL')
            if diag['signal_strength'] is None or score > diag['signal_strength']['score']:
                diag['signal_strength'] = {'type': 'SELL', 'score': score, 'pattern': pattern}
            diag['filters']['signal_strength'] = score >= effective_min_score
            if not diag['filters']['signal_strength']:
                diag['reasons_failed'].append(f"SELL score {score:.1f} < {effective_min_score:.1f}")
        except Exception as e:
            diag['reasons_failed'].append(f"Ошибка расчета SELL score: {e}")
    
    # Определяем итоговый статус
    all_filters_pass = all(diag['filters'].values())
    has_triggers = buy_triggers >= min_triggers or sell_triggers >= min_triggers
    
    if not all_filters_pass:
        diag['status'] = 'filtered_out'
    elif not has_triggers:
        diag['status'] = 'no_triggers'
        diag['reasons_failed'].append(f"Недостаточно триггеров: BUY={buy_triggers:.1f}, SELL={sell_triggers:.1f} < {min_triggers}")
    elif potential_buy and last['rsi'] > 85:
        diag['status'] = 'rsi_overbought'
        diag['reasons_failed'].append(f"RSI {last['rsi']:.1f} > 85 (перекупленность)")
    elif potential_sell and last['rsi'] < 15:
        diag['status'] = 'rsi_oversold'
        diag['reasons_failed'].append(f"RSI {last['rsi']:.1f} < 15 (перепроданность)")
    elif diag['signal_strength'] and not diag['filters'].get('signal_strength', False):
        diag['status'] = 'weak_signal'
    else:
        diag['status'] = 'signal_generated'
    
    return diag

def simulate_day_signals(hours_back=24):
    """
    Симулирует работу бота за последние N часов, проверяя сигналы каждые SIGNAL_COOLDOWN_MINUTES минут.
    """
    print(f"\n==== СИМУЛЯЦИЯ РАБОТЫ БОТА ЗА {hours_back} ЧАСОВ ====")
    print(f"Интервал проверки: каждые {SIGNAL_COOLDOWN_MINUTES} минут")
    print(f"Таймфрейм: {TIMEFRAME}")
    
    # Рассчитываем количество проверок
    total_checks = int(hours_back * 60 / SIGNAL_COOLDOWN_MINUTES)
    print(f"Всего проверок: {total_checks}")
    
    simulation_results = {
        'total_checks': total_checks,
        'signals_found': 0,
        'symbols_tested': len(TEST_SYMBOLS),
        'filter_stats': {
            'volume': 0,
            'spread': 0,
            'adx': 0,
            'no_triggers': 0,
            'weak_signal': 0,
            'rsi_extreme': 0,
            'signal_generated': 0
        },
        'hourly_breakdown': {},
        'symbol_performance': {}
    }
    
    # Симулируем каждую проверку
    current_time = datetime.now(timezone.utc)
    start_time = current_time - timedelta(hours=hours_back)
    
    print(f"Период симуляции: {start_time.strftime('%d.%m %H:%M')} - {current_time.strftime('%d.%m %H:%M')} UTC")
    print("\nИдет симуляция...\n")
    
    for check_num in range(total_checks):
        check_time = start_time + timedelta(minutes=check_num * SIGNAL_COOLDOWN_MINUTES)
        hour_key = check_time.strftime('%H:00')
        
        if hour_key not in simulation_results['hourly_breakdown']:
            simulation_results['hourly_breakdown'][hour_key] = {'signals': 0, 'checks': 0}
        
        simulation_results['hourly_breakdown'][hour_key]['checks'] += len(TEST_SYMBOLS)
        
        # Проверяем каждый символ в это время
        for symbol in TEST_SYMBOLS:
            if symbol not in simulation_results['symbol_performance']:
                simulation_results['symbol_performance'][symbol] = {
                    'total_checks': 0,
                    'signals': 0,
                    'filter_fails': 0,
                    'last_fail_reason': None
                }
            
            simulation_results['symbol_performance'][symbol]['total_checks'] += 1
            
            # Имитируем получение данных на это время
            # (В реальной симуляции здесь бы были исторические данные)
            try:
                df = get_ohlcv(symbol, 24)  # Получаем данные
                if df.empty:
                    simulation_results['symbol_performance'][symbol]['filter_fails'] += 1
                    simulation_results['symbol_performance'][symbol]['last_fail_reason'] = 'Нет данных'
                    continue
                
                df = analyze(df)
                if df.empty:
                    simulation_results['symbol_performance'][symbol]['filter_fails'] += 1
                    simulation_results['symbol_performance'][symbol]['last_fail_reason'] = 'Ошибка анализа'
                    continue
                
                # Проверяем фильтры
                diag = check_signal_filters(df, symbol)
                
                if diag['status'] == 'signal_generated':
                    simulation_results['signals_found'] += 1
                    simulation_results['hourly_breakdown'][hour_key]['signals'] += 1
                    simulation_results['symbol_performance'][symbol]['signals'] += 1
                else:
                    simulation_results['symbol_performance'][symbol]['filter_fails'] += 1
                    simulation_results['symbol_performance'][symbol]['last_fail_reason'] = diag['status']
                
                # Собираем статистику фильтров
                if diag['status'] in simulation_results['filter_stats']:
                    simulation_results['filter_stats'][diag['status']] += 1
                
            except Exception as e:
                simulation_results['symbol_performance'][symbol]['filter_fails'] += 1
                simulation_results['symbol_performance'][symbol]['last_fail_reason'] = f'Ошибка: {str(e)[:50]}'
    
    return simulation_results

def run_detailed_diagnostic(quick_mode=False):
    """
    Запускает детальную диагностику текущего состояния всех пар.
    
    Args:
        quick_mode (bool): Если True, тестирует только топовые 8 монет для быстрой проверки
    """
    symbols_to_test = QUICK_TEST_SYMBOLS if quick_mode else TEST_SYMBOLS
    mode_text = "БЫСТРАЯ" if quick_mode else "ПОЛНАЯ"
    
    print(f"\n==== {mode_text} ДИАГНОСТИКА СИГНАЛОВ {datetime.now().strftime('%d.%m.%Y %H:%M:%S')} ====")
    print(f"Анализируем {len(symbols_to_test)} пар на предмет возможных сигналов...")
    if quick_mode:
        print("⚡ БЫСТРЫЙ РЕЖИМ: Тестируем только топовые эффективные монеты")
    print()
    
    diagnostic_results = {
        'timestamp': datetime.now().isoformat(),
        'total_symbols': len(symbols_to_test),
        'quick_mode': quick_mode,
        'results': {}
    }
    
    signals_found = 0
    total_analyzed = 0
    filter_summary = {
        'volume': 0,
        'spread': 0, 
        'adx': 0,
        'weak_signal': 0,
        'no_triggers': 0,
        'rsi_extreme': 0
    }
    
    for symbol in symbols_to_test:
        print(f"🔍 Анализ {symbol}...")
        
        try:
            # Получаем данные
            df = get_ohlcv(symbol)
            if df.empty:
                print(f"   ❌ Нет данных\n")
                continue
            
            # Анализируем
            df = analyze(df)
            if df.empty:
                print(f"   ❌ Ошибка анализа\n")
                continue
            
            total_analyzed += 1
            
            # Получаем диагностику
            diag = check_signal_filters(df, symbol)
            diagnostic_results['results'][symbol] = diag
            
            # Получаем объём для отображения
            volume_mln = diag['metrics']['volume_24h'] / 1_000_000
            
            if diag['status'] == 'signal_generated':
                signals_found += 1
                strength = diag['signal_strength']
                signal_type = strength['type']
                score = strength['score']
                
                print(f"   🎯 СИГНАЛ: {signal_type} | Score: {score:.1f}")
                print(f"   📊 RSI: {diag['metrics']['rsi']:.1f} | ADX: {diag['metrics']['adx']:.1f} | Vol: {volume_mln:.1f}M")
                print(f"   🔥 Триггеры: BUY={diag['triggers']['buy']:.1f}, SELL={diag['triggers']['sell']:.1f}")
                
                # Показываем активные триггеры
                active_triggers = []
                for trigger, active in diag['triggers'].items():
                    if trigger not in ['buy', 'sell'] and active:
                        active_triggers.append(trigger.replace('_', ' ').title())
                
                if active_triggers:
                    print(f"   ⚡ Активные: {', '.join(active_triggers)}")
                print()
            else:
                # Показываем почему сигнал не прошел
                print(f"   ⚠️  Статус: {diag['status'].upper()}")
                print(f"   📊 RSI: {diag['metrics']['rsi']:.1f} | ADX: {diag['metrics']['adx']:.1f} | Vol: {volume_mln:.1f}M")
                
                if diag['reasons_failed']:
                    print(f"   ❌ Причины: {'; '.join(diag['reasons_failed'][:2])}")
                
                # Считаем статистику фильтров
                if not diag['filters'].get('volume', True):
                    filter_summary['volume'] += 1
                elif not diag['filters'].get('spread', True):
                    filter_summary['spread'] += 1
                elif not diag['filters'].get('adx', True):
                    filter_summary['adx'] += 1
                elif diag['status'] == 'weak_signal':
                    filter_summary['weak_signal'] += 1
                elif diag['status'] == 'no_triggers':
                    filter_summary['no_triggers'] += 1
                elif diag['status'] in ['rsi_overbought', 'rsi_oversold']:
                    filter_summary['rsi_extreme'] += 1
                
                print()
                
        except Exception as e:
            print(f"   💥 Ошибка: {e}\n")
    
    # Финальная статистика
    print(f"{'='*80}")
    print(f"📋 ИТОГИ ДИАГНОСТИКИ:")
    print(f"🔍 Проанализировано пар: {total_analyzed}")
    print(f"🎯 Найдено сигналов: {signals_found}")
    
    if total_analyzed > 0:
        success_rate = (signals_found / total_analyzed) * 100
        print(f"📈 Процент успеха: {success_rate:.1f}%")
        
        print(f"\n🚫 АНАЛИЗ ФИЛЬТРАЦИИ:")
        total_filtered = sum(filter_summary.values())
        if total_filtered > 0:
            print(f"   📉 Низкий объём: {filter_summary['volume']} ({filter_summary['volume']/total_filtered*100:.1f}%)")
            print(f"   📊 Высокий спред: {filter_summary['spread']} ({filter_summary['spread']/total_filtered*100:.1f}%)")
            print(f"   📈 Слабый тренд (ADX): {filter_summary['adx']} ({filter_summary['adx']/total_filtered*100:.1f}%)")
            print(f"   🎯 Слабый сигнал: {filter_summary['weak_signal']} ({filter_summary['weak_signal']/total_filtered*100:.1f}%)")
            print(f"   ⚡ Мало триггеров: {filter_summary['no_triggers']} ({filter_summary['no_triggers']/total_filtered*100:.1f}%)")
            print(f"   🌡️ Экстремальный RSI: {filter_summary['rsi_extreme']} ({filter_summary['rsi_extreme']/total_filtered*100:.1f}%)")
        
        # Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ:")
        if filter_summary['volume'] > total_analyzed * 0.3:
            print(f"   📉 Снизить MIN_VOLUME_USDT с {MIN_VOLUME_USDT/1_000_000:.1f}M до {MIN_VOLUME_USDT*0.8/1_000_000:.1f}M")
        if filter_summary['adx'] > total_analyzed * 0.2:
            print(f"   📈 Снизить MIN_ADX с {MIN_ADX} до {MIN_ADX-2}")
        if filter_summary['weak_signal'] > total_analyzed * 0.2:
            print(f"   🎯 Снизить MIN_COMPOSITE_SCORE с {MIN_COMPOSITE_SCORE} до {MIN_COMPOSITE_SCORE-0.2}")
        if filter_summary['no_triggers'] > total_analyzed * 0.15:
            print(f"   ⚡ Уменьшить пороги триггеров или расширить RSI диапазоны")
    
    print(f"{'='*80}")
    
    # Сохраняем результаты
    diagnostic_results['summary'] = {
        'signals_found': signals_found,
        'total_analyzed': total_analyzed,
        'success_rate': (signals_found / total_analyzed * 100) if total_analyzed > 0 else 0,
        'filter_summary': filter_summary
    }
    
    return diagnostic_results

def save_test_results(results):
    """Сохраняет результаты тестирования в файл."""
    try:
        # Загружаем существующие результаты
        if os.path.exists(DAILY_TEST_FILE):
            with open(DAILY_TEST_FILE, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = []
        
        # Добавляем новый результат
        all_results.append(results)
        
        # Оставляем только последние 7 дней
        if len(all_results) > 7:
            all_results = all_results[-7:]
        
        # Сохраняем
        with open(DAILY_TEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Результаты сохранены в {DAILY_TEST_FILE}")
        
    except Exception as e:
        print(f"❌ Ошибка сохранения результатов: {e}")

def run_full_analysis(quick_mode=False):
    """Запускает полный анализ: текущую диагностику + симуляцию за день."""
    mode_text = "БЫСТРЫЙ" if quick_mode else "ПОЛНЫЙ"
    print(f"🚀 ЗАПУСК {mode_text}О АНАЛИЗА ТОРГОВЫХ СИГНАЛОВ")
    print("=" * 80)
    
    # 1. Текущая диагностика
    diagnostic_results = run_detailed_diagnostic(quick_mode)
    
    # 2. Симуляция за день
    simulation_results = simulate_day_signals(24)
    
    # 3. Отчет по симуляции
    print(f"\n==== РЕЗУЛЬТАТЫ СИМУЛЯЦИИ ЗА СУТКИ ====")
    print(f"📊 Всего проверок: {simulation_results['total_checks']} x {simulation_results['symbols_tested']} = {simulation_results['total_checks'] * simulation_results['symbols_tested']:,}")
    print(f"🎯 Найдено сигналов: {simulation_results['signals_found']}")
    
    if simulation_results['total_checks'] > 0:
        signals_per_hour = simulation_results['signals_found'] / 24
        print(f"⏰ Сигналов в час: {signals_per_hour:.1f}")
        print(f"📈 Частота сигналов: {simulation_results['signals_found'] / (simulation_results['total_checks'] * simulation_results['symbols_tested']) * 100:.3f}%")
    
    # Почасовой анализ
    print(f"\n⏰ ПОЧАСОВАЯ АКТИВНОСТЬ:")
    for hour in sorted(simulation_results['hourly_breakdown'].keys()):
        hour_data = simulation_results['hourly_breakdown'][hour]
        if hour_data['signals'] > 0:
            print(f"   {hour}: {hour_data['signals']} сигналов из {hour_data['checks']} проверок")
    
    # Топ символы
    print(f"\n🏆 ТОП-5 АКТИВНЫХ ПАР:")
    symbol_stats = [(symbol, data['signals']) for symbol, data in simulation_results['symbol_performance'].items()]
    symbol_stats.sort(key=lambda x: x[1], reverse=True)
    
    for i, (symbol, signals) in enumerate(symbol_stats[:5]):
        if signals > 0:
            total_checks = simulation_results['symbol_performance'][symbol]['total_checks']
            rate = signals / total_checks * 100 if total_checks > 0 else 0
            print(f"   {i+1}. {symbol}: {signals} сигналов ({rate:.1f}%)")
    
    # Объединяем результаты
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'diagnostic': diagnostic_results,
        'simulation': simulation_results,
        'config_snapshot': {
            'MIN_COMPOSITE_SCORE': MIN_COMPOSITE_SCORE,
            'MIN_ADX': MIN_ADX,
            'MIN_VOLUME_USDT': MIN_VOLUME_USDT,
            'SIGNAL_COOLDOWN_MINUTES': SIGNAL_COOLDOWN_MINUTES,
            'TIMEFRAME': TIMEFRAME
        }
    }
    
    # Сохраняем результаты
    save_test_results(full_results)
    
    # Финальные рекомендации
    print(f"\n{'='*80}")
    print("🎯 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ:")
    
    current_signals = diagnostic_results['summary']['signals_found']
    projected_daily = simulation_results['signals_found']
    
    if projected_daily >= 10:
        print("✅ ОТЛИЧНО! Прогнозируется 10+ сигналов в день")
    elif projected_daily >= 5:
        print("⚠️ ХОРОШО, но можно улучшить до 10+ сигналов")
        print("💡 Попробуйте немного снизить пороги фильтров")
    else:
        print("❌ МАЛО СИГНАЛОВ! Нужно существенно снизить пороги")
        print(f"💡 Рекомендации:")
        print(f"   - Снизить MIN_COMPOSITE_SCORE до {MIN_COMPOSITE_SCORE - 0.5}")
        print(f"   - Снизить MIN_ADX до {MIN_ADX - 3}")
        print(f"   - Снизить MIN_VOLUME_USDT до {int(MIN_VOLUME_USDT * 0.7):,}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    # Можно запускать разные режимы
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'quick' or command == 'q':
            # Быстрая диагностика (только топовые монеты)
            print("🚀 БЫСТРЫЙ ТЕСТ ФИЛЬТРОВ")
            run_detailed_diagnostic(quick_mode=True)
        elif command == 'diag':
            # Полная диагностика
            run_detailed_diagnostic(quick_mode=False)
        elif command == 'sim':
            # Только симуляция
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            simulate_day_signals(hours)
        elif command == 'full':
            # Полный анализ
            run_full_analysis(quick_mode=False)
        else:
            print("❌ Неизвестная команда!")
            print("📖 Доступные команды:")
            print("   py test_signals.py quick  - Быстрый тест фильтров (8 топовых монет)")
            print("   py test_signals.py diag   - Полная диагностика всех монет")
            print("   py test_signals.py sim    - Симуляция за сутки")
            print("   py test_signals.py full   - Полный анализ")
    else:
        # По умолчанию - быстрый режим
        print("🚀 БЫСТРЫЙ ТЕСТ ФИЛЬТРОВ ПО УМОЛЧАНИЮ")
        print("💡 Используйте 'py test_signals.py full' для полного анализа")
        run_detailed_diagnostic(quick_mode=True)