import ccxt
import pandas as pd
import ta
import numpy as np
import json
import os
from datetime import datetime, timedelta, timezone
from config import *
from crypto_signal_bot import analyze, evaluate_signal_strength, get_24h_volume, SYMBOLS
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()])

# Инициализация биржи
EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'
    }
})

def get_historical_data(symbol, hours_back=168):  # 7 дней по умолчанию
    """Получает исторические данные для анализа."""
    try:
        # Рассчитываем лимит свечей
        candles_needed = int(hours_back * 60 / 15) + 100  # 15м таймфрейм + запас
        
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=candles_needed)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"Ошибка получения данных для {symbol}: {e}")
        return pd.DataFrame()

def simulate_signal_generation(df, symbol):
    """
    Симулирует генерацию сигналов на исторических данных.
    Возвращает список сигналов с их результатами.
    """
    if df.empty or len(df) < MIN_15M_CANDLES + 50:
        return []
    
    # Анализируем данные
    df_analyzed = analyze(df.copy())
    if df_analyzed.empty:
        return []
    
    signals = []
    last_signal_time = None
    
    # Проходим по каждой свече, начиная с достаточного количества для анализа
    for i in range(MIN_15M_CANDLES, len(df_analyzed) - 20):  # Оставляем 20 свечей для анализа результата
        current_df = df_analyzed.iloc[:i+1].copy()
        current_time = current_df.iloc[-1]['timestamp']
        
        # Проверяем кулдаун
        if last_signal_time and (current_time - last_signal_time).total_seconds() < SIGNAL_COOLDOWN_MINUTES * 60:
            continue
        
        # Получаем данные текущей и предыдущей свечи
        last = current_df.iloc[-1]
        prev = current_df.iloc[-2]
        
        # Базовые фильтры
        volume = get_24h_volume(symbol) if i == len(df_analyzed) - 21 else 1_000_000  # Упрощаем для исторических данных
        if volume < MIN_VOLUME_USDT * 0.5:  # Более мягкий фильтр для истории
            continue
        
        if last['spread_pct'] > MAX_SPREAD_PCT:
            continue
        
        if last['adx'] < MIN_ADX:
            continue
        
        # Проверяем триггеры покупки
        buy_triggers = 0
        sell_triggers = 0
        
        # EMA кроссовер
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            buy_triggers += 1
        elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
            buy_triggers += 0.5
        
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            sell_triggers += 1
        elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
            sell_triggers += 0.5
        
        # MACD
        if 'macd' in current_df.columns:
            if last['macd'] > last['macd_signal']:
                buy_triggers += 0.5
            if last['macd'] < last['macd_signal']:
                sell_triggers += 0.5
        
        # Bollinger Bands
        if 'bollinger_low' in current_df.columns:
            bb_position = (last['close'] - last['bollinger_low']) / (last['bollinger_high'] - last['bollinger_low'])
            if bb_position <= 0.3:
                buy_triggers += 0.5
            if bb_position >= 0.7:
                sell_triggers += 0.5
        
        # VWAP
        if USE_VWAP and 'vwap' in current_df.columns:
            vwap_dev = last.get('vwap_deviation', 0)
            if vwap_dev <= 0 and vwap_dev >= -VWAP_DEVIATION_THRESHOLD * 2:
                buy_triggers += 0.3
            if vwap_dev >= 0 and vwap_dev <= VWAP_DEVIATION_THRESHOLD * 2:
                sell_triggers += 0.3
        
        # Проверяем минимальные триггеры
        min_triggers = 1.0
        
        signal_type = None
        if buy_triggers >= min_triggers and last['rsi'] <= 85:
            signal_type = 'BUY'
        elif sell_triggers >= min_triggers and last['rsi'] >= 15:
            signal_type = 'SELL'
        
        if signal_type:
            try:
                # Оцениваем силу сигнала
                score, pattern = evaluate_signal_strength(current_df, symbol, signal_type)
                
                if score >= MIN_COMPOSITE_SCORE:
                    # Анализируем результат сигнала
                    entry_price = last['close']
                    entry_time = current_time
                    
                    # Смотрим что случилось в следующие N свечей
                    future_data = df_analyzed.iloc[i+1:i+21]  # Следующие 20 свечей (~5 часов)
                    
                    if len(future_data) >= 10:
                        signal_result = analyze_signal_outcome(
                            entry_price, signal_type, future_data, last['atr']
                        )
                        
                        signal_info = {
                            'symbol': symbol,
                            'type': signal_type,
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'score': score,
                            'pattern': pattern,
                            'rsi': last['rsi'],
                            'adx': last['adx'],
                            'triggers': {'buy': buy_triggers, 'sell': sell_triggers},
                            'result': signal_result
                        }
                        
                        signals.append(signal_info)
                        last_signal_time = current_time
                        
            except Exception as e:
                logging.warning(f"Ошибка оценки сигнала {symbol} в {current_time}: {e}")
                continue
    
    return signals

def analyze_signal_outcome(entry_price, signal_type, future_data, atr):
    """
    Анализирует результат сигнала на основе будущих данных.
    """
    if future_data.empty:
        return {'status': 'no_data', 'pnl_pct': 0, 'max_profit': 0, 'max_loss': 0, 'duration': 0}
    
    # Рассчитываем TP и SL на основе ATR
    tp_distance = atr * TP_ATR_MULT
    sl_distance = atr * SL_ATR_MULT
    
    if signal_type == 'BUY':
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:  # SELL
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance
    
    max_profit = 0
    max_loss = 0
    exit_price = None
    exit_reason = None
    duration_candles = 0
    
    for idx, candle in future_data.iterrows():
        duration_candles += 1
        current_price = candle['close']
        
        # Рассчитываем текущий P&L
        if signal_type == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Проверяем TP/SL
            if candle['high'] >= tp_price:
                exit_price = tp_price
                exit_reason = 'tp'
                break
            elif candle['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = 'sl'
                break
                
        else:  # SELL
            pnl_pct = (entry_price - current_price) / entry_price
            
            # Проверяем TP/SL
            if candle['low'] <= tp_price:
                exit_price = tp_price
                exit_reason = 'tp'
                break
            elif candle['high'] >= sl_price:
                exit_price = sl_price
                exit_reason = 'sl'
                break
        
        # Отслеживаем максимальную прибыль и убыток
        max_profit = max(max_profit, pnl_pct)
        max_loss = min(max_loss, pnl_pct)
    
    # Если не сработали TP/SL, берем последнюю цену
    if exit_price is None:
        exit_price = future_data.iloc[-1]['close']
        exit_reason = 'timeout'
    
    # Финальный P&L
    if signal_type == 'BUY':
        final_pnl = (exit_price - entry_price) / entry_price
    else:
        final_pnl = (entry_price - exit_price) / entry_price
    
    return {
        'status': 'completed',
        'exit_reason': exit_reason,
        'exit_price': exit_price,
        'pnl_pct': final_pnl * 100,
        'max_profit': max_profit * 100,
        'max_loss': max_loss * 100,
        'duration_candles': duration_candles,
        'duration_hours': duration_candles * 0.25  # 15м = 0.25 часа
    }

def run_historical_analysis(hours_back=168, max_symbols=10):
    """
    Запускает исторический анализ сигналов за указанный период.
    """
    print(f"🕒 АНАЛИЗ ИСТОРИЧЕСКОЙ ЭФФЕКТИВНОСТИ СИГНАЛОВ")
    print(f"Период: {hours_back} часов ({hours_back/24:.1f} дней)")
    print(f"Анализируем топ-{max_symbols} символов...")
    print("="*70)
    
    # Берем первые N символов для анализа
    test_symbols = SYMBOLS[:max_symbols]
    all_signals = []
    
    for i, symbol in enumerate(test_symbols):
        print(f"\n📈 Анализ {symbol} ({i+1}/{len(test_symbols)})")
        
        # Получаем исторические данные
        df = get_historical_data(symbol, hours_back)
        if df.empty:
            print(f"   ❌ Нет данных")
            continue
        
        # Симулируем сигналы
        signals = simulate_signal_generation(df, symbol)
        print(f"   🎯 Найдено сигналов: {len(signals)}")
        
        if signals:
            # Краткая статистика по символу
            profitable = sum(1 for s in signals if s['result']['pnl_pct'] > 0)
            win_rate = profitable / len(signals) * 100
            avg_pnl = np.mean([s['result']['pnl_pct'] for s in signals])
            
            print(f"   📊 Прибыльных: {profitable}/{len(signals)} ({win_rate:.1f}%)")
            print(f"   💰 Средний P&L: {avg_pnl:.2f}%")
            
            all_signals.extend(signals)
    
    # Общая статистика
    if not all_signals:
        print("\n❌ Сигналы не найдены!")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 ОБЩАЯ СТАТИСТИКА ({len(all_signals)} сигналов)")
    
    # Базовая статистика
    profitable_signals = [s for s in all_signals if s['result']['pnl_pct'] > 0]
    losing_signals = [s for s in all_signals if s['result']['pnl_pct'] <= 0]
    
    win_rate = len(profitable_signals) / len(all_signals) * 100
    avg_profit = np.mean([s['result']['pnl_pct'] for s in profitable_signals]) if profitable_signals else 0
    avg_loss = np.mean([s['result']['pnl_pct'] for s in losing_signals]) if losing_signals else 0
    total_pnl = sum(s['result']['pnl_pct'] for s in all_signals)
    
    print(f"🎯 Винрейт: {win_rate:.1f}% ({len(profitable_signals)}/{len(all_signals)})")
    print(f"📈 Средняя прибыль: {avg_profit:.2f}%")
    print(f"📉 Средний убыток: {avg_loss:.2f}%")
    print(f"💰 Общий P&L: {total_pnl:.2f}%")
    
    # Статистика по типам сигналов
    buy_signals = [s for s in all_signals if s['type'] == 'BUY']
    sell_signals = [s for s in all_signals if s['type'] == 'SELL']
    
    if buy_signals:
        buy_winrate = sum(1 for s in buy_signals if s['result']['pnl_pct'] > 0) / len(buy_signals) * 100
        buy_pnl = sum(s['result']['pnl_pct'] for s in buy_signals)
        print(f"🟢 LONG сигналы: {len(buy_signals)} шт., винрейт {buy_winrate:.1f}%, P&L {buy_pnl:.2f}%")
    
    if sell_signals:
        sell_winrate = sum(1 for s in sell_signals if s['result']['pnl_pct'] > 0) / len(sell_signals) * 100
        sell_pnl = sum(s['result']['pnl_pct'] for s in sell_signals)
        print(f"🔴 SHORT сигналы: {len(sell_signals)} шт., винрейт {sell_winrate:.1f}%, P&L {sell_pnl:.2f}%")
    
    # Статистика по силе сигналов
    print(f"\n📈 АНАЛИЗ ПО СИЛЕ СИГНАЛОВ:")
    score_ranges = [
        (8.0, float('inf'), 'Экстремально сильные'),
        (7.0, 8.0, 'Очень сильные'),
        (6.5, 7.0, 'Сильные'),
        (6.0, 6.5, 'Умеренные'),
        (0, 6.0, 'Слабые')
    ]
    
    for min_score, max_score, label in score_ranges:
        range_signals = [s for s in all_signals if min_score <= s['score'] < max_score]
        if range_signals:
            range_winrate = sum(1 for s in range_signals if s['result']['pnl_pct'] > 0) / len(range_signals) * 100
            range_pnl = sum(s['result']['pnl_pct'] for s in range_signals)
            avg_score = np.mean([s['score'] for s in range_signals])
            print(f"   {label} ({avg_score:.1f}): {len(range_signals)} сигналов, {range_winrate:.1f}% винрейт, {range_pnl:.2f}% P&L")
    
    # Анализ причин выхода
    print(f"\n🚪 ПРИЧИНЫ ВЫХОДА:")
    exit_reasons = {}
    for signal in all_signals:
        reason = signal['result']['exit_reason']
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'pnl': 0}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += signal['result']['pnl_pct']
    
    for reason, data in exit_reasons.items():
        avg_pnl = data['pnl'] / data['count']
        print(f"   {reason.upper()}: {data['count']} сигналов ({data['count']/len(all_signals)*100:.1f}%), средний P&L: {avg_pnl:.2f}%")
    
    # Топ и худшие сигналы
    print(f"\n🏆 ТОП-3 ЛУЧШИХ СИГНАЛА:")
    best_signals = sorted(all_signals, key=lambda x: x['result']['pnl_pct'], reverse=True)[:3]
    for i, signal in enumerate(best_signals):
        print(f"   {i+1}. {signal['symbol']} {signal['type']} - {signal['result']['pnl_pct']:.2f}% (score: {signal['score']:.1f})")
    
    print(f"\n💸 ТОП-3 ХУДШИХ СИГНАЛА:")
    worst_signals = sorted(all_signals, key=lambda x: x['result']['pnl_pct'])[:3]
    for i, signal in enumerate(worst_signals):
        print(f"   {i+1}. {signal['symbol']} {signal['type']} - {signal['result']['pnl_pct']:.2f}% (score: {signal['score']:.1f})")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    
    if win_rate < 60:
        print(f"   ⚠️ Низкий винрейт ({win_rate:.1f}%). Рассмотрите повышение MIN_COMPOSITE_SCORE")
    
    if avg_loss < -3:
        print(f"   ⚠️ Большие убытки ({avg_loss:.2f}%). Проверьте настройки SL")
    
    if len(buy_signals) == 0 or len(sell_signals) == 0:
        print(f"   ⚠️ Дисбаланс типов сигналов. Проверьте триггеры")
    
    # Расчет оптимального минимального скора
    optimal_score = find_optimal_min_score(all_signals)
    if optimal_score > MIN_COMPOSITE_SCORE:
        print(f"   💡 Рекомендуемый MIN_COMPOSITE_SCORE: {optimal_score:.1f} (текущий: {MIN_COMPOSITE_SCORE})")
    
    # Сохранение результатов
    results = {
        'timestamp': datetime.now().isoformat(),
        'period_hours': hours_back,
        'total_signals': len(all_signals),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'signals': all_signals[:50]  # Сохраняем только первые 50 для экономии места
    }
    
    with open('historical_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✅ Результаты сохранены в historical_analysis.json")
    print("="*70)

def find_optimal_min_score(signals):
    """Находит оптимальный минимальный скор для максимизации прибыли."""
    scores_to_test = np.arange(5.0, 8.5, 0.1)
    best_score = MIN_COMPOSITE_SCORE
    best_result = -float('inf')
    
    for test_score in scores_to_test:
        filtered_signals = [s for s in signals if s['score'] >= test_score]
        
        if len(filtered_signals) < 10:  # Минимум 10 сигналов для статистики
            continue
        
        profitable = sum(1 for s in filtered_signals if s['result']['pnl_pct'] > 0)
        win_rate = profitable / len(filtered_signals)
        total_pnl = sum(s['result']['pnl_pct'] for s in filtered_signals)
        
        # Критерий: винрейт > 60% и положительный общий P&L
        if win_rate > 0.6 and total_pnl > best_result:
            best_result = total_pnl
            best_score = test_score
    
    return best_score

def run_3day_filter_optimizer():
    """
    Улучшенный подбор фильтров для реального бота по историческим данным за 3 дня:
    - Динамические фильтры (min_adx, min_score)
    - Исключение монет с плохой историей
    - Фильтр по времени (UTC)
    - Анализ распределения сигналов
    - Рекомендации по фильтрам и TP/SL
    """
    print("\n🚀 УЛУЧШЕННАЯ ОПТИМИЗАЦИЯ ФИЛЬТРОВ ПО ИСТОРИИ (3 дня)")
    print("="*80)
    hours_back = 72
    max_symbols = 15
    active_hours_utc = [8,9,10,11,12,13,14,15,16,17,18,19,20,21]  # только активные часы
    min_score_base = 3.5
    min_score_night = 4.2
    min_adx_base = 16
    min_adx_high_vol = 20
    min_adx_low_vol = 10
    min_volume = 400_000
    tp_mult = 1.3
    sl_mult = 2.3
    min_winrate = 70
    min_tp_sl_ratio = 1.7
    min_signals = 75
    # 1. Сбор сигналов по всем монетам
    all_signals = []
    mon_stats = {}
    for symbol in SYMBOLS[:max_symbols]:
        df = get_historical_data(symbol, hours_back)
        if df.empty:
            continue
        df_an = analyze(df.copy())
        if df_an.empty:
            continue
        signals = []
        last_signal_time = None
        for i in range(MIN_15M_CANDLES, len(df_an) - 20):
            current_df = df_an.iloc[:i+1].copy()
            last = current_df.iloc[-1]
            prev = current_df.iloc[-2]
            now = last['timestamp']
            hour_utc = now.hour
            # Фильтр по времени (UTC)
            if hour_utc not in active_hours_utc:
                continue
            # Динамический min_score
            min_score = min_score_base if hour_utc in active_hours_utc else min_score_night
            # Динамический min_adx по волатильности
            vol = last.get('volatility', 0.02)
            if vol > HIGH_VOLATILITY_THRESHOLD:
                min_adx = min_adx_high_vol
            elif vol < LOW_VOLATILITY_THRESHOLD:
                min_adx = min_adx_low_vol
            else:
                min_adx = min_adx_base
            # Spread
            if last['spread_pct'] > MAX_SPREAD_PCT:
                continue
            # ADX
            if last['adx'] < min_adx:
                continue
            # Объем (упрощенно)
            volume = 1_000_000
            if volume < min_volume:
                continue
            # Триггеры
            buy_triggers = 0
            sell_triggers = 0
            if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
                buy_triggers += 1
            elif last['close'] > last['ema_fast'] and last['close'] > prev['close']:
                buy_triggers += 0.5
            if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
                sell_triggers += 1
            elif last['close'] < last['ema_fast'] and last['close'] < prev['close']:
                sell_triggers += 0.5
            if 'macd' in current_df.columns:
                if last['macd'] > last['macd_signal']:
                    buy_triggers += 0.5
                if last['macd'] < last['macd_signal']:
                    sell_triggers += 0.5
            if 'bollinger_low' in current_df.columns:
                bb_position = (last['close'] - last['bollinger_low']) / (last['bollinger_high'] - last['bollinger_low'])
                if bb_position <= 0.3:
                    buy_triggers += 0.5
                if bb_position >= 0.7:
                    sell_triggers += 0.5
            if USE_VWAP and 'vwap' in current_df.columns:
                vwap_dev = last.get('vwap_deviation', 0)
                if vwap_dev <= 0 and vwap_dev >= -VWAP_DEVIATION_THRESHOLD * 2:
                    buy_triggers += 0.3
                if vwap_dev >= 0 and vwap_dev <= VWAP_DEVIATION_THRESHOLD * 2:
                    sell_triggers += 0.3
            min_triggers = 1.0
            signal_type = None
            if buy_triggers >= min_triggers and last['rsi'] <= 85:
                signal_type = 'BUY'
            elif sell_triggers >= min_triggers and last['rsi'] >= 15:
                signal_type = 'SELL'
            if signal_type:
                try:
                    score, pattern = evaluate_signal_strength(current_df, symbol, signal_type)
                    if score >= min_score:
                        entry_price = last['close']
                        entry_time = now
                        future_data = df_an.iloc[i+1:i+21]
                        if len(future_data) >= 10:
                            atr = last['atr']
                            tp_distance = atr * tp_mult
                            sl_distance = atr * sl_mult
                            if signal_type == 'BUY':
                                tp_price = entry_price + tp_distance
                                sl_price = entry_price - sl_distance
                            else:
                                tp_price = entry_price - tp_distance
                                sl_price = entry_price + sl_distance
                            result = None
                            for idx, candle in future_data.iterrows():
                                if signal_type == 'BUY':
                                    if candle['high'] >= tp_price:
                                        result = 'tp'
                                        break
                                    elif candle['low'] <= sl_price:
                                        result = 'sl'
                                        break
                                else:
                                    if candle['low'] <= tp_price:
                                        result = 'tp'
                                        break
                                    elif candle['high'] >= sl_price:
                                        result = 'sl'
                                        break
                            if not result:
                                result = 'timeout'
                            signals.append({
                                'symbol': symbol,
                                'type': signal_type,
                                'entry_time': entry_time,
                                'score': score,
                                'result': result,
                                'volatility': vol,
                                'hour': hour_utc,
                                'tp_pct': ((tp_price - entry_price) / entry_price * 100) if signal_type == 'BUY' else ((entry_price - tp_price) / entry_price * 100),
                                'sl_pct': ((entry_price - sl_price) / entry_price * 100) if signal_type == 'BUY' else ((sl_price - entry_price) / entry_price * 100)
                            })
                        last_signal_time = now
                except Exception as e:
                    logging.warning(f"Ошибка оценки сигнала {symbol} в {now}: {e}")
                    continue
        all_signals.extend(signals)
        # Статистика по монете
        tp_signals = [s for s in signals if s['result'] == 'tp']
        sl_signals = [s for s in signals if s['result'] == 'sl']
        winrate = len(tp_signals) / len(signals) * 100 if signals else 0
        mon_stats[symbol] = {
            'signals': len(signals),
            'winrate': winrate,
            'tp': len(tp_signals),
            'sl': len(sl_signals)
        }
    # Исключаем монеты с winrate < 40%
    good_symbols = [s for s, stat in mon_stats.items() if stat['winrate'] >= 40 and stat['signals'] > 0]
    print(f"\nМонеты с winrate >= 40%: {good_symbols}")
    filtered_signals = [s for s in all_signals if s['symbol'] in good_symbols]
    # Глобальная статистика
    tp_signals = [s for s in filtered_signals if s['result'] == 'tp']
    sl_signals = [s for s in filtered_signals if s['result'] == 'sl']
    winrate = len(tp_signals) / len(filtered_signals) * 100 if filtered_signals else 0
    avg_tp = np.mean([s['tp_pct'] for s in tp_signals]) if tp_signals else 0
    avg_sl = abs(np.mean([s['sl_pct'] for s in sl_signals])) if sl_signals else 0
    tp_sl_ratio = (avg_tp / avg_sl) if avg_sl > 0 else 0
    print(f"\n=== ГЛОБАЛЬНАЯ СТАТИСТИКА ===")
    print(f"Сигналов: {len(filtered_signals)}, TP: {len(tp_signals)}, SL: {len(sl_signals)}, Winrate: {winrate:.1f}%, TP/SL: {tp_sl_ratio:.2f}")
    # Анализ по score
    for rng in [(3,4),(4,5),(5,6),(6,7),(7,8),(8,10)]:
        group = [s for s in filtered_signals if rng[0]<=s['score']<rng[1]]
        if group:
            tp = [s for s in group if s['result']=='tp']
            wr = len(tp)/len(group)*100 if group else 0
            print(f"Score {rng[0]}-{rng[1]}: {len(group)} сигналов, winrate={wr:.1f}%")
    # Анализ по времени
    for h in active_hours_utc:
        group = [s for s in filtered_signals if s['hour']==h]
        if group:
            tp = [s for s in group if s['result']=='tp']
            wr = len(tp)/len(group)*100 if group else 0
            print(f"Час {h}: {len(group)} сигналов, winrate={wr:.1f}%")
    # Рекомендации
    print("\n=== РЕКОМЕНДАЦИИ ===")
    if winrate < min_winrate:
        print(f"   ⚠️ Повысить min_score или min_adx, либо уменьшить TP")
    if tp_sl_ratio < min_tp_sl_ratio:
        print(f"   ⚠️ Уменьшить SL или увеличить TP")
    if len(filtered_signals) < min_signals:
        print(f"   ⚠️ Смягчить фильтры или расширить список монет")
    print("\nЛучшие параметры:")
    print(f"min_score={min_score_base}, min_adx={min_adx_base}, min_volume={min_volume}, TP_ATR_MULT={tp_mult}, SL_ATR_MULT={sl_mult}")
    print(f"Монеты для торговли: {good_symbols}")
    print("="*80)

def test_new_settings():
    """
    Тестирует новые строгие настройки на исторических данных
    для оценки улучшения винрейта.
    """
    print(f"🔧 ТЕСТИРОВАНИЕ ИСПРАВЛЕННЫХ НАСТРОЕК")
    print(f"Цель: повысить винрейт с 41.6% до 60%+")
    print("="*70)
    
    # Тестируем на топ-10 символах за последние 3 дня
    test_symbols = SYMBOLS[:15]  # Больше символов для статистики
    hours_back = 72  # 3 дня
    all_signals = []
    
    for i, symbol in enumerate(test_symbols):
        print(f"\n📈 Тестирование {symbol} ({i+1}/{len(test_symbols)})")
        
        # Получаем исторические данные
        df = get_historical_data(symbol, hours_back)
        if df.empty:
            print(f"   ❌ Нет данных")
            continue
        
        # Симулируем сигналы с новыми настройками
        signals = simulate_signal_generation(df, symbol)
        print(f"   🎯 Найдено сигналов: {len(signals)}")
        
        if signals:
            # Краткая статистика по символу
            profitable = sum(1 for s in signals if s['result']['pnl_pct'] > 0)
            win_rate = profitable / len(signals) * 100
            avg_pnl = np.mean([s['result']['pnl_pct'] for s in signals])
            
            print(f"   📊 Винрейт: {win_rate:.1f}% ({profitable}/{len(signals)})")
            print(f"   💰 Средний P&L: {avg_pnl:.2f}%")
            
            all_signals.extend(signals)
    
    if not all_signals:
        print("\n❌ Сигналы не найдены с новыми настройками!")
        return
    
    # Анализ результатов
    print(f"\n{'='*70}")
    print(f"📊 РЕЗУЛЬТАТЫ ИСПРАВЛЕННОЙ СИСТЕМЫ ({len(all_signals)} сигналов)")
    
    # Основная статистика
    profitable_signals = [s for s in all_signals if s['result']['pnl_pct'] > 0]
    losing_signals = [s for s in all_signals if s['result']['pnl_pct'] <= 0]
    
    new_win_rate = len(profitable_signals) / len(all_signals) * 100
    avg_profit = np.mean([s['result']['pnl_pct'] for s in profitable_signals]) if profitable_signals else 0
    avg_loss = np.mean([s['result']['pnl_pct'] for s in losing_signals]) if losing_signals else 0
    total_pnl = sum(s['result']['pnl_pct'] for s in all_signals)
    
    print(f"🎯 НОВЫЙ винрейт: {new_win_rate:.1f}% (было 41.6%)")
    print(f"📈 Средняя прибыль: {avg_profit:.2f}%")
    print(f"📉 Средний убыток: {avg_loss:.2f}%")
    print(f"💰 Общий P&L: {total_pnl:.2f}%")
    print(f"🔢 Сигналов в день: {len(all_signals) / (hours_back/24):.1f}")
    
    # Сравнение с предыдущими результатами
    improvement = new_win_rate - 41.6
    if improvement > 0:
        print(f"✅ УЛУЧШЕНИЕ: +{improvement:.1f}% винрейта!")
    else:
        print(f"❌ Ухудшение: {improvement:.1f}% винрейта")
    
    # Анализ по силе сигналов
    print(f"\n📈 КАЧЕСТВО СИГНАЛОВ:")
    score_ranges = [
        (10.0, float('inf'), 'Отличные'),
        (8.0, 10.0, 'Хорошие'),
        (6.5, 8.0, 'Средние'),
        (5.0, 6.5, 'Слабые'),
        (0, 5.0, 'Очень слабые')
    ]
    
    for min_score, max_score, label in score_ranges:
        range_signals = [s for s in all_signals if min_score <= s['score'] < max_score]
        if range_signals:
            range_winrate = sum(1 for s in range_signals if s['result']['pnl_pct'] > 0) / len(range_signals) * 100
            range_pnl = sum(s['result']['pnl_pct'] for s in range_signals)
            avg_score = np.mean([s['score'] for s in range_signals])
            print(f"   {label} ({avg_score:.1f}): {len(range_signals)} сигналов, {range_winrate:.1f}% винрейт, {range_pnl:.2f}% P&L")
    
    # Анализ по типам
    buy_signals = [s for s in all_signals if s['type'] == 'BUY']
    sell_signals = [s for s in all_signals if s['type'] == 'SELL']
    
    if buy_signals:
        buy_winrate = sum(1 for s in buy_signals if s['result']['pnl_pct'] > 0) / len(buy_signals) * 100
        buy_pnl = sum(s['result']['pnl_pct'] for s in buy_signals)
        print(f"🟢 LONG: {len(buy_signals)} сигналов, {buy_winrate:.1f}% винрейт, {buy_pnl:.2f}% P&L")
    
    if sell_signals:
        sell_winrate = sum(1 for s in sell_signals if s['result']['pnl_pct'] > 0) / len(sell_signals) * 100
        sell_pnl = sum(s['result']['pnl_pct'] for s in sell_signals)
        print(f"🔴 SHORT: {len(sell_signals)} сигналов, {sell_winrate:.1f}% винрейт, {sell_pnl:.2f}% P&L")
    
    # Анализ выходов
    print(f"\n🚪 АНАЛИЗ ВЫХОДОВ:")
    exit_reasons = {}
    for signal in all_signals:
        reason = signal['result']['exit_reason']
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'pnl': 0}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += signal['result']['pnl_pct']
    
    for reason, data in exit_reasons.items():
        avg_pnl = data['pnl'] / data['count']
        percentage = data['count'] / len(all_signals) * 100
        print(f"   {reason.upper()}: {data['count']} ({percentage:.1f}%), средний P&L: {avg_pnl:.2f}%")
    
    # Рекомендации
    print(f"\n💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ:")
    
    if new_win_rate >= 55:
        print(f"   ✅ Отличный результат! Винрейт {new_win_rate:.1f}% достиг цели")
    elif new_win_rate >= 50:
        print(f"   ✅ Хороший результат! Винрейт {new_win_rate:.1f}% значительно улучшен")
    elif new_win_rate >= 45:
        print(f"   ⚠️ Умеренное улучшение. Винрейт {new_win_rate:.1f}% лучше, но можно еще строже")
    else:
        print(f"   ❌ Недостаточное улучшение. Нужны еще более строгие фильтры")
    
    if len(all_signals) < hours_back / 24 * 2:  # Меньше 2 сигналов в день
        print(f"   ⚠️ Очень мало сигналов ({len(all_signals) / (hours_back/24):.1f}/день). Возможно, слишком строго")
    elif len(all_signals) > hours_back / 24 * 8:  # Больше 8 сигналов в день
        print(f"   ⚠️ Много сигналов ({len(all_signals) / (hours_back/24):.1f}/день). Можно быть строже")
    else:
        print(f"   ✅ Оптимальное количество сигналов ({len(all_signals) / (hours_back/24):.1f}/день)")
    
    if avg_loss < -3:
        print(f"   ⚠️ Большие убытки ({avg_loss:.2f}%). Рассмотрите увеличение SL_ATR_MULT")
    
    # Профит фактор
    total_profit = sum(s['result']['pnl_pct'] for s in profitable_signals)
    total_loss = abs(sum(s['result']['pnl_pct'] for s in losing_signals))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    print(f"📊 Профит фактор: {profit_factor:.2f} (цель: >1.5)")
    
    if profit_factor >= 1.5:
        print(f"   ✅ Отличная прибыльность!")
    elif profit_factor >= 1.2:
        print(f"   ✅ Хорошая прибыльность")
    else:
        print(f"   ⚠️ Низкая прибыльность. Нужно улучшить TP/SL соотношение")
    
    print("="*70)
    return {
        'win_rate': new_win_rate,
        'total_signals': len(all_signals),
        'profit_factor': profit_factor,
        'signals_per_day': len(all_signals) / (hours_back/24),
        'improvement': improvement
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'optimize':
        run_3day_filter_optimizer()
    else:
        hours = 168  # 7 дней по умолчанию
        symbols = 10
        
        if len(sys.argv) > 2:
            hours = int(sys.argv[2])
        if len(sys.argv) > 3:
            symbols = int(sys.argv[3])
        
        run_historical_analysis(hours, symbols) 