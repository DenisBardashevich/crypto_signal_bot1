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

if __name__ == "__main__":
    import sys
    
    hours = 168  # 7 дней по умолчанию
    symbols = 10
    
    if len(sys.argv) > 1:
        hours = int(sys.argv[1])
    if len(sys.argv) > 2:
        symbols = int(sys.argv[2])
    
    run_historical_analysis(hours, symbols) 