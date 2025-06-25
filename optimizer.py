import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import *
from crypto_signal_bot import analyze, evaluate_signal_strength, SYMBOLS
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

EXCHANGE = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

def get_historical_data(symbol, hours_back=72):
    candles_needed = int(hours_back * 60 / 15) + 100
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=candles_needed)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    except Exception as e:
        logging.warning(f"Ошибка загрузки {symbol}: {e}")
        return pd.DataFrame()

def simulate_signals(df, symbol, params):
    # Повторяет фильтрацию как в check_signals, но для истории и с подстановкой параметров
    if df.empty or len(df) < MIN_15M_CANDLES + 50:
        return []
    df_analyzed = analyze(df.copy())
    if df_analyzed.empty:
        return []
    signals = []
    last_signal_time = None
    for i in range(MIN_15M_CANDLES, len(df_analyzed) - 20):
        current_df = df_analyzed.iloc[:i+1].copy()
        current_time = current_df.iloc[-1]['timestamp']
        if last_signal_time and (current_time - last_signal_time).total_seconds() < SIGNAL_COOLDOWN_MINUTES * 60:
            continue
        last = current_df.iloc[-1]
        prev = current_df.iloc[-2]
        # Фильтры
        if last['spread_pct'] > MAX_SPREAD_PCT:
            continue
        if last['adx'] < params['MIN_ADX']:
            continue
        # Объем (упрощенно для истории)
        volume = 1_000_000
        if volume < params['MIN_VOLUME_USDT'] * 0.5:
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
                if score >= params['MIN_COMPOSITE_SCORE']:
                    entry_price = last['close']
                    entry_time = current_time
                    future_data = df_analyzed.iloc[i+1:i+21]
                    if len(future_data) >= 10:
                        # TP/SL расчет
                        atr = last['atr']
                        tp_distance = atr * params['TP_ATR_MULT']
                        sl_distance = atr * params['SL_ATR_MULT']
                        if signal_type == 'BUY':
                            tp_price = entry_price + tp_distance
                            sl_price = entry_price - sl_distance
                        else:
                            tp_price = entry_price - tp_distance
                            sl_price = entry_price + sl_distance
                        result = None
                        tp_hit = False
                        sl_hit = False
                        for idx, candle in future_data.iterrows():
                            if signal_type == 'BUY':
                                if candle['high'] >= tp_price:
                                    result = 'tp'
                                    tp_hit = True
                                    break
                                elif candle['low'] <= sl_price:
                                    result = 'sl'
                                    sl_hit = True
                                    break
                            else:
                                if candle['low'] <= tp_price:
                                    result = 'tp'
                                    tp_hit = True
                                    break
                                elif candle['high'] >= sl_price:
                                    result = 'sl'
                                    sl_hit = True
                                    break
                        if not result:
                            result = 'timeout'
                        signals.append({
                            'symbol': symbol,
                            'type': signal_type,
                            'entry_time': entry_time,
                            'score': score,
                            'result': result,
                            'tp_pct': ((tp_price - entry_price) / entry_price * 100) if signal_type == 'BUY' else ((entry_price - tp_price) / entry_price * 100),
                            'sl_pct': ((entry_price - sl_price) / entry_price * 100) if signal_type == 'BUY' else ((sl_price - entry_price) / entry_price * 100)
                        })
                        last_signal_time = current_time
            except Exception as e:
                logging.warning(f"Ошибка оценки сигнала {symbol} в {current_time}: {e}")
                continue
    return signals

def optimize_filters():
    hours_back = 72
    min_score_range = [3.0, 3.5, 4.0, 4.5]
    min_adx_range = [14, 16, 18, 20]
    min_volume_range = [400_000, 600_000, 800_000]
    tp_mult_range = [1.1, 1.3, 1.5]
    sl_mult_range = [2.0, 2.3, 2.6]
    best_result = None
    best_params = None
    for min_score in min_score_range:
        for min_adx in min_adx_range:
            for min_vol in min_volume_range:
                for tp_mult in tp_mult_range:
                    for sl_mult in sl_mult_range:
                        params = {
                            'MIN_COMPOSITE_SCORE': min_score,
                            'MIN_ADX': min_adx,
                            'MIN_VOLUME_USDT': min_vol,
                            'TP_ATR_MULT': tp_mult,
                            'SL_ATR_MULT': sl_mult
                        }
                        all_signals = []
                        for symbol in SYMBOLS[:10]:
                            df = get_historical_data(symbol, hours_back)
                            if df.empty:
                                continue
                            signals = simulate_signals(df, symbol, params)
                            all_signals.extend(signals)
                        if len(all_signals) < 75:
                            print(f"Параметры {params}: мало сигналов ({len(all_signals)})")
                            continue
                        tp_signals = [s for s in all_signals if s['result'] == 'tp']
                        sl_signals = [s for s in all_signals if s['result'] == 'sl']
                        winrate = len(tp_signals) / len(all_signals) * 100 if all_signals else 0
                        avg_tp = np.mean([s['tp_pct'] for s in tp_signals]) if tp_signals else 0
                        avg_sl = abs(np.mean([s['sl_pct'] for s in sl_signals])) if sl_signals else 0
                        tp_sl_ratio = (avg_tp / avg_sl) if avg_sl > 0 else 0
                        print(f"Параметры {params}: сигналов={len(all_signals)}, TP={len(tp_signals)}, SL={len(sl_signals)}, winrate={winrate:.1f}%, TP/SL={tp_sl_ratio:.2f}")
                        if winrate >= 70 and tp_sl_ratio >= 1.7:
                            print(f"   ✅ Найдено! Параметры подходят.")
                            if best_result is None or winrate > best_result['winrate'] or (winrate == best_result['winrate'] and tp_sl_ratio > best_result['tp_sl_ratio']):
                                best_result = {
                                    'signals': len(all_signals),
                                    'winrate': winrate,
                                    'tp_sl_ratio': tp_sl_ratio,
                                    'params': params
                                }
                                best_params = params
    print("\n=== ЛУЧШИЕ НАЙДЕННЫЕ ПАРАМЕТРЫ ===")
    if best_result:
        print(f"Сигналов: {best_result['signals']}, Winrate: {best_result['winrate']:.1f}%, TP/SL: {best_result['tp_sl_ratio']:.2f}")
        print(f"Параметры: {best_params}")
    else:
        print("❌ Не найдено ни одной комбинации, удовлетворяющей всем условиям!")

if __name__ == '__main__':
    optimize_filters() 