#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обновления config.py с лучшими найденными параметрами
"""

import json
import re
from datetime import datetime

def update_config_with_best_params():
    """Обновляет config.py с лучшими найденными параметрами"""
    try:
        # Загружаем лучшие результаты
        with open('best_params_fixed.json', 'r', encoding='utf-8') as f:
            best_results = json.load(f)
        
        # Берем лучшую идеальную комбинацию
        if best_results['perfect_results']:
            best_perfect = max(best_results['perfect_results'], key=lambda x: x['winrate'])
            params = best_perfect['params']
            
            print(f"🏆 ОБНОВЛЕНИЕ config.py С ЛУЧШИМИ ПАРАМЕТРАМИ")
            print(f"Winrate: {best_perfect['winrate']:.1f}%")
            print(f"Сигналов/день: {best_perfect['signals_per_day']:.1f}")
            print(f"TP/SL (кол-во): {best_perfect['tp_sl_count_ratio']:.2f}")
            print(f"TP/SL (прибыль): {best_perfect['tp_sl_profit_ratio']:.2f}")
            print(f"Хорошие символы: {best_perfect['good_symbols']}")
            
        else:
            # Если нет идеальных, берем лучший по winrate
            best_overall = best_results['best_by_winrate']
            params = best_overall['params']
            
            print(f"🏆 ОБНОВЛЕНИЕ config.py С ЛУЧШИМ WINRATE")
            print(f"Winrate: {best_overall['winrate']:.1f}%")
            print(f"Сигналов/день: {best_overall['signals_per_day']:.1f}")
        
        # Читаем текущий config.py
        with open('config.py', 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Маппинг параметров оптимизатора к параметрам config.py
        param_mapping = {
            'min_score': 'MIN_COMPOSITE_SCORE',
            'min_adx': 'MIN_ADX',
            'short_min_adx': 'SHORT_MIN_ADX',
            'short_min_rsi': 'SHORT_MIN_RSI',
            'long_max_rsi': 'LONG_MAX_RSI',
            'rsi_min': 'RSI_MIN',
            'rsi_max': 'RSI_MAX',
            'tp_mult': 'TP_ATR_MULT',
            'sl_mult': 'SL_ATR_MULT',
            'min_volume': 'MIN_VOLUME_USDT',
            'max_spread': 'MAX_SPREAD_PCT',
            'min_bb_width': 'MIN_BB_WIDTH',
            'rsi_extreme_oversold': 'RSI_EXTREME_OVERSOLD',
            'rsi_extreme_overbought': 'RSI_EXTREME_OVERBOUGHT',
            'min_candle_body_pct': 'MIN_CANDLE_BODY_PCT',
            'max_wick_to_body_ratio': 'MAX_WICK_TO_BODY_RATIO',
            'signal_cooldown_minutes': 'SIGNAL_COOLDOWN_MINUTES',
            'min_triggers_active_hours': 'MIN_TRIGGERS_ACTIVE_HOURS',
            'min_triggers_inactive_hours': 'MIN_TRIGGERS_INACTIVE_HOURS',
            'bb_squeeze_threshold': 'BB_SQUEEZE_THRESHOLD',
            'macd_signal_window': 'MACD_SIGNAL_WINDOW',
            'stoch_rsi_k': 'STOCH_RSI_K',
            'stoch_rsi_d': 'STOCH_RSI_D',
            'stoch_rsi_length': 'STOCH_RSI_LENGTH',
            'stoch_rsi_smooth': 'STOCH_RSI_SMOOTH',
            'min_volume_ma_ratio': 'MIN_VOLUME_MA_RATIO',
            'min_volume_consistency': 'MIN_VOLUME_CONSISTENCY',
            'max_rsi_volatility': 'MAX_RSI_VOLATILITY',
            'require_macd_histogram': 'REQUIRE_MACD_HISTOGRAM_CONFIRMATION',
            'weight_rsi': 'WEIGHT_RSI',
            'weight_macd': 'WEIGHT_MACD',
            'weight_bb': 'WEIGHT_BB',
            'weight_vwap': 'WEIGHT_VWAP',
            'weight_volume': 'WEIGHT_VOLUME',
            'weight_adx': 'WEIGHT_ADX',
            'short_boost_multiplier': 'SHORT_BOOST_MULTIPLIER',
            'long_penalty_in_downtrend': 'LONG_PENALTY_IN_DOWNTREND',
            'mtf_confluence_weight': 'MTF_CONFLUENCE_WEIGHT',
            'RSI_WINDOW': 'RSI_WINDOW',
            'MA_FAST': 'MA_FAST',
            'MA_SLOW': 'MA_SLOW',
            'ATR_WINDOW': 'ATR_WINDOW',
            'TRAIL_ATR_MULT': 'TRAIL_ATR_MULT',
            'TP_MIN': 'TP_MIN',
            'SL_MIN': 'SL_MIN',
            'BB_WINDOW': 'BB_WINDOW',
            'BB_STD_DEV': 'BB_STD_DEV',
            'MACD_FAST': 'MACD_FAST',
            'MACD_SLOW': 'MACD_SLOW',
            'MACD_SIGNAL': 'MACD_SIGNAL',
            'STOCH_RSI_K': 'STOCH_RSI_K',
            'STOCH_RSI_D': 'STOCH_RSI_D',
            'STOCH_RSI_LENGTH': 'STOCH_RSI_LENGTH',
            'STOCH_RSI_SMOOTH': 'STOCH_RSI_SMOOTH',
            'MIN_TP_SL_DISTANCE': 'MIN_TP_SL_DISTANCE',
        }
        
        # Обновляем каждую строку
        updated_count = 0
        for opt_param, config_param in param_mapping.items():
            if opt_param in params:
                # Ищем строку с параметром в config.py
                pattern = rf'^{config_param}\s*=\s*[^#\n]+'
                replacement = f"{config_param} = {params[opt_param]}"
                new_content = re.sub(pattern, replacement, config_content, flags=re.MULTILINE)
                if new_content != config_content:
                    config_content = new_content
                    updated_count += 1
                    print(f"  ✅ {config_param} = {params[opt_param]}")
        
        # Добавляем комментарий о том, что файл обновлен
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""# 🛡️ ОПТИМИЗИРОВАННЫЙ КОНФИГ
# Обновлено: {timestamp}
# Автоматическая оптимизация завершена
# Исключены проблемные фильтры: Volatility, EMA Separation, Momentum
# Найдено {len(best_results['perfect_results'])} идеальных комбинаций
# Лучший winrate: {best_perfect['winrate']:.1f}%
# Сигналов/день: {best_perfect['signals_per_day']:.1f}

"""
        
        # Удаляем старый заголовок если есть
        config_content = re.sub(r'^# 🛡️.*?\n', '', config_content, flags=re.DOTALL)
        
        # Добавляем новый заголовок
        config_content = header + config_content
        
        # Сохраняем обновленный config.py
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
            
        print(f"\n✅ config.py успешно обновлен!")
        print(f"Обновлено параметров: {updated_count}")
        print(f"Файл сохранен с новым заголовком")
        
        # Создаем резервную копию
        backup_filename = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        with open(backup_filename, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"Резервная копия сохранена: {backup_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обновления config.py: {e}")
        return False

def show_best_params():
    """Показывает лучшие найденные параметры"""
    try:
        with open('best_params_fixed.json', 'r', encoding='utf-8') as f:
            best_results = json.load(f)
        
        print(f"📊 ЛУЧШИЕ НАЙДЕННЫЕ ПАРАМЕТРЫ")
        print("="*60)
        
        if best_results['perfect_results']:
            best_perfect = max(best_results['perfect_results'], key=lambda x: x['winrate'])
            print(f"🏆 ЛУЧШАЯ ИДЕАЛЬНАЯ КОМБИНАЦИЯ:")
            print(f"  Winrate: {best_perfect['winrate']:.1f}%")
            print(f"  Сигналов/день: {best_perfect['signals_per_day']:.1f}")
            print(f"  TP/SL (кол-во): {best_perfect['tp_sl_count_ratio']:.2f}")
            print(f"  TP/SL (прибыль): {best_perfect['tp_sl_profit_ratio']:.2f}")
            print(f"  Хорошие символы: {best_perfect['good_symbols']}")
            
            print(f"\n📋 КЛЮЧЕВЫЕ ПАРАМЕТРЫ:")
            key_params = [
                'min_score', 'min_adx', 'rsi_min', 'rsi_max', 
                'tp_mult', 'sl_mult', 'min_volume', 'max_spread'
            ]
            for param in key_params:
                if param in best_perfect['params']:
                    print(f"  {param}: {best_perfect['params'][param]}")
        
        print(f"\n📈 СТАТИСТИКА:")
        print(f"  Всего результатов: {best_results['all_results_count']}")
        print(f"  Идеальных комбинаций: {best_results['perfect_results_count']}")
        
    except Exception as e:
        print(f"❌ Ошибка чтения результатов: {e}")

if __name__ == '__main__':
    print("🔧 ОБНОВЛЕНИЕ КОНФИГУРАЦИИ")
    print("="*60)
    
    # Показываем лучшие параметры
    show_best_params()
    
    print(f"\n" + "="*60)
    
    # Обновляем config.py
    success = update_config_with_best_params()
    
    if success:
        print(f"\n🎉 ОБНОВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print(f"Теперь можно запускать бота с оптимизированными параметрами:")
        print(f"  py crypto_signal_bot.py")
    else:
        print(f"\n❌ ОБНОВЛЕНИЕ НЕ УДАЛОСЬ!")
        print(f"Проверьте файл best_params_fixed.json") 