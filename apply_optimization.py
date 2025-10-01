"""
Скрипт для применения результатов оптимизации к config.py
"""

import json
import sys
from pathlib import Path

def apply_optimization(json_file):
    """Применить результаты оптимизации к config.py"""
    
    # Загружаем результаты
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    params = results['final_params']
    
    # Читаем текущий config.py
    with open('config.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Карта параметров для замены
    replacements = {
        'MA_FAST': params['ma_fast'],
        'MA_SLOW': params['ma_slow'],
        'RSI_WINDOW': params['rsi_window'],
        'RSI_MIN': params['rsi_min'],
        'RSI_MAX': params['rsi_max'],
        'MIN_ADX': params['min_adx'],
        'MACD_FAST': params['macd_fast'],
        'MACD_SLOW': params['macd_slow'],
        'MACD_SIGNAL': params['macd_signal'],
        'ADX_WINDOW': params['adx_window'],
        'ATR_WINDOW': params['atr_window'],
        'TP_ATR_MULT': params['tp_atr_mult'],
        'SL_ATR_MULT': params['sl_atr_mult'],
        'TP_MIN': params['tp_min'],
        'SL_MIN': params['sl_min'],
        'WEIGHT_RSI': params['weight_rsi'],
        'WEIGHT_MACD': params['weight_macd'],
        'WEIGHT_ADX': params['weight_adx'],
        'SIGNAL_COOLDOWN_MINUTES': params['signal_cooldown'],
    }
    
    # Заменяем значения
    new_lines = []
    for line in lines:
        updated = False
        for param_name, param_value in replacements.items():
            if line.startswith(f'{param_name} ='):
                # Форматируем значение
                if isinstance(param_value, float):
                    new_line = f'{param_name} = {param_value:.3f}  # Оптимизировано\n'
                else:
                    new_line = f'{param_name} = {param_value}  # Оптимизировано\n'
                new_lines.append(new_line)
                updated = True
                print(f"✅ Обновлено: {param_name} = {param_value}")
                break
        
        if not updated:
            new_lines.append(line)
    
    # Сохраняем backup
    backup_file = 'config_backup.py'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\n💾 Создан backup: {backup_file}")
    
    # Сохраняем новый config
    with open('config.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"✅ Config.py обновлен!")
    
    # Выводим статистику
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА ОПТИМИЗАЦИИ:")
    print("=" * 60)
    print(f"Этап 1 (качество сигналов): {results['stage1']['score']:.2f}")
    print(f"Этап 2 (прибыльность): {results['stage2']['pnl']:+.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("❌ Использование: py apply_optimization.py <файл_с_результатами.json>")
        print("\nПример: py apply_optimization.py optimization_results_20251001_120000.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not Path(json_file).exists():
        print(f"❌ Файл не найден: {json_file}")
        sys.exit(1)
    
    apply_optimization(json_file)

