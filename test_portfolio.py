#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест для проверки логики виртуального портфеля
"""

import json
from datetime import datetime

def test_portfolio_logic():
    """Тестирует логику добавления и расчета удачности сделок"""
    
    # Тестовые данные
    test_portfolio = {
        "BTC/USDT:USDT": [
            # Удачная LONG сделка
            {
                "action": "BUY",
                "side": "long", 
                "price": 50000,
                "time": "2025-01-15 10:00",
                "score": 6.5,
                "operation": "OPEN"
            },
            {
                "action": "SELL",
                "side": "long",
                "price": 52000, 
                "time": "2025-01-15 12:00",
                "operation": "CLOSE"
            },
            # Неудачная SHORT сделка
            {
                "action": "SELL",
                "side": "short",
                "price": 51000,
                "time": "2025-01-15 14:00", 
                "score": 5.2,
                "operation": "OPEN"
            },
            {
                "action": "BUY",
                "side": "short",
                "price": 52500,
                "time": "2025-01-15 16:00",
                "operation": "CLOSE"
            }
        ],
        "ETH/USDT:USDT": [
            # Удачная SHORT сделка
            {
                "action": "SELL",
                "side": "short",
                "price": 3000,
                "time": "2025-01-15 09:00",
                "score": 7.1, 
                "operation": "OPEN"
            },
            {
                "action": "BUY",
                "side": "short", 
                "price": 2850,
                "time": "2025-01-15 11:00",
                "operation": "CLOSE"
            }
        ],
        "open_trades": {}
    }
    
    # Функция для анализа (как в боте)
    def analyze_portfolio(portfolio):
        report = []
        total_win = 0
        total_loss = 0
        
        for symbol, trades in portfolio.items():
            if symbol == 'open_trades':
                continue
                
            # Группируем сделки по парам открытие-закрытие
            symbol_trades = []
            open_trade = None
            
            for trade in trades:
                operation = trade.get('operation', None)
                if operation == 'OPEN':
                    open_trade = trade
                elif operation == 'CLOSE' and open_trade is not None:
                    symbol_trades.append((open_trade, trade))
                    open_trade = None
            
            # Анализируем завершенные сделки
            for open_trade, close_trade in symbol_trades:
                side = open_trade['side'].upper()
                entry = float(open_trade['price'])
                exit = float(close_trade['price'])
                
                # Расчет P&L в процентах
                if side == 'LONG':
                    pnl_pct = ((exit - entry) / entry) * 100
                    result = 'УДАЧНО' if exit > entry else 'НЕУДАЧНО'
                else:  # SHORT
                    pnl_pct = ((entry - exit) / entry) * 100
                    result = 'УДАЧНО' if exit < entry else 'НЕУДАЧНО'
                
                if result == 'УДАЧНО':
                    total_win += 1
                else:
                    total_loss += 1
                
                # Добавляем детали в отчет
                score = open_trade.get('score', 'N/A')
                report.append(f"{symbol} {side}: {result} ({pnl_pct:+.2f}%) - Score: {score}")
        
        # Общая статистика
        if total_win + total_loss > 0:
            winrate = (total_win / (total_win + total_loss)) * 100
            report.append(f"\nВсего удачных: {total_win}")
            report.append(f"Всего неудачных: {total_loss}")
            report.append(f"Винрейт: {winrate:.1f}%")
        else:
            report.append("\nНет завершённых сделок.")
            
        return report, total_win, total_loss
    
    # Выполняем анализ
    print("🧪 ТЕСТ ЛОГИКИ ПОРТФЕЛЯ")
    print("=" * 50)
    
    report, wins, losses = analyze_portfolio(test_portfolio)
    
    for line in report:
        print(line)
    
    print("\n" + "=" * 50)
    print("🔍 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
    print("BTC LONG: УДАЧНО (+4.00%) - вход 50000, выход 52000")
    print("BTC SHORT: НЕУДАЧНО (-2.94%) - вход 51000, выход 52500") 
    print("ETH SHORT: УДАЧНО (+5.00%) - вход 3000, выход 2850")
    print("Винрейт: 66.7% (2 удачных из 3)")
    
    # Проверка правильности
    expected_wins = 2
    expected_losses = 1
    expected_winrate = 66.7
    
    actual_winrate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    
    print(f"\n✅ РЕЗУЛЬТАТ ТЕСТА:")
    print(f"Удачных: {wins} (ожидалось {expected_wins}) - {'✅' if wins == expected_wins else '❌'}")
    print(f"Неудачных: {losses} (ожидалось {expected_losses}) - {'✅' if losses == expected_losses else '❌'}")
    print(f"Винрейт: {actual_winrate:.1f}% (ожидалось {expected_winrate}%) - {'✅' if abs(actual_winrate - expected_winrate) < 0.1 else '❌'}")

if __name__ == "__main__":
    test_portfolio_logic() 