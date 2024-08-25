# strategy.py

def trading_strategy(action, market_data, balance):
    # Tu jest logika strategii, która ustawia odpowiednio take profit, stop loss i oblicza reward
    # Na przykład:
    take_profit = market_data['close'] * 1.02
    stop_loss = market_data['close'] * 0.98

    if action == 1:  # Long
        if market_data['high'] >= take_profit:
            reward = (take_profit - market_data['close']) / market_data['close']
            balance += balance * reward  # Dodanie zysku do balansu
        elif market_data['low'] <= stop_loss:
            reward = (stop_loss - market_data['close']) / market_data['close']
            balance += balance * reward  # Dodanie straty do balansu
        else:
            reward = 0  # Neutralne, jeśli żaden z warunków nie został spełniony
    elif action == 2:  # Short
        if market_data['low'] <= stop_loss:
            reward = (market_data['close'] - stop_loss) / market_data['close']
            balance += balance * reward  # Dodanie zysku do balansu
        elif market_data['high'] >= take_profit:
            reward = (market_data['close'] - take_profit) / market_data['close']
            balance += balance * reward  # Dodanie straty do balansu
        else:
            reward = 0  # Neutralne, jeśli żaden z warunków nie został spełniony
    else:
        reward = 0  # Dla przypadku braku akcji

    return reward, balance  # Zwracamy nagrodę i zaktualizowany balans
