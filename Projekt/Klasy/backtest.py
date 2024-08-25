from strategy import trading_strategy

def backtest_strategy(agent, df, initial_balance=10000):
    balance = initial_balance
    portfolio = []
    positions = trading_strategy(agent, df)

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        for pos in positions.copy():
            direction, entry_price, take_profit, stop_loss = pos

            # Long position handling
            if direction == 'long':
                if current_price >= take_profit:  # Take profit hit
                    profit = (take_profit - entry_price) / entry_price * 100
                    balance += balance * (profit / 100)
                    agent.remember(df.iloc[i], 1, 1 + profit, df.iloc[i], True)  # Nagroda za zysk
                    print(f"Zrealizowano zysk: {profit:.2f}% z Long po cenie {current_price}")
                    positions.remove(pos)
                elif current_price <= stop_loss:  # Stop loss hit
                    loss = (entry_price - stop_loss) / entry_price * 100
                    balance -= balance * (loss / 100)
                    agent.remember(df.iloc[i], 1, -1 - loss, df.iloc[i], True)  # Kara za stratę
                    print(f"Zrealizowano stratę: {loss:.2f}% z Long po cenie {current_price}")
                    positions.remove(pos)

            # Short position handling
            elif direction == 'short':
                if current_price <= take_profit:  # Take profit hit
                    profit = (entry_price - take_profit) / entry_price * 100
                    balance += balance * (profit / 100)
                    agent.remember(df.iloc[i], 2, 1 + profit, df.iloc[i], True)  # Nagroda za zysk
                    print(f"Zrealizowano zysk: {profit:.2f}% z Short po cenie {current_price}")
                    positions.remove(pos)
                elif current_price >= stop_loss:  # Stop loss hit
                    loss = (stop_loss - entry_price) / entry_price * 100
                    balance -= balance * (loss / 100)
                    agent.remember(df.iloc[i], 2, -1 - loss, df.iloc[i], True)  # Kara za stratę
                    print(f"Zrealizowano stratę: {loss:.2f}% z Short po cenie {current_price}")
                    positions.remove(pos)

    print(f"Końcowy balans: {balance}")
    return balance
