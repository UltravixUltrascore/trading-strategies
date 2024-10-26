import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed

# Funzione per scaricare i dati
def get_data(tickers, start_date, end_date):
    df = yf.download(tickers, start=start_date, end=end_date)
    df['Returns'] = df['Adj Close'].pct_change()  # Rendimenti giornalieri
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df['Portfolio_Value'] = 1000.0  # Valore iniziale del portafoglio
    return df

# Funzione per calcolare la SMA dinamica
def calculate_dynamic_sma(df, base_window=50, vol_window=10, min_window=20, max_window=100):
    df['Volatility'] = df['Returns'].rolling(window=vol_window).std() * np.sqrt(252)
    df['SMA_Window'] = (base_window / df['Volatility']).clip(lower=min_window, upper=max_window).fillna(min_window).astype(int)
    
    sma_dynamic = []
    for idx in range(len(df)):
        window = df['SMA_Window'].iloc[idx]
        if idx >= window:
            sma = df['Adj Close'].iloc[idx - window + 1 : idx + 1].mean()
        else:
            sma = np.nan
        sma_dynamic.append(sma)
    df['SMA_Dynamic'] = sma_dynamic
    return df

# Funzione di backtest con SMA dinamica e risk management ottimizzabile
def backtest_with_dynamic_sma(df, base_window=50, base_risk_per_trade=0.01, stop_loss_perc=0.2, take_profit_perc=0.5, trailing_stop=False, trailing_stop_perc=0.2):
    df = calculate_dynamic_sma(df, base_window=base_window)
    df['Position'] = 0.0
    df['Strategy_Returns'] = 0.0

    portfolio_value = 1000.0
    trade_open = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    entry_dates = []
    exit_dates = []
    entry_prices = []
    exit_prices = []
    positions_sizes = []
    portfolio_values_entry = []
    portfolio_values_exit = []
    trades_log = []

    for index in range(1, len(df)):
        current_price = df['Adj Close'].iloc[index]
        previous_position = df['Position'].iloc[index - 1]
        df.at[df.index[index], 'Position'] = previous_position

        if trade_open:
            if current_price <= stop_loss or current_price >= take_profit:
                df.at[df.index[index], 'Position'] = 0
                trade_open = False
                exit_dates.append(df.index[index])
                exit_prices.append(current_price)
                portfolio_values_exit.append(portfolio_value)

                pnl = (current_price - entry_price) * positions_sizes[-1] * portfolio_values_entry[-1] / entry_price
                trade_log = {
                    'Data Entrata': entry_dates[-1],
                    'Data Uscita': exit_dates[-1],
                    'Prezzo Entrata': entry_prices[-1],
                    'Prezzo Uscita': exit_prices[-1],
                    'Esposizione': positions_sizes[-1],
                    'Valore Portafoglio Entrata': portfolio_values_entry[-1],
                    'Valore Portafoglio Uscita': portfolio_value,
                    'PnL': pnl
                }
                trades_log.append(trade_log)

        else:
            if df['Adj Close'].iloc[index] > df['SMA_Dynamic'].iloc[index]:
                volatility = df['Volatility'].iloc[index]
                position_size = base_risk_per_trade / volatility if volatility != 0 else base_risk_per_trade
                df.at[df.index[index], 'Position'] = position_size
                trade_open = True
                entry_price = current_price
                entry_dates.append(df.index[index])
                entry_prices.append(entry_price)
                positions_sizes.append(position_size)
                portfolio_values_entry.append(portfolio_value)

                stop_loss = entry_price * (1 - stop_loss_perc)
                take_profit = entry_price * (1 + take_profit_perc)

        # Use current_position instead of previous_position
        current_position = df.at[df.index[index], 'Position']
        df.at[df.index[index], 'Strategy_Returns'] = current_position * df['Returns'].iloc[index]
        df.at[df.index[index], 'Portfolio_Value'] = portfolio_value * (1 + df['Strategy_Returns'].iloc[index])
        portfolio_value = df['Portfolio_Value'].iloc[index]

    trades_df = pd.DataFrame(trades_log)
    return df, trades_df


# Funzione per calcolare le metriche di performance
def calculate_performance_metrics(df):
    portfolio_values = df['Portfolio_Value']
    cumulative_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    daily_returns = df['Strategy_Returns'].fillna(0)
    annual_volatility = np.std(daily_returns) * np.sqrt(252)
    risk_free_rate = 0.01
    sharpe_ratio = (np.mean(daily_returns) * 252 - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return cumulative_return, sharpe_ratio, max_drawdown

# Funzione per plottare l'Equity Curve
def plot_equity_curve(df):
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df['Portfolio_Value'], label='Equity Curve', color='green')
    plt.title('Equity Curve')
    plt.xlabel('Data')
    plt.ylabel('Valore del Portafoglio')
    plt.legend()
    plt.grid()
    plt.show()

# Funzione per plottare il prezzo e la SMA dinamica con i segnali
def plot_sma_signals(df, entry_dates, exit_dates, entry_prices, exit_prices):
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df['Adj Close'], label='Prezzo Sottostante', color='blue')
    plt.plot(df.index, df['SMA_Dynamic'], label='SMA Dinamica', color='orange', linestyle='--')

    # Aggiungi segnali di ingresso e uscita
    plt.scatter(entry_dates, entry_prices, color='green', marker='^', label='Segnali di Entrata', s=100)
    plt.scatter(exit_dates, exit_prices, color='red', marker='v', label='Segnali di Uscita', s=100)

    plt.title('Prezzo e SMA Dinamica con Segnali')
    plt.xlabel('Data')
    plt.ylabel('Prezzo')
    plt.legend()
    plt.grid()
    plt.show()

# Funzione obiettivo per l'ottimizzazione bayesiana
def objective_function(base_window, stop_loss_perc, take_profit_perc, base_risk_per_trade, df):
    base_window = int(base_window)
    stop_loss_perc = float(stop_loss_perc)
    take_profit_perc = float(take_profit_perc)
    base_risk_per_trade = float(base_risk_per_trade)
    
    df_test, trades_df = backtest_with_dynamic_sma(
        df.copy(), 
        base_window=base_window,
        stop_loss_perc=stop_loss_perc,
        take_profit_perc=take_profit_perc,
        base_risk_per_trade=base_risk_per_trade,
        trailing_stop=False
    )
    
    cumulative_return, sharpe_ratio, max_drawdown = calculate_performance_metrics(df_test)
    
    # return cumulative_return
    # return cumulative_return + sharpe_ratio * 0.5  # Pesi diversi possono essere assegnati
    return cumulative_return - abs(max_drawdown) * 0.5  # Penalizza drawdown maggiori


# Parallelizziamo il backtest
def parallel_backtest(params, df):
    return objective_function(params['base_window'], params['stop_loss_perc'], params['take_profit_perc'], params['base_risk_per_trade'], df)

# Inizializziamo i dati
tickers = "SPY"
start_date = "2015-01-01"
end_date = "2022-01-01"
df = get_data(tickers, start_date, end_date)

# Parametri per l'ottimizzazione (aggiunto base_risk_per_trade)
pbounds = {
    'base_window': (20, 100),  # Finestra per la SMA
    'stop_loss_perc': (0.01, 0.05),  # Percentuale di stop-loss
    'take_profit_perc': (0.5, 2.0),  # Percentuale di take-profit
    'base_risk_per_trade': (0.01, 0.2)  # Percentuale di rischio per trade
}

# Ottimizzazione bayesiana con il parametro risk size
optimizer = BayesianOptimization(
    f=lambda base_window, stop_loss_perc, take_profit_perc, base_risk_per_trade: parallel_backtest({'base_window': base_window, 'stop_loss_perc': stop_loss_perc, 'take_profit_perc': take_profit_perc, 'base_risk_per_trade': base_risk_per_trade}, df), 
    pbounds=pbounds,  
    verbose=2,  
    random_state=42
)

# Esecuzione ottimizzazione con parallelizzazione del backtest
optimizer.maximize(init_points=10, n_iter=100)

# Estrazione dei parametri ottimizzati
params_ottimizzati = optimizer.max['params']
params_ottimizzati['base_window'] = int(params_ottimizzati['base_window'])

# Esecuzione backtest con i parametri ottimizzati
df_test, trades_df = backtest_with_dynamic_sma(
    df.copy(), 
    base_window=params_ottimizzati['base_window'],
    stop_loss_perc=params_ottimizzati['stop_loss_perc'], 
    take_profit_perc=params_ottimizzati['take_profit_perc'],
    base_risk_per_trade=params_ottimizzati['base_risk_per_trade'],
    trailing_stop=False
)

# Calcola le metriche di performance
cumulative_return, sharpe_ratio, max_drawdown = calculate_performance_metrics(df_test)
print(f"Rendimento Cumulativo: {cumulative_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Plot dell'Equity Curve
plot_equity_curve(df_test)

# Plot dei segnali SMA
plot_sma_signals(df_test, trades_df['Data Entrata'], trades_df['Data Uscita'], trades_df['Prezzo Entrata'], trades_df['Prezzo Uscita'])

print("\nDettagli delle operazioni:")
print(trades_df)
