import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# Function to download data
def get_data(tickers, start_date, end_date):
    df = yf.download(tickers, start=start_date, end=end_date)
    df['Returns'] = df['Adj Close'].pct_change()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df['Portfolio_Value'] = 1000.0
    return df

# Function to calculate dynamic SMA with corrections
def calculate_dynamic_sma(df, base_window=50, vol_window=10, min_window=20, max_window=100):
    df['Volatility'] = df['Returns'].rolling(window=vol_window).std() * np.sqrt(252)
    
    # Replace 0 with NaN in 'Volatility' column
    df['Volatility'] = df['Volatility'].replace(0, np.nan)
    
    # Forward fill NaN values
    df['Volatility'] = df['Volatility'].ffill()
    
    # Backward fill NaN values
    df['Volatility'] = df['Volatility'].bfill()
    
    # Calculate the dynamic SMA window
    df['SMA_Window'] = (base_window / df['Volatility']).clip(lower=min_window, upper=max_window).fillna(min_window).astype(int)
    
    # Initialize the SMA_Dynamic list
    sma_dynamic = []
    for idx in range(len(df)):
        window = df['SMA_Window'].iloc[idx]
        if idx >= window:
            sma = df['Adj Close'].iloc[idx - window + 1: idx + 1].mean()
        else:
            sma = np.nan
        sma_dynamic.append(sma)
    
    df['SMA_Dynamic'] = sma_dynamic
    df.dropna(subset=['SMA_Dynamic'], inplace=True)
    
    return df


# Backtest function with dynamic SMA and risk management
def backtest_with_dynamic_sma(df, base_window=50, base_risk_per_trade=0.01, vol_window=10):
    df = calculate_dynamic_sma(df, base_window=base_window, vol_window=vol_window)
    
    if df.empty:
        print("DataFrame is empty. Exiting backtest.")
        return df, pd.DataFrame()
    
    df['Position'] = 0.0
    df['Strategy_Returns'] = 0.0
    portfolio_value = 1000.0
    trade_open = False

    entry_dates, exit_dates = [], []
    entry_prices, exit_prices = [], []
    positions_sizes = []
    portfolio_values_entry, portfolio_values_exit = [], []
    trades_log = []

    for index in range(1, len(df)):
        current_price = df['Adj Close'].iloc[index]
        previous_position = df['Position'].iloc[index - 1]
        df.at[df.index[index], 'Position'] = previous_position

        if trade_open:
            if current_price < df['SMA_Dynamic'].iloc[index]:
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
            if current_price > df['SMA_Dynamic'].iloc[index]:
                volatility = df['Volatility'].iloc[index]
                position_size = base_risk_per_trade / volatility if volatility != 0 else base_risk_per_trade
                df.at[df.index[index], 'Position'] = position_size
                trade_open = True
                entry_price = current_price
                entry_dates.append(df.index[index])
                entry_prices.append(entry_price)
                positions_sizes.append(position_size)
                portfolio_values_entry.append(portfolio_value)

        current_position = df.at[df.index[index], 'Position']
        df.at[df.index[index], 'Strategy_Returns'] = current_position * df['Returns'].iloc[index]
        df.at[df.index[index], 'Portfolio_Value'] = portfolio_value * (1 + df['Strategy_Returns'].iloc[index])
        portfolio_value = df['Portfolio_Value'].iloc[index]

    trades_df = pd.DataFrame(trades_log)
    return df, trades_df

# Function to calculate performance metrics
def calculate_performance_metrics(df):
    if df.empty or 'Portfolio_Value' not in df.columns or df['Portfolio_Value'].iloc[0] == 0:
        return np.nan, np.nan, np.nan

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

# Funzione obiettivo che ottimizza solo cumulative_return e sharpe_ratio
def objective_function(base_window, base_risk_per_trade, vol_window, df):
    try:
        base_window = int(base_window)
        base_risk_per_trade = float(base_risk_per_trade)
        vol_window = int(vol_window)
        
        # Esegui il backtest con i parametri attuali
        df_test, trades_df = backtest_with_dynamic_sma(
            df.copy(),
            base_window=base_window,
            base_risk_per_trade=base_risk_per_trade,
            vol_window=vol_window
        )
        
        # Calcola il rendimento cumulativo e lo sharpe ratio
        cumulative_return, sharpe_ratio, _ = calculate_performance_metrics(df_test)
        
        # Verifica che non ci siano NaN nei risultati
        if not np.isfinite(cumulative_return) or not np.isfinite(sharpe_ratio):
            return -1e6  # Restituisce un valore negativo molto grande per risultati non validi
        
        # Ottimizza solo su cumulative_return e sharpe_ratio
        objective_value = cumulative_return + sharpe_ratio  # Ottimizza su entrambi i valori positivi
        
        if not np.isfinite(objective_value):
            return -1e6  # Verifica che l'obiettivo sia finito
        
        return objective_value
    except Exception as e:
        print(f"Exception occurred: {e}")
        return -1e6


# Initialize the data
tickers = "AAPL"
start_date = "2019-01-01"
end_date = "2022-01-01"
df = get_data(tickers, start_date, end_date)

# Parameters for optimization
pbounds = {
    'base_window': (10, 50),
    'vol_window': (5, 20),
    'base_risk_per_trade': (0.1, 0.3)
}

# Bayesian optimization
optimizer = BayesianOptimization(
    f=lambda base_window, base_risk_per_trade, vol_window: objective_function(base_window, base_risk_per_trade, vol_window, df),
    pbounds=pbounds,
    verbose=2,
    random_state=42
)

# Run optimization
optimizer.maximize(init_points=10, n_iter=100)

# Extract optimized parameters
params_ottimizzati = optimizer.max['params']
params_ottimizzati['base_window'] = int(params_ottimizzati['base_window'])
params_ottimizzati['vol_window'] = int(params_ottimizzati['vol_window'])
params_ottimizzati['base_risk_per_trade'] = float(params_ottimizzati['base_risk_per_trade'])

# Run backtest with optimized parameters
df_test, trades_df = backtest_with_dynamic_sma(
    df.copy(),
    base_window=params_ottimizzati['base_window'],
    vol_window=params_ottimizzati['vol_window'],
    base_risk_per_trade=params_ottimizzati['base_risk_per_trade']
)

# Calculate performance metrics
cumulative_return, sharpe_ratio, max_drawdown = calculate_performance_metrics(df_test)
print(f"Rendimento Cumulativo: {cumulative_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Plot Equity Curve
def plot_equity_curve(df):
    if df.empty or 'Portfolio_Value' not in df.columns:
        print("No data to plot.")
        return
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df['Portfolio_Value'], label='Equity Curve', color='green')
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.show()

# Plot SMA signals
def plot_sma_signals(df, entry_dates, exit_dates, entry_prices, exit_prices):
    if df.empty or 'Adj Close' not in df.columns or 'SMA_Dynamic' not in df.columns:
        print("No data to plot.")
        return
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df['Adj Close'], label='Underlying Price', color='blue')
    plt.plot(df.index, df['SMA_Dynamic'], label='Dynamic SMA', color='orange', linestyle='--')

    plt.scatter(entry_dates, entry_prices, color='green', marker='^', label='Entry Signals', s=100)
    plt.scatter(exit_dates, exit_prices, color='red', marker='v', label='Exit Signals', s=100)

    plt.title('Price and Dynamic SMA with Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

plot_equity_curve(df_test)
plot_sma_signals(df_test, trades_df['Data Entrata'], trades_df['Data Uscita'], trades_df['Prezzo Entrata'], trades_df['Prezzo Uscita'])
print("\nTrade Details:")
print(trades_df)
