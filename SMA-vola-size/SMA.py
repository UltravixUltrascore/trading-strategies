import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Funzione per scaricare i dati
def get_data(tickers, start_date, end_date):
    df = yf.download(tickers, start=start_date, end=end_date)
    df['Returns'] = df['Adj Close'].pct_change().shift(-1)  # Rendimenti giornalieri
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df['Portfolio_Value'] = 1000.0  # Valore iniziale del portafoglio
    return df

# Funzione per calcolare il drawdown massimo
def calculate_max_drawdown(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min()

# Funzione per calcolare il Sortino Ratio
def sortino_ratio(returns, target_return=0):
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.nanstd(downside_returns)
    if downside_std == 0:
        return np.inf
    sortino = np.mean(excess_returns) / downside_std
    return sortino

# Calcolo della media mobile semplice (SMA)
def calculate_sma(df, window):
    df['SMA'] = df['Adj Close'].rolling(window=window).mean()
    return df

# Funzione per calcolare la volatilità a 10 giorni
def calculate_10d_volatility(df):
    df['10d_volatility'] = df['Returns'].rolling(window=10).std() * np.sqrt(252)
    df['10d_volatility'] = df['10d_volatility'].bfill()
    return df

# Funzione di backtest con SMA e gestione dinamica della posizione
def backtest_with_sma(df, sma_window=50, base_risk_per_trade=0.01):
    df = calculate_sma(df, sma_window)
    df = calculate_10d_volatility(df)
    df['Position'] = 0.0  # Inizializza la colonna delle posizioni come float
    df['Strategy_Returns'] = 0.0  # Inizializza i rendimenti

    portfolio_value = 1000.0  # Valore iniziale del portafoglio
    trade_open = False
    position_size = 0

    # Liste per registrare ingressi, uscite e dettagli delle operazioni
    entry_dates = []
    exit_dates = []
    entry_prices = []
    exit_prices = []
    trades_log = []  # Lista per il log delle operazioni

    for index in range(1, len(df)):
        # Segnale di acquisto se il prezzo supera la SMA
        if df['Adj Close'].iloc[index] > df['SMA'].iloc[index] and not trade_open:
            volatility = df['10d_volatility'].iloc[index]
            position_size = base_risk_per_trade / volatility if volatility != 0 else base_risk_per_trade
            df.at[df.index[index], 'Position'] = position_size
            trade_open = True
            entry_dates.append(df.index[index])
            entry_prices.append(df['Adj Close'].iloc[index])

        # Segnale di vendita se il prezzo scende sotto la SMA
        elif df['Adj Close'].iloc[index] < df['SMA'].iloc[index] and trade_open:
            df.at[df.index[index], 'Position'] = 0
            trade_open = False
            exit_dates.append(df.index[index])
            exit_prices.append(df['Adj Close'].iloc[index])

            # Calcolo del risultato dell'operazione e log
            trade_log = {
                'Entry Date': entry_dates[-1],
                'Exit Date': exit_dates[-1],
                'Entry Price': entry_prices[-1],
                'Exit Price': exit_prices[-1],
                'PnL': exit_prices[-1] - entry_prices[-1]
            }
            trades_log.append(trade_log)

        # Calcolo dei rendimenti della strategia
        df.at[df.index[index], 'Strategy_Returns'] = df['Position'].iloc[index] * df['Returns'].iloc[index]
        df.at[df.index[index], 'Portfolio_Value'] = portfolio_value * (1 + df['Strategy_Returns'].iloc[index])
        portfolio_value = df['Portfolio_Value'].iloc[index]

    # Creare un DataFrame dal log delle operazioni
    trades_df = pd.DataFrame(trades_log)
    
    return df, entry_dates, exit_dates, entry_prices, exit_prices, trades_df

# Funzione per plottare i risultati con ingressi, uscite e la SMA
def plot_equity_curve_with_trades(df, entry_dates, exit_dates, entry_prices, exit_prices, sma_window):
    plt.figure(figsize=(12, 8))
    
    # Plot del prezzo del sottostante
    plt.plot(df.index, df['Adj Close'], label='Prezzo Sottostante', color='blue')
    
    # Plot della media mobile
    plt.plot(df.index, df['SMA'], label=f'SMA {sma_window}', color='orange', linestyle='--')
    
    # Aggiungi ingressi e uscite
    plt.scatter(entry_dates, entry_prices, color='green', marker='^', label='Ingresso', s=100)
    plt.scatter(exit_dates, exit_prices, color='red', marker='v', label='Uscita', s=100)
    
    # Dettagli del grafico
    plt.title(f'Backtest con Ingressi/Uscite su SMA {sma_window}')
    plt.xlabel('Data')
    plt.ylabel('Prezzo')
    plt.legend()
    plt.grid()
    plt.show()

# Parametri iniziali
tickers = "AAPL"
start_date = "2015-01-01"
end_date = "2022-01-01"

# Scarica i dati
df = get_data(tickers, start_date, end_date)

# Testa diverse finestre SMA
sma_windows = [20, 50, 100, 200]  # Puoi provare con altre finestre di media mobile
for window in sma_windows:
    print(f"\nBacktest con SMA {window}")
    df_test, entry_dates, exit_dates, entry_prices, exit_prices, trades_df = backtest_with_sma(df.copy(), sma_window=window)
    max_drawdown = calculate_max_drawdown(df_test['Portfolio_Value'])
    sortino = sortino_ratio(df_test['Strategy_Returns'])
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sortino Ratio: {sortino:.2f}")
    
    # Mostra il log delle operazioni
    print("\nDettagli delle operazioni:")
    print(trades_df)

    # Plotta il grafico con ingressi, uscite e la SMA
    plot_equity_curve_with_trades(df_test, entry_dates, exit_dates, entry_prices, exit_prices, window)
