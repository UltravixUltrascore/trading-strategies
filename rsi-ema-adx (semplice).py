import yfinance as yf
import pandas as pd
import numpy as np
import ta  # Per gli indicatori tecnici
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# Funzione per scaricare i dati
def get_data(tickers, start_date, end_date):
    try:
        df = yf.download(tickers, start=start_date, end=end_date)
        df['Returns'] = df['Adj Close'].pct_change().shift(-1)  # Rendimenti giornalieri
        df.dropna(inplace=True)  # Rimuovi valori NaN
        df['Date'] = df.index
        return df
    except Exception as e:
        print(f"Errore durante il download dei dati per {tickers}: {e}")
        return pd.DataFrame()

# Funzione per calcolare gli indicatori tecnici
def technical_indicators(df, ema_short=50, ema_long=200):
    df['EMA_50'] = df['Adj Close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_200'] = df['Adj Close'].ewm(span=ema_long, adjust=False).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Adj Close'], window=14).rsi()
    df['EMA_Momentum'] = df['EMA_50'] - df['EMA_200']  # EMA Momentum come differenza tra EMA50 e EMA200
    return df

# Funzione per generare segnali long e short
def generate_signals(df, rsi_threshold=40, stop_loss=0.03, take_profit=0.05):
    # Condizioni per il segnale long
    df['Long_Signal'] = np.where(
        (df['RSI'] < rsi_threshold) & (df['EMA_Momentum'] > 0), 1, 0
    )
    
    # Condizioni per il segnale short
    df['Short_Signal'] = np.where(
        (df['RSI'] > 70) & (df['EMA_Momentum'] < 0), -1, 0
    )
    
    # Combina Long e Short Signal
    df['Signal'] = df['Long_Signal'] + df['Short_Signal']

    # Stop Loss e Take Profit
    df['Stop_Loss'] = -stop_loss
    df['Take_Profit'] = take_profit

    return df

# Funzione per il backtest della strategia
def backtest_strategy(df, ticker):
    initial_portfolio_value = 1000  # Valore iniziale del portafoglio

    # Calcola i rendimenti della strategia
    df['Strategy_Returns'] = df['Returns'] * df['Signal']

    # Implementazione Stop Loss e Take Profit
    df['Strategy_Returns'] = np.where(
        df['Strategy_Returns'] < df['Stop_Loss'], df['Stop_Loss'], df['Strategy_Returns']
    )
    df['Strategy_Returns'] = np.where(
        df['Strategy_Returns'] > df['Take_Profit'], df['Take_Profit'], df['Strategy_Returns']
    )

    # Calcola il valore del portafoglio
    df['Portfolio_Value'] = initial_portfolio_value * (1 + df['Strategy_Returns']).cumprod()

    # Gestione dei NaN nei dati
    df = df.dropna(subset=['Strategy_Returns', 'Portfolio_Value'])

    # Calcolo delle metriche
    total_returns = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) - 1  # Rendimento totale
    num_years = len(df) / 252  # Supponendo 252 giorni di trading all'anno
    expected_return_annualized = (1 + total_returns) ** (1 / num_years) - 1  # Rendimento annualizzato
    expected_return_annualized *= 100  # Convertito in percentuale

    # Sharpe Ratio
    sharpe = np.mean(df['Strategy_Returns']) / np.std(df['Strategy_Returns']) * np.sqrt(252) if np.std(df['Strategy_Returns']) != 0 else 0

    # Max Drawdown
    max_drawdown = calculate_max_drawdown(df['Portfolio_Value'])

    # Stampa delle metriche
    print(f"\n*** Backtest {ticker} ***")
    print(f"Rendimento cumulativo totale: {total_returns * 100:.2f}%")
    print(f"Rendimento atteso annualizzato: {expected_return_annualized:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

    # Grafico dell'equity curve
    plot_equity_curve(df, ticker)

# Funzione per il grafico dell'andamento del portafoglio
def plot_equity_curve(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df['Portfolio_Value'], label='Valore Portafoglio', color='blue')
    ax.set_title(f'Andamento del Portafoglio ({ticker})')
    ax.set_ylabel('Valore Portafoglio')
    ax.set_xlabel('Data')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Funzione per calcolare il drawdown massimo
def calculate_max_drawdown(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min()

# Funzione per eseguire l'ottimizzazione Bayesiana
def optimize_strategy(df, ticker):
    def target_function(ema_short, ema_long, rsi_threshold, stop_loss, take_profit):
        # Applica indicatori e segnali
        df_opt = technical_indicators(df.copy(), ema_short=int(ema_short), ema_long=int(ema_long))
        df_opt = generate_signals(df_opt, rsi_threshold=int(rsi_threshold), stop_loss=stop_loss, take_profit=take_profit)

        # Calcola i rendimenti della strategia
        df_opt['Strategy_Returns'] = df_opt['Returns'] * df_opt['Signal']
        df_opt['Portfolio_Value'] = 1000 * (1 + df_opt['Strategy_Returns']).cumprod()

        # Calcolo del Sharpe Ratio
        sharpe_ratio = np.mean(df_opt['Strategy_Returns']) / np.std(df_opt['Strategy_Returns']) * np.sqrt(252) if np.std(df_opt['Strategy_Returns']) != 0 else 0
        return sharpe_ratio

    optimizer = BayesianOptimization(
        f=target_function,
        pbounds={
            'ema_short': (20, 100),  # Intervallo per EMA corta
            'ema_long': (100, 300),  # Intervallo per EMA lunga
            'rsi_threshold': (40, 60),  # RSI per entrare long
            'stop_loss': (0.01, 0.1),  # Stop loss tra 1% e 5%
            'take_profit': (0.01, 0.2)  # Take profit tra 1% e 10%
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=30)
    
    return optimizer.max['params']

def main():
    tickers = ['AAPL', 'MSFT', 'JNJ', 'KO', 'BABA']  # Ticker da testare
    start_date = '2010-01-01'
    end_date = '2024-10-01'

    for ticker in tickers:
        # Scarica i dati
        df = get_data(ticker, start_date, end_date)
        if df.empty:
            continue

        # Ottimizzazione dei parametri
        print(f"\n*** Ottimizzazione {ticker} ***")
        best_params = optimize_strategy(df, ticker)
        print(f"Migliori parametri per {ticker}: {best_params}")

        # Applica i migliori parametri
        df = technical_indicators(df, ema_short=int(best_params['ema_short']), ema_long=int(best_params['ema_long']))
        df = generate_signals(df, rsi_threshold=int(best_params['rsi_threshold']), stop_loss=best_params['stop_loss'], take_profit=best_params['take_profit'])

        # Esegui il backtest con i migliori parametri
        backtest_strategy(df, ticker)

if __name__ == '__main__':
    main()
