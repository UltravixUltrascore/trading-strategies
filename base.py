import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Funzione per scaricare i dati
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['Returns'] = df['Adj Close'].pct_change()
    return df.dropna()

# Funzione di backtest semplice
def simple_backtest(df, stop_loss, take_profit):
    df = df.copy()  # Crea una copia per evitare modifiche in-place
    df.loc[:, 'Position'] = 0  # Inizializza posizione
    df.loc[:, 'Strategy_Returns'] = 0.0  # Inizializza rendimenti della strategia
    entry_price = None

    for i in range(1, len(df)):
        # Segnale di acquisto (esempio)
        if df['Returns'].iloc[i] < -stop_loss and df['Position'].iloc[i-1] == 0:
            entry_price = df['Adj Close'].iloc[i]
            df.loc[df.index[i], 'Position'] = 1  # Usa .loc[] per assegnare il valore
        # Segnale di vendita (esempio)
        elif df['Returns'].iloc[i] > take_profit and df['Position'].iloc[i-1] == 1:
            exit_price = df['Adj Close'].iloc[i]
            df.loc[df.index[i], 'Position'] = 0  # Usa .loc[] per assegnare il valore
            # Calcola il rendimento del trade
            df.loc[df.index[i], 'Strategy_Returns'] = (exit_price - entry_price) / entry_price

    # Calcola il rendimento cumulativo
    df.loc[:, 'Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    return df

# Parametri fissi
stop_loss = 0.02
take_profit = 0.03

# Scarica i dati e fai il backtest
df = get_data('AAPL', '2015-01-01', '2020-01-01')
df = simple_backtest(df, stop_loss, take_profit)

# Visualizza i risultati
plt.plot(df.index, df['Cumulative_Returns'], label='Strategia')
plt.title('Backtest Strategia Semplice')
plt.show()
