import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Funzione per scaricare dati e calcolare i rendimenti
def get_data(tickers, start_date, end_date):
    df = yf.download(tickers=tickers, start=start_date, end=end_date)['Adj Close']
    returns = df.pct_change().dropna()
    return returns

# Funzione per calcolare la matrice di correlazione dinamica con una finestra mobile
def rolling_correlation(returns, asset1, asset2, window=60):
    # Estrai solo i dati dei due asset
    rolling_corr = returns[[asset1, asset2]].rolling(window=window).corr().unstack().dropna()
    return rolling_corr

# Funzione per plot della correlazione dinamica tra due asset specifici
def plot_dynamic_correlation(corr_matrix, asset1, asset2):
    try:
        # Estrai la correlazione dinamica tra i due asset
        dynamic_corr = corr_matrix[(asset1, asset2)]
        
        # Plotta la correlazione dinamica
        plt.figure(figsize=(10, 6))
        plt.plot(dynamic_corr.index, dynamic_corr, label=f'{asset1}-{asset2} Correlation')
        plt.title(f'Dynamic Correlation between {asset1} and {asset2}')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True)
        plt.show()
    except KeyError:
        print(f"Errore: Non è stato possibile trovare la correlazione tra {asset1} e {asset2}.")

# Imposta i parametri
tickers = ["AAPL", "MSFT", "SPY", "TSLA", "JNJ", "KO", "MCD"]
start_date = "2015-01-01"
end_date = "2022-01-01"

# Scarica i dati e calcola i rendimenti
returns = get_data(tickers, start_date, end_date)

# Calcola la correlazione dinamica tra AAPL e MSFT
corr_matrix = rolling_correlation(returns, 'AAPL', 'MSFT')

# Esempio: plot della correlazione dinamica tra AAPL e MSFT
plot_dynamic_correlation(corr_matrix, 'AAPL', 'MSFT')
