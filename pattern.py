import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Funzione per scaricare i dati storici da yFinance
def get_data(tickers, start_date, end_date):
    df = yf.download(tickers=tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    return df

# Funzione per rilevare pattern Doji
def detect_doji(df):
    doji_cond = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
    df['Doji'] = doji_cond
    return df

# Funzione per rilevare pattern Engulfing Bullish e Bearish
def detect_engulfing(df):
    df['Engulfing_Bullish'] = ((df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1)))
    df['Engulfing_Bearish'] = ((df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1)))
    return df

# Funzione per calcolare l'ATR
def calculate_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-P'] = abs(df['High'] - df['Close'].shift(1))
    df['L-P'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-P', 'L-P']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df['ATR'] = df['ATR'].ffill()  # Usa ffill() invece di fillna(method='ffill')
    return df

# Aggiungiamo colonne per la posizione basata sui pattern
def trading_strategy(df, risk_per_trade=0.01, initial_capital=10000):
    df['Position'] = 0  # Nessuna posizione di default

    # Regole di base:
    # Se abbiamo un Engulfing Bullish, entriamo Long (acquisto)
    df.loc[df['Engulfing_Bullish'] == True, 'Position'] = 1

    # Se abbiamo un Engulfing Bearish, entriamo Short (vendita)
    df.loc[df['Engulfing_Bearish'] == True, 'Position'] = -1

    # Quando rileviamo un Doji, chiudiamo la posizione
    df.loc[df['Doji'] == True, 'Position'] = 0

    # Facciamo un forward fill per mantenere la posizione fino al segnale successivo
    df['Position'] = df['Position'].ffill().fillna(0)

    # Impostiamo stop-loss dinamici in base all'ATR
    df['Stop_Loss'] = df['Close'] - 2 * df['ATR']  # 2x ATR sotto il prezzo di chiusura
    df['Take_Profit'] = df['Close'] + 2 * df['ATR']  # 2x ATR sopra il prezzo di chiusura

    return df

# Funzione per applicare la strategia su più ticker
def apply_strategy_to_tickers(df, tickers):
    portfolio = pd.DataFrame()

    for ticker in tickers:
        print(f'Processing {ticker}...')
        
        # Ottieni il sottodataframe per ciascun ticker
        df_ticker = df[ticker].copy()

        # Aggiungiamo la colonna dei ritorni per ciascun ticker
        df_ticker['Returns'] = df_ticker['Close'].pct_change()

        # Calcoliamo l'ATR prima di applicare la strategia
        df_ticker = calculate_atr(df_ticker)

        # Rileviamo i pattern
        df_ticker = detect_doji(df_ticker)
        df_ticker = detect_engulfing(df_ticker)

        # Applichiamo la strategia
        df_ticker = trading_strategy(df_ticker)

        # Calcoliamo i rendimenti basati sulla strategia
        df_ticker['Strategy_Returns'] = df_ticker['Position'].shift(1) * df_ticker['Returns']

        # Calcoliamo il valore del portafoglio
        initial_capital = 10000  # Capitale iniziale
        df_ticker['Portfolio_Value'] = initial_capital * (1 + df_ticker['Strategy_Returns'].cumsum())

        # Aggiungiamo il valore del portafoglio alla tabella di output complessiva
        portfolio[ticker] = df_ticker['Portfolio_Value']
    
    return portfolio

# Inizializziamo i dati
tickers = ["AAPL", "MSFT", "SPY", "TSLA", "JNJ", "KO", "MCD"]
start_date = "2019-01-01"
end_date = "2022-01-01"

# Scarichiamo i dati storici
df = get_data(tickers, start_date, end_date)

# Applichiamo la strategia su ciascun ticker
portfolio = apply_strategy_to_tickers(df, tickers)


# Plot Equity Curves per ogni ticker
plt.figure(figsize=(10, 6))
for ticker in tickers:
    plt.plot(portfolio.index, portfolio[ticker], label=f'Equity Curve {ticker}')

plt.title('Equity Curve basata su Pattern Tecnici per più Asset')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid()
plt.show()
