import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import seaborn as sns

# Funzione per scaricare e preparare i dati
def get_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df['Returns'] = df['Close'].pct_change()  # Calcolo dei ritorni giornalieri
    df.dropna(inplace=True)
    return df

# Funzione per calcolare il drawdown massimo
def max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    drawdown = 1 - cum_returns / cum_returns.cummax()
    return drawdown.max()

# Funzione per calcolare lo Sharpe Ratio annualizzato
def sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate / 252  # Tasso privo di rischio giornaliero
    return np.sqrt(252) * excess_returns.mean() / returns.std()

# Funzione per applicare la strategia con due medie mobili, stop-loss e take-profit
def apply_strategy(df, sma_short, sma_long, stop_loss, take_profit, transaction_cost=0.001, report=False):
    df['SMA_short'] = df['Close'].rolling(window=int(sma_short)).mean()
    df['SMA_long'] = df['Close'].rolling(window=int(sma_long)).mean()

    # Genera segnali basati sulle medie mobili
    df['Signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
    df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = 0

    # Inizializza variabili per tracciare posizioni e operazioni
    position_open = False
    buy_price = 0
    buy_date = None
    cumulative_return = 1
    trades = []
    positions = np.zeros(len(df))

    for i in range(len(df)):
        if position_open:
            positions[i] = 1
            current_price = df['Close'].iloc[i]
            # Gestione dello stop-loss e del take-profit
            if current_price <= buy_price * (1 - stop_loss):
                sell_price = current_price
                sell_date = df.index[i]
                trade_return = (sell_price / buy_price - 1)
                cumulative_return *= (1 + trade_return)  # Corretto uso della moltiplicazione
                trades.append((buy_date, buy_price, sell_date, sell_price, trade_return))
                position_open = False
            elif current_price >= buy_price * (1 + take_profit):
                sell_price = current_price
                sell_date = df.index[i]
                trade_return = (sell_price / buy_price - 1)
                cumulative_return *= (1 + trade_return)  # Corretto uso della moltiplicazione
                trades.append((buy_date, buy_price, sell_date, sell_price, trade_return))
                position_open = False
            elif df['Signal'].iloc[i] == 0:
                sell_price = current_price
                sell_date = df.index[i]
                trade_return = (sell_price / buy_price - 1)
                cumulative_return *= (1 + trade_return)  # Corretto uso della moltiplicazione
                trades.append((buy_date, buy_price, sell_date, sell_price, trade_return))
                position_open = False
        else:
            positions[i] = 0
            if df['Signal'].iloc[i] == 1:
                buy_price = df['Close'].iloc[i]
                buy_date = df.index[i]
                position_open = True
                positions[i] = 1

    df['Position'] = positions

    # Rendimenti della strategia e aggiunta di costi di transazione
    df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1)
    df['Strategy_Returns'] -= transaction_cost * df['Position'].diff().abs()
    
    # Calcolo del rendimento cumulativo corretto
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    # Calcolo di metriche di performance
    sharpe = sharpe_ratio(df['Strategy_Returns'])
    max_dd = max_drawdown(df['Strategy_Returns'])

    if report:
        print("\nDettagli delle operazioni:\n")
        for trade in trades:
            print(f"Acquisto: {trade[0].date()} a {trade[1]:.2f}, Vendita: {trade[2].date()} a {trade[3]:.2f}, Rendimento: {trade[4] * 100:.2f}%")

    final_cumulative_return = df['Cumulative_Strategy_Returns'].iloc[-1] - 1  # Corretto calcolo del rendimento cumulativo finale
    return final_cumulative_return, df, trades, sharpe, max_dd

# Funzione per il backtest Monte Carlo
def montecarlo_backtest(df, n_simulations, best_params):
    montecarlo_results = []
    for _ in range(n_simulations):
        df_sim = df.copy()
        df_sim['Returns'] = np.random.permutation(df_sim['Returns'])  # Permuta casualmente i rendimenti
        df_sim['Close'] = df_sim['Close'].iloc[0] * (1 + df_sim['Returns']).cumprod()  # Ricostruisce i prezzi
        final_return, _, _, _, _ = apply_strategy(df_sim, **best_params)
        montecarlo_results.append(final_return)
    return montecarlo_results

# Parametri
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'

# Scaricamento dei dati
df = get_data(ticker, start_date, end_date)

# Suddividere i dati in training (70%) e test (30%)
train_size = 0.7
train_length = int(len(df) * train_size)
df_train, df_test = df[:train_length], df[train_length:]

# Funzione obiettivo per l'ottimizzazione bayesiana
def objective_function(sma_short, sma_long, stop_loss, take_profit):
    return apply_strategy(
        df_train.copy(),
        sma_short=sma_short,
        sma_long=sma_long,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )[0]

# Ottimizzazione bayesiana per trovare i migliori parametri
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds={
        'sma_short': (10, 50),  # Finestra media mobile breve
        'sma_long': (100, 200),  # Finestra media mobile lunga
        'stop_loss': (0.01, 0.1),  # Stop-loss tra 1% e 10%
        'take_profit': (0.05, 0.2)  # Take-profit tra 5% e 20%
    },
    random_state=42,
    verbose=2
)

# Eseguire l'ottimizzazione
optimizer.maximize(init_points=10, n_iter=50)

# Ottieni i migliori parametri
best_params = optimizer.max['params']
print(f"Migliori parametri trovati: {best_params}")

# Esegui il backtest con la rendicontazione per i dati di training
final_return_train, df_train, trades_train, sharpe_train, max_dd_train = apply_strategy(
    df_train.copy(),
    sma_short=best_params['sma_short'],
    sma_long=best_params['sma_long'],
    stop_loss=best_params['stop_loss'],
    take_profit=best_params['take_profit'],
    report=True  # Mostra il report delle operazioni
)

print(f"\nRendimento cumulativo finale sui dati di training: {final_return_train * 100:.2f}%")
print(f"Sharpe Ratio sui dati di training: {sharpe_train:.2f}")
print(f"Drawdown massimo sui dati di training: {max_dd_train * 100:.2f}%")

# Esegui il backtest con la rendicontazione per i dati di test
final_return_test, df_test, trades_test, sharpe_test, max_dd_test = apply_strategy(
    df_test.copy(),
    sma_short=best_params['sma_short'],
    sma_long=best_params['sma_long'],
    stop_loss=best_params['stop_loss'],
    take_profit=best_params['take_profit'],
    report=True  # Mostra il report delle operazioni
)

print(f"\nRendimento cumulativo finale sui dati di test: {final_return_test * 100:.2f}%")
print(f"Sharpe Ratio sui dati di test: {sharpe_test:.2f}")
print(f"Drawdown massimo sui dati di test: {max_dd_test * 100:.2f}%")

# Visualizzazione dei risultati del backtest sui dati di test
df_test['Cumulative_Market_Returns'] = (1 + df_test['Returns']).cumprod()
plt.figure(figsize=(10, 5))
plt.plot(df_test['Cumulative_Strategy_Returns'], label='Rendimento Strategia (%)')
plt.plot(df_test['Cumulative_Market_Returns'], label='Rendimento Mercato (%)')
plt.legend()
plt.title('Performance Strategia sui Dati di Test vs Buy & Hold')
plt.ylabel('Rendimento (%)')
plt.show()

# Esegui il backtest Monte Carlo
n_simulations = 1000
montecarlo_results = montecarlo_backtest(df_test.copy(), n_simulations, best_params)

# Analisi dei risultati del Monte Carlo
mean_return = np.mean(montecarlo_results)
std_return = np.std(montecarlo_results)
print(f"\nRendimento medio Monte Carlo: {mean_return * 100:.2f}%")
print(f"Deviazione standard dei rendimenti: {std_return * 100:.2f}%")

# Plot dei risultati Monte Carlo
plt.figure(figsize=(10, 5))
sns.histplot(montecarlo_results, bins=50, color='blue', kde=True)
plt.title('Distribuzione dei Ritorni - Simulazioni Monte Carlo')
plt.xlabel('Rendimento (%)')
plt.ylabel('Frequenza')
plt.show()
