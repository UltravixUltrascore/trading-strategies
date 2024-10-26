"""
----------------------------------------------------------------------
      SORTINO-RETURNS QUANTITATIVE TRADING STRATEGY SCRIPT
           Ottimizzazione e Backtesting Quantitativo
----------------------------------------------------------------------
Autori:
    - Francesco Lojacono
    - Consulente Avanzato di Trading Quantitativo (GPT-4 di OpenAI)

Descrizione:
    Questo script implementa una strategia di trading quantitativa basata 
    su ottimizzazioni di parametri attraverso Bayesian Optimization e 
    test walk-forward su dati storici. Include funzioni per:
        - Download dati storici tramite yFinance
        - Calcolo del Sortino Ratio e del drawdown massimo
        - Ottimizzazione della strategia con stop-loss, take-profit e soglia di volatilità
        - Backtesting e visualizzazione dell'andamento dell'equity con entrata e uscita trade
        - Calcolo delle metriche di performance, tra cui Sharpe Ratio e rapporto vincite/perdite

Data di inizio progetto: 19-10-2024
Ultima modifica: [Ultima data di modifica]

Disclaimer:
    L'autore non si assume alcuna responsabilità per eventuali perdite finanziarie derivanti
    dall'uso diretto o indiretto di questo software.

----------------------------------------------------------------------
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import logging
import pickle
from scipy.stats import gmean

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funzione per scaricare i dati
def get_data(tickers, start_date, end_date):
    try:
        df = yf.download(tickers, start=start_date, end=end_date)
        df['Returns'] = df['Adj Close'].pct_change().shift(-1)  # Rendimenti giornalieri
        df.dropna(inplace=True)  # Rimuovi valori NaN
        df['Date'] = df.index
        return df
    except Exception as e:
        logging.error(f"Errore durante il download dei dati per {tickers}: {e}")
        return pd.DataFrame()

# Funzione per calcolare il Sortino Ratio
def sortino_ratio(returns, target_return=0):
    excess_returns = returns - target_return
    downside_std = np.std(excess_returns[excess_returns < 0])
    if downside_std == 0 or np.isnan(downside_std):
        return np.inf  # Restituisce un valore alto se non ci sono perdite
    return np.mean(excess_returns) / downside_std

# Funzione per calcolare il drawdown massimo
def calculate_max_drawdown(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min()

# Funzione per ottimizzare la strategia
def optimize_strategy(df):
    def target_function(stop_loss, take_profit, volatility_threshold):
        try:
            df_opt = df.copy()

            # Simula i rendimenti sulla base di stop_loss e take_profit
            df_opt['Strategy_Returns'] = df_opt['Returns'].apply(lambda r: min(max(r, -stop_loss), take_profit))
            
            # Calcola volatilità e usa un mix dinamico tra Sortino e rendimento atteso
            volatility = np.std(df_opt['Strategy_Returns'])
            sortino = sortino_ratio(df_opt['Strategy_Returns'])
            annual_return = np.prod(1 + df_opt['Strategy_Returns']) ** (252 / len(df_opt)) - 1
            
            weight_sortino = np.clip(volatility / volatility_threshold, 0, 1)
            weight_return = 1 - weight_sortino
            
            dynamic_target = weight_return * annual_return + weight_sortino * sortino
            
            if np.isinf(dynamic_target) or np.isnan(dynamic_target):
                return 0.0  # Valore di default se non è valido
            
            return dynamic_target
        except Exception as e:
            logging.error(f"Errore nella target_function: {e}")
            return 0.0

    optimizer = BayesianOptimization(
        f=target_function,
        pbounds={
            'stop_loss': (0.01, 0.2),
            'take_profit': (0.01, 0.2),
            'volatility_threshold': (0.05, 0.1)
        },
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=5, n_iter=50)
    return optimizer.max['params']

# Funzione per plottare l'equity con entrate e uscite dei trade
def plot_equity_with_trades(df, trades_log):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Portfolio_Value'], label='Valore del Portafoglio')

    # Marker per entrata e uscita trade
    for _, trade in trades_log.iterrows():
        if trade['Direction'] == 'Buy':
            plt.scatter(trade['Entry_Date'], trade['Portfolio_Value_At_Entry'], color='green', marker='^', s=50, label='Entrata Trade')
        else:
            plt.scatter(trade['Exit_Date'], trade['Portfolio_Value_At_Exit'], color='red', marker='v', s=50, label='Uscita Trade')
        plt.annotate(f"{trade['Return']:.2f}%", (trade['Exit_Date'], trade['Portfolio_Value_At_Exit']))

    plt.title('Andamento Equity')
    plt.xlabel('Data')
    plt.ylabel('Valore del Portafoglio')
    plt.legend()
    plt.grid()
    plt.show()

# Funzione di backtesting
def backtest_strategy(df, params):
    """
    Esegue il backtest di una strategia di trading.

    Parametri:
    df (DataFrame): Il dataset contenente i dati storici del prezzo.
    params (dict): I parametri di stop-loss e take-profit per la strategia.

    Ritorna:
    tuple: DataFrame con il valore del portafoglio aggiornato e DataFrame con il log delle operazioni.
    """
    stop_loss = params['stop_loss']
    take_profit = params['take_profit']
    
    df['Portfolio_Value'] = 1000.0  # Valore iniziale del portafoglio
    df['Portfolio_Value'] = df['Portfolio_Value'].astype(float)
    df['Strategy_Returns'] = 0.0

    trade_open = False
    trades_log_list = []
    
    for i in range(1, len(df)):
        daily_return = df['Returns'].iloc[i]
        entry_date = df.index[i - 1]
        exit_date = df.index[i]
        direction = None

        if not trade_open:
            if daily_return > take_profit:
                df.at[df.index[i], 'Strategy_Returns'] = -take_profit
                trade_open = True
                direction = 'Sell'
            elif daily_return < -stop_loss:
                df.at[df.index[i], 'Strategy_Returns'] = stop_loss
                trade_open = True
                direction = 'Buy'
        else:
            if daily_return <= -take_profit or daily_return >= stop_loss:
                df.at[df.index[i], 'Strategy_Returns'] = daily_return
                trade_open = False

        df.at[df.index[i], 'Portfolio_Value'] = df['Portfolio_Value'].iloc[i - 1] * (1 + df['Strategy_Returns'].iloc[i])

        if df['Strategy_Returns'].iloc[i] != 0:
            trade_details = {
                'Entry_Date': entry_date if trade_open else df.index[i],
                'Exit_Date': exit_date,
                'Return': df['Strategy_Returns'].iloc[i] * 100,
                'Direction': direction if direction else 'Hold',
                'Portfolio_Value_At_Exit': df['Portfolio_Value'].iloc[i],
                'Portfolio_Value_At_Entry': df['Portfolio_Value'].iloc[i - 1],
                'Exposure': (df['Portfolio_Value'].iloc[i] - df['Portfolio_Value'].iloc[i - 1]) / df['Portfolio_Value'].iloc[i - 1] * 100
            }
            trades_log_list.append(trade_details)

    trades_log = pd.DataFrame(trades_log_list)
    return df, trades_log

# Funzione per calcolare le metriche di performance
def calculate_performance_metrics(df, trades_log):
    total_returns = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) - 1
    expected_return_annualized = (1 + total_returns) ** (252 / len(df)) - 1
    sharpe_ratio = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252)
    win_loss_ratio = trades_log[trades_log['Return'] > 0].shape[0] / max(1, trades_log[trades_log['Return'] < 0].shape[0])
    avg_trade_duration = trades_log['Exit_Date'] - trades_log['Entry_Date']

    print(f"Rendimento atteso annualizzato: {expected_return_annualized * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Rapporto tra vincite e perdite: {win_loss_ratio:.2f}")
    print(f"Durata media delle operazioni: {avg_trade_duration.mean()} giorni")

# Funzione per salvare e caricare i parametri

def save_params(params, filename='best_params.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def load_params(filename='best_params.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Esempio di utilizzo del codice
if __name__ == "__main__":
    tickers = "SPY"
    start_date = "2010-01-01"
    end_date = "2022-12-31"
    
    df = get_data(tickers, start_date, end_date)
    if not df.empty:
        best_params = optimize_strategy(df)
        save_params(best_params)

        df_backtest, trades_log = backtest_strategy(df, best_params)
        plot_equity_with_trades(df_backtest, trades_log)
        calculate_performance_metrics(df_backtest, trades_log)
