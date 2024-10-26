import yfinance as yf 
import pandas as pd
import numpy as np
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

# Funzione per calcolare il Sortino Ratio
def sortino_ratio(returns, target_return=0):
    excess_returns = returns - target_return
    downside_std = np.std(excess_returns[excess_returns < 0])
    
    if downside_std == 0 or np.isnan(downside_std):
        return np.inf  # Restituisce un valore alto se non ci sono perdite
    
    sortino = np.mean(excess_returns) / downside_std
    return sortino

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
            print(f"Errore nella target_function: {e}")
            return 0.0

    optimizer = BayesianOptimization(
        f=target_function,
        pbounds={
            'stop_loss': (0.01, 0.2),
            'take_profit': (0.9, 0.98),
            'volatility_threshold': (0.05, 0.1)
        },
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=5, n_iter=50)
    return optimizer.max['params']

def plot_equity_with_trades(df, trades_log):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Portfolio_Value'], label='Valore del Portafoglio')

    # Marker per entrata e uscita trade
    entry_dates = trades_log['Entry_Date']
    exit_dates = trades_log['Exit_Date']
    entry_values = trades_log['Portfolio_Value_At_Entry']
    exit_values = trades_log['Portfolio_Value_At_Exit']

    # Scatter plot per segnare entrate e uscite
    plt.scatter(entry_dates, entry_values, color='green', marker='^', label='Entrata Trade', s=50)
    plt.scatter(exit_dates, exit_values, color='red', marker='v', label='Uscita Trade', s=50)

    # Filtra per escludere linee verticali troppo ravvicinate
    min_days_between_trades = pd.Timedelta(days=3)
    last_exit = exit_dates[0]
    for exit in exit_dates[1:]:
        if exit - last_exit >= min_days_between_trades:
            plt.axvline(exit, color='red', linestyle='--', alpha=0.3)
        last_exit = exit

    plt.title('Andamento Equity')
    plt.xlabel('Data')
    plt.ylabel('Valore del Portafoglio')
    plt.legend()
    plt.grid()
    
    # Salva il grafico come file anziché visualizzarlo
    plt.savefig("Walk Forward Opt - Sortino")
    plt.close()


def backtest_strategy(df, params):
    stop_loss = params['stop_loss']
    take_profit = params['take_profit']
    
    df['Portfolio_Value'] = 1000.0
    df['Portfolio_Value'] = df['Portfolio_Value'].astype(float)
    df['Strategy_Returns'] = 0.0

    trade_open = False
    trades_log_list = []

    print(f"Inizio del backtest inverso con i parametri: Stop Loss: {stop_loss}, Take Profit: {take_profit}")
    
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

    total_returns = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) - 1
    expected_return_annualized = (1 + total_returns) ** (252 / len(df)) - 1
    sortino = sortino_ratio(df['Strategy_Returns'])
    max_drawdown = calculate_max_drawdown(df['Portfolio_Value'])

    print(f"\n*** Backtest Inverso ***")
    print(f"Rendimento cumulativo totale: {total_returns * 100:.2f}%")
    print(f"Rendimento atteso annualizzato: {expected_return_annualized * 100:.2f}%")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

    # Salva il grafico su file, specificando il nome
    plot_equity_with_trades(df, trades_log, filename=f"equity_plot_{params['stop_loss']}_{params['take_profit']}.png")

    return df, trades_log




# Funzione di walk-forward optimization aggiornata
def walk_forward_optimization(df, optimization_window=3, test_window=3):
    total_months = (df.index[-1].year - df.index[0].year) * 12 + (df.index[-1].month - df.index[0].month)
    cycles = (total_months - optimization_window) // test_window
    
    all_trades = []  # Lista per registrare tutti i trades

    for cycle in range(cycles):
        start_optimization = df.index[0] + pd.DateOffset(months=cycle * test_window)
        end_optimization = start_optimization + pd.DateOffset(months=optimization_window)
        end_test = end_optimization + pd.DateOffset(months=test_window)
        
        optimization_data = df[(df.index >= start_optimization) & (df.index < end_optimization)]
        test_data = df[(df.index >= end_optimization) & (df.index < end_test)]
        
        if optimization_data.empty or test_data.empty:
            print(f"Ciclo {cycle + 1}: Dati insufficienti per ottimizzazione o test.")
            continue
        
        print(f"\n*** Ottimizzazione per il ciclo {cycle + 1} ***")
        best_params = optimize_strategy(optimization_data)
        
        print(f"\n*** Backtest per il ciclo {cycle + 1} ***")
        df_test, trades_log = backtest_strategy(test_data.copy(), best_params)
        
        all_trades.append(trades_log)  # Aggiunge i trades del ciclo corrente
        
        print(f"Ciclo {cycle + 1}:")
        print(f"Periodo di Ottimizzazione: {start_optimization.date()} - {end_optimization.date()}")
        print(f"Periodo di Test: {end_optimization.date()} - {end_test.date()}\n")

    # Concatenare tutti i log dei trades
    all_trades_df = pd.concat(all_trades, ignore_index=True)
    print("\nDettagli complessivi dei trades:\n", all_trades_df)
    all_trades_df.to_csv('tutti_trades_walkforward.csv', index=False)


# Scarica i dati
tickers = "SPY"
start_date = "2010-01-01"
end_date = "2022-12-31"
df = get_data(tickers, start_date, end_date)

# Esegui walk-forward optimization
walk_forward_optimization(df)
