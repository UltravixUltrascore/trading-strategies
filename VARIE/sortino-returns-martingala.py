import yfinance as yf
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Funzione per scaricare i dati
def get_data(tickers, start_date, end_date):
    try:
        print(f"Scaricando dati per {tickers} da {start_date} a {end_date}...")
        df = yf.download(tickers, start=start_date, end=end_date)
        df['Returns'] = df['Adj Close'].pct_change()  # Rendimenti giornalieri
        df.ffill(inplace=True)  # Riempie i valori NaN usando il metodo forward fill
        df.bfill(inplace=True)  # Se ci sono ancora NaN, usa il metodo backward fill
        df['Date'] = df.index
        print(f"Dati scaricati con successo per {tickers}.")
        return df
    except Exception as e:
        print(f"Errore durante il download dei dati per {tickers}: {e}")
        return pd.DataFrame()

# Funzione per generare segnali di trading
def generate_signals(df):
    print("Generando segnali di trading...")
    df['Signal'] = np.where(df['Returns'] > 0, 1, -1)  # Long su rendimenti positivi, Short su negativi
    print("Segnali generati.")
    return df

# Funzione per calcolare il Sortino Ratio
def sortino_ratio(returns, target_return=0):
    print("Calcolando il Sortino Ratio...")
    excess_returns = returns - target_return
    downside_std = np.std(excess_returns[excess_returns < 0])
    if downside_std == 0 or np.isnan(downside_std):
        return np.inf  # Restituisce un valore alto se non ci sono perdite
    sortino = np.mean(excess_returns) / downside_std
    print(f"Sortino Ratio calcolato: {sortino}")
    return sortino

# Funzione per calcolare il rendimento atteso annualizzato
def annualized_return(returns, periods_per_year=252):
    print("Calcolando il rendimento atteso annualizzato...")
    total_return = np.prod(1 + returns) - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    print(f"Rendimento atteso annualizzato: {annualized_return}")
    return annualized_return

# Funzione per calcolare la volatilità (deviazione standard dei rendimenti)
def calculate_volatility(returns):
    print("Calcolando la volatilità dei rendimenti...")
    volatility = np.std(returns)
    print(f"Volatilità calcolata: {volatility}")
    return volatility

# Funzione per calcolare il drawdown massimo
def calculate_max_drawdown(portfolio_values):
    print("Calcolando il drawdown massimo...")
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    print(f"Drawdown massimo: {max_drawdown}")
    return max_drawdown

# Funzione per calcolare l'Omega Ratio
def omega_ratio(returns, threshold=0):
    print("Calcolando l'Omega Ratio...")
    gains = np.sum(returns[returns > threshold])
    losses = abs(np.sum(returns[returns < threshold]))
    if losses == 0:
        return 1000.0  # Limita Omega Ratio a un valore massimo
    omega = gains / losses
    omega = min(omega, 1000.0)  # Limita valori troppo alti di Omega Ratio
    print(f"Omega Ratio calcolato: {omega}")
    return omega

# Funzione per ottimizzare la strategia (Bayesian Optimization con Omega Ratio)
def optimize_strategy(df):
    def target_function(stop_loss, take_profit, volatility_threshold):
        try:
            print(f"Valutando target_function con stop_loss={stop_loss}, take_profit={take_profit}, volatility_threshold={volatility_threshold}...")
            df_opt = df.copy()
            df_opt = generate_signals(df_opt)
            df_opt['Strategy_Returns'] = df_opt['Returns'] * df_opt['Signal']
            df_opt['Strategy_Returns'] = np.clip(df_opt['Strategy_Returns'], -stop_loss, take_profit)
            if df_opt['Strategy_Returns'].abs().max() > 1:
                print("Rendimento eccessivo, scartato.")
                return 0.0
            df_opt['Portfolio_Value'] = 1000 * (1 + df_opt['Strategy_Returns']).cumprod()
            omega = omega_ratio(df_opt['Strategy_Returns'], threshold=0)
            volatility = calculate_volatility(df_opt['Strategy_Returns'])
            sortino = sortino_ratio(df_opt['Strategy_Returns'], target_return=0)
            weight_sortino = np.clip(volatility / volatility_threshold, 0, 1)
            weight_return = 1 - weight_sortino
            annual_ret = annualized_return(df_opt['Strategy_Returns'])
            dynamic_target = weight_return * annual_ret + weight_sortino * sortino
            if np.isinf(dynamic_target) or np.isnan(dynamic_target):
                print("Valore target non valido, ritorno 0.0.")
                return 0.0
            print(f"Valore target: {dynamic_target}")
            return dynamic_target
        except Exception as e:
            print(f"Errore nella target_function: {e}")
            return 0.0

    optimizer = BayesianOptimization(
        f=target_function,
        pbounds={
            'stop_loss': (0.01, 0.03),
            'take_profit': (0.01, 0.03),
            'volatility_threshold': (0.01, 0.05)
        },
        random_state=42,
        verbose=2
    )

    try:
        print("Inizio ottimizzazione della strategia...")
        optimizer.maximize(init_points=5, n_iter=50)
        print("Ottimizzazione completata.")
    except Exception as e:
        print(f"Errore durante l'ottimizzazione: {e}")

    if optimizer.max and np.isfinite(optimizer.max['target']):
        print(f"Parametri ottimali trovati: {optimizer.max['params']}")
        return optimizer.max['params']
    else:
        print("Bayesian Optimization non ha trovato parametri validi. Utilizzo parametri di default.")
        return {
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'volatility_threshold': 0.03
        }

# Backtest con montante in perdita (Martingala)
def backtest_strategy(df, params, ticker, montante_in_perdita=True, incremento_montante=1.2, commission_rate=0.001):
    print("Inizio backtest della strategia...")
    df = generate_signals(df)

    stop_loss = params['stop_loss']
    take_profit = params['take_profit']
    initial_portfolio_value = 1000
    max_exposure = 1.2  # Ridotto ulteriormente l'esposizione massima
    df['Strategy_Returns'] = df['Returns'] * df['Signal']
    df['Strategy_Returns'] = np.clip(df['Strategy_Returns'], -stop_loss, take_profit)

    # Impostare esposizione iniziale e valore di portafoglio
    df['Exposure'] = 1.0
    df['Portfolio_Value'] = initial_portfolio_value
    
    # Evita l'incremento eccessivo dell'esposizione e gestisci meglio i rendimenti negativi
    for i in range(1, len(df)):
        # Se il portafoglio scende rispetto a due giorni fa, incrementa l'esposizione
        if i >= 2 and df['Portfolio_Value'].iloc[i-1] < df['Portfolio_Value'].iloc[i-2]:
            df.at[df.index[i], 'Exposure'] = min(df['Exposure'].iloc[i-1] * incremento_montante, max_exposure)
        else:
            df.at[df.index[i], 'Exposure'] = 1.0  # Reset esposizione a 1

    # Calcola rendimenti aggiustati
    df['Adjusted_Returns'] = df['Strategy_Returns'] * df['Exposure']
    df['Adjusted_Returns'] = np.clip(df['Adjusted_Returns'], -stop_loss, take_profit)

    # Limita le perdite (evita crash del cumprod)
    df['Adjusted_Returns'] = np.where(df['Adjusted_Returns'] < -1, -0.9999, df['Adjusted_Returns'])

    # Calcola il valore del portafoglio con cumprod
    df['Portfolio_Value'] = initial_portfolio_value * (1 + df['Adjusted_Returns']).cumprod()

    # Controllo del numero di operazioni (evitare troppi cambiamenti di segnale)
    num_trades = df['Signal'].diff().abs().sum() / 2  # Conta i cambiamenti di segnale come operazioni
    commission_cost = num_trades * commission_rate * initial_portfolio_value
    df['Portfolio_Value'] -= commission_cost
    print(f"Costo delle commissioni totale: {commission_cost}")

    # Rimuovi i valori NaN
    df = df.dropna(subset=['Adjusted_Returns', 'Portfolio_Value'])

    # Calcola le statistiche finali
    total_returns = (df['Portfolio_Value'].iloc[-1] / initial_portfolio_value) - 1
    num_years = len(df) / 252
    expected_return_annualized = (1 + total_returns) ** (1 / num_years) - 1
    expected_return_annualized *= 100

    omega = omega_ratio(df['Adjusted_Returns'], threshold=0)
    max_drawdown = calculate_max_drawdown(df['Portfolio_Value'])

    # Report finale
    print(f"\n*** Backtest ***")
    print(f"Rendimento cumulativo totale: {total_returns * 100:.2f}%")
    print(f"Rendimento atteso annualizzato: {expected_return_annualized:.2f}%")
    print(f"Omega Ratio: {omega:.2f}")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

    plot_equity_curve(df, ticker)

# Funzione per il grafico dell'equity curve
def plot_equity_curve(df, ticker):
    print("Plotting equity curve...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df['Portfolio_Value'], label='Valore Portafoglio', color='blue')
    ax.set_title(f'Andamento del Portafoglio ({ticker})')
    ax.set_ylabel('Valore Portafoglio')
    ax.set_xlabel('Data')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Funzione principale
def main():
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2024-01-01'
    print(f"Inizio del processo principale per il ticker {ticker}...")
    df = get_data(ticker, start_date, end_date)
    if not df.empty:
        params = optimize_strategy(df)
        print(f"\nParametri Ottimali: {params}")
        backtest_strategy(df, params, ticker)
    print("Processo principale completato.")

if __name__ == '__main__':
    main()
