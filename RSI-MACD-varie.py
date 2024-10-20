import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import logging
import json  # Cambiato da pickle a json
from scipy.stats import gmean
import yaml
import argparse

# Impostazione del seed random per la riproducibilità
np.random.seed(42)

# Configurazione avanzata del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os  # Importa il modulo os per controllare se il file esiste

class TradingStrategy:
    def __init__(self, config_path='config.yaml'):
        # Controlla se il file di configurazione esiste
        if not os.path.exists(config_path):
            logging.warning(f"File di configurazione non trovato: {config_path}. Verranno utilizzati i valori predefiniti.")
            # Carica una configurazione predefinita
            self.config = {
                'tickers': 'SPY',
                'start_date': '2010-01-01',
                'end_date': '2022-12-31',
                'param_bounds': {
                    'stop_loss': (0.01, 1),
                    'take_profit': (0.01, 0.5),
                    'volatility_threshold': (0.05, 3),
                    'position_size': (0.1, 2.0)
                },
                'initial_portfolio_value': 1000,  # Valore iniziale del portafoglio configurabile
                'plot_annotations': True  # Opzione per abilitare/disabilitare le annotazioni nei grafici
            }
        else:
            self.load_config(config_path)

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError as e:
            logging.error(f"File di configurazione non trovato: {e}")
            self.config = {}
        except yaml.YAMLError as e:
            logging.error(f"Errore nel parsing del file YAML: {e}")
            self.config = {}

    # Funzione per scaricare i dati
    def get_data(self, tickers, start_date, end_date):
        try:
            df = yf.download(tickers, start=start_date, end=end_date)  # Rimuovi l'argomento 'retries'
            if df.empty:
                logging.error(f"Nessun dato scaricato per {tickers}")
                return pd.DataFrame()
            df['Returns'] = np.log(df['Adj Close'].pct_change() + 1).fillna(0)  # Rendimenti giornalieri
            df.dropna(inplace=True)  # Rimuovi valori NaN
            return df
        except Exception as e:
            logging.error(f"Errore durante il download dei dati per {tickers}: {e}")
        return pd.DataFrame()

    # Funzione per calcolare il Sortino Ratio
    def sortino_ratio(self, returns, target_return=0):
        excess_returns = returns - target_return
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) == 0:
            return np.inf
        downside_std = np.std(negative_returns, ddof=1)
        if downside_std == 0 or np.isnan(downside_std):
            return np.inf
        return np.mean(excess_returns) / downside_std

    # Funzione per calcolare il drawdown massimo
    def calculate_max_drawdown(self, portfolio_values):
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        return drawdown.min()

    # Funzione per calcolare il Conditional Value at Risk (CVaR)
    def calculate_cvar(self, returns, confidence_level=0.95):
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = returns[returns <= var].mean()
        return cvar

    # Funzione per ottimizzare la strategia
    def optimize_strategy(self, df):
        # Estrae i limiti dei parametri dal file di configurazione o imposta i valori di default
        param_bounds = self.config.get('param_bounds', {
            'stop_loss': (0.01, 1),
            'take_profit': (0.01, 0.5),  # Range esteso
            'volatility_threshold': (0.05, 3),
            'position_size': (0.1, 2.0)
        })

        def target_function(stop_loss, take_profit, volatility_threshold, position_size):
            try:
                # Inizializzazione delle variabili
                df_opt = df.copy()
                df_opt['Strategy_Returns'] = 0.0
                daily_returns = df_opt['Returns'].values * position_size

                # Generazione dei segnali
                df_opt['Signal'] = self.generate_signals(daily_returns, stop_loss, take_profit)

                # Calcolo dei rendimenti della strategia
                df_opt['Strategy_Returns'] = df_opt['Signal'].shift(1) * daily_returns
                df_opt['Strategy_Returns'] = df_opt['Strategy_Returns'].fillna(0)

                # Calcolo del valore cumulativo del portafoglio
                df_opt['Portfolio_Value'] = (1 + df_opt['Strategy_Returns']).cumprod()

                # Calcolo delle metriche di performance
                sortino = self.sortino_ratio(df_opt['Strategy_Returns'])
                max_drawdown = self.calculate_max_drawdown(df_opt['Portfolio_Value'])
                annual_return = df_opt['Portfolio_Value'].iloc[-1] ** (252 / len(df_opt)) - 1

                # Funzione obiettivo (massimizza il rendimento annuale aggiustato per drawdown e volatilità)
                if max_drawdown == 0:
                    return 0.0

                # Calcolo della volatilità
                volatility = np.std(df_opt['Strategy_Returns']) * np.sqrt(252)

                # Peso dinamico
                dynamic_target = annual_return / (-max_drawdown * volatility_threshold * volatility)

                if np.isinf(dynamic_target) or np.isnan(dynamic_target):
                    return 0.0

                return dynamic_target
            except Exception as e:
                logging.error(f"Errore nella target_function: {e}")
                return 0.0

        optimizer = BayesianOptimization(
            f=target_function,
            pbounds=param_bounds,
            random_state=42,
            verbose=2
        )

        optimizer.maximize(init_points=5, n_iter=50)  # Ridotto per i test iniziali
        return optimizer.max['params']

    # Funzione per generare i segnali di trading
    def generate_signals(self, daily_returns, stop_loss, take_profit):
        signal = np.zeros_like(daily_returns)
        signal = np.where(daily_returns < -stop_loss, 1, signal)  # Segnale di acquisto
        signal = np.where(daily_returns > take_profit, -1, signal)  # Segnale di vendita
        return signal

    # Funzione di backtesting
    def backtest_strategy(self, df, params):
        stop_loss = params['stop_loss']
        take_profit = params['take_profit']
        position_size = params['position_size']
        initial_portfolio_value = self.config.get('initial_portfolio_value', 1000)

        df = df.copy()
        df['Strategy_Returns'] = 0.0
        daily_returns = df['Returns'] * position_size

        # Inizializzazione delle variabili di trade
        position = 0
        entry_price = 0.0
        trades_log_list = []

        df['Signal'] = 0
        df.loc[(daily_returns < -stop_loss) & (position == 0), 'Signal'] = 1
        df.loc[(daily_returns > take_profit) & (position == 0), 'Signal'] = -1

        # Iterazione per il log dei trades
        for i in range(1, len(df)):
            if df['Signal'].iloc[i] == 1:
                position = 1
                entry_price = df['Adj Close'].iloc[i]
                entry_date = df.index[i]
                direction = 'Buy'
            elif df['Signal'].iloc[i] == -1:
                position = -1
                entry_price = df['Adj Close'].iloc[i]
                entry_date = df.index[i]
                direction = 'Sell'
            elif df['Signal'].iloc[i] == 0 and position != 0:
                if position == 1:
                    # Chiudi posizione long
                    df.loc[df.index[i], 'Strategy_Returns'] = (df['Adj Close'].iloc[i] - entry_price) / entry_price
                    trades_log_list.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': df.index[i],
                        'Return': df.at[df.index[i], 'Strategy_Returns'] * 100,
                        'Direction': 'Buy',
                        'Entry_Price': entry_price,
                        'Exit_Price': df['Adj Close'].iloc[i]
                    })
                elif position == -1:
                    # Chiudi posizione short
                    df.loc[df.index[i], 'Strategy_Returns'] = (entry_price - df['Adj Close'].iloc[i]) / entry_price
                    trades_log_list.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': df.index[i],
                        'Return': df.at[df.index[i], 'Strategy_Returns'] * 100,
                        'Direction': 'Sell',
                        'Entry_Price': entry_price,
                        'Exit_Price': df['Adj Close'].iloc[i]
                    })
                position = 0

        df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
        df['Portfolio_Value'] = (1 + df['Strategy_Returns']).cumprod() * initial_portfolio_value

        trades_log = pd.DataFrame(trades_log_list)
        # Calcolo di metriche aggiuntive
        max_drawdown = self.calculate_max_drawdown(df['Portfolio_Value'])
        cvar = self.calculate_cvar(df['Strategy_Returns'])
        sortino = self.sortino_ratio(df['Strategy_Returns'])

        logging.info(f"Max Drawdown: {max_drawdown:.2%}")
        logging.info(f"CVaR: {cvar:.2%}")
        logging.info(f"Sortino Ratio: {sortino:.2f}")

        return df, trades_log

    # Funzione per plottare l'equity con entrate e uscite dei trade
    def plot_equity_with_trades(self, df, trades_log):
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Portfolio_Value'], label='Valore del Portafoglio')

        for _, trade in trades_log.iterrows():
            if trade['Direction'] == 'Buy':
                plt.scatter(trade['Entry_Date'], df.loc[trade['Entry_Date'], 'Portfolio_Value'], color='green', marker='^', s=100)
                plt.scatter(trade['Exit_Date'], df.loc[trade['Exit_Date'], 'Portfolio_Value'], color='red', marker='v', s=100)
                if self.config.get('plot_annotations', True):
                    plt.annotate(f"{trade['Return']:.2f}%", (trade['Exit_Date'], df.loc[trade['Exit_Date'], 'Portfolio_Value']),
                                 textcoords="offset points", xytext=(0,10), ha='center')
            else:
                plt.scatter(trade['Entry_Date'], df.loc[trade['Entry_Date'], 'Portfolio_Value'], color='red', marker='v', s=100)
                plt.scatter(trade['Exit_Date'], df.loc[trade['Exit_Date'], 'Portfolio_Value'], color='green', marker='^', s=100)
                if self.config.get('plot_annotations', True):
                    plt.annotate(f"{trade['Return']:.2f}%", (trade['Exit_Date'], df.loc[trade['Exit_Date'], 'Portfolio_Value']),
                                 textcoords="offset points", xytext=(0,10), ha='center')

        plt.title('Andamento Equity')
        plt.xlabel('Data')
        plt.ylabel('Valore del Portafoglio')
        plt.legend()
        plt.grid()
        plt.show()

    # Funzione per plottare il drawdown nel tempo
    def plot_drawdown(self, df):
        running_max = np.maximum.accumulate(df['Portfolio_Value'])
        drawdown = (df['Portfolio_Value'] - running_max) / running_max

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, drawdown, color='red', label='Drawdown')
        plt.title('Drawdown nel tempo')
        plt.xlabel('Data')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path al file di configurazione')
    args = parser.parse_args()

    strategy = TradingStrategy(config_path=args.config)
    tickers = strategy.config.get('tickers', 'SPY')
    start_date = strategy.config.get('start_date', '2010-01-01')
    end_date = strategy.config.get('end_date', '2022-12-31')

    df = strategy.get_data(tickers, start_date, end_date)
    if not df.empty:
        best_params = strategy.optimize_strategy(df)
        try:
            with open('best_params.json', 'w') as f:
                json.dump(best_params, f)
        except (OSError, IOError) as e:
            logging.error(f"Errore durante il salvataggio dei parametri ottimizzati: {e}")

        df_backtest, trades_log = strategy.backtest_strategy(df, best_params)
        strategy.plot_equity_with_trades(df_backtest, trades_log)
        strategy.plot_drawdown(df_backtest)
    else:
        logging.error("DataFrame dei prezzi vuoto. Terminazione del programma.")