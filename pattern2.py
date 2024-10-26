import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization
import logging
import gc
import ta  # Per calcolare l'ATR

# Configurazione logging
logging.basicConfig(filename='trading_strategy.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Funzione per scaricare i dati storici da yFinance
def get_data(tickers, start_date, end_date):
    df = yf.download(tickers=tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    
    # Risolvi il multi-indice e resetta
    df = df.stack(level=0).reset_index()
    df.rename(columns={"level_1": "Ticker"}, inplace=True)
    return df

# Funzione per rilevare pattern Doji
def detect_doji(df):
    df['Doji'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1).astype(int)
    return df

# Funzione per rilevare pattern Engulfing Bullish e Bearish
def detect_engulfing(df):
    prev_close = df['Close'].shift(1)
    prev_open = df['Open'].shift(1)
    df['Engulfing_Bullish'] = ((prev_close < prev_open) & (df['Close'] > df['Open']) &
                               (df['Open'] <= prev_close) & (df['Close'] >= prev_open)).astype(int)
    df['Engulfing_Bearish'] = ((prev_close > prev_open) & (df['Close'] < df['Open']) &
                                (df['Open'] >= prev_close) & (df['Close'] <= prev_open)).astype(int)
    return df

# Funzione per rilevare Hammer e Shooting Star
def detect_hammer_shooting_star(df):
    hammer_cond = ((df['Close'] > df['Open']) &
                   ((df['Open'] - df['Low']) >= 2 * (df['Close'] - df['Open'])) &
                   ((df['High'] - df['Close']) <= (df['Close'] - df['Open'])))
    shooting_star_cond = ((df['Open'] > df['Close']) &
                          ((df['High'] - df['Open']) >= 2 * (df['Open'] - df['Close'])) &
                          ((df['Close'] - df['Low']) <= (df['Open'] - df['Close'])))
    df['Hammer'] = hammer_cond.astype(int)
    df['Shooting_Star'] = shooting_star_cond.astype(int)
    return df

# Funzione per rilevare Three White Soldiers e Three Black Crows
def detect_three_soldiers_crows(df):
    white_soldiers_cond = ((df['Close'] > df['Open']) &
                           (df['Close'].shift(1) > df['Open'].shift(1)) &
                           (df['Close'].shift(2) > df['Open'].shift(2)) &
                           (df['Close'] > df['Close'].shift(1)) &
                           (df['Close'].shift(1) > df['Close'].shift(2)))
    black_crows_cond = ((df['Close'] < df['Open']) &
                        (df['Close'].shift(1) < df['Open'].shift(1)) &
                        (df['Close'].shift(2) < df['Open'].shift(2)) &
                        (df['Close'] < df['Close'].shift(1)) &
                        (df['Close'].shift(1) < df['Close'].shift(2)))
    df['Three_White_Soldiers'] = white_soldiers_cond.astype(int)
    df['Three_Black_Crows'] = black_crows_cond.astype(int)
    return df

# Funzione per rilevare Morning Star ed Evening Star
def detect_morning_evening_star(df):
    morning_star_cond = ((df['Close'].shift(2) < df['Open'].shift(2)) &
                         (df['Close'].shift(1).between(df['Open'].shift(1), df['Close'].shift(1))) &
                         (df['Close'] > df['Open']) &
                         (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2))
    evening_star_cond = ((df['Close'].shift(2) > df['Open'].shift(2)) &
                         (df['Close'].shift(1).between(df['Open'].shift(1), df['Close'].shift(1))) &
                         (df['Close'] < df['Open']) &
                         (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2))
    df['Morning_Star'] = morning_star_cond.astype(int)
    df['Evening_Star'] = evening_star_cond.astype(int)
    return df

# Funzione per rilevare il pattern Piercing
def detect_piercing(df):
    prev_close = df['Close'].shift(1)
    prev_open = df['Open'].shift(1)
    prev_low = df['Low'].shift(1)
    piercing_cond = ((df['Close'] > df['Open']) & (prev_close < prev_open) &
                     (df['Open'] < prev_low) & (df['Close'] > (prev_open + prev_close) / 2))
    df['Piercing'] = piercing_cond.astype(int)
    return df

# Funzione per rilevare il pattern Dark Cloud Cover
def detect_dark_cloud(df):
    prev_close = df['Close'].shift(1)
    prev_open = df['Open'].shift(1)
    prev_high = df['High'].shift(1)
    dark_cloud_cond = ((df['Close'] < df['Open']) & (prev_close > prev_open) &
                       (df['Open'] > prev_high) & (df['Close'] < (prev_open + prev_close) / 2))
    df['Dark_Cloud_Cover'] = dark_cloud_cond.astype(int)
    return df

# Funzione per rilevare pattern Harami Bullish e Bearish
def detect_harami(df):
    prev_close = df['Close'].shift(1)
    prev_open = df['Open'].shift(1)
    harami_bullish_cond = ((df['Close'] > df['Open']) & (prev_close < prev_open) &
                           (df['Open'] > prev_close) & (df['Close'] < prev_open))
    harami_bearish_cond = ((df['Close'] < df['Open']) & (prev_close > prev_open) &
                           (df['Open'] < prev_close) & (df['Close'] > prev_open))
    df['Harami_Bullish'] = harami_bullish_cond.astype(int)
    df['Harami_Bearish'] = harami_bearish_cond.astype(int)
    return df

# Funzione per applicare tutti i pattern tecnici
def apply_patterns(df):
    df = detect_doji(df)
    df = detect_engulfing(df)
    df = detect_hammer_shooting_star(df)
    df = detect_three_soldiers_crows(df)
    df = detect_morning_evening_star(df)
    df = detect_piercing(df)
    df = detect_dark_cloud(df)
    df = detect_harami(df)
    return df

# Funzione per ottimizzare gli iperparametri con Bayesian Optimization
def optimize_hyperparameters(X_train, y_train):
    def xgb_cv(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
        model = XGBClassifier(n_estimators=int(n_estimators),
                              max_depth=int(max_depth),
                              learning_rate=learning_rate,
                              subsample=subsample,
                              colsample_bytree=colsample_bytree,
                              random_state=42,
                              n_jobs=-1)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
        return scores.mean()

    pbounds = {
        'n_estimators': (50, 100),
        'max_depth': (5, 15),
        'learning_rate': (0.01, 0.1),
        'subsample': (0.01, 0.1),
        'colsample_bytree': (0.01, 0.1)
    }

    optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=10, n_iter=30)

    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])

    return best_params

# Walk Forward Optimization (WFO)
def walk_forward_optimization(X, y, df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    f1_scores = []
    models = []
    predictions = pd.Series(index=X.index, dtype=int)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_params = optimize_hyperparameters(X_train, y_train)
        model = XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        models.append(model)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
        predictions.iloc[test_index] = y_pred

    print(f"F1 Score medio durante WFO: {np.mean(f1_scores)}")
    return models, f1_scores, predictions

def apply_atr_stop_loss(df, atr_multiplier=2):
    # Calcolo dell'ATR
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    # Stop loss basato su ATR
    df['Stop_Loss'] = df['Close'] - (df['ATR'] * atr_multiplier)
    # Segnale di stop se il prezzo scende sotto lo stop loss
    df['Stop_Signal'] = (df['Close'] < df['Stop_Loss']).astype(int)
    return df


# Funzione per applicare la strategia
def apply_ml_strategy_aggregated(df):
    df = df.copy()

    # Applica i pattern tecnici
    df = apply_patterns(df)

    features = ['Doji', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Hammer', 
                'Shooting_Star', 'Three_White_Soldiers', 'Three_Black_Crows', 
                'Morning_Star', 'Evening_Star']

    # Convertiamo 'Ticker_Code' in tipo int
    df['Ticker_Code'] = pd.Categorical(df['Ticker']).codes

    # Calcolo di 'Target' all'interno di ciascun ticker
    df['Target'] = df.groupby('Ticker')['Close'].shift(-1) > df['Close']
    df['Target'] = df['Target'].astype(int)

    # Definizione di X (feature) e y (target)
    X = df[features + ['Ticker_Code']]
    y = df['Target']

    # Rimozione dei valori NaN
    X = X.dropna()
    y = y.loc[X.index]
    df = df.loc[X.index]

    models, f1_scores, predictions = walk_forward_optimization(X, y, df)

    df['Prediction'] = predictions
    df['Position'] = df.groupby('Ticker')['Prediction'].shift(1)

    df['Returns'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).groupby(df['Ticker']).cumprod()
    df['Portfolio_Value'] = df['Cumulative_Returns'] * 10000

    return df, models, f1_scores

# Funzione per plot dei risultati WFO
def plot_wfo_results(portfolio, tickers):
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        plt.plot(portfolio.index, portfolio[ticker], label=f'Equity Curve {ticker}')
    plt.title('Equity Curve basata su Pattern Tecnici con WFO per più Asset')
    plt.xlabel('Data')
    plt.ylabel('Valore del Portafoglio')
    plt.legend()
    plt.grid()
    plt.show()

# Inizializzazione e applicazione della strategia
tickers = ["AAPL", "MSFT", "SPY", "QQQ", "TSLA", "JNJ", "KO", "MCD", "AMZN", "GOOGL", "NFLX", "NVDA", "META", "BA", "DIS"]
start_date = "2019-01-01"
end_date = "2022-01-01"
df = get_data(tickers, start_date, end_date)

df_aggregated, models, f1_scores = apply_ml_strategy_aggregated(df)

# Creazione del portafoglio basato sui valori aggregati del dataframe
portfolio = df_aggregated.pivot(index='Date', columns='Ticker', values='Portfolio_Value')

# Verifica e impostazione dell'indice Data come datetime nel portafoglio
portfolio.index = pd.to_datetime(df_aggregated['Date'].unique())

# Rimuovi eventuali valori NaT (Not a Time)
portfolio = portfolio.dropna()

# Visualizzazione del grafico dei risultati WFO
plot_wfo_results(portfolio, tickers)

# Funzione per calcolare le metriche di performance
def calculate_performance_metrics(df):
    mean_daily_return = df['Strategy_Returns'].mean()
    std_daily_return = df['Strategy_Returns'].std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Rolling_Max'] = df['Cumulative_Returns'].cummax()
    df['Drawdown'] = df['Cumulative_Returns'] / df['Rolling_Max'] - 1
    max_drawdown = df['Drawdown'].min()
    negative_return_std = df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std()
    sortino_ratio = (mean_daily_return / negative_return_std) * np.sqrt(252)
    return sharpe_ratio, max_drawdown, sortino_ratio

# Applicazione delle metriche
sharpe_ratio, max_drawdown, sortino_ratio = calculate_performance_metrics(df_aggregated)

print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Max Drawdown: {max_drawdown}")
print(f"Sortino Ratio: {sortino_ratio}")

# Analisi delle operazioni
def analyze_trades(df):
    trades = df[df['Position'].diff().abs() == 1]
    num_trades = len(trades)
    mean_profit = trades['Strategy_Returns'].mean()
    total_return = trades['Strategy_Returns'].sum()
    print(f"Numero di operazioni: {num_trades}")
    print(f"Profitto medio per operazione: {mean_profit}")
    print(f"Rendimento totale dalle operazioni: {total_return}")

# Applicazione dell'analisi
analyze_trades(df_aggregated)
