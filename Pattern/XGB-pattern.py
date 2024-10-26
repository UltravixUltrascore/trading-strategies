import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
import logging
import gc
import shap
from sklearn.metrics import f1_score

# Configurazione logging
logging.basicConfig(filename='trading_strategy.log', level=logging.INFO, format='%(asctime)s - %(message)s')


# Funzione per scaricare i dati storici da yFinance
def get_data(tickers, start_date, end_date):
    df = yf.download(tickers=tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    return df

# Funzione per scaricare i dati del benchmark
def get_benchmark_data(start_date, end_date):
    benchmark = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=True)
    benchmark['Benchmark_Returns'] = benchmark['Close'].pct_change().fillna(0)
    benchmark['Benchmark_Cumulative'] = (1 + benchmark['Benchmark_Returns']).cumprod()
    return benchmark

# Funzione per rilevare pattern Doji
def detect_doji(df):
    df = df.copy()
    df['Doji'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
    return df

# Funzione per rilevare pattern Engulfing Bullish e Bearish
def detect_engulfing(df):
    df = df.copy()
    prev_close = df['Close'].shift(1)
    prev_open = df['Open'].shift(1)

    # Engulfing Bullish
    prev_bearish = prev_close < prev_open
    curr_bullish = df['Close'] > df['Open']
    df['Engulfing_Bullish'] = prev_bearish & curr_bullish & (df['Open'] <= prev_close) & (df['Close'] >= prev_open)

    # Engulfing Bearish
    prev_bullish = prev_close > prev_open
    curr_bearish = df['Close'] < df['Open']
    df['Engulfing_Bearish'] = prev_bullish & curr_bearish & (df['Open'] >= prev_close) & (df['Close'] <= prev_open)

    return df

# Funzione per calcolare l'ATR
def calculate_atr(df, period=14):
    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-P'] = abs(df['High'] - df['Close'].shift(1))
    df['L-P'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-P', 'L-P']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean().ffill()
    return df


# Funzione per rilevare Hammer e Shooting Star
def detect_hammer_shooting_star(df):
    df = df.copy()

    # Hammer
    hammer_cond = (
        (df['Close'] > df['Open']) &
        ((df['Open'] - df['Low']) >= 2 * (df['Close'] - df['Open'])) &
        ((df['High'] - df['Close']) <= (df['Close'] - df['Open']))
    )

    # Shooting Star
    shooting_star_cond = (
        (df['Open'] > df['Close']) &
        ((df['High'] - df['Open']) >= 2 * (df['Open'] - df['Close'])) &
        ((df['Close'] - df['Low']) <= (df['Open'] - df['Close']))
    )

    df['Hammer'] = hammer_cond
    df['Shooting_Star'] = shooting_star_cond
    return df

# Funzione per rilevare Three White Soldiers e Three Black Crows
def detect_three_soldiers_crows(df):
    df = df.copy()

    # Three White Soldiers
    white_soldiers_cond = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'].shift(2) > df['Open'].shift(2)) &
        (df['Close'] > df['Close'].shift(1)) &
        (df['Close'].shift(1) > df['Close'].shift(2))
    )

    # Three Black Crows
    black_crows_cond = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Close'] < df['Close'].shift(1)) &
        (df['Close'].shift(1) < df['Close'].shift(2))
    )

    df['Three_White_Soldiers'] = white_soldiers_cond
    df['Three_Black_Crows'] = black_crows_cond
    return df

# Funzione per rilevare Morning Star ed Evening Star
def detect_morning_evening_star(df):
    df = df.copy()

    # Morning Star
    morning_star_cond = (
        (df['Close'].shift(2) < df['Open'].shift(2)) &
        ((df['Close'].shift(1) > df['Open'].shift(1)) | (df['Close'].shift(1) < df['Open'].shift(1))) &
        (df['Close'] > df['Open']) &
        (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2)
    )

    # Evening Star
    evening_star_cond = (
        (df['Close'].shift(2) > df['Open'].shift(2)) &
        ((df['Close'].shift(1) > df['Open'].shift(1)) | (df['Close'].shift(1) < df['Open'].shift(1))) &
        (df['Close'] < df['Open']) &
        (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2)
    )

    df['Morning_Star'] = morning_star_cond
    df['Evening_Star'] = evening_star_cond
    return df



# Sostituire la funzione per ottimizzare gli iperparametri con XGBClassifier
def optimize_hyperparameters(X_train, y_train):
    def xgb_cv(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
        model = XGBClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
            enable_categorical=False
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        return scores.mean()

    pbounds = {
        'n_estimators': (100, 200),
        'max_depth': (3.0, 20),
        'learning_rate': (0.01, 0.2),
        'subsample': (0.01, 0.2),
        'colsample_bytree': (0.01, 0.2)
    }

    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=10, n_iter=50)

    # Recuperiamo i migliori iperparametri
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])

    return best_params

# Funzione per calcolare le metriche di performance
def calculate_performance_metrics(df):
    mean_daily_return = df['Strategy_Returns'].mean()
    std_daily_return = df['Strategy_Returns'].std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)

    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).groupby(df['Ticker']).cumprod()
    df['Rolling_Max'] = df.groupby('Ticker')['Cumulative_Returns'].cummax()
    df['Drawdown'] = df['Cumulative_Returns'] / df['Rolling_Max'] - 1
    max_drawdown = df['Drawdown'].min()

    negative_return_std = df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std()
    sortino_ratio = (mean_daily_return / negative_return_std) * np.sqrt(252)

    logging.info(f"Sharpe Ratio: {sharpe_ratio}")
    logging.info(f"Max Drawdown: {max_drawdown}")
    logging.info(f"Sortino Ratio: {sortino_ratio}")

    return sharpe_ratio, max_drawdown, sortino_ratio

# Funzione per preparare il dataset aggregato per tutti i ticker
def prepare_aggregated_dataset(df, tickers):
    df_list = []

    for ticker in tickers:
        df_ticker = df[ticker].copy()
        df_ticker.reset_index(inplace=True)  # Reset dell'indice per avere 'Date' come colonna
        df_ticker['Ticker'] = ticker
        df_ticker = detect_doji(df_ticker)
        df_ticker = detect_engulfing(df_ticker)
        df_ticker = detect_hammer_shooting_star(df_ticker)
        df_ticker = detect_three_soldiers_crows(df_ticker)
        df_ticker = detect_morning_evening_star(df_ticker)
        df_list.append(df_ticker)

    df_combined = pd.concat(df_list, ignore_index=True)
    return df_combined

# Funzione per applicare la strategia con un unico modello aggregato
def apply_ml_strategy_aggregated(df):
    df = df.copy()

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

    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    models = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Ottimizzazione degli iperparametri
        best_params = optimize_hyperparameters(X_train, y_train)

        # Creazione del modello XGBoost con gli iperparametri ottimizzati
        model = XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            random_state=42,
            n_jobs=-1,
            enable_categorical=True
        )
        model.fit(X_train, y_train)
        models.append(model)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        gc.collect()

    best_model_index = np.argmax(accuracies)
    best_model = models[best_model_index]

    # Predizioni finali
    df['Prediction'] = best_model.predict(X)

    # Shift della 'Prediction' all'interno di ciascun ticker
    df['Position'] = df.groupby('Ticker')['Prediction'].shift(1)

    # Calcolo dei rendimenti all'interno di ciascun ticker
    df['Returns'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).groupby(df['Ticker']).cumprod()

    # Calcolo del valore del portafoglio
    df['Portfolio_Value'] = df['Cumulative_Returns'] * 10000

    # Impostazione dell'indice per la visualizzazione
    df.set_index('Date', inplace=True)

    # Restituiamo anche il dataset delle feature X
    return df, best_model, X

# Funzione per applicare la strategia su più ticker usando il dataset aggregato
def apply_strategy_with_aggregated_model(df, tickers):
    df_aggregated = prepare_aggregated_dataset(df, tickers)
    df_result, best_model, X = apply_ml_strategy_aggregated(df_aggregated)

    portfolio = {}
    for ticker in tickers:
        df_ticker = df_result[df_result['Ticker'] == ticker]
        portfolio[ticker] = df_ticker['Portfolio_Value']

    portfolio_df = pd.DataFrame(portfolio)
    portfolio_df.index = pd.to_datetime(portfolio_df.index)

    # Restituiamo anche X insieme al portafoglio e al modello
    return portfolio_df, best_model, X


# Inizializziamo i dati
tickers = ["AAPL", "MSFT", "SPY", "TSLA", "JNJ", "KO", "MCD"]
start_date = "2015-01-01"
end_date = "2020-01-01"
df = get_data(tickers, start_date, end_date)

# Applicazione della strategia su tutti i ticker
portfolio, best_model, X = apply_strategy_with_aggregated_model(df, tickers)

# Visualizzazione dell'Equity Curve
plt.figure(figsize=(10, 6))
for ticker in tickers:
    plt.plot(portfolio.index, portfolio[ticker], label=f'Equity Curve {ticker}')
plt.title('Equity Curve basata su Pattern Tecnici per più Asset')
plt.xlabel('Data')
plt.ylabel('Valore del Portafoglio')
plt.legend()
plt.grid()
plt.show()


# ---- Inizio del codice SHAP ----
# Calcolo dei valori SHAP per il modello XGBoost allenato (best_model)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X)

# Plot della sintesi dei valori SHAP (importanza delle feature globali)
shap.summary_plot(shap_values, X)

# Inizializza JS per la visualizzazione interattiva
shap.initjs()

# Plot della singola predizione per interpretare una specifica decisione
shap.force_plot(explainer.expected_value, shap_values.values[0, :], X.iloc[0, :])
# ---- Fine del codice SHAP ----

# Analisi dell'importanza delle feature
def feature_importance_analysis(model, features):
    # Ottieni le importanze delle feature dal modello XGBoost
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type='gain')

    # Mappa gli indici delle feature ai nomi delle feature
    feature_map = {f'f{idx}': feat for idx, feat in enumerate(features + ['Ticker_Code'])}

    # Converti il dizionario delle importanze in un DataFrame
    importance_df = pd.DataFrame([
        {'Feature': feature_map.get(feat, feat), 'Importance': score}
        for feat, score in importance_dict.items()
    ])

    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

# Eseguire l'analisi dell'importanza delle feature
features = ['ATR', 'Doji', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Hammer',
            'Shooting_Star', 'Three_White_Soldiers', 'Three_Black_Crows',
            'Morning_Star', 'Evening_Star']
feature_importance_analysis(best_model, features)
