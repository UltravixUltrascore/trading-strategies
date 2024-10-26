import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from bayes_opt import BayesianOptimization
import logging
import gc
import shap

# Configurazione logging
logging.basicConfig(filename='trading_strategy.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Funzione per scaricare i dati storici da yFinance
def get_data(tickers, start_date, end_date):
    df = yf.download(tickers=tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    return df

# Funzione per rilevare pattern Doji
def detect_doji(df):
    df = df.copy()
    df['Doji'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
    df['Doji'] = df['Doji'].astype(int)
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
    df['Engulfing_Bullish'] = df['Engulfing_Bullish'].astype(int)

    # Engulfing Bearish
    prev_bullish = prev_close > prev_open
    curr_bearish = df['Close'] < df['Open']
    df['Engulfing_Bearish'] = prev_bullish & curr_bearish & (df['Open'] >= prev_close) & (df['Close'] <= prev_open)
    df['Engulfing_Bearish'] = df['Engulfing_Bearish'].astype(int)

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
    df['Hammer'] = hammer_cond.astype(int)

    # Shooting Star
    shooting_star_cond = (
        (df['Open'] > df['Close']) &
        ((df['High'] - df['Open']) >= 2 * (df['Open'] - df['Close'])) &
        ((df['Close'] - df['Low']) <= (df['Open'] - df['Close']))
    )
    df['Shooting_Star'] = shooting_star_cond.astype(int)
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
    df['Three_White_Soldiers'] = white_soldiers_cond.astype(int)

    # Three Black Crows
    black_crows_cond = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Close'] < df['Close'].shift(1)) &
        (df['Close'].shift(1) < df['Close'].shift(2))
    )
    df['Three_Black_Crows'] = black_crows_cond.astype(int)
    return df

# Funzione per rilevare Morning Star ed Evening Star
def detect_morning_evening_star(df):
    df = df.copy()

    # Morning Star
    morning_star_cond = (
        (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Close'].shift(1).between(df['Open'].shift(1), df['Close'].shift(1))) &
        (df['Close'] > df['Open']) &
        (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2)
    )
    df['Morning_Star'] = morning_star_cond.astype(int)

    # Evening Star
    evening_star_cond = (
        (df['Close'].shift(2) > df['Open'].shift(2)) &
        (df['Close'].shift(1).between(df['Open'].shift(1), df['Close'].shift(1))) &
        (df['Close'] < df['Open']) &
        (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2)
    )
    df['Evening_Star'] = evening_star_cond.astype(int)
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
            enable_categorical=True
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
        return scores.mean()

    pbounds = {
        'n_estimators': (70, 120),
        'max_depth': (3.0, 15),
        'learning_rate': (0.01, 0.1),
        'subsample': (0.01, 1.0),
        'colsample_bytree': (0.01, 1.0)
    }

    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=5, n_iter=20)

    # Recuperiamo i migliori iperparametri
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])

    return best_params

# Funzione per applicare Stop-Loss e Take-Profit decisi dal modello XGB
def apply_dynamic_stop_loss_take_profit(df, model, features):
    df = df.copy()
    df['Stop_Loss'] = np.nan
    df['Take_Profit'] = np.nan

    # Calcola i livelli di Stop-Loss e Take-Profit basati sulle predizioni del modello
    for idx in df.index:
        if df.loc[idx, 'Position'] == 1:
            feature_values = df.loc[idx, features + ['Ticker_Code']].values.reshape(1, -1)
            df.loc[idx, 'Take_Profit'] = model.predict(feature_values)[0] * df.loc[idx, 'Close'] * 1.02  # Take-Profit
            df.loc[idx, 'Stop_Loss'] = model.predict(feature_values)[0] * df.loc[idx, 'Close'] * 0.98  # Stop-Loss

    # Determina se lo Stop-Loss o il Take-Profit sono stati raggiunti
    df['SL_Hit'] = (df['Low'] <= df['Stop_Loss'])
    df['TP_Hit'] = (df['High'] >= df['Take_Profit'])

    # Prezzo di uscita predefinito è la chiusura
    df['Exit_Price'] = df['Close']

    # Caso 1: Solo Stop-Loss colpito
    df.loc[df['SL_Hit'] & ~df['TP_Hit'], 'Exit_Price'] = df['Stop_Loss']
    # Caso 2: Solo Take-Profit colpito
    df.loc[df['TP_Hit'] & ~df['SL_Hit'], 'Exit_Price'] = df['Take_Profit']
    # Caso 3: Entrambi colpiti - assumiamo che lo Stop-Loss sia colpito per primo
    df.loc[df['SL_Hit'] & df['TP_Hit'], 'Exit_Price'] = df['Stop_Loss']

    # Calcolo dei rendimenti
    df['Strategy_Returns'] = 0.0
    df.loc[df['Position'] == 1, 'Strategy_Returns'] = (df['Exit_Price'] - df['Open']) / df['Open']
    df.loc[df['Position'] != 1, 'Strategy_Returns'] = 0.0

    return df

# Funzione per applicare la strategia con un unico modello aggregato
def apply_ml_strategy_aggregated(df):
    df = df.copy()

    features = ['Doji', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Hammer',
                'Three_Black_Crows', 'Bullish_Patterns', 'Bearish_Patterns']
    df['Ticker_Code'] = pd.Categorical(df['Ticker']).codes
    df['Ticker_Code'] = df['Ticker_Code'].astype(int)

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

    # Applicazione della Walk Forward Optimization
    models, f1_scores, predictions = walk_forward_optimization(X, y, df, n_splits=5)

    df['Prediction'] = predictions

    # Shift della 'Prediction' all'interno di ciascun ticker
    df['Position'] = df.groupby('Ticker')['Prediction'].shift(1)

    # Applica Stop-Loss e Take-Profit dinamici con il modello
    df = apply_dynamic_stop_loss_take_profit(df, models[-1], features)

    # Calcolo dei rendimenti cumulativi e del valore del portafoglio
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).groupby(df['Ticker']).cumprod()
    df['Portfolio_Value'] = df['Cumulative_Returns'] * 10000

    return df, models, X

# Funzione per applicare la strategia su più ticker usando il dataset aggregato
def apply_strategy_with_aggregated_model(df, tickers):
    df_aggregated = prepare_aggregated_dataset(df, tickers)
    df_result, models, X = apply_ml_strategy_aggregated(df_aggregated)

    portfolio = {}
    for ticker in tickers:
        df_ticker = df_result[df_result['Ticker'] == ticker]
        portfolio[ticker] = df_ticker['Portfolio_Value']

    portfolio_df = pd.DataFrame(portfolio)
    portfolio_df.index = pd.to_datetime(portfolio_df.index)

    # Restituiamo anche X insieme al portafoglio e ai modelli
    return portfolio_df, models, X, df_result

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

        # Feature engineering: combinazione di pattern
        df_ticker['Bullish_Patterns'] = df_ticker[['Three_White_Soldiers', 'Morning_Star']].sum(axis=1)
        df_ticker['Bearish_Patterns'] = df_ticker[['Shooting_Star', 'Evening_Star']].sum(axis=1)

        df_list.append(df_ticker)

    df_combined = pd.concat(df_list, ignore_index=True)
    return df_combined

# Funzione per la Walk Forward Optimization
def walk_forward_optimization(X, y, df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    f1_scores = []
    models = []
    predictions = pd.Series(index=X.index, dtype=int)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Ottimizzazione degli iperparametri
        best_params = optimize_hyperparameters(X_train, y_train)

        # Modello XGBoost con i parametri ottimizzati
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

        # Test del modello su X_test
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

        # Memorizza le predizioni
        predictions.iloc[test_index] = y_pred

    print(f"F1 Score medio durante WFO: {np.mean(f1_scores)}")
    return models, f1_scores, predictions

# Inizializziamo i dati
tickers = ["AAPL", "MSFT", "SPY", "TSLA", "JNJ", "KO", "MCD"]
start_date = "2015-01-01"
end_date = "2020-01-01"
df = get_data(tickers, start_date, end_date)

# Applicazione della strategia su tutti i ticker
portfolio, models, X, df_result = apply_strategy_with_aggregated_model(df, tickers)

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
# Calcolo dei valori SHAP per l'ultimo modello XGBoost allenato
explainer = shap.TreeExplainer(models[-1])
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