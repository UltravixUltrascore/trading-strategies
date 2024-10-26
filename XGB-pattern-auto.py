import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization
import logging
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

# Funzione per ottimizzare gli iperparametri con XGBClassifier
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
        df_ticker['Bullish_Patterns'] = df_ticker[['Three_White_Soldiers', 'Morning_Star']].sum(axis=1)
        df_ticker['Bearish_Patterns'] = df_ticker[['Shooting_Star', 'Evening_Star']].sum(axis=1)

        df_list.append(df_ticker)

    df_combined = pd.concat(df_list, ignore_index=True)
    return df_combined

# Funzione per applicare Stop-Loss e Take-Profit decisi dal modello XGB
def apply_dynamic_stop_loss_take_profit(df, model, features):
    df = df.copy()
    df['Stop_Loss'] = np.nan
    df['Take_Profit'] = np.nan

    for idx, row in df.iterrows():
        if row['Position'] == 1.0:
            feature_values = row[features + ['Ticker_Code']].values.reshape(1, -1)
            prob = model.predict_proba(feature_values)[0, 1]  # Probabilità di classe 1

            take_profit_multiplier = 1 + (prob * 0.05)  # Tra 1.0 e 1.05
            stop_loss_multiplier = 1 - ((1 - prob) * 0.05)  # Tra 0.95 e 1.0

            df.at[idx, 'Take_Profit'] = row['Close'] * take_profit_multiplier
            df.at[idx, 'Stop_Loss'] = row['Close'] * stop_loss_multiplier

    df['SL_Hit'] = (df['Low'] <= df['Stop_Loss'])
    df['TP_Hit'] = (df['High'] >= df['Take_Profit'])

    df['Exit_Price'] = df['Close']
    df.loc[df['SL_Hit'] & ~df['TP_Hit'], 'Exit_Price'] = df['Stop_Loss']
    df.loc[df['TP_Hit'] & ~df['SL_Hit'], 'Exit_Price'] = df['Take_Profit']
    df.loc[df['SL_Hit'] & df['TP_Hit'], 'Exit_Price'] = df['Stop_Loss']

    df['Strategy_Returns'] = 0.0
    df.loc[df['Position'] == 1, 'Strategy_Returns'] = (df['Exit_Price'] - df['Open']) / df['Open']
    return df

# Funzione per applicare la strategia con un unico modello aggregato
def apply_ml_strategy_aggregated(df):
    df = df.copy()

    features = ['Doji', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Hammer',
                'Three_Black_Crows', 'Bullish_Patterns', 'Bearish_Patterns']
    df['Ticker_Code'] = pd.Categorical(df['Ticker']).codes

    df['Target'] = df.groupby('Ticker')['Close'].shift(-1) > df['Close']
    df['Target'] = df['Target'].astype(int)

    X = df[features + ['Ticker_Code']]
    y = df['Target']

    X = X.dropna()
    y = y.loc[X.index]
    df = df.loc[X.index]

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.reset_index(drop=True, inplace=True)
    df['Date_Ticker'] = df['Date'].astype(str) + '_' + df['Ticker']
    df.set_index('Date_Ticker', inplace=True)

    models, f1_scores, predictions = walk_forward_optimization(X, y, df, n_splits=5)
    df['Prediction'] = predictions

    df['Position'] = df.groupby('Ticker')['Prediction'].shift(1).fillna(0)
    df = apply_dynamic_stop_loss_take_profit(df, models[-1], features)

    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).groupby(df['Ticker']).cumprod()
    df['Portfolio_Value'] = df['Cumulative_Returns'] * 10000

    return df, models, X

# Funzione per applicare la strategia su più ticker usando il dataset aggregato
def apply_strategy_with_aggregated_model(df, tickers):
    df_aggregated = prepare_aggregated_dataset(df, tickers)
    df_result, models, X = apply_ml_strategy_aggregated(df_aggregated)

    portfolio_list = []
    for ticker in tickers:
        df_ticker = df_result[df_result['Ticker'] == ticker].copy()
        df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
        df_ticker.set_index('Date', inplace=True)
        portfolio_series = df_ticker['Portfolio_Value'].rename(ticker)
        portfolio_list.append(portfolio_series)

    # Concatenare le serie di portafoglio lungo l'asse delle colonne
    portfolio_df = pd.concat(portfolio_list, axis=1)

    return portfolio_df, models, X, df_result

# Funzione di Walk Forward Optimization
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
            n_jobs=-1,
            enable_categorical=True
        )
        model.fit(X_train, y_train)
        models.append(model)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

        predictions.iloc[test_index] = y_pred

    print(f"F1 Score medio durante WFO: {np.mean(f1_scores)}")
    return models, f1_scores, predictions

# Inizializziamo i dati
tickers = ["AAPL", "MSFT", "SPY", "TSLA", "JNJ", "KO", "MCD"]
start_date = "2015-01-01"
end_date = "2020-01-01"
df = get_data(tickers, start_date, end_date)

# Applicazione della strategia su tutti i ticker
portfolio_df, models, X, df_result = apply_strategy_with_aggregated_model(df, tickers)

# Verifica che l'indice sia il campo temporale corretto
portfolio_df.index = pd.to_datetime(portfolio_df.index)

# Controllo del DataFrame portfolio_df
print(portfolio_df.head())
print(portfolio_df.tail())

# Visualizzazione dell'Equity Curve
plt.figure(figsize=(10, 6))
for ticker in tickers:
    if ticker in portfolio_df.columns:
        plt.plot(portfolio_df.index, portfolio_df[ticker], label=f'Equity Curve {ticker}')
    else:
        print(f"Ticker {ticker} non presente in portfolio_df.columns")
plt.title('Equity Curve basata su Pattern Tecnici per più Asset')
plt.xlabel('Data')
plt.ylabel('Valore del Portafoglio')
plt.legend()
plt.grid()
plt.show()

# ---- Inizio del codice SHAP ----
explainer = shap.TreeExplainer(models[-1])
shap_values = explainer(X)

shap.summary_plot(shap_values, X)
shap.initjs()
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

    # Ordina i dati per importanza
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    # Verifica che il DataFrame non sia vuoto
    if importance_df.empty:
        print("Il DataFrame delle importanze è vuoto. Nessuna feature ha un'importanza significativa.")
    else:
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--')
        plt.show()

# Eseguire l'analisi dell'importanza delle feature utilizzando l'ultimo modello
features = ['Doji', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Hammer',
            'Three_Black_Crows', 'Bullish_Patterns', 'Bearish_Patterns']
feature_importance_analysis(models[-1], features)
