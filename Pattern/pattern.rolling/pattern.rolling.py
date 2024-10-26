import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import logging

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

# Funzione per aggiungere la media mobile
def add_moving_average(df, period=50):
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(window=period).mean()
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

# Funzione per applicare la strategia con Random Forest
def apply_ml_strategy(df):
    df = df.dropna().copy()

    # Definizione delle features e target
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'SMA50', 'Doji', 'Engulfing_Bullish', 'Engulfing_Bearish',
                'Hammer', 'Shooting_Star', 'Three_White_Soldiers', 'Three_Black_Crows', 'Morning_Star', 'Evening_Star']
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    X = df[features]
    y = df['Target']

    # Suddivisione del dataset in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modello Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Previsioni
    df['Prediction'] = model.predict(X)

    # Metriche di performance
    accuracy = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))
    
    logging.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

    # Determinazione delle posizioni (decisioni di trading)
    df['Position'] = df['Prediction'].shift(1)

    # Calcolo dei rendimenti
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Returns'] = df['Position'] * df['Returns']

    # Calcolo del valore del portafoglio
    initial_capital = 10000
    df['Portfolio_Value'] = initial_capital * (1 + df['Strategy_Returns'].cumsum())

    # Salvataggio delle decisioni di trading
    df[['Position', 'Returns', 'Strategy_Returns', 'Portfolio_Value']].to_csv('trading_decisions.csv')

    return df

# Funzione per applicare la strategia su più ticker
def apply_strategy_to_tickers(df, tickers):
    portfolio = pd.DataFrame()

    for ticker in tickers:
        print(f'Processing {ticker}...')
        df_ticker = df[ticker].copy()

        # Calcolo dell'ATR e aggiunta della SMA
        df_ticker = calculate_atr(df_ticker)
        df_ticker = add_moving_average(df_ticker)

        # Rilevamento dei pattern tecnici
        df_ticker = detect_doji(df_ticker)
        df_ticker = detect_engulfing(df_ticker)
        df_ticker = detect_hammer_shooting_star(df_ticker)
        df_ticker = detect_three_soldiers_crows(df_ticker)
        df_ticker = detect_morning_evening_star(df_ticker)

        # Applicazione della strategia basata su Random Forest
        df_ticker = apply_ml_strategy(df_ticker)

        # Aggiunta del valore del portafoglio al portafoglio complessivo
        portfolio[ticker] = df_ticker['Portfolio_Value']

    return portfolio

# Funzione per la Walk Forward Optimization
def walk_forward_optimization(df, tickers, window_size, step_size):
    results = []
    tscv = TimeSeriesSplit(n_splits=int((len(df) - window_size) / step_size))

    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        # Applichiamo la strategia su ogni sottoinsieme
        portfolio_train = apply_strategy_to_tickers(train_df, tickers)
        portfolio_test = apply_strategy_to_tickers(test_df, tickers)
        
        # Valutiamo la performance del test set
        test_performance = portfolio_test.mean(axis=1).iloc[-1]
        results.append(test_performance)
        
        # Visualizzazione delle Equity Curves per ogni finestra
        plt.figure(figsize=(10, 6))
        for ticker in tickers:
            plt.plot(portfolio_test.index, portfolio_test[ticker], label=f'Equity Curve {ticker} - Fold {i+1}')
        plt.title(f'Equity Curve per Fold {i+1}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid()
        plt.show()

    return results


# Inizializziamo i dati
tickers = ["AAPL", "MSFT", "SPY", "TSLA", "JNJ", "KO", "MCD"]
start_date = "2015-01-01"
end_date = "2020-01-01"

# Scarichiamo i dati storici
df = get_data(tickers, start_date, end_date)

# Applichiamo la strategia con Walk Forward Optimization
window_size = 365  # 1 anno
step_size = 90  # 3 mesi
wfo_results = walk_forward_optimization(df, tickers, window_size, step_size)

# Applichiamo la strategia su ciascun ticker e memorizziamo il portafoglio
portfolio = apply_strategy_to_tickers(df, tickers)

# Visualizziamo le performance delle finestre
print(wfo_results)

# Analisi dell'importanza delle feature
def feature_importance_analysis(model, features):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

# Eseguire l'analisi dell'importanza delle feature
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'SMA50', 'Doji', 'Engulfing_Bullish', 'Engulfing_Bearish', 'Hammer', 'Shooting_Star', 'Three_White_Soldiers', 'Three_Black_Crows', 'Morning_Star', 'Evening_Star']
X = df[features]
y = df['Target']
feature_importance_analysis(RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y), features)


# Plot delle Equity Curves
plt.figure(figsize=(10, 6))
for ticker in tickers:
    plt.plot(portfolio.index, portfolio[ticker], label=f'Equity Curve {ticker}')

plt.title('Equity Curve basata su Pattern Tecnici per più Asset')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid()
plt.show()


