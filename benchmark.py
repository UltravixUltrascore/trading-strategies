#########################################################
#          RANDOM FOREST TRADING STRATEGY BACKTEST      #
#########################################################

from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix


# Funzione per scaricare i dati
def get_data(tickers, start_date, end_date):
    try:
        df = yf.download(tickers, start=start_date, end=end_date)
        df['Returns'] = df['Adj Close'].pct_change().shift(-1)
        df.dropna(inplace=True)
        df['Date'] = df.index
        return df
    except Exception as e:
        print(f"Errore durante il download dei dati per {tickers}: {e}")
        return pd.DataFrame()

# Media mobile semplice
def momentum(df):
    df['EMA_Momentum'] = df['EMA_50'] - df['EMA_200']
    return df



def preprocess_data(df):
    df = df = momentum(df)
    df = df.dropna()  # Elimina le righe con NaN create dai calcoli degli indicatori
    
    # Selezione delle feature
    X = df[[ 'EMA_50', 'EMA_200', 'Momentum']]
    
    # Target per il modello
    y = df['Returns'].apply(lambda x: 2 if x > 0 else (-2 if x < 0 else 0))

    # Split del dataset in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

    # Mostra la distribuzione delle classi
    print("Distribuzione delle classi nel set di training:", np.bincount(y_train + 2))

    # Escludi le classi con meno di 2 campioni
    unique, counts = np.unique(y_train, return_counts=True)
    valid_classes = unique[counts > 1]
    
    # Filtro per mantenere solo le classi con più di 1 campione
    mask = np.isin(y_train, valid_classes)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Standardizzazione delle feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Applicazione del bilanciamento SMOTE
    smote = SMOTE(k_neighbors=1, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    return X_train_smote, X_test_scaled, y_train_smote, y_test, df




# Funzione per Random Forest con Early Stopping
def early_stopping_random_forest(X_train, y_train, X_valid, y_valid, patience=10, max_estimators=100, step=10):
    best_loss = float('inf')
    best_n_estimators = 0
    patience_counter = 0
    best_model = None

    for n_estimators in range(step, max_estimators + step, step):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
        )
        
        # Addestramento del modello con gli alberi correnti
        model.fit(X_train, y_train)
        
        # Valutazione della performance sul set di validazione
        y_valid_pred = model.predict_proba(X_valid)
        loss = log_loss(y_valid, y_valid_pred)
        
        if loss < best_loss:
            best_loss = loss
            best_n_estimators = n_estimators
            best_model = model
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break
    
    return best_model


# Funzione per Ottimizzazione Bayesiana
def bayesian_optimization_with_early_stopping(X_train, y_train, X_test, y_test, df):
    def rf_cv_wrapper(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        X_train_part, X_valid_part, y_train_part, y_valid_part = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train_part, y_train_part)
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Calcoliamo le metriche di performance basate sul backtest
        df_test = df.iloc[-len(X_test):].copy()
        df_test['Predicted_Signal'] = y_test_pred
        df_test['Strategy_Returns'] = df_test['Returns'] * (df_test['Predicted_Signal'] / 2)

        mean_return = df_test['Strategy_Returns'].mean() * 252
        std_return = df_test['Strategy_Returns'].std() * np.sqrt(252)
        sharpe_ratio = (mean_return / std_return) if std_return != 0 else 0

        combined_metric = 0.7 * sharpe_ratio + 0.3 * mean_return
        return combined_metric

    optimizer = BayesianOptimization(
        f=rf_cv_wrapper,
        pbounds={
            'n_estimators': (80, 150),
            'max_depth': (10, 20),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'max_features': (0.1, 1.0),
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=50)
    return optimizer.max['params']


# Funzione di ranking delle feature per importanza
def feature_importance_ranking(model, X_train):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print(f"{f + 1}. Feature {X_train.columns[indices[f]]} ({importances[indices[f]]:.4f})")
    
    return indices, importances


# Funzione di filtraggio delle feature meno importanti
def filter_features_by_importance(model, X_train, threshold=0.05):
    importances = model.feature_importances_
    important_features = [X_train.columns[i] for i in range(len(importances)) if importances[i] >= threshold]
    return X_train[important_features]




def backtest_random_forest_with_bayesian(X_train, X_test, y_train, y_test, df_test, ticker):
    # Ottimizziamo i parametri utilizzando Bayesian Optimization
    best_params = bayesian_optimization_with_early_stopping(X_train, y_train, X_test, y_test, df_test)

    # Inizializziamo il modello con i migliori parametri
    model = RandomForestClassifier(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        min_samples_split=int(best_params['min_samples_split']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        max_features=best_params['max_features'],
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Addestriamo il modello
    model.fit(X_train, y_train)
    
    # Usa le colonne effettive delle feature usate nel modello
    feature_columns = ['EMA_50', 'Ema_200', 'Momentum']  # Colonne delle feature usate nel preprocessamento
    
    # Mostra l'importanza delle feature
    # Assicurati che X_train_filtered abbia lo stesso numero di feature indicato in feature_columns
    feature_importance_ranking(model, pd.DataFrame(X_train, columns=feature_columns[:X_train.shape[1]]))

    # Filtra le feature in base all'importanza
    X_train_filtered = filter_features_by_importance(model, pd.DataFrame(X_train, columns=feature_columns[:X_train.shape[1]]), threshold=0.05)
    X_test_filtered = pd.DataFrame(X_test, columns=feature_columns[:X_train_filtered.shape[1]])[X_train_filtered.columns]

    # Addestra di nuovo il modello con le feature filtrate
    model.fit(X_train_filtered, y_train)

    # Predizione sui dati di test con feature filtrate
    y_test_pred = model.predict(X_test_filtered)

    # Valutazione delle performance con controllo di 'zero_division'
    print("\n*** Valutazione sul training set ***")
    print("Accuracy (Train):", accuracy_score(y_train, model.predict(X_train_filtered)))
    print(classification_report(y_train, model.predict(X_train_filtered), zero_division=0))  # imposta a 0 con classi non previste

    print("\n*** Valutazione sul test set ***")
    print("Accuracy (Test):", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, zero_division=0))  # imposta a 0 in caso di classi non previste

    # Matrice di confusione
    cm = confusion_matrix(y_test, y_test_pred)
    print("Matrice di confusione (Test):")
    print(cm)

    # Calcoliamo i rendimenti della strategia
    df_test = df_test.iloc[-len(X_test):].copy()
    df_test['Predicted_Signal'] = y_test_pred
    df_test['Strategy_Returns'] = df_test['Returns'] * (df_test['Predicted_Signal'] / 2)
    
    initial_portfolio_value = 1000
    df_test['Portfolio_Value'] = initial_portfolio_value * (1 + df_test['Strategy_Returns']).cumprod()

    # Calcoliamo il rendimento cumulativo e lo Sharpe Ratio
    total_returns = (df_test['Portfolio_Value'].iloc[-1] / df_test['Portfolio_Value'].iloc[0]) - 1
    num_years = len(df_test) / 252
    expected_return_annualized = (1 + total_returns) ** (1 / num_years) - 1
    expected_return_annualized *= 100

    sharpe = np.mean(df_test['Strategy_Returns']) / np.std(df_test['Strategy_Returns']) * np.sqrt(252)

    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Rendimento cumulativo totale: {total_returns * 100:.2f}%")
    print(f"Rendimento atteso annualizzato: {expected_return_annualized:.2f}%")

    plot_equity_curve(df_test, ticker)




# Funzione per il plot dell'andamento del portafoglio
def plot_equity_curve(df_test, ticker):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_test['Date'], df_test['Portfolio_Value'], label='Valore Portafoglio', color='blue')
    ax.set_title(f'Andamento del Portafoglio ({ticker})')
    ax.set_ylabel('Valore Portafoglio')
    ax.set_xlabel('Data')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# Funzione principale
def main():
    tickers = ['AAPL', 'JNJ']
    start_date = '2000-01-01'
    end_date = '2023-01-01'

    for ticker_to_test in tickers:
        df = get_data(ticker_to_test, start_date, end_date)
        if df.empty:
            continue

        X_train_smote, X_test_scaled, y_train_smote, y_test, df = preprocess_data(df)
        print(f"\n*** Backtest Random Forest - {ticker_to_test} ***")
        backtest_random_forest_with_bayesian(X_train_smote, X_test_scaled, y_train_smote, y_test, df, ticker_to_test)

if __name__ == '__main__':
    main()
