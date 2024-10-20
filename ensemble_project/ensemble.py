import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from bayes_opt import BayesianOptimization
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns

# Funzione per scaricare e preparare i dati
def get_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df['Returns'] = df['Close'].pct_change()

    # Calcolo delle medie mobili
    df['ema50'] = df['Close'].rolling(window=50).mean()
    df['ema100'] = df['Close'].rolling(window=100).mean()

    # Feature aggiuntive
    df['ema_Ratio'] = df['ema50'] / df['ema100']
    df['Momentum'] = df['Close'].diff(5)

    df.dropna(inplace=True)
    return df

# Funzione per creare il target (segnale di trading)
def create_target(df):
    df['Signal'] = np.where(df['ema50'] > df['ema100'], 1, 0)
    return df

# Funzione per preparare i dati per il modello
def prepare_data(df):
    features = ['ema50', 'ema100', 'ema_Ratio', 'Momentum']
    X = df[features]
    y = df['Signal']

    # Standardizziamo le feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Funzione per creare e addestrare la rete neurale
def build_model(hidden_layer1, hidden_layer2, alpha, validation_fraction):
    model = MLPClassifier(hidden_layer_sizes=(int(hidden_layer1), int(hidden_layer2)), 
                          activation='relu', solver='adam', 
                          max_iter=1000, random_state=42, 
                          validation_fraction=validation_fraction, 
                          early_stopping=True, alpha=alpha)
    return model

# Funzione per eseguire il Monte Carlo Backtest
def montecarlo_simulation(index, df, model):
    df_sim = df.copy()

    # Permuta casualmente i rendimenti giornalieri
    df_sim['Returns'] = np.random.permutation(df_sim['Returns'])

    # Ricostruisce il prezzo dai nuovi rendimenti
    df_sim['Close'] = df_sim['Close'].iloc[0] * (1 + df_sim['Returns']).cumprod()

    # Predici i segnali con il modello addestrato
    X_sim = df_sim[['ema50', 'ema100', 'ema_Ratio', 'Momentum']]
    X_sim_scaled = StandardScaler().fit_transform(X_sim)
    df_sim['Predicted_Signal'] = model.predict(X_sim_scaled)
    df_sim['Predicted_Signal'] = df_sim['Predicted_Signal'].shift(1)

    # Simulazione della strategia
    df_sim['Strategy_Returns'] = np.where(df_sim['Predicted_Signal'] == 1, df_sim['Returns'], 0)
    df_sim['Cumulative_Strategy_Returns'] = (1 + df_sim['Strategy_Returns']).cumprod()
    
    return df_sim['Cumulative_Strategy_Returns'].iloc[-1] - 1

# Funzione per eseguire il Monte Carlo Backtest con multiprocessing
def montecarlo_backtest(df, model, n_simulations):
    montecarlo_results = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(montecarlo_simulation, range(n_simulations), [df]*n_simulations, [model]*n_simulations)
    
    montecarlo_results = list(results)
    return montecarlo_results

# Funzione per valutare il modello
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice di confusione:")
    print(cm)

# Funzione per visualizzare i tracciati dei segnali di trading
def plot_strategy(df, test_portfolio, title):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[len(df) - len(test_portfolio):], test_portfolio, label='Portafoglio Strategia')
    plt.plot(df.index, (1 + df['Returns']).cumprod(), label='Buy & Hold')
    plt.legend()
    plt.title(title)
    plt.ylabel('Valore del Portafoglio')
    plt.show()

# Funzione per aggiungere il confronto fra la strategia e Bue & Hold
def plot_comparison(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Cumulative_Strategy_Returns'], label='Strategia di Trading')
    plt.plot(df.index, df['Cumulative_Market_Returns'], label='Buy & Hold')
    plt.legend()
    plt.title('Confronto tra la Strategia di Trading e Buy & Hold')
    plt.xlabel('Data')
    plt.ylabel('Valore Cumulativo')
    plt.show()

# Blocco principale
if __name__ == '__main__':
    # Parametri
    ticker = 'MSFT'
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    # Scaricamento e preparazione dei dati
    df = get_data(ticker, start_date, end_date)
    df = create_target(df)

    # Prepara i dati
    X, y = prepare_data(df)

    # Suddivisione in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    # Creazione del modello ottimizzato
    model = build_model(hidden_layer1=123, hidden_layer2=124, alpha=0.0056, validation_fraction=0.229)

    # Addestramento
    model.fit(X_train, y_train)

    # Valutazione del modello sui dati di test
    evaluate_model(model, X_test, y_test)

    # Monte Carlo Backtest
    n_simulations = 1000
    montecarlo_results = montecarlo_backtest(df, model, n_simulations)

    # Visualizzazione dei risultati Monte Carlo
    plt.figure(figsize=(10, 5))
    sns.histplot(montecarlo_results, bins=50, color='blue', kde=True)
    plt.title('Distribuzione dei Ritorni - Monte Carlo')
    plt.xlabel('Rendimento (%)')
    plt.ylabel('Frequenza')
    plt.show()

    mean_return = np.mean(montecarlo_results)
    std_return = np.std(montecarlo_results)
    print(f"Rendimento medio Monte Carlo: {mean_return * 100:.2f}%")
    print(f"Deviazione standard dei rendimenti: {std_return * 100:.2f}%")
