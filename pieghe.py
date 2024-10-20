import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Funzione per scaricare e preparare i dati
def get_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df['Returns'] = df['Close'].pct_change()

    # Calcolo delle medie mobili
    df['ema50'] = df['Close'].rolling(window=50).mean()
    df['ema200'] = df['Close'].rolling(window=200).mean()

    # Feature aggiuntive
    df['ema_Ratio'] = df['ema50'] / df['ema200']  # Rapporto tra ema50 e ema200
    df['Momentum'] = df['Close'].diff(5)  # Momento a 5 giorni

    # Rimozione dei valori NaN
    df.dropna(inplace=True)

    return df

# Funzione per creare il target (segnale di trading)
def create_target(df):
    df['Signal'] = np.where(df['ema50'] > df['ema200'], 1, 0)  # 1 per long, 0 per short/nessuna posizione
    return df

# Funzione per preparare i dati per il modello
def prepare_data(df):
    features = ['ema50', 'ema200', 'ema_Ratio', 'Momentum']
    X = df[features]
    y = df['Signal']

    # Standardizziamo le feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Funzione per costruire la rete neurale con Scikit-learn
def build_model():
    model = MLPClassifier(hidden_layer_sizes=(96, 49), activation='relu', solver='adam', 
                      max_iter=1000, random_state=42, 
                      validation_fraction=0.2622, early_stopping=True, alpha=0.008469)
    return model

# Parametri
ticker = 'BABA'
start_date = '2010-01-01'
end_date = '2023-01-01'

# Scaricamento e preparazione dei dati
df = get_data(ticker, start_date, end_date)
df = create_target(df)

# Prepara i dati
X, y = prepare_data(df)

# Definiamo K per la cross-validation
kfold = KFold(n_splits=5, shuffle=True)  # Cross-validation a 5 pieghe (fold)

# Costruzione del modello
model = build_model()

# Eseguire la cross-validation e calcolare l'accuratezza per ogni fold
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Output dei risultati
print(f"Accuratezza per ciascuna piega: {cv_scores}")
print(f"Accuratezza media: {cv_scores.mean():.4f}")
print(f"Deviazione standard delle accuratezze: {cv_scores.std():.4f}")
