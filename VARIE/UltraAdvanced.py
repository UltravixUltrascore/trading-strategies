# models/supervised.py

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from bayes_opt import BayesianOptimization
import pandas as pd
from sklearn.decomposition import PCA

class SupervisedModels:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': self._create_xgboost_model()
        }
        self.pipelines = {}

    def _create_xgboost_model(self):
        return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', reg_alpha=0.1, reg_lambda=0.1)

    def create_pipeline(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not recognized.")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),  # PCA to reduce dimensionality, keeping 95% variance
            ('model', self.models[model_name])
        ])
        self.pipelines[model_name] = pipeline

    def train_model(self, X_train, y_train, model_name='random_forest'):
        if model_name not in self.pipelines:
            self.create_pipeline(model_name)
        tscv = TimeSeriesSplit(n_splits=5)

        def objective_function(n_estimators, max_depth, learning_rate):
            # Reset model to original state to avoid cumulative parameter changes
            self.models[model_name] = self._create_xgboost_model()
            self.models[model_name].set_params(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=learning_rate
            )
            scores = cross_val_score(self.pipelines[model_name], X_train, y_train, cv=tscv, scoring='accuracy')
            return scores.mean()

        param_bounds = {
            'n_estimators': (50, 150),
            'max_depth': (3, 7),
            'learning_rate': (0.01, 0.1)
        } if model_name == 'xgboost' else {
            'n_estimators': (50, 150),
            'max_depth': (3, 7)
        }

        optimizer = BayesianOptimization(f=objective_function, pbounds=param_bounds, random_state=42)
        optimizer.maximize(init_points=5, n_iter=20)

        best_params = optimizer.max['params']
        self.models[model_name].set_params(
            n_estimators=int(best_params['n_estimators']),
            max_depth=int(best_params['max_depth']),
            learning_rate=best_params.get('learning_rate', 0.1)
        )
        self.pipelines[model_name].fit(X_train, y_train)
        print(f"Best parameters for {model_name}: {best_params}")

    def predict(self, X_test, model_name='random_forest'):
        if model_name not in self.pipelines:
            raise ValueError(f"Model {model_name} pipeline not initialized.")
        return self.pipelines[model_name].predict(X_test)

# feature_engineering.py

import pandas as pd
from arch import arch_model
import numpy as np

class FeatureEngineering:
    def __init__(self):
        pass

    def create_features(self, data):
        data['RSI'] = self.calculate_rsi(data['Adj Close'])
        data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Adj Close'])
        data['ATR'] = self.calculate_atr(data)
        data['Historical_Volatility'] = data['Adj Close'].pct_change().rolling(14).std()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()  # Simple Moving Average (50 days)
        data['SMA_200'] = data['Adj Close'].rolling(window=200).mean()  # Simple Moving Average (200 days)
        data['Ulcer_Index'] = self.calculate_ulcer_index(data['Adj Close'])  # Ulcer Index
        data['GARCH_Volatility'] = self.calculate_garch_volatility(data['Adj Close'])  # GARCH Volatility
        data = self.add_time_features(data)
        data = self.handle_missing_values(data)
        return data

    def handle_missing_values(self, data):
        # Use different strategies to handle missing values depending on the nature of the feature
        for column in data.columns:
            if column in ['Volume', 'Historical_Volatility', 'ATR']:
                data[column].fillna(data[column].median(), inplace=True)  # Use median for features with high variance
            elif column in ['RSI', 'MACD', 'MACD_Signal']:
                data[column].fillna(data[column].mean(), inplace=True)  # Use mean for smoother features
            else:
                data[column] = data[column].interpolate(method='linear')  # Use linear interpolation for other features
        return data

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, short_window=12, long_window=26, signal_window=9):
        short_ema = prices.ewm(span=short_window, adjust=False).mean()
        long_ema = prices.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def calculate_atr(self, data, window=14):
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Adj Close'].shift()).abs()
        low_close = (data['Low'] - data['Adj Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window, min_periods=1).mean()  # Adjusted to account for gaps in data
        return atr

    def calculate_ulcer_index(self, prices, window=14):
        rolling_max = prices.rolling(window=window).max()
        drawdown = (prices - rolling_max) / rolling_max * 100
        ulcer_index = (drawdown ** 2).rolling(window=window).mean().apply(np.sqrt)
        return ulcer_index

    def calculate_garch_volatility(self, prices):
        returns = prices.pct_change().dropna() * 100
        model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_fit = model.fit(disp='off')
        return garch_fit.conditional_volatility

    def add_time_features(self, data):
        data['Day'] = data.index.day
        data['Month'] = data.index.month
        data['Weekday'] = data.index.weekday
        return data

# data_handler.py

import yfinance as yf
import os

class DataHandler:
    def __init__(self, cache_dir='cache/'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_data(self, ticker, start_date, end_date):
        cache_file = f"{self.cache_dir}{ticker}_data.csv"
        if os.path.exists(cache_file):
            data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            # Check if the existing data covers the requested date range
            if data.index.min() <= pd.to_datetime(start_date) and data.index.max() >= pd.to_datetime(end_date):
                data = data.loc[start_date:end_date]
            else:
                # Download missing data
                new_data = yf.download(ticker, start=start_date, end=end_date)
                data = pd.concat([data, new_data])
                data = data[~data.index.duplicated(keep='last')]
                data.to_csv(cache_file)
                data = data.loc[start_date:end_date]
        else:
            data = yf.download(ticker, start=start_date, end=end_date)
            data.to_csv(cache_file)
        return data

    def update_data(self, ticker, last_date):
        new_data = yf.download(ticker, start=last_date)
        cache_file = f"{self.cache_dir}{ticker}_data.csv"
        if os.path.exists(cache_file):
            existing_data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            data = pd.concat([existing_data, new_data])
            data = data[~data.index.duplicated(keep='last')]  # Remove duplicates
            data.to_csv(cache_file)
        else:
            new_data.to_csv(cache_file)
            data = new_data
        return data

# risk_manager.py

class RiskManager:
    def __init__(self, risk_per_trade=0.01):
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(self, capital, atr, entry_price, atr_long_term_mean):
        risk_amount = capital * self.risk_per_trade
        stop_loss = atr * 2
        position_size = risk_amount / stop_loss
        max_position = capital / entry_price

        # Reduce position size if ATR is significantly higher than its long-term average
        if atr > 1.5 * atr_long_term_mean:
            position_size *= 0.5
            print("Warning: ATR is significantly higher than its long-term average. Reducing position size.")

        position_size = min(position_size, max_position)

        # Automate reduction of position sizes or implement stop-loss measures
        if position_size < 1:
            print("Warning: Position size is very small, consider adjusting risk management strategy.")
            position_size = 0  # Avoid taking positions that are too small to be meaningful

        return position_size

# signal_generator.py

import numpy as np

class SignalGenerator:
    def __init__(self, model):
        self.model = model

    def generate_signals(self, data, model_type='rf'):
        features = data.drop(columns=['Adj Close', 'Signal'], errors='ignore')
        predictions = self.model.predict(features, model_type=model_type)
        data['Signal'] = np.where(predictions > 0, 1, -1)
        return data

# backtester.py

import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def run_backtest(self, data):
        data['Market_Return'] = data['Adj Close'].pct_change()
        data['Strategy_Return'] = data['Market_Return'] * data['Signal'].shift(1)
        data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod() * self.initial_capital
        final_return = data['Cumulative_Return'].iloc[-1]
        performance_metrics = self.calculate_performance_metrics(data)
        return final_return, performance_metrics

    def calculate_performance_metrics(self, data):
        returns = data['Strategy_Return'].dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(data['Cumulative_Return'])
        return {
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown
        }

    def calculate_sortino_ratio(self, returns):
        negative_returns = returns[returns < 0]
        expected_return = returns.mean()
        downside_std = negative_returns.std()
        if downside_std == 0:
            return np.nan  # Avoid division by zero
        sortino_ratio = (expected_return / downside_std) * np.sqrt(252)
        return sortino_ratio

    def calculate_max_drawdown(self, cumulative_returns):
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return max_drawdown

# main.py

from sklearn.model_selection import TimeSeriesSplit
import yaml
import numpy as np


def main():
    # Carica la configurazione
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ticker = config['data']['ticker']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    initial_capital = config['backtesting']['initial_capital']
    risk_per_trade = config['risk_management']['risk_per_trade']

    # 1. Gestione dei dati
    data_handler = DataHandler()
    data = data_handler.download_data(ticker, start_date, end_date)

    # 2. Feature Engineering
    fe = FeatureEngineering()
    data = fe.create_features(data)
    features = fe.select_features(data)

    # Target variable
    data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
    data['Target'] = ((data['SMA_20'] > data['SMA_50']) & (data['SMA_20'].shift(-1) <= data['SMA_50'])).astype(int)
    data = fe.handle_missing_values(data)  # Use the feature engineering method to handle missing values
    X = features
    y = data['Target']

    # Separate out-of-sample data
    train_size = int(0.8 * len(X))
    X_train, X_oos = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_oos = y.iloc[:train_size], y.iloc[train_size:]

    # TimeSeriesSplit to avoid data leakage
    n_splits = 5 if len(X_train) > 100 else 3  # Adjusting split ratio based on dataset size
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 3. Modelli Supervisionati
    models = SupervisedModels()
    for train_index, test_index in tscv.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        models.train_model(X_train_fold, y_train_fold, model_name='xgboost')

    # 4. Generazione dei Segnali con trailing stop dinamico
    signal_generator = SignalGenerator(models)
    data_with_signals_train = data.loc[X_train.index].copy()  # Use train dataset for generating signals
    data_with_signals_train = signal_generator.generate_signals(data_with_signals_train, model_type='xgboost')

    # Implement dynamic trailing stop based on ATR
    data_with_signals_train['Trailing_Stop'] = data_with_signals_train['Adj Close'] - (data_with_signals_train['ATR'] * data_with_signals_train['ATR'].rolling(window=14).mean())

    # 5. Gestione del Rischio con dimensionamento dinamico delle posizioni
    risk_manager = RiskManager(risk_per_trade=risk_per_trade)
    data_with_signals_train['Position_Size'] = data_with_signals_train.apply(lambda row: risk_manager.calculate_position_size(initial_capital, row['ATR'], row['Adj Close']), axis=1)

    
    
