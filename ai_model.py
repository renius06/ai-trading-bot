import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from typing import Tuple, Dict, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta


class TradingAIModel:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger('TradingAIModel')
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['returns_2'] = df['close'].pct_change(2)
        features['returns_5'] = df['close'].pct_change(5)
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Technical indicators as features
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi']
        if 'macd' in df.columns:
            features['macd'] = df['macd']
        if 'macd_signal' in df.columns:
            features['macd_signal'] = df['macd_signal']
        if 'sma_20' in df.columns:
            features['price_sma20_ratio'] = df['close'] / df['sma_20']
        if 'sma_50' in df.columns:
            features['price_sma50_ratio'] = df['close'] / df['sma_50']
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        return features.dropna()
    
    def prepare_labels(self, df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.02) -> pd.Series:
        """Prepare labels for supervised learning"""
        future_returns = df['close'].pct_change(lookahead).shift(-lookahead)
        
        labels = pd.Series(index=df.index, dtype=int)
        labels[future_returns > threshold] = 2  # BUY
        labels[future_returns < -threshold] = 0  # SELL
        labels[(future_returns >= -threshold) & (future_returns <= threshold)] = 1  # HOLD
        
        return labels.dropna()
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train Random Forest model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
    
    def train_lstm(self, X: pd.DataFrame, y: pd.Series, sequence_length: int = 20) -> Dict:
        """Train LSTM model for time series prediction"""
        # Prepare sequences
        def create_sequences(data, labels, seq_length):
            sequences = []
            seq_labels = []
            
            for i in range(len(data) - seq_length):
                sequences.append(data[i:i + seq_length])
                seq_labels.append(labels.iloc[i + seq_length])
            
            return np.array(sequences), np.array(seq_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)
        
        # Convert labels to categorical
        y_cat = tf.keras.utils.to_categorical(y_seq, num_classes=3)
        
        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_cat[:split_idx], y_cat[split_idx:]
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'history': history.history
        }
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the AI model"""
        # Prepare features and labels
        X = self.prepare_features(df)
        y = self.prepare_labels(df)
        
        # Align features and labels
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) < 100:
            raise ValueError("Insufficient data for training. Need at least 100 samples.")
        
        self.logger.info(f"Training {self.model_type} model with {len(X)} samples")
        
        if self.model_type == 'random_forest':
            return self.train_random_forest(X, y)
        elif self.model_type == 'lstm':
            return self.train_lstm(X, y)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Make prediction on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(df)
        
        if len(X) == 0:
            return 1, 0.0  # HOLD with no confidence
        
        # Get the latest features
        latest_features = X.iloc[-1:].values
        
        # Scale features
        latest_features_scaled = self.scaler.transform(latest_features)
        
        if self.model_type == 'random_forest':
            # Get prediction probabilities
            probabilities = self.model.predict_proba(latest_features_scaled)[0]
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
        elif self.model_type == 'lstm':
            # For LSTM, we need sequence data
            if len(X) < 20:  # sequence_length
                return 1, 0.0
            
            # Prepare sequence
            X_scaled = self.scaler.transform(X)
            sequence = X_scaled[-20:].reshape(1, 20, -1)
            
            # Predict
            probabilities = self.model.predict(sequence)[0]
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Model loaded from {filepath}")


class EnsembleModel:
    def __init__(self):
        self.models = []
        self.weights = []
    
    def add_model(self, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Make ensemble prediction"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        confidences = []
        
        for model in self.models:
            pred, conf = model.predict(df)
            predictions.append(pred)
            confidences.append(conf)
        
        # Weighted voting
        weighted_votes = np.zeros(3)  # 3 classes: SELL, HOLD, BUY
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            weighted_votes[pred] += conf * self.weights[i]
        
        final_prediction = np.argmax(weighted_votes)
        final_confidence = weighted_votes[final_prediction] / sum(self.weights)
        
        return final_prediction, final_confidence


if __name__ == "__main__":
    # Example usage
    from trading_bot import AITradingBot
    
    # Create bot and get data
    bot = AITradingBot({'initial_balance': 10000})
    df = bot.get_market_data('BTC/USDT', limit=1000)
    df = bot.calculate_technical_indicators(df)
    
    # Train AI model
    ai_model = TradingAIModel(model_type='random_forest')
    training_results = ai_model.train(df)
    
    print(f"Model trained with accuracy: {training_results['accuracy']:.3f}")
    
    # Make prediction
    prediction, confidence = ai_model.predict(df)
    actions = ['SELL', 'HOLD', 'BUY']
    print(f"Prediction: {actions[prediction]} with confidence: {confidence:.3f}")
    
    # Save model
    ai_model.save_model('models/trading_model.joblib')
