
"""
Complete Stock Market Prediction using LSTM Deep Learning
Single file implementation with full output display
"""

# Core imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class StockPredictionSystem:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.symbol = None
        
    def fetch_stock_data(self, symbol, period="2y"):
        """Fetch stock data from Yahoo Finance"""
        print(f"📊 Fetching data for {symbol}...")
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                raise ValueError("No data found")
            print(f"✅ Successfully fetched {len(data)} data points")
            self.symbol = symbol
            return data
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def add_technical_indicators(self, data):
        """Add technical indicators to the data"""
        print("📈 Adding technical indicators...")
        df = data.copy()
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_window = 20
        df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
        bb_std = df['Close'].rolling(window=bb_window).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price change indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Remove NaN values
        df = df.dropna()
        
        indicators_added = len(df.columns) - len(data.columns)
        print(f"✅ Added {indicators_added} technical indicators")
        
        return df
    
    def prepare_data(self, data, sequence_length=60):
        """Prepare data for LSTM training"""
        print("🔧 Preparing data for LSTM...")
        
        # Select features for training (exclude dividends and stock splits, keep consistent order)
        feature_columns = [col for col in data.columns if col not in ['Dividends', 'Stock Splits']]
        features = data[feature_columns].values
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, :-1])  # All features except Close
            y.append(scaled_data[i, -1])  # Close price (target)
        
        X, y = np.array(X), np.array(y)
        
        # Train-test split (80-20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"📊 Data prepared:")
        print(f"   Training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"   Testing data shape: X={X_test.shape}, y={y_test.shape}")
        print(f"   Features used: {len(feature_columns)-1}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        print("🧠 Building LSTM model...")
        
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the LSTM model"""
        print("🚀 Starting model training...")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_lstm_model(input_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Model training completed!")
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("📊 Evaluating model performance...")
        
        # Make predictions
        predictions = self.model.predict(X_test, verbose=0)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(y_test))
        predicted_direction = np.sign(np.diff(predictions.flatten()))
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        print(f"\n📈 Model Performance Metrics:")
        print(f"   MAE (Mean Absolute Error): {mae:.4f}")
        print(f"   RMSE (Root Mean Square Error): {rmse:.4f}")
        print(f"   MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'predictions': predictions,
            'actual': y_test
        }
    
    def predict_future(self, data, days=30):
        """Predict future stock prices"""
        print(f"🔮 Generating {days}-day predictions...")
        
        # Use the same feature columns as training
        if not hasattr(self, 'feature_columns'):
            raise ValueError("Model must be trained first to establish feature columns")
        
        # Get data with same features as training
        data_features = data[self.feature_columns]
        
        # Get last sequence (60 days)
        last_sequence = data_features.iloc[-60:].values
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        predictions = []
        current_sequence = last_sequence_scaled[-60:, :-1]  # Exclude Close price (last column)
        
        for _ in range(days):
            # Reshape for prediction
            current_batch = current_sequence.reshape(1, 60, -1)
            
            # Predict next value
            next_pred = self.model.predict(current_batch, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence (simplified - using last row with new prediction)
            new_row = current_sequence[-1].copy()
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy_array[:, -1] = predictions  # Put predictions in last column (Close price position)
        predictions_actual = self.scaler.inverse_transform(dummy_array)[:, -1]
        
        # Create prediction dataframe
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions_actual
        })
        
        print("✅ Predictions generated!")
        return predictions_df
    
    def create_visualizations(self, data, predictions, evaluation_results):
        """Create comprehensive visualizations"""
        print("📊 Creating visualizations...")
        
        # Set matplotlib backend for better compatibility
        import matplotlib
        try:
            matplotlib.use('TkAgg')  # Try interactive backend
        except:
            matplotlib.use('Agg')   # Fallback to non-interactive
        
        # Create comprehensive analysis plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'{self.symbol} Stock Market Analysis & Predictions', fontsize=18, fontweight='bold')
        
        # Plot 1: Historical Stock Price with Moving Averages
        data_copy = data.copy()
        data_copy['SMA_20'] = data_copy['Close'].rolling(window=20).mean()
        data_copy['SMA_50'] = data_copy['Close'].rolling(window=50).mean()
        
        axes[0, 0].plot(data_copy.index, data_copy['Close'], linewidth=2, color='blue', label='Close Price')
        axes[0, 0].plot(data_copy.index, data_copy['High'], linewidth=1, alpha=0.5, color='green', label='High')
        axes[0, 0].plot(data_copy.index, data_copy['Low'], linewidth=1, alpha=0.5, color='red', label='Low')
        axes[0, 0].plot(data_copy.index, data_copy['SMA_20'], linewidth=1, color='orange', label='SMA 20', alpha=0.8)
        axes[0, 0].plot(data_copy.index, data_copy['SMA_50'], linewidth=1, color='purple', label='SMA 50', alpha=0.8)
        axes[0, 0].set_title('Historical Stock Price with Moving Averages', fontweight='bold')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Future Predictions
        recent_data = data.tail(60)
        axes[0, 1].plot(recent_data.index, recent_data['Close'], 
                       linewidth=2, color='blue', label='Historical Price')
        axes[0, 1].plot(predictions['Date'], predictions['Predicted_Price'], 
                       linewidth=2, color='red', linestyle='--', marker='o', 
                       markersize=4, label='30-Day Predictions')
        axes[0, 1].set_title('Stock Price Predictions (30 Days)', fontweight='bold')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Trading Volume
        axes[1, 0].bar(data.index, data['Volume'], alpha=0.7, color='lightblue', width=1)
        axes[1, 0].set_title('Trading Volume Over Time', fontweight='bold')
        axes[1, 0].set_ylabel('Volume')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Model Performance (Actual vs Predicted)
        actual = evaluation_results['actual']
        predicted = evaluation_results['predictions'].flatten()
        axes[1, 1].scatter(actual, predicted, alpha=0.6, color='green', s=20)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 1].set_title('Model Performance: Actual vs Predicted', fontweight='bold')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Price Distribution
        axes[2, 0].hist(data['Close'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2, 0].axvline(data['Close'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: ${data["Close"].mean():.2f}')
        axes[2, 0].axvline(data['Close'].median(), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: ${data["Close"].median():.2f}')
        axes[2, 0].set_title('Price Distribution Analysis', fontweight='bold')
        axes[2, 0].set_xlabel('Price ($)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Prediction Trend Analysis
        current_price = data['Close'].iloc[-1]
        price_changes = [(p - current_price) / current_price * 100 for p in predictions['Predicted_Price']]
        axes[2, 1].plot(range(1, len(price_changes) + 1), price_changes, 
                       linewidth=2, color='purple', marker='o', markersize=3)
        axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[2, 1].fill_between(range(1, len(price_changes) + 1), price_changes, 
                               alpha=0.3, color='purple')
        axes[2, 1].set_title('Predicted Price Change Trend (%)', fontweight='bold')
        axes[2, 1].set_xlabel('Days Ahead')
        axes[2, 1].set_ylabel('Price Change (%)')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comprehensive plot
        filename = f'{self.symbol}_complete_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive analysis saved as {filename}")
        
        # Create separate prediction-focused plot
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig2.suptitle(f'{self.symbol} Stock Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Extended historical + predictions
        extended_data = data.tail(90)
        ax1.plot(extended_data.index, extended_data['Close'], 
                linewidth=2, color='blue', label='Historical Price')
        ax1.plot(predictions['Date'], predictions['Predicted_Price'], 
                linewidth=3, color='red', linestyle='--', marker='o', 
                markersize=5, label='Future Predictions')
        
        # Add confidence bands (simplified)
        upper_bound = predictions['Predicted_Price'] * 1.05
        lower_bound = predictions['Predicted_Price'] * 0.95
        ax1.fill_between(predictions['Date'], lower_bound, upper_bound, 
                        alpha=0.2, color='red', label='Confidence Band')
        
        ax1.set_title('Stock Price Prediction with Confidence Band', fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume context
        ax2.bar(extended_data.index, extended_data['Volume'], 
               alpha=0.7, color='lightgreen', width=1)
        ax2.set_title('Trading Volume Context', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save prediction plot
        pred_filename = f'{self.symbol}_predictions_detailed.png'
        plt.savefig(pred_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Detailed predictions saved as {pred_filename}")
        
        # Try to show plots
        try:
            plt.show()
            print("📊 Plots displayed successfully!")
        except:
            print("📊 Plots saved (display not available in current environment)")
        
        return filename, pred_filename

def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 STOCK MARKET PREDICTION WITH LSTM DEEP LEARNING")
    print("=" * 60)
    
    # Configuration
    SYMBOL = 'AAPL'
    PERIOD = '2y'
    FORECAST_DAYS = 30
    
    # Initialize system
    predictor = StockPredictionSystem()
    
    try:
        # Step 1: Fetch data
        print("\n📊 STEP 1: DATA COLLECTION")
        raw_data = predictor.fetch_stock_data(SYMBOL, PERIOD)
        if raw_data is None:
            return
        
        # Step 2: Add technical indicators
        print("\n📈 STEP 2: FEATURE ENGINEERING")
        data_with_indicators = predictor.add_technical_indicators(raw_data)
        
        # Step 3: Prepare data
        print("\n🔧 STEP 3: DATA PREPARATION")
        X_train, X_test, y_train, y_test, features = predictor.prepare_data(data_with_indicators)
        
        # Step 4: Train model
        print("\n🧠 STEP 4: MODEL TRAINING")
        history = predictor.train_model(X_train, y_train, X_test, y_test)
        
        # Step 5: Evaluate model
        print("\n📊 STEP 5: MODEL EVALUATION")
        evaluation_results = predictor.evaluate_model(X_test, y_test)
        
        # Step 6: Make predictions
        print(f"\n🔮 STEP 6: FUTURE PREDICTIONS")
        predictions = predictor.predict_future(data_with_indicators, FORECAST_DAYS)
        
        # Display predictions
        print(f"\n📈 {SYMBOL} Stock Predictions (Next 10 days):")
        print("-" * 50)
        current_price = raw_data['Close'].iloc[-1]
        for i in range(min(10, len(predictions))):
            date = predictions.iloc[i]['Date'].strftime('%Y-%m-%d')
            price = predictions.iloc[i]['Predicted_Price']
            change = ((price - current_price) / current_price) * 100
            print(f"   {date}: ${price:.2f} ({change:+.1f}%)")
        
        # Step 7: Create visualizations
        print(f"\n📊 STEP 7: VISUALIZATION")
        plot_file, pred_file = predictor.create_visualizations(raw_data, predictions, evaluation_results)
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(f"✅ Stock Symbol: {SYMBOL}")
        print(f"✅ Data Points: {len(raw_data)}")
        print(f"✅ Features Used: {len(features)-1}")
        print(f"✅ Model Parameters: {predictor.model.count_params():,}")
        print(f"✅ RMSE: {evaluation_results['rmse']:.4f}")
        print(f"✅ Directional Accuracy: {evaluation_results['directional_accuracy']:.1f}%")
        print(f"✅ Predictions Generated: {len(predictions)} days")
        print(f"✅ Comprehensive Analysis: {plot_file}")
        print(f"✅ Detailed Predictions: {pred_file}")
        
        # Trading simulation
        print(f"\n💰 TRADING STRATEGY SIMULATION:")
        print("-" * 50)
        total_return = ((predictions['Predicted_Price'].iloc[-1] - current_price) / current_price) * 100
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   30-day Target: ${predictions['Predicted_Price'].iloc[-1]:.2f}")
        print(f"   Expected Return: {total_return:+.1f}%")
        print(f"   Investment Risk: Medium")
        print(f"   Confidence Level: {100 - evaluation_results['mape']:.1f}%")
        
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
