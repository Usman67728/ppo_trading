"""
Data Preprocessing for Gold Metals Trading
Handles data loading, cleaning, and feature engineering
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os


class GoldDataPreprocessor:
    """
    Preprocesses Gold Metals trading data for PPO training
    """
    
    def __init__(self, data_path: str = "Gold_Metals_M1.csv"):
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate the dataset"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} not found")
        
        print(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path, parse_dates=['time'])
        
        # Basic validation
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Loaded {len(self.data)} rows of data")
        print(f"Date range: {self.data['time'].min()} to {self.data['time'].max()}")
        
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Cleaning data...")
        original_length = len(self.data)
        
        # Remove duplicates
        self.data = self.data.drop_duplicates(subset=['time'])
        
        # Sort by time
        self.data = self.data.sort_values('time').reset_index(drop=True)
        
        # Handle missing values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with invalid prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            self.data = self.data[self.data[col] > 0]
        
        # Ensure high >= low
        self.data = self.data[self.data['high'] >= self.data['low']]
        
        # Ensure high >= open, close and low <= open, close
        self.data = self.data[
            (self.data['high'] >= self.data['open']) &
            (self.data['high'] >= self.data['close']) &
            (self.data['low'] <= self.data['open']) &
            (self.data['low'] <= self.data['close'])
        ]
        
        cleaned_length = len(self.data)
        print(f"Cleaned data: {original_length} -> {cleaned_length} rows")
        
        return self.data
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """Add basic technical indicators to the dataset"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Adding basic technical indicators...")
        
        # Only add basic price features
        self.data['price_change'] = self.data['close'].pct_change()
        self.data['price_range'] = (self.data['high'] - self.data['low']) / self.data['close']
        
        # Simple moving averages
        self.data['ma_5'] = self.data['close'].rolling(window=5).mean()
        self.data['ma_20'] = self.data['close'].rolling(window=20).mean()
        
        # Simple RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility indicators
        self.data['volatility'] = self.data['close'].rolling(window=20).std()
        self.data['atr'] = self._calculate_atr()
        
        # Support and resistance levels
        self.data['resistance'] = self.data['high'].rolling(window=20).max()
        self.data['support'] = self.data['low'].rolling(window=20).min()
        self.data['resistance_distance'] = (self.data['resistance'] - self.data['close']) / self.data['close']
        self.data['support_distance'] = (self.data['close'] - self.data['support']) / self.data['close']
        
        # Time-based features
        self.data['hour'] = self.data['time'].dt.hour
        self.data['day_of_week'] = self.data['time'].dt.dayofweek
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
        
        # Market session indicators
        self.data['is_london_session'] = ((self.data['hour'] >= 8) & (self.data['hour'] < 16)).astype(int)
        self.data['is_ny_session'] = ((self.data['hour'] >= 13) & (self.data['hour'] < 21)).astype(int)
        self.data['is_asian_session'] = ((self.data['hour'] >= 0) & (self.data['hour'] < 8)).astype(int)
        
        # Fill NaN values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        print("Technical indicators added successfully")
        return self.data
    
    def _calculate_atr(self, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def add_metals_correlations(self) -> pd.DataFrame:
        """Add correlation features with other metals"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Adding metals correlation features...")
        
        # Calculate correlations with other metals
        metals = ['XAGUSD', 'XPTUSD', 'XPDUSD']
        
        for metal in metals:
            if f'{metal}_close' in self.data.columns:
                # Price correlation
                self.data[f'{metal}_correlation'] = self.data['close'].rolling(window=20).corr(self.data[f'{metal}_close'])
                
                # Price ratio
                self.data[f'{metal}_ratio'] = self.data['close'] / self.data[f'{metal}_close']
                
                # Relative strength
                self.data[f'{metal}_relative_strength'] = (
                    self.data['close'].pct_change() - self.data[f'{metal}_close'].pct_change()
                )
        
        # Fill NaN values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        print("Metals correlation features added")
        return self.data
    
    def create_training_splits(self, train_ratio: float = 0.7, 
                             val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training, validation, and test sets"""
        if self.data is None:
            raise ValueError("Data not processed. Call preprocessing methods first.")
        
        print("Creating training splits...")
        
        # Sort by time to ensure chronological order
        data_sorted = self.data.sort_values('time').reset_index(drop=True)
        
        n_total = len(data_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data_sorted[:n_train]
        val_data = data_sorted[n_train:n_train + n_val]
        test_data = data_sorted[n_train + n_val:]
        
        print(f"Training set: {len(train_data)} rows")
        print(f"Validation set: {len(val_data)} rows")
        print(f"Test set: {len(test_data)} rows")
        
        return train_data, val_data, test_data
    
    def get_feature_columns(self) -> list:
        """Get list of feature columns for training"""
        # Exclude non-feature columns
        exclude_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        feature_columns = [col for col in self.data.columns if col not in exclude_columns]
        
        print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
        return feature_columns
    
    def preprocess_full_pipeline(self) -> pd.DataFrame:
        """Run the complete preprocessing pipeline"""
        print("Starting full preprocessing pipeline...")
        
        # Load data
        self.load_data()
        
        # Clean data
        self.clean_data()
        
        # Add technical indicators
        self.add_technical_indicators()
        
        # Add metals correlations
        self.add_metals_correlations()
        
        # Store processed data
        self.processed_data = self.data.copy()
        
        print("Preprocessing pipeline completed successfully")
        return self.processed_data
    
    def save_processed_data(self, output_path: str = "processed_gold_data.csv"):
        """Save processed data to file"""
        if self.processed_data is None:
            raise ValueError("No processed data to save. Run preprocessing first.")
        
        self.processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        
        return output_path
