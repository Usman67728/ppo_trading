"""
Real-time Data Fetcher for Live Trading
Handles continuous data updates from MetaTrader 5
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any
import logging
from queue import Queue


class RealTimeDataFetcher:
    """
    Fetches real-time data from MetaTrader 5 for live trading
    """
    
    def __init__(self, symbol: str = "XAUUSD", update_interval: int = 60):
        self.symbol = symbol
        self.update_interval = update_interval
        self.is_running = False
        self.data_thread = None
        self.latest_data = None
        self.data_callbacks = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize MT5
        self._initialize_mt5()
    
    def _initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5 connection"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Ensure symbol is visible
            if not mt5.symbol_select(self.symbol, True):
                self.logger.error(f"Failed to select symbol {self.symbol}")
                return False
            
            self.logger.info(f"MT5 data fetcher initialized for {self.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            return False
    
    def add_data_callback(self, callback: Callable[[pd.DataFrame], None]):
        """Add callback function for new data"""
        self.data_callbacks.append(callback)
    
    def get_latest_data(self, lookback_minutes: int = 20) -> Optional[pd.DataFrame]:
        """Get latest market data"""
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            # Get rates
            rates = mt5.copy_rates_range(
                self.symbol,
                mt5.TIMEFRAME_M1,
                start_time,
                end_time
            )
            
            if rates is None or len(rates) == 0:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Add volume spike feature
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
                df['vol_spike'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None
    
    def get_tick_data(self) -> Optional[Dict[str, Any]]:
        """Get latest tick data"""
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None
            
            return {
                'time': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'flags': tick.flags
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching tick data: {e}")
            return None
    
    def get_symbol_info(self) -> Optional[Dict[str, Any]]:
        """Get symbol information"""
        try:
            info = mt5.symbol_info(self.symbol)
            if info is None:
                return None
            
            return {
                'symbol': info.name,
                'point': info.point,
                'digits': info.digits,
                'spread': info.spread,
                'trade_mode': info.trade_mode,
                'trade_stops_level': info.trade_stops_level,
                'trade_freeze_level': info.trade_freeze_level,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step,
                'margin_initial': info.margin_initial,
                'margin_maintenance': info.margin_maintenance
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching symbol info: {e}")
            return None
    
    def start_data_fetching(self):
        """Start continuous data fetching"""
        if self.is_running:
            return
        
        self.is_running = True
        self.data_thread = threading.Thread(target=self._data_fetching_loop)
        self.data_thread.start()
        
        self.logger.info("Data fetching started")
    
    def stop_data_fetching(self):
        """Stop continuous data fetching"""
        self.is_running = False
        
        if self.data_thread:
            self.data_thread.join()
        
        self.logger.info("Data fetching stopped")
    
    def _data_fetching_loop(self):
        """Main data fetching loop"""
        while self.is_running:
            try:
                # Get latest data
                data = self.get_latest_data()
                if data is not None:
                    self.latest_data = data
                    
                    # Notify callbacks
                    for callback in self.data_callbacks:
                        try:
                            callback(data)
                        except Exception as e:
                            self.logger.error(f"Error in data callback: {e}")
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in data fetching loop: {e}")
                time.sleep(10)
    
    def get_current_price(self) -> Optional[float]:
        """Get current market price"""
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None
            
            return (tick.bid + tick.ask) / 2  # Mid price
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            # Get current time
            now = datetime.now()
            
            # Check if it's weekend
            if now.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Check trading hours (simplified - 24/5 for forex)
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False


class DataProcessor:
    """
    Processes real-time data for the trading agent
    """
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.data_buffer = []
        self.processed_data = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def add_data(self, data: pd.DataFrame):
        """Add new data to buffer"""
        try:
            # Add to buffer
            self.data_buffer.append(data)
            
            # Keep only recent data
            if len(self.data_buffer) > 10:  # Keep last 10 updates
                self.data_buffer = self.data_buffer[-10:]
            
            # Process data
            self._process_data()
            
        except Exception as e:
            self.logger.error(f"Error adding data: {e}")
    
    def _process_data(self):
        """Process accumulated data"""
        try:
            if not self.data_buffer:
                return
            
            # Combine all data
            combined_data = pd.concat(self.data_buffer, ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=['time']).sort_values('time')
            
            # Get last lookback_window rows
            if len(combined_data) >= self.lookback_window:
                self.processed_data = combined_data.tail(self.lookback_window)
            else:
                # Pad with last available data
                last_row = combined_data.iloc[-1:].copy()
                padding = pd.concat([last_row] * (self.lookback_window - len(combined_data)), ignore_index=True)
                self.processed_data = pd.concat([padding, combined_data], ignore_index=True)
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed data for the agent"""
        return self.processed_data
    
    def get_latest_state(self) -> Optional[np.ndarray]:
        """Get latest state array for the agent"""
        try:
            if self.processed_data is None:
                return None
            
            # Create state array (same as in trading_env.py)
            price_data = self.processed_data[['open', 'high', 'low', 'close']].values
            price_normalized = (price_data - price_data.mean()) / (price_data.std() + 1e-8)
            
            volume_normalized = (self.processed_data['volume'].values - self.processed_data['volume'].mean()) / (self.processed_data['volume'].std() + 1e-8)
            
            # Technical indicators (simplified)
            rsi_normalized = np.random.uniform(-1, 1, len(self.processed_data))  # Placeholder
            macd_normalized = np.random.uniform(-1, 1, len(self.processed_data))  # Placeholder
            
            # Moving averages (simplified)
            ma5_normalized = np.random.uniform(-0.1, 0.1, len(self.processed_data))  # Placeholder
            ma20_normalized = np.random.uniform(-0.1, 0.1, len(self.processed_data))  # Placeholder
            
            # Position info
            position_info = np.zeros(len(self.processed_data))
            balance_ratio = np.ones(len(self.processed_data))
            unrealized_pnl = np.zeros(len(self.processed_data))
            
            # Combine all features
            state = np.column_stack([
                price_normalized,
                volume_normalized.reshape(-1, 1),
                rsi_normalized.reshape(-1, 1),
                macd_normalized.reshape(-1, 1),
                ma5_normalized.reshape(-1, 1),
                ma20_normalized.reshape(-1, 1),
                position_info.reshape(-1, 1),
                balance_ratio.reshape(-1, 1),
                unrealized_pnl.reshape(-1, 1)
            ])
            
            return state.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error creating state array: {e}")
            return None


def main():
    """Test the data fetcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Data Fetcher")
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    # Create data fetcher
    fetcher = RealTimeDataFetcher(args.symbol, args.interval)
    
    # Create data processor
    processor = DataProcessor()
    
    # Add callback
    def data_callback(data):
        print(f"New data received: {len(data)} rows")
        processor.add_data(data)
        
        # Get processed state
        state = processor.get_latest_state()
        if state is not None:
            print(f"State shape: {state.shape}")
    
    fetcher.add_data_callback(data_callback)
    
    try:
        # Start data fetching
        fetcher.start_data_fetching()
        
        print("Data fetching started. Press Ctrl+C to stop.")
        
        # Monitor for a while
        time.sleep(300)  # 5 minutes
        
    except KeyboardInterrupt:
        print("\nStopping data fetching...")
        fetcher.stop_data_fetching()
        print("Data fetching stopped")


if __name__ == "__main__":
    main()
