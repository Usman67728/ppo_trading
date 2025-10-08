"""
Live Trading Module for PPO Trading Agent
Handles real-time data fetching, margin trading, and live execution
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import threading
from queue import Queue
import logging

from trading_config import load_trading_config, TradingConfig
from ppo_agent import PPOTradingAgent
from data_preprocessor import GoldDataPreprocessor


class LiveTradingManager:
    """
    Manages live trading operations with real-time data and margin trading
    """
    
    def __init__(self, config_path: str = "trading_config.json"):
        self.config = load_trading_config(config_path)
        self.agent = None
        self.is_trading = False
        self.current_position = None
        self.account_info = None
        self.symbol_info = None
        self.data_queue = Queue()
        self.trading_thread = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize MT5
        self._initialize_mt5()
        
    def _initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5 connection"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Get account info
            self.account_info = mt5.account_info()
            if self.account_info is None:
                self.logger.error("Failed to get account info")
                return False
            
            # Get symbol info
            self.symbol_info = mt5.symbol_info(self.config.symbol)
            if self.symbol_info is None:
                self.logger.error(f"Symbol {self.config.symbol} not found")
                return False
            
            # Ensure symbol is visible
            if not mt5.symbol_select(self.config.symbol, True):
                self.logger.error(f"Failed to select symbol {self.config.symbol}")
                return False
            
            self.logger.info(f"MT5 initialized successfully")
            self.logger.info(f"Account: {self.account_info.login}")
            self.logger.info(f"Balance: {self.account_info.balance}")
            self.logger.info(f"Equity: {self.account_info.equity}")
            self.logger.info(f"Margin: {self.account_info.margin}")
            self.logger.info(f"Free margin: {self.account_info.margin_free}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            return False
    
    def load_trained_model(self, model_path: str) -> bool:
        """Load the trained PPO model"""
        try:
            # Create a dummy environment for the agent
            from trading_env import GoldTradingEnv
            
            # Create sample data for environment initialization
            sample_data = pd.DataFrame({
                'time': pd.date_range('2023-01-01', periods=100, freq='1min'),
                'open': np.random.uniform(1800, 2000, 100),
                'high': np.random.uniform(1800, 2000, 100),
                'low': np.random.uniform(1800, 2000, 100),
                'close': np.random.uniform(1800, 2000, 100),
                'volume': np.random.randint(10, 100, 100)
            })
            
            env = GoldTradingEnv(sample_data, initial_balance=10000.0)
            self.agent = PPOTradingAgent(env, model_path=model_path)
            
            self.logger.info(f"Trained model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_latest_data(self, lookback_minutes: int = 20) -> pd.DataFrame:
        """Fetch latest market data from MT5"""
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            # Get rates
            rates = mt5.copy_rates_range(
                self.config.symbol,
                mt5.TIMEFRAME_M1,
                start_time,
                end_time
            )
            
            if rates is None or len(rates) == 0:
                self.logger.warning("No data received from MT5")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Add volume spike feature
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
                df['vol_spike'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
            
            # Add other metals data if available
            metals = ['XAGUSD', 'XPTUSD', 'XPDUSD']
            for metal in metals:
                try:
                    metal_rates = mt5.copy_rates_range(
                        metal,
                        mt5.TIMEFRAME_M1,
                        start_time,
                        end_time
                    )
                    if metal_rates is not None and len(metal_rates) > 0:
                        metal_df = pd.DataFrame(metal_rates)
                        metal_df['time'] = pd.to_datetime(metal_df['time'], unit='s')
                        df = df.merge(
                            metal_df[['time', 'high', 'low', 'close']].rename(columns={
                                'high': f'{metal}_high',
                                'low': f'{metal}_low',
                                'close': f'{metal}_close'
                            }),
                            on='time',
                            how='left'
                        )
                except:
                    pass  # Skip if metal not available
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None
    
    def preprocess_live_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess live data for the model"""
        try:
            # Use the same preprocessing as training
            preprocessor = GoldDataPreprocessor()
            preprocessor.data = data.copy()
            preprocessor.add_technical_indicators()
            preprocessor.add_metals_correlations()
            
            # Get the latest state
            processed_data = preprocessor.data.ffill().bfill()
            
            # Create state representation (last 20 timesteps)
            lookback_window = 20
            if len(processed_data) < lookback_window:
                # Pad with the last available data
                last_row = processed_data.iloc[-1:].copy()
                padding = pd.concat([last_row] * (lookback_window - len(processed_data)), ignore_index=True)
                processed_data = pd.concat([padding, processed_data], ignore_index=True)
            
            # Get the last lookback_window rows
            recent_data = processed_data.tail(lookback_window)
            
            # Create state array
            state = self._create_state_array(recent_data)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return None
    
    def _create_state_array(self, data: pd.DataFrame) -> np.ndarray:
        """Create state array from processed data"""
        try:
            # Normalize price data
            price_data = data[['open', 'high', 'low', 'close']].values
            price_normalized = (price_data - price_data.mean()) / (price_data.std() + 1e-8)
            
            # Normalize volume
            volume_normalized = (data['volume'].values - data['volume'].mean()) / (data['volume'].std() + 1e-8)
            
            # Technical indicators
            rsi_normalized = (data['rsi'].values - 50) / 50
            macd_normalized = data['macd'].values / (data['macd'].std() + 1e-8)
            
            # Moving averages
            ma5_normalized = (data['ma5'].values - data['close'].values) / (data['close'].values + 1e-8)
            ma20_normalized = (data['ma20'].values - data['close'].values) / (data['close'].values + 1e-8)
            
            # Position info (current position, balance ratio, unrealized P&L)
            position_info = np.zeros(len(data))
            balance_ratio = np.ones(len(data))
            unrealized_pnl = np.zeros(len(data))
            
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
    
    def get_trading_signal(self, state: np.ndarray) -> Tuple[int, float]:
        """Get trading signal from the model"""
        try:
            if self.agent is None:
                return 0, 0.0  # Hold action, no confidence
            
            # Get action and confidence from model
            action, _ = self.agent.predict(state, deterministic=True)
            
            # Calculate confidence (simplified)
            confidence = 0.8  # This should be calculated from the model's output
            
            return action[0], confidence
            
        except Exception as e:
            self.logger.error(f"Error getting trading signal: {e}")
            return 0, 0.0
    
    def calculate_position_size(self, action: int, current_price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            if action == 0:  # Hold
                return 0.0
            
            # Get current account info
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0
            
            # Calculate available margin
            free_margin = account_info.margin_free
            margin_requirement = self.config.margin_trading['margin_requirement']
            
            # Calculate maximum position size based on free margin
            max_position_value = free_margin / margin_requirement
            max_lots = max_position_value / (current_price * 100)  # Assuming 100 units per lot
            
            # Apply risk management
            risk_per_trade = self.config.max_risk_per_trade
            max_risk_amount = account_info.balance * risk_per_trade
            
            # Calculate position size based on risk
            if action in [1, 2]:  # Buy or Sell
                position_size = min(
                    self.config.lot_size,
                    max_lots * 0.1,  # Use only 10% of available margin
                    max_risk_amount / (current_price * 100)
                )
                
                return max(0.01, position_size)  # Minimum 0.01 lot
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def execute_trade(self, action: int, position_size: float, current_price: float) -> bool:
        """Execute trade on MT5"""
        try:
            if action == 0:  # Hold
                return True
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.symbol,
                "volume": position_size,
                "type": mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "deviation": self.config.trading_fees['slippage_pips'],
                "magic": 12345,
                "comment": "PPO Trading Agent",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit
            if action == 1:  # Buy
                request["sl"] = current_price - (self.config.stop_loss_pips * 0.0001)
                request["tp"] = current_price + (self.config.take_profit_pips * 0.0001)
            else:  # Sell
                request["sl"] = current_price + (self.config.stop_loss_pips * 0.0001)
                request["tp"] = current_price - (self.config.take_profit_pips * 0.0001)
            
            # Send trade request
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Trade execution failed: {result.retcode} - {result.comment}")
                return False
            
            self.logger.info(f"Trade executed successfully: {action}, size: {position_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all open positions"""
        try:
            positions = mt5.positions_get(symbol=self.config.symbol)
            if positions is None:
                return True
            
            for position in positions:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "magic": 12345,
                    "comment": "Close position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(f"Failed to close position {position.ticket}")
                    return False
            
            self.logger.info("All positions closed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
            return False
    
    def start_live_trading(self, model_path: str) -> bool:
        """Start live trading"""
        try:
            # Load model
            if not self.load_trained_model(model_path):
                return False
            
            # Start trading thread
            self.is_trading = True
            self.trading_thread = threading.Thread(target=self._trading_loop)
            self.trading_thread.start()
            
            self.logger.info("Live trading started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting live trading: {e}")
            return False
    
    def stop_live_trading(self) -> bool:
        """Stop live trading"""
        try:
            self.is_trading = False
            
            if self.trading_thread:
                self.trading_thread.join()
            
            # Close all positions
            self.close_all_positions()
            
            self.logger.info("Live trading stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping live trading: {e}")
            return False
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.is_trading:
            try:
                # Get latest data
                data = self.get_latest_data()
                if data is None or len(data) < 20:
                    time.sleep(10)
                    continue
                
                # Preprocess data
                state = self.preprocess_live_data(data)
                if state is None:
                    time.sleep(10)
                    continue
                
                # Get trading signal
                action, confidence = self.get_trading_signal(state)
                
                # Check confidence threshold
                if confidence < self.config.confidence_threshold:
                    self.logger.info(f"Low confidence: {confidence:.2f}, holding position")
                    time.sleep(60)
                    continue
                
                # Get current price
                current_price = data['close'].iloc[-1]
                
                # Calculate position size
                position_size = self.calculate_position_size(action, current_price)
                
                # Execute trade if position size > 0
                if position_size > 0:
                    self.execute_trade(action, position_size, current_price)
                
                # Wait for next iteration
                time.sleep(self.config.data_fetching['update_interval_seconds'])
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def get_account_status(self) -> Dict[str, Any]:
        """Get current account status"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {}
            
            positions = mt5.positions_get(symbol=self.config.symbol)
            position_count = len(positions) if positions else 0
            
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'positions': position_count,
                'profit': account_info.profit
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account status: {e}")
            return {}


def main():
    """Main function for live trading"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Trading with PPO Agent")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--config", default="trading_config.json", help="Configuration file")
    
    args = parser.parse_args()
    
    # Create trading manager
    manager = LiveTradingManager(args.config)
    
    try:
        # Start live trading
        if manager.start_live_trading(args.model):
            print("Live trading started. Press Ctrl+C to stop.")
            
            # Monitor status
            while True:
                status = manager.get_account_status()
                print(f"Balance: {status.get('balance', 0):.2f}, "
                      f"Equity: {status.get('equity', 0):.2f}, "
                      f"Positions: {status.get('positions', 0)}")
                time.sleep(30)
                
        else:
            print("Failed to start live trading")
            
    except KeyboardInterrupt:
        print("\nStopping live trading...")
        manager.stop_live_trading()
        print("Live trading stopped")


if __name__ == "__main__":
    main()
