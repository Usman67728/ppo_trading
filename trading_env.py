"""
Trading Environment for Gold Metals PPO Agent
Handles state representation, action space, and reward calculation
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from gymnasium import spaces


class GoldTradingEnv(gym.Env):
    """
    Custom trading environment for Gold Metals trading with PPO
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0, 
                 transaction_cost: float = 0.001, lookback_window: int = 20,
                 lot_size: float = 0.35, commission_per_lot: float = 5.0,
                 spread_pips: float = 2.0, slippage_pips: float = 1.0):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.lot_size = lot_size
        self.commission_per_lot = commission_per_lot
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        
        # Trading state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.position_size = 0.0
        self.entry_price = 0.0
        
        # Action space: [0: hold, 1: buy, 2: sell, 3: close_position]
        self.action_space = spaces.Discrete(4)
        
        # State space: normalized OHLCV + technical indicators + position info
        # Features: OHLC, volume, RSI, MACD, moving averages, position info
        n_features = 12  # OHLC + volume + RSI + MACD + MA5 + MA20 + position + balance_ratio + unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(lookback_window, n_features), dtype=np.float32
        )
        
        # Calculate technical indicators
        self.add_technical_indicators()
        
    def add_technical_indicators(self):
        """Add technical indicators to the dataset"""
        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['close'].ewm(span=12).mean()
        exp2 = self.data['close'].ewm(span=26).mean()
        self.data['macd'] = exp1 - exp2
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
        
        # Moving averages
        self.data['ma5'] = self.data['close'].rolling(window=5).mean()
        self.data['ma20'] = self.data['close'].rolling(window=20).mean()
        
        # Fill NaN values
        self.data = self.data.ffill().bfill()
        
        # Replace any remaining NaN or infinite values
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.fillna(0)
        
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step < self.lookback_window:
            # Pad with zeros if not enough history
            padding = np.zeros((self.lookback_window - self.current_step - 1, 12))
            current_data = self.data.iloc[:self.current_step + 1]
        else:
            current_data = self.data.iloc[self.current_step - self.lookback_window + 1:self.current_step + 1]
        
        # Normalize price data
        price_data = current_data[['open', 'high', 'low', 'close']].values
        price_mean = price_data.mean()
        price_std = price_data.std() + 1e-8
        price_normalized = (price_data - price_mean) / price_std
        
        # Normalize volume
        volume_data = current_data['volume'].values
        volume_mean = volume_data.mean()
        volume_std = volume_data.std() + 1e-8
        volume_normalized = (volume_data - volume_mean) / volume_std
        
        # Technical indicators
        rsi_data = current_data['rsi'].values
        rsi_normalized = (rsi_data - 50) / 50  # Normalize RSI to [-1, 1]
        
        macd_data = current_data['macd'].values
        macd_std = macd_data.std() + 1e-8
        macd_normalized = macd_data / macd_std
        
        # Moving averages normalized
        ma5_data = current_data['ma5'].values
        ma20_data = current_data['ma20'].values
        close_data = current_data['close'].values
        
        ma5_normalized = (ma5_data - close_data) / (close_data + 1e-8)
        ma20_normalized = (ma20_data - close_data) / (close_data + 1e-8)
        
        # Position information
        position_info = np.full(len(current_data), self.position)
        balance_ratio = np.full(len(current_data), self.balance / self.initial_balance)
        
        # Unrealized P&L
        if self.position != 0:
            current_price = self.data.iloc[self.current_step]['close']
            unrealized_pnl = (current_price - self.entry_price) * self.position * self.position_size
        else:
            unrealized_pnl = 0
        unrealized_pnl_ratio = unrealized_pnl / self.initial_balance
        unrealized_pnl_array = np.full(len(current_data), unrealized_pnl_ratio)
        
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
            unrealized_pnl_array.reshape(-1, 1)
        ])
        
        if self.current_step < self.lookback_window:
            state = np.vstack([padding, state])
        
        # Check for NaN or infinite values and replace them
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip values to prevent extreme values
        state = np.clip(state, -10.0, 10.0)
        
        return state.astype(np.float32)
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on action and market performance"""
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0.0
        
        # Position-based rewards
        if self.position != 0:
            # Unrealized P&L reward
            unrealized_pnl = (current_price - self.entry_price) * self.position * self.position_size
            reward += unrealized_pnl / self.initial_balance * 100  # Scale reward
            
            # Risk management penalty for holding too long
            if self.current_step > 0:
                reward -= 0.01  # Small penalty for holding positions
        
        # Action-specific rewards
        if action == 1 and self.position == 0:  # Buy when no position
            # Reward for good entry timing (simplified)
            if self.current_step > 0:
                price_change = (current_price - self.data.iloc[self.current_step - 1]['close']) / self.data.iloc[self.current_step - 1]['close']
                reward += price_change * 10  # Reward for buying on uptrend
                
        elif action == 2 and self.position == 0:  # Sell when no position
            if self.current_step > 0:
                price_change = (current_price - self.data.iloc[self.current_step - 1]['close']) / self.data.iloc[self.current_step - 1]['close']
                reward += -price_change * 10  # Reward for selling on downtrend
                
        elif action == 3 and self.position != 0:  # Close position
            if self.position != 0:
                realized_pnl = (current_price - self.entry_price) * self.position * self.position_size
                reward += realized_pnl / self.initial_balance * 100
                
        # Penalty for invalid actions
        if (action in [1, 2] and self.position != 0) or (action == 3 and self.position == 0):
            reward -= 1.0
            
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.position_size = self.lot_size
            self.entry_price = current_price
            
            # Calculate costs: commission + spread + slippage
            commission_cost = self.commission_per_lot * self.position_size
            spread_cost = (self.spread_pips * 0.0001) * current_price * self.position_size * 100
            slippage_cost = (self.slippage_pips * 0.0001) * current_price * self.position_size * 100
            
            total_cost = commission_cost + spread_cost + slippage_cost
            self.balance -= total_cost
            
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.position_size = self.lot_size
            self.entry_price = current_price
            
            # Calculate costs: commission + spread + slippage
            commission_cost = self.commission_per_lot * self.position_size
            spread_cost = (self.spread_pips * 0.0001) * current_price * self.position_size * 100
            slippage_cost = (self.slippage_pips * 0.0001) * current_price * self.position_size * 100
            
            total_cost = commission_cost + spread_cost + slippage_cost
            self.balance -= total_cost
            
        elif action == 3 and self.position != 0:  # Close position
            if self.position == 1:  # Close long
                pnl = (current_price - self.entry_price) * self.position_size * 100
                commission_cost = self.commission_per_lot * self.position_size
                spread_cost = (self.spread_pips * 0.0001) * current_price * self.position_size * 100
                slippage_cost = (self.slippage_pips * 0.0001) * current_price * self.position_size * 100
                
                total_cost = commission_cost + spread_cost + slippage_cost
                self.balance += pnl - total_cost
            else:  # Close short
                pnl = (self.entry_price - current_price) * self.position_size * 100
                commission_cost = self.commission_per_lot * self.position_size
                spread_cost = (self.spread_pips * 0.0001) * current_price * self.position_size * 100
                slippage_cost = (self.slippage_pips * 0.0001) * current_price * self.position_size * 100
                
                total_cost = commission_cost + spread_cost + slippage_cost
                self.balance += pnl - total_cost
            
            self.position = 0
            self.position_size = 0
            self.entry_price = 0
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Get next state
        if not done:
            next_state = self._get_state()
        else:
            next_state = self._get_state()  # Final state
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'position_size': self.position_size,
            'current_price': current_price
        }
        
        return next_state, reward, done, truncated, info
    
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        
        initial_state = self._get_state()
        info = {
            'balance': self.balance,
            'position': self.position,
            'position_size': self.position_size
        }
        
        return initial_state, info
    
    def render(self, mode='human'):
        """Render current state"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
                  f"Position: {self.position}, Price: {self.data.iloc[self.current_step]['close']:.2f}")
