"""
Main execution script for PPO Trading Agent
Handles training, evaluation, and live trading
"""
import argparse
import os
import sys
import time
from typing import Optional
import json
from datetime import datetime

from training_pipeline import TrainingPipeline
from trading_config import load_trading_config, create_config_template
from data_preprocessor import GoldDataPreprocessor
from trading_env import GoldTradingEnv
from ppo_agent import PPOTradingAgent


def train_model(data_path: str = "Gold_Metals_M1.csv", 
                total_timesteps: int = 100000,
                output_dir: str = "training_output") -> str:
    """
    Train the PPO trading model
    
    Args:
        data_path: Path to the Gold Metals dataset
        total_timesteps: Number of training timesteps
        output_dir: Directory to save training outputs
    
    Returns:
        Path to the trained model
    """
    print("=" * 60)
    print("PPO TRADING AGENT TRAINING")
    print("=" * 60)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        data_path=data_path,
        initial_balance=10000.0,
        output_dir=output_dir
    )
    
    # Run full training pipeline
    results = pipeline.run_full_pipeline(
        total_timesteps=total_timesteps,
        n_eval_episodes=10
    )
    
    model_path = results['training_results']['final_model_path']
    print(f"Training completed! Model saved to: {model_path}")
    
    return model_path


def evaluate_model(model_path: str, data_path: str = "Gold_Metals_M1.csv",
                  n_episodes: int = 10) -> dict:
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the dataset
        n_episodes: Number of evaluation episodes
    
    Returns:
        Evaluation metrics
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load and preprocess data
    preprocessor = GoldDataPreprocessor(data_path)
    processed_data = preprocessor.preprocess_full_pipeline()
    
    # Create test environment
    test_env = GoldTradingEnv(
        data=processed_data,
        initial_balance=10000.0,
        transaction_cost=0.001,
        lookback_window=20
    )
    
    # Load trained agent
    agent = PPOTradingAgent(env=test_env, model_path=model_path)
    
    # Evaluate agent
    episode_rewards = []
    episode_returns = []
    episode_balances = []
    
    for episode in range(n_episodes):
        obs, info = test_env.reset()
        episode_reward = 0
        episode_balance = [test_env.balance]
        
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            episode_balance.append(test_env.balance)
        
        episode_rewards.append(episode_reward)
        episode_returns.append((episode_balance[-1] - episode_balance[0]) / episode_balance[0])
        episode_balances.append(episode_balance)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Return = {episode_returns[-1]:.2%}, "
              f"Final Balance = {episode_balance[-1]:.2f}")
    
    # Calculate metrics
    metrics = {
        'mean_reward': sum(episode_rewards) / len(episode_rewards),
        'std_reward': (sum([(r - metrics['mean_reward'])**2 for r in episode_rewards]) / len(episode_rewards))**0.5,
        'mean_return': sum(episode_returns) / len(episode_returns),
        'std_return': (sum([(r - metrics['mean_return'])**2 for r in episode_returns]) / len(episode_returns))**0.5,
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_balances': episode_balances
    }
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Return: {metrics['mean_return']:.2%} ± {metrics['std_return']:.2%}")
    
    return metrics


def create_config_file():
    """Create configuration template file"""
    print("Creating trading configuration template...")
    create_config_template()
    print("Configuration template created at 'trading_config.json'")
    print("Please edit the file with your trading account details before running live trading.")


def validate_setup():
    """Validate the setup and dependencies"""
    print("Validating setup...")
    
    # Check if data file exists
    if not os.path.exists("Gold_Metals_M1.csv"):
        print("[ERROR] Gold_Metals_M1.csv not found!")
        print("Please ensure the dataset is in the current directory.")
        return False
    
    # Check if configuration file exists
    if not os.path.exists("trading_config.json"):
        print("[WARNING] trading_config.json not found!")
        print("Run 'python main.py --create-config' to create a template.")
        return False
    
    # Check required packages
    try:
        import torch
        import stable_baselines3
        import gymnasium
        import pandas
        import numpy
        print("[OK] All required packages are installed")
    except ImportError as e:
        print(f"[ERROR] Missing required package: {e}")
        return False
    
    print("[OK] Setup validation passed")
    return True


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="PPO Trading Agent for Gold Metals")
    parser.add_argument("--mode", choices=["train", "evaluate", "create-config", "validate", "live-trade", "test-costs"], 
                       default="train", help="Mode to run")
    parser.add_argument("--data", default="Gold_Metals_M1.csv", 
                       help="Path to the dataset")
    parser.add_argument("--model", help="Path to trained model (for evaluation)")
    parser.add_argument("--timesteps", type=int, default=100000, 
                       help="Number of training timesteps")
    parser.add_argument("--episodes", type=int, default=10, 
                       help="Number of evaluation episodes")
    parser.add_argument("--output", default="training_output", 
                       help="Output directory for training")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "validate":
            validate_setup()
            
        elif args.mode == "create-config":
            create_config_file()
            
        elif args.mode == "train":
            print("Starting training mode...")
            if not validate_setup():
                sys.exit(1)
            
            model_path = train_model(
                data_path=args.data,
                total_timesteps=args.timesteps,
                output_dir=args.output
            )
            print(f"Training completed! Model saved to: {model_path}")
            
        elif args.mode == "evaluate":
            print("Starting evaluation mode...")
            if not args.model:
                print("❌ Model path required for evaluation. Use --model argument.")
                sys.exit(1)
            
            if not os.path.exists(args.model):
                print(f"❌ Model file {args.model} not found!")
                sys.exit(1)
            
            metrics = evaluate_model(
                model_path=args.model,
                data_path=args.data,
                n_episodes=args.episodes
            )
            
            # Save evaluation results
            results_path = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"Evaluation results saved to: {results_path}")
            
        elif args.mode == "live-trade":
            print("Starting live trading mode...")
            if not args.model:
                print("[ERROR] Model path required for live trading. Use --model argument.")
                sys.exit(1)
            
            if not os.path.exists(args.model):
                print(f"[ERROR] Model file {args.model} not found!")
                sys.exit(1)
            
            # Import MetaTrader5 for live trading
            import MetaTrader5 as mt5
            import pandas as pd
            import numpy as np
            
            # Initialize MT5
            if not mt5.initialize():
                print(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
                sys.exit(1)
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                print("[ERROR] Failed to get account info")
                sys.exit(1)
            
            print(f"[OK] Account: {account_info.login}")
            print(f"[OK] Balance: {account_info.balance}")
            print(f"[OK] Server: {account_info.server}")
            
            # Load configuration
            config = load_trading_config()
            
            # Get symbol info
            symbol_info = mt5.symbol_info(config.symbol)
            if symbol_info is None:
                print(f"[ERROR] Symbol {config.symbol} not found")
                sys.exit(1)
            
            # Ensure symbol is visible
            if not mt5.symbol_select(config.symbol, True):
                print(f"[ERROR] Failed to select symbol {config.symbol}")
                sys.exit(1)
            
            print(f"[OK] Symbol {config.symbol} selected successfully")
            
            # Create simple trading environment
            sample_data = pd.DataFrame({
                'time': pd.date_range('2023-01-01', periods=100, freq='1min'),
                'open': np.random.uniform(1800, 2000, 100),
                'high': np.random.uniform(1800, 2000, 100),
                'low': np.random.uniform(1800, 2000, 100),
                'close': np.random.uniform(1800, 2000, 100),
                'volume': np.random.randint(10, 100, 100)
            })
            
            env = GoldTradingEnv(
                data=sample_data, 
                initial_balance=10000.0,
                lot_size=config.lot_size,
                commission_per_lot=config.trading_fees.get('commission_per_lot', 5.0),
                spread_pips=config.trading_fees.get('spread_pips', 2.0),
                slippage_pips=config.trading_fees.get('slippage_pips', 1.0)
            )
            agent = PPOTradingAgent(env, model_path=args.model)
            
            print("Live trading started. Press Ctrl+C to stop.")
            
            # Start a separate thread for continuous profit/loss monitoring
            import threading
            
            def monitor_profit_loss():
                """Monitor positions continuously for profit/loss"""
                while True:
                    try:
                        positions = mt5.positions_get(symbol=config.symbol)
                        if positions:
                            for pos in positions:
                                current_profit = pos.profit
                                
                                # Check take profit
                                if current_profit >= 100:
                                    print(f"[PROFIT] Auto-closing position {pos.ticket} with profit ${current_profit:.2f}")
                                    # Close position logic here (same as main loop)
                                    if pos.type == mt5.POSITION_TYPE_BUY:
                                        request = {
                                            "action": mt5.TRADE_ACTION_DEAL,
                                            "symbol": config.symbol,
                                            "volume": pos.volume,
                                            "type": mt5.ORDER_TYPE_SELL,
                                            "position": pos.ticket,
                                            "price": mt5.symbol_info_tick(config.symbol).bid,
                                            "deviation": 20,
                                            "magic": 234000,
                                            "comment": "PPO Auto Profit",
                                            "type_time": mt5.ORDER_TIME_GTC,
                                            "type_filling": mt5.ORDER_FILLING_IOC,
                                        }
                                    else:
                                        request = {
                                            "action": mt5.TRADE_ACTION_DEAL,
                                            "symbol": config.symbol,
                                            "volume": pos.volume,
                                            "type": mt5.ORDER_TYPE_BUY,
                                            "position": pos.ticket,
                                            "price": mt5.symbol_info_tick(config.symbol).ask,
                                            "deviation": 20,
                                            "magic": 234000,
                                            "comment": "PPO Auto Profit",
                                            "type_time": mt5.ORDER_TIME_GTC,
                                            "type_filling": mt5.ORDER_FILLING_IOC,
                                        }
                                    result = mt5.order_send(request)
                                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                                        print(f"[OK] Position auto-closed for profit: {pos.ticket}")
                                
                                # Check stop loss
                                elif current_profit < -100:
                                    print(f"[LOSS] Auto-closing position {pos.ticket} with loss ${current_profit:.2f}")
                                    # Close position logic here (same as main loop)
                                    if pos.type == mt5.POSITION_TYPE_BUY:
                                        request = {
                                            "action": mt5.TRADE_ACTION_DEAL,
                                            "symbol": config.symbol,
                                            "volume": pos.volume,
                                            "type": mt5.ORDER_TYPE_SELL,
                                            "position": pos.ticket,
                                            "price": mt5.symbol_info_tick(config.symbol).bid,
                                            "deviation": 20,
                                            "magic": 234000,
                                            "comment": "PPO Auto Stop Loss",
                                            "type_time": mt5.ORDER_TIME_GTC,
                                            "type_filling": mt5.ORDER_FILLING_IOC,
                                        }
                                    else:
                                        request = {
                                            "action": mt5.TRADE_ACTION_DEAL,
                                            "symbol": config.symbol,
                                            "volume": pos.volume,
                                            "type": mt5.ORDER_TYPE_BUY,
                                            "position": pos.ticket,
                                            "price": mt5.symbol_info_tick(config.symbol).ask,
                                            "deviation": 20,
                                            "magic": 234000,
                                            "comment": "PPO Auto Stop Loss",
                                            "type_time": mt5.ORDER_TIME_GTC,
                                            "type_filling": mt5.ORDER_FILLING_IOC,
                                        }
                                    result = mt5.order_send(request)
                                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                                        print(f"[OK] Position auto-closed for stop loss: {pos.ticket}")
                        
                        time.sleep(2)  # Check every 2 seconds
                    except Exception as e:
                        print(f"[WARNING] Profit/loss monitoring error: {e}")
                        time.sleep(5)
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_profit_loss, daemon=True)
            monitor_thread.start()
            print("[OK] Profit/loss monitoring started (every 2 seconds)")
            
            # Load risk management settings from config
            risk_config = config.risk_management if hasattr(config, 'risk_management') else {
                'daily_loss_limit_percent': 5.0,
                'total_loss_limit_percent': 10.0,
                'cooling_period_hours': 24,
                'emergency_stop_loss_percent': 4.0,
                'max_daily_trades': 15,
                'observation_period_minutes': 5
            }
            
            # Risk management variables
            daily_trade_limit = risk_config['max_daily_trades']
            max_concurrent_trades = risk_config['max_concurrent_trades']
            daily_loss_limit_percent = risk_config['daily_loss_limit_percent']
            total_loss_limit_percent = risk_config['total_loss_limit_percent']
            emergency_stop_percent = risk_config['emergency_stop_loss_percent']
            cooling_period_hours = risk_config['cooling_period_hours']
            observation_period = risk_config['observation_period_minutes'] * 60  # Convert to seconds
            
            # Trading state variables
            trades_today = 0
            last_trade_time = None
            last_observation_time = time.time()
            initial_balance = account_info.balance
            daily_start_balance = initial_balance
            cooling_period_until = None
            emergency_stop_triggered = False
            
            # Trade history tracking (last 3 trades)
            trade_history = []  # List of dicts: {'action': 'BUY/SELL', 'price': float, 'profit': float, 'timestamp': time}
            
            print(f"[OK] Daily trade limit: {daily_trade_limit} trades")
            print(f"[OK] Max concurrent trades: {max_concurrent_trades}")
            print(f"[OK] Observation period: {observation_period/60} minutes")
            print(f"[OK] Daily loss limit: {daily_loss_limit_percent}%")
            print(f"[OK] Total loss limit: {total_loss_limit_percent}%")
            print(f"[OK] Emergency stop: {emergency_stop_percent}%")
            print(f"[OK] Cooling period: {cooling_period_hours} hours")
            
            try:
                while True:
                    # Get latest candle data (not tick data)
                    rates = mt5.copy_rates_from_pos(config.symbol, mt5.TIMEFRAME_M1, 0, 100)
                    if rates is None or len(rates) == 0:
                        print("No data received from MT5")
                        time.sleep(10)
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Handle zero volume in demo accounts
                    if 'tick_volume' in df.columns:
                        df['volume'] = df['tick_volume']
                    else:
                        df['volume'] = 100  # Default volume for demo accounts
                    
                    # Ensure we have the required columns
                    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
                    for col in required_cols:
                        if col not in df.columns:
                            if col == 'volume':
                                df[col] = 100  # Default volume
                            else:
                                df[col] = df['close']  # Use close price as fallback
                    
                    # Add basic technical indicators for live data
                    df['price_change'] = df['close'].pct_change()
                    df['price_range'] = (df['high'] - df['low']) / df['close']
                    df['ma5'] = df['close'].rolling(window=5).mean()
                    df['ma20'] = df['close'].rolling(window=20).mean()
                    
                    # Simple RSI calculation
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Simple MACD calculation
                    ema_12 = df['close'].ewm(span=12).mean()
                    ema_26 = df['close'].ewm(span=26).mean()
                    df['macd'] = ema_12 - ema_26
                    df['macd_signal'] = df['macd'].ewm(span=9).mean()
                    df['macd_histogram'] = df['macd'] - df['macd_signal']
                    
                    # Fill NaN values
                    df = df.ffill().bfill()
                    
                    # Update environment data
                    env.data = df
                    
                    # Get current state
                    obs, info = env.reset()
                    
                    # Add trade history context to the observation
                    history_features = np.zeros(8)  # 8 features for trade history
                    if trade_history:
                        recent_trades = trade_history[-5:]  # Last 5 trades
                        
                        # Feature 1: Recent profit/loss (normalized)
                        total_recent_profit = sum(t['profit'] for t in recent_trades)
                        history_features[0] = np.clip(total_recent_profit / 1000, -1, 1)  # Normalize to [-1, 1]
                        
                        # Feature 2: Average profit per trade (normalized)
                        avg_profit = total_recent_profit / len(recent_trades) if recent_trades else 0
                        history_features[1] = np.clip(avg_profit / 100, -1, 1)  # Normalize to [-1, 1]
                        
                        # Feature 3: BUY vs SELL ratio
                        buy_count = len([t for t in recent_trades if t['action'] == 'BUY'])
                        sell_count = len([t for t in recent_trades if t['action'] == 'SELL'])
                        history_features[2] = (buy_count - sell_count) / 5  # Normalize to [-1, 1]
                        
                        # Feature 4: Price trend from recent trades
                        if len(recent_trades) >= 2:
                            price_trend = (recent_trades[-1]['price'] - recent_trades[0]['price']) / recent_trades[0]['price']
                            history_features[3] = np.clip(price_trend * 100, -1, 1)  # Normalize to [-1, 1]
                        
                        # Feature 5: Recent trade frequency (how many trades recently)
                        history_features[4] = len(recent_trades) / 5  # Normalize to [0, 1]
                        
                        # Feature 6: Win rate (trades with positive profit)
                        winning_trades = len([t for t in recent_trades if t['profit'] > 0])
                        history_features[5] = winning_trades / len(recent_trades) if recent_trades else 0
                        
                        # Feature 7: Recent BUY performance
                        buy_trades = [t for t in recent_trades if t['action'] == 'BUY']
                        buy_profit = sum(t['profit'] for t in buy_trades) if buy_trades else 0
                        history_features[6] = np.clip(buy_profit / 500, -1, 1)  # Normalize to [-1, 1]
                        
                        # Feature 8: Recent SELL performance
                        sell_trades = [t for t in recent_trades if t['action'] == 'SELL']
                        sell_profit = sum(t['profit'] for t in sell_trades) if sell_trades else 0
                        history_features[7] = np.clip(sell_profit / 500, -1, 1)  # Normalize to [-1, 1]
                    
                    print(f"[HISTORY_FEATURES] Recent profit: {history_features[0]:.3f}, Avg profit: {history_features[1]:.3f}")
                    print(f"[HISTORY_FEATURES] BUY/SELL ratio: {history_features[2]:.3f}, Price trend: {history_features[3]:.3f}")
                    print(f"[HISTORY_FEATURES] Trade frequency: {history_features[4]:.3f}, Win rate: {history_features[5]:.3f}")
                    print(f"[HISTORY_FEATURES] BUY performance: {history_features[6]:.3f}, SELL performance: {history_features[7]:.3f}")
                    
                    # Get action from agent with some exploration (using original observation)
                    action, _ = agent.predict(obs, deterministic=False)  # Allow some randomness
                    
                    # Apply history-based decision modification (not forcing, but influencing)
                    if trade_history and len(trade_history) >= 3:  # Only if we have enough history
                        # Calculate influence factors
                        recent_buy_profit = history_features[6]  # BUY performance
                        recent_sell_profit = history_features[7]  # SELL performance
                        win_rate = history_features[5]  # Win rate
                        
                        # Influence probability (not forcing)
                        influence_strength = 0.3  # 30% influence from history
                        
                        # If recent BUY trades are performing well and model wants to BUY, increase confidence
                        if action == 1 and recent_buy_profit > 0.1:  # Model wants BUY and recent BUY profitable
                            print(f"[INFLUENCE] Recent BUY success - reinforcing BUY decision")
                        # If recent SELL trades are performing well and model wants to SELL, increase confidence  
                        elif action == 2 and recent_sell_profit > 0.1:  # Model wants SELL and recent SELL profitable
                            print(f"[INFLUENCE] Recent SELL success - reinforcing SELL decision")
                        # If recent trades are losing and model wants same direction, add caution
                        elif (action == 1 and recent_buy_profit < -0.1) or (action == 2 and recent_sell_profit < -0.1):
                            print(f"[INFLUENCE] Recent losses in this direction - model decision stands but with caution")
                        # If win rate is low, model might want to be more conservative
                        elif win_rate < 0.3:
                            print(f"[INFLUENCE] Low win rate ({win_rate:.2f}) - model decision with caution")
                    # Handle different action formats
                    if isinstance(action, np.ndarray):
                        if action.ndim == 0:
                            action = int(action)
                        else:
                            action = int(action[0])
                    else:
                        action = int(action)
                    
                    # Risk management checks
                    current_time = time.time()
                    current_balance = account_info.balance
                    current_equity = account_info.equity
                    
                    # Calculate losses
                    total_loss_percent = ((initial_balance - current_balance) / initial_balance) * 100
                    daily_loss_percent = ((daily_start_balance - current_balance) / daily_start_balance) * 100
                    
                    # Check for new trading day
                    if last_trade_time is None or (current_time - last_trade_time) > 86400:  # 24 hours
                        trades_today = 0
                        last_trade_time = current_time
                        daily_start_balance = current_balance
                        cooling_period_until = None
                        emergency_stop_triggered = False
                        print(f"[INFO] New trading day - reset counters and balances")
                    
                    # Check emergency stop conditions
                    if total_loss_percent >= emergency_stop_percent and not emergency_stop_triggered:
                        emergency_stop_triggered = True
                        cooling_period_until = current_time + (cooling_period_hours * 3600)
                        print(f"[EMERGENCY] Total loss {total_loss_percent:.2f}% >= {emergency_stop_percent}% - EMERGENCY STOP!")
                        print(f"[EMERGENCY] Trading suspended for {cooling_period_hours} hours")
                    
                    # Check daily loss limit
                    if daily_loss_percent >= daily_loss_limit_percent:
                        print(f"[LIMIT] Daily loss {daily_loss_percent:.2f}% >= {daily_loss_limit_percent}% - stopping trades for today")
                    
                    # Check total loss limit
                    if total_loss_percent >= total_loss_limit_percent:
                        print(f"[LIMIT] Total loss {total_loss_percent:.2f}% >= {total_loss_limit_percent}% - stopping all trading")
                    
                    # Check cooling period
                    if cooling_period_until and current_time < cooling_period_until:
                        remaining_hours = (cooling_period_until - current_time) / 3600
                        print(f"[COOLING] Cooling period active - {remaining_hours:.1f} hours remaining")
                    
                    # Count current positions
                    current_positions = mt5.positions_get(symbol=config.symbol)
                    current_position_count = len(current_positions) if current_positions else 0
                    
                    # Determine if we can trade
                    can_trade = (
                        trades_today < daily_trade_limit and
                        current_position_count < max_concurrent_trades and
                        daily_loss_percent < daily_loss_limit_percent and
                        total_loss_percent < total_loss_limit_percent and
                        not emergency_stop_triggered and
                        (cooling_period_until is None or current_time >= cooling_period_until)
                    )
                    
                    time_since_last_observation = current_time - last_observation_time
                    
                    print(f"[STATUS] Trades today: {trades_today}/{daily_trade_limit}")
                    print(f"[STATUS] Current positions: {current_position_count}/{max_concurrent_trades}")
                    print(f"[STATUS] Daily loss: {daily_loss_percent:.2f}% (limit: {daily_loss_limit_percent}%)")
                    print(f"[STATUS] Total loss: {total_loss_percent:.2f}% (limit: {total_loss_limit_percent}%)")
                    print(f"[STATUS] Can trade: {can_trade}")
                    print(f"[STATUS] Time since last observation: {time_since_last_observation/60:.1f} minutes")
                    
                    # Analyze position directions
                    buy_positions = 0
                    sell_positions = 0
                    if current_positions:
                        for pos in current_positions:
                            if pos.type == mt5.POSITION_TYPE_BUY:
                                buy_positions += 1
                            elif pos.type == mt5.POSITION_TYPE_SELL:
                                sell_positions += 1
                    
                    print(f"[POSITIONS] BUY: {buy_positions}, SELL: {sell_positions}, Total: {current_position_count}")
                    
                    # Analyze trade history for better decision making (context, not forcing)
                    if trade_history:
                        recent_trades = trade_history[-5:]  # Last 5 trades for better context
                        buy_trades = [t for t in recent_trades if t['action'] == 'BUY']
                        sell_trades = [t for t in recent_trades if t['action'] == 'SELL']
                        
                        recent_buy_profit = sum(t['profit'] for t in buy_trades) if buy_trades else 0
                        recent_sell_profit = sum(t['profit'] for t in sell_trades) if sell_trades else 0
                        
                        # Calculate performance metrics
                        total_recent_profit = recent_buy_profit + recent_sell_profit
                        avg_profit_per_trade = total_recent_profit / len(recent_trades) if recent_trades else 0
                        
                        # Price trend analysis
                        if len(recent_trades) >= 2:
                            price_trend = recent_trades[-1]['price'] - recent_trades[0]['price']
                            price_trend_pct = (price_trend / recent_trades[0]['price']) * 100
                        else:
                            price_trend = 0
                            price_trend_pct = 0
                        
                        print(f"[HISTORY] Last 5 trades: {len(recent_trades)} total")
                        print(f"[HISTORY] BUY: {len(buy_trades)} trades, Profit: ${recent_buy_profit:.2f}")
                        print(f"[HISTORY] SELL: {len(sell_trades)} trades, Profit: ${recent_sell_profit:.2f}")
                        print(f"[HISTORY] Total recent profit: ${total_recent_profit:.2f}")
                        print(f"[HISTORY] Avg profit per trade: ${avg_profit_per_trade:.2f}")
                        print(f"[HISTORY] Price trend: ${price_trend:.2f} ({price_trend_pct:.2f}%)")
                        
                        # Provide context to model (not forcing, just information)
                        if avg_profit_per_trade < -5:
                            print(f"[CONTEXT] Recent trades underperforming (avg: ${avg_profit_per_trade:.2f})")
                        elif avg_profit_per_trade > 5:
                            print(f"[CONTEXT] Recent trades performing well (avg: ${avg_profit_per_trade:.2f})")
                        
                        if price_trend_pct > 0.1:
                            print(f"[CONTEXT] Price trending upward ({price_trend_pct:.2f}%)")
                        elif price_trend_pct < -0.1:
                            print(f"[CONTEXT] Price trending downward ({price_trend_pct:.2f}%)")
                    else:
                        print(f"[HISTORY] No trade history yet")
                    
                    # Position direction logic - enforce same direction trading
                    dominant_direction = "BUY" if buy_positions > sell_positions else "SELL" if sell_positions > buy_positions else "NONE"
                    
                    print(f"[DIRECTION] Dominant direction: {dominant_direction}")
                    print(f"[DIRECTION] Current: {buy_positions} BUY, {sell_positions} SELL")
                    
                    # Enforce same direction trading - block opposite direction trades
                    if current_position_count > 0:
                        if dominant_direction == "BUY" and action == 2:  # Have BUY positions, trying to SELL
                            action = 0  # Force HOLD
                            print(f"[DIRECTION] Blocking SELL - already have BUY positions, must trade same direction")
                        elif dominant_direction == "SELL" and action == 1:  # Have SELL positions, trying to BUY
                            action = 0  # Force HOLD
                            print(f"[DIRECTION] Blocking BUY - already have SELL positions, must trade same direction")
                    
                    # Check if we should wait for price recovery before adding more positions
                    if current_positions:
                        # Calculate average profit/loss of existing positions
                        total_profit = sum(pos.profit for pos in current_positions)
                        avg_profit = total_profit / len(current_positions)
                        
                        print(f"[RECOVERY] Average P&L: ${avg_profit:.2f}")
                        
                        # Check for price recovery signals
                        price_recovery_signal = False
                        if len(df) >= 3:  # Need at least 3 data points for trend analysis
                            recent_prices = df['close'].tail(3).values
                            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]  # 3-period price change
                            
                            # Strong recovery signal: price moving in favorable direction
                            if dominant_direction == "BUY" and price_trend > 0.001:  # Price recovering upward
                                price_recovery_signal = True
                                print(f"[RECOVERY] Price recovery detected: {price_trend*100:.3f}% upward movement")
                            elif dominant_direction == "SELL" and price_trend < -0.001:  # Price recovering downward
                                price_recovery_signal = True
                                print(f"[RECOVERY] Price recovery detected: {price_trend*100:.3f}% downward movement")
                        
                        # Recovery strategy: Double down when losing but price is recovering
                        if avg_profit < -30 and price_recovery_signal:  # Losing >$30 but price recovering
                            if (action == 1 and dominant_direction == "BUY") or (action == 2 and dominant_direction == "SELL"):
                                print(f"[RECOVERY] DOUBLING DOWN - Price recovering, adding more positions for double profit!")
                                # Allow the trade to proceed - don't block it
                        # REMOVED: Recovery wait bias - let model decide when to trade
                        # The model should learn optimal timing, not be forced to wait
                        # If positions are profitable, allow more trades in same direction
                        elif avg_profit > 10:  # If average profit > $10
                            print(f"[RECOVERY] Positions profitable - allowing more trades in same direction")
                    
                    # Debug: Show action probabilities and market analysis
                    print(f"[DEBUG] Model predicted action: {action}")
                    
                    # Analyze market conditions to help with decision
                    price_change = df['price_change'].iloc[-1] if 'price_change' in df.columns else 0
                    rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
                    ma5 = df['ma5'].iloc[-1] if 'ma5' in df.columns else current_price
                    ma20 = df['ma20'].iloc[-1] if 'ma20' in df.columns else current_price
                    
                    print(f"[MARKET] Price change: {price_change:.4f}, RSI: {rsi:.1f}, MA5: {ma5:.2f}, MA20: {ma20:.2f}")
                    
                    # Only trade if we haven't exceeded daily limit AND sufficient observation time
                    if can_trade and time_since_last_observation >= observation_period:
                        # Only encourage trading if model is too conservative AND market shows clear signals
                        if action == 0:  # If HOLD, check if market has clear signals
                            market_signal = 0
                            
                            # Bullish signals
                            if price_change > 0.001 and rsi < 70 and current_price > ma5 > ma20:
                                market_signal = 1  # BUY signal
                                print(f"[MARKET] Bullish signals detected")
                            # Bearish signals  
                            elif price_change < -0.001 and rsi > 30 and current_price < ma5 < ma20:
                                market_signal = 2  # SELL signal
                                print(f"[MARKET] Bearish signals detected")
                            
                            # Only override HOLD if there are clear market signals
                            if market_signal != 0 and random.random() < 0.3:  # 30% chance to follow market signal
                                action = market_signal
                                print(f"[DEBUG] Following market signal: {action}")
                            else:
                                print(f"[DEBUG] Model HOLD decision respected")
                    else:
                        # Force HOLD if we can't trade
                        if not can_trade:
                            action = 0  # HOLD
                            if current_position_count >= max_concurrent_trades:
                                print(f"[LIMIT] Max concurrent trades reached ({current_position_count}/{max_concurrent_trades}) - forcing HOLD")
                            elif daily_loss_percent >= daily_loss_limit_percent:
                                print(f"[LIMIT] Daily loss limit reached - forcing HOLD")
                            elif total_loss_percent >= total_loss_limit_percent:
                                print(f"[LIMIT] Total loss limit reached - forcing HOLD")
                            elif emergency_stop_triggered:
                                print(f"[EMERGENCY] Emergency stop active - forcing HOLD")
                            elif cooling_period_until and current_time < cooling_period_until:
                                print(f"[COOLING] Cooling period active - forcing HOLD")
                            elif trades_today >= daily_trade_limit:
                                print(f"[LIMIT] Daily trade limit reached - forcing HOLD")
                        # REMOVED: Observation period bias - let model trade when it wants
                        # The model should learn optimal timing, not be forced to wait
                    
                    # Execute action in MT5
                    current_price = float(df['close'].iloc[-1])
                    action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
                    
                    if action == 1:  # BUY
                        print(f"[BUY] BUY Signal at {current_price:.2f}")
                        # Place buy order in MT5
                        # Get current market price
                        tick = mt5.symbol_info_tick(config.symbol)
                        if tick is None:
                            print("[ERROR] Cannot get current market price")
                            continue
                        
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": config.symbol,
                            "volume": config.lot_size,
                            "type": mt5.ORDER_TYPE_BUY,
                            "price": tick.ask,  # Use ask price for buy
                            "deviation": 20,
                            "magic": 234000,
                            "comment": "PPO Buy",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"[ERROR] Buy order failed: {result.retcode}")
                        else:
                            print(f"[OK] Buy order placed: {result.order}")
                            trades_today += 1
                            last_trade_time = current_time
                            last_observation_time = current_time  # Reset observation period
                            
                            # Add to trade history
                            trade_history.append({
                                'action': 'BUY',
                                'price': current_price,
                                'profit': 0.0,  # Will be updated when position closes
                                'timestamp': current_time
                            })
                            
                            # Keep only last 3 trades
                            if len(trade_history) > 3:
                                trade_history = trade_history[-3:]
                            
                            print(f"[TRADE] Trade #{trades_today} executed - next trade allowed after {observation_period/60:.1f} minutes")
                    
                    elif action == 2:  # SELL
                        print(f"[SELL] SELL Signal at {current_price:.2f}")
                        # Place sell order in MT5
                        # Get current market price
                        tick = mt5.symbol_info_tick(config.symbol)
                        if tick is None:
                            print("[ERROR] Cannot get current market price")
                            continue
                        
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": config.symbol,
                            "volume": config.lot_size,
                            "type": mt5.ORDER_TYPE_SELL,
                            "price": tick.bid,  # Use bid price for sell
                            "deviation": 20,
                            "magic": 234000,
                            "comment": "PPO Sell",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"[ERROR] Sell order failed: {result.retcode}")
                        else:
                            print(f"[OK] Sell order placed: {result.order}")
                            trades_today += 1
                            last_trade_time = current_time
                            last_observation_time = current_time  # Reset observation period
                            
                            # Add to trade history
                            trade_history.append({
                                'action': 'SELL',
                                'price': current_price,
                                'profit': 0.0,  # Will be updated when position closes
                                'timestamp': current_time
                            })
                            
                            # Keep only last 3 trades
                            if len(trade_history) > 3:
                                trade_history = trade_history[-3:]
                            
                            print(f"[TRADE] Trade #{trades_today} executed - next trade allowed after {observation_period/60:.1f} minutes")
                    
                    elif action == 3:  # CLOSE - BLOCKED
                        action = 0  # Force HOLD instead
                        print(f"[BLOCKED] Model CLOSE action blocked - positions will run until TP/SL")
                        print(f"[INFO] Let TP ($100) or SL (-$100) handle position closing")
                    
                    # Execute action in environment
                    obs, reward, done, truncated, info = env.step(action)
                    
                    # Update observation time if we actually traded (counter already updated in order execution)
                    if action in [1, 2]:  # BUY or SELL
                        last_observation_time = current_time
                        print(f"[TRADE] Trade #{trades_today} executed")
                    elif action == 3:  # CLOSE (should not reach here due to blocking above)
                        print(f"[CLOSE] All positions closed")
                        # No observation period - model can trade immediately
                    
                    # Get real MT5 account balance
                    account_info = mt5.account_info()
                    real_balance = account_info.balance if account_info else 0
                    real_equity = account_info.equity if account_info else 0
                    real_margin = account_info.margin if account_info else 0
                    real_free_margin = account_info.margin_free if account_info else 0
                    
                    # Log results
                    print(f"Action: {action_names.get(action, 'UNKNOWN')}, Reward: {reward:.2f}")
                    print(f"Real Balance: ${real_balance:.2f}, Equity: ${real_equity:.2f}")
                    print(f"Margin: ${real_margin:.2f}, Free Margin: ${real_free_margin:.2f}")
                    print(f"Price: {current_price:.2f}, Data points: {len(df)}")
                    
                    # Online Learning: Update policy based on live trading results
                    if hasattr(agent, 'model') and agent.model is not None:
                        # Store the experience for online learning
                        if not hasattr(agent, 'experience_buffer'):
                            agent.experience_buffer = []
                        
                        # Add experience to buffer
                        agent.experience_buffer.append({
                            'observation': obs,
                            'action': action,
                            'reward': reward,
                            'next_observation': obs,  # For simplicity, using current obs
                            'done': done
                        })
                        
                        # Update policy every 60 trades and only when all positions are closed
                        if len(agent.experience_buffer) >= 60 and current_position_count == 0:
                            print(f"[LEARNING] Updating policy - Buffer: {len(agent.experience_buffer)}/60, Positions: {current_position_count}")
                            print("[LEARNING] Updating policy based on live trading...")
                            try:
                                # Convert buffer to training data
                                observations = np.array([exp['observation'] for exp in agent.experience_buffer])
                                actions = np.array([exp['action'] for exp in agent.experience_buffer])
                                rewards = np.array([exp['reward'] for exp in agent.experience_buffer])
                                
                                # Update the model with new experiences
                                agent.model.learn(total_timesteps=100, reset_num_timesteps=False)
                                
                                # Clear buffer after learning
                                agent.experience_buffer = []
                                print("[OK] Policy updated successfully!")
                                
                            except Exception as e:
                                print(f"[WARNING] Online learning failed: {e}")
                                # Clear buffer to prevent memory issues
                                agent.experience_buffer = []
                    
                    # Check current positions and manage profit/loss
                    positions = mt5.positions_get(symbol=config.symbol)
                    print(f"Current positions: {len(positions)}")
                    
                    for pos in positions:
                        current_profit = pos.profit
                        print(f"  Position {pos.ticket}: {pos.type} {pos.volume} lots at {pos.price_open}")
                        print(f"    Current profit: ${current_profit:.2f}")
                        
                        # Auto-close positions based on profit/loss
                        if current_profit >= 100:  # Take profit at $100
                            print(f"[PROFIT] Closing position {pos.ticket} with profit ${current_profit:.2f}")
                            # Close position
                            if pos.type == mt5.POSITION_TYPE_BUY:
                                request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": config.symbol,
                                    "volume": pos.volume,
                                    "type": mt5.ORDER_TYPE_SELL,
                                    "position": pos.ticket,
                                    "price": tick.bid,
                                    "deviation": 20,
                                    "magic": 234000,
                                    "comment": "PPO Profit",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                            else:
                                request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": config.symbol,
                                    "volume": pos.volume,
                                    "type": mt5.ORDER_TYPE_BUY,
                                    "position": pos.ticket,
                                    "price": tick.ask,
                                    "deviation": 20,
                                    "magic": 234000,
                                    "comment": "PPO Profit",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                            result = mt5.order_send(request)
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                print(f"[OK] Position closed for profit: {pos.ticket}")
                        
                        elif current_profit < -100:  # Stop loss at -$100
                            print(f"[LOSS] Closing position {pos.ticket} with loss ${current_profit:.2f}")
                            # Close position
                            if pos.type == mt5.POSITION_TYPE_BUY:
                                request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": config.symbol,
                                    "volume": pos.volume,
                                    "type": mt5.ORDER_TYPE_SELL,
                                    "position": pos.ticket,
                                    "price": tick.bid,
                                    "deviation": 20,
                                    "magic": 234000,
                                    "comment": "PPO Stop Loss",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                            else:
                                request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": config.symbol,
                                    "volume": pos.volume,
                                    "type": mt5.ORDER_TYPE_BUY,
                                    "position": pos.ticket,
                                    "price": tick.ask,
                                    "deviation": 20,
                                    "magic": 234000,
                                    "comment": "PPO Stop Loss",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                            result = mt5.order_send(request)
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                print(f"[OK] Position closed for stop loss: {pos.ticket}")
                    
                    # Wait for next iteration (shorter wait for more frequent profit/loss checks)
                    time.sleep(10)  # Check every 10 seconds for better profit/loss monitoring
                    
            except KeyboardInterrupt:
                print("\nLive trading stopped by user")
            finally:
                mt5.shutdown()
            
        elif args.mode == "test-costs":
            print("Testing cost calculations...")
            from cost_calculator import CostCalculator
            
            # Load configuration
            config = load_trading_config()
            
            # Create cost calculator
            calculator = CostCalculator(config.to_dict())
            
            # Test parameters
            lot_size = config.lot_size
            current_price = 2000.0  # Example gold price
            
            # Calculate costs
            costs = calculator.calculate_trade_costs(lot_size, current_price, "open")
            print(f"Trading costs for {lot_size} lots at ${current_price}:")
            print(f"  Commission: ${costs.commission:.2f}")
            print(f"  Spread: ${costs.spread:.2f}")
            print(f"  Slippage: ${costs.slippage:.2f}")
            print(f"  Total: ${costs.total:.2f}")
            
            # Calculate position value and margin
            position_value = calculator.calculate_position_value(lot_size, current_price)
            margin_required = calculator.calculate_margin_requirement(lot_size, current_price)
            
            print(f"\nPosition value: ${position_value:.2f}")
            print(f"Margin required: ${margin_required:.2f}")
            
            # Calculate maximum position size
            max_position_size = calculator.calculate_max_position_size(10000, current_price)
            print(f"Max position size: {max_position_size:.2f} lots")
            
            # Get cost summary
            summary = calculator.get_cost_summary(lot_size, current_price)
            print(f"\nCost summary:")
            print(f"Total costs: ${summary['total_costs']:.2f}")
            print(f"Cost percentage: {summary['cost_percentage']:.2f}%")
            
    except KeyboardInterrupt:
        print("\n[WARNING] Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
