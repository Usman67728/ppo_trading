"""
Test script to verify the PPO trading agent setup
"""
import os
import sys
import pandas as pd
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print("[OK] PyTorch imported successfully")
    except ImportError:
        print("[ERROR] PyTorch not found")
        return False
    
    try:
        import stable_baselines3
        print("[OK] Stable Baselines3 imported successfully")
    except ImportError:
        print("[ERROR] Stable Baselines3 not found")
        return False
    
    try:
        import gymnasium
        print("[OK] Gymnasium imported successfully")
    except ImportError:
        print("[ERROR] Gymnasium not found")
        return False
    
    try:
        import pandas
        print("[OK] Pandas imported successfully")
    except ImportError:
        print("[ERROR] Pandas not found")
        return False
    
    try:
        import numpy
        print("[OK] NumPy imported successfully")
    except ImportError:
        print("[ERROR] NumPy not found")
        return False
    
    return True

def test_data_file():
    """Test if the data file exists and is readable"""
    print("\nTesting data file...")
    
    if not os.path.exists("Gold_Metals_M1.csv"):
        print("[ERROR] Gold_Metals_M1.csv not found")
        return False
    
    try:
        df = pd.read_csv("Gold_Metals_M1.csv", parse_dates=['time'])
        print(f"[OK] Data file loaded successfully: {len(df)} rows")
        
        # Check required columns
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"[ERROR] Missing columns: {missing_columns}")
            return False
        
        print("[OK] All required columns present")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error reading data file: {e}")
        return False

def test_environment_creation():
    """Test if the trading environment can be created"""
    print("\nTesting environment creation...")
    
    try:
        from trading_env import GoldTradingEnv
        
        # Create sample data
        sample_data = pd.DataFrame({
            'time': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.random.uniform(1800, 2000, 100),
            'high': np.random.uniform(1800, 2000, 100),
            'low': np.random.uniform(1800, 2000, 100),
            'close': np.random.uniform(1800, 2000, 100),
            'volume': np.random.randint(10, 100, 100)
        })
        
        # Create environment
        env = GoldTradingEnv(sample_data, initial_balance=10000.0)
        
        # Test reset
        obs, info = env.reset()
        print(f"[OK] Environment created successfully")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"[OK] Environment step successful")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error creating environment: {e}")
        return False

def test_agent_creation():
    """Test if the PPO agent can be created"""
    print("\nTesting agent creation...")
    
    try:
        from ppo_agent import PPOTradingAgent
        from trading_env import GoldTradingEnv
        
        # Create sample data
        sample_data = pd.DataFrame({
            'time': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.random.uniform(1800, 2000, 100),
            'high': np.random.uniform(1800, 2000, 100),
            'low': np.random.uniform(1800, 2000, 100),
            'close': np.random.uniform(1800, 2000, 100),
            'volume': np.random.randint(10, 100, 100)
        })
        
        # Create environment
        env = GoldTradingEnv(sample_data, initial_balance=10000.0)
        
        # Create agent
        agent = PPOTradingAgent(env)
        print("[OK] PPO agent created successfully")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error creating agent: {e}")
        return False

def test_data_preprocessing():
    """Test if data preprocessing works"""
    print("\nTesting data preprocessing...")
    
    try:
        from data_preprocessor import GoldDataPreprocessor
        
        # Create sample data
        sample_data = pd.DataFrame({
            'time': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.random.uniform(1800, 2000, 100),
            'high': np.random.uniform(1800, 2000, 100),
            'low': np.random.uniform(1800, 2000, 100),
            'close': np.random.uniform(1800, 2000, 100),
            'volume': np.random.randint(10, 100, 100)
        })
        
        # Save sample data
        sample_data.to_csv("test_data.csv", index=False)
        
        # Test preprocessor
        preprocessor = GoldDataPreprocessor("test_data.csv")
        processed_data = preprocessor.preprocess_full_pipeline()
        
        print("[OK] Data preprocessing successful")
        print(f"   Processed data shape: {processed_data.shape}")
        print(f"   Features: {len(preprocessor.get_feature_columns())}")
        
        # Clean up
        os.remove("test_data.csv")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error in data preprocessing: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PPO TRADING AGENT SETUP TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_file,
        test_environment_creation,
        test_agent_creation,
        test_data_preprocessing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("[OK] All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Run: python main.py --mode create-config")
        print("2. Edit trading_config.json with your account details")
        print("3. Run: python main.py --mode train")
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Ensure Gold_Metals_M1.csv is in the current directory")
        print("3. Check Python version compatibility")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
