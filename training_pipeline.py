"""
Training Pipeline for PPO Trading Agent
Handles model training, evaluation, and performance tracking
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
import json
from datetime import datetime

from trading_env import GoldTradingEnv
from ppo_agent import PPOTradingAgent, TradingCallback
from data_preprocessor import GoldDataPreprocessor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class TrainingPipeline:
    """
    Complete training pipeline for PPO trading agent
    """
    
    def __init__(self, data_path: str = "Gold_Metals_M1.csv", 
                 initial_balance: float = 10000.0,
                 output_dir: str = "training_output"):
        
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        # Initialize components
        self.preprocessor = GoldDataPreprocessor(data_path)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_env = None
        self.val_env = None
        self.test_env = None
        self.agent = None
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'validation_rewards': [],
            'training_losses': []
        }
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare and split data for training"""
        print("Preparing data for training...")
        
        # Run full preprocessing pipeline
        processed_data = self.preprocessor.preprocess_full_pipeline()
        
        # Create training splits
        self.train_data, self.val_data, self.test_data = self.preprocessor.create_training_splits()
        
        # Save processed data
        self.preprocessor.save_processed_data(f"{self.output_dir}/processed_data.csv")
        
        print(f"Data preparation completed:")
        print(f"  Training: {len(self.train_data)} samples")
        print(f"  Validation: {len(self.val_data)} samples")
        print(f"  Test: {len(self.test_data)} samples")
        
        return self.train_data, self.val_data, self.test_data
    
    def create_environments(self) -> Tuple[GoldTradingEnv, GoldTradingEnv, GoldTradingEnv]:
        """Create training, validation, and test environments"""
        print("Creating trading environments...")
        
        # Training environment
        self.train_env = GoldTradingEnv(
            data=self.train_data,
            initial_balance=self.initial_balance,
            transaction_cost=0.001,
            lookback_window=20
        )
        
        # Validation environment
        self.val_env = GoldTradingEnv(
            data=self.val_data,
            initial_balance=self.initial_balance,
            transaction_cost=0.001,
            lookback_window=20
        )
        
        # Test environment
        self.test_env = GoldTradingEnv(
            data=self.test_data,
            initial_balance=self.initial_balance,
            transaction_cost=0.001,
            lookback_window=20
        )
        
        # Wrap environments with monitoring
        self.train_env = Monitor(self.train_env, f"{self.output_dir}/logs/train")
        self.val_env = Monitor(self.val_env, f"{self.output_dir}/logs/val")
        self.test_env = Monitor(self.test_env, f"{self.output_dir}/logs/test")
        
        print("Environments created successfully")
        return self.train_env, self.val_env, self.test_env
    
    def create_agent(self, model_path: str = None) -> PPOTradingAgent:
        """Create PPO trading agent"""
        print("Creating PPO trading agent...")
        
        self.agent = PPOTradingAgent(
            env=self.train_env,
            model_path=model_path,
            learning_rate=1e-4,  # Reduced learning rate
            n_steps=2048,
            batch_size=64,
            n_epochs=5,  # Reduced epochs
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )
        
        print("PPO agent created successfully")
        return self.agent
    
    def train_agent(self, total_timesteps: int = 100000, 
                   eval_freq: int = 10000, save_freq: int = 10000) -> Dict[str, Any]:
        """Train the PPO agent"""
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Create callbacks
        callbacks = []
        if self.val_env is not None:
            callback = TradingCallback(
                eval_env=self.val_env,
                eval_freq=eval_freq,
                verbose=1
            )
            callbacks.append(callback)
        
        # Train the agent
        start_time = datetime.now()
        self.agent.train(
            total_timesteps=total_timesteps,
            eval_env=self.val_env,
            save_freq=save_freq,
            log_interval=1
        )
        end_time = datetime.now()
        
        # Save final model
        final_model_path = f"{self.output_dir}/models/final_model.zip"
        self.agent.save(final_model_path)
        
        # Calculate training time
        training_time = (end_time - start_time).total_seconds()
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final model saved to {final_model_path}")
        
        return {
            'training_time': training_time,
            'total_timesteps': total_timesteps,
            'final_model_path': final_model_path
        }
    
    def evaluate_agent(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        print(f"Evaluating agent on {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        episode_returns = []
        episode_sharpe_ratios = []
        
        for episode in range(n_episodes):
            obs, info = self.test_env.reset()
            episode_reward = 0
            episode_length = 0
            returns = []
            
            done = False
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.test_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Calculate returns
                if 'balance' in info:
                    returns.append(info['balance'])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Calculate episode return
            if len(returns) > 1:
                episode_return = (returns[-1] - returns[0]) / returns[0]
                episode_returns.append(episode_return)
                
                # Calculate Sharpe ratio
                if len(returns) > 1:
                    returns_array = np.array(returns)
                    returns_pct = np.diff(returns_array) / returns_array[:-1]
                    if len(returns_pct) > 1 and np.std(returns_pct) > 0:
                        sharpe = np.mean(returns_pct) / np.std(returns_pct) * np.sqrt(252)
                        episode_sharpe_ratios.append(sharpe)
        
        # Calculate evaluation metrics
        evaluation_metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_return': np.mean(episode_returns) if episode_returns else 0,
            'mean_sharpe': np.mean(episode_sharpe_ratios) if episode_sharpe_ratios else 0,
            'episode_rewards': episode_rewards,
            'episode_returns': episode_returns,
            'episode_sharpe_ratios': episode_sharpe_ratios
        }
        
        print(f"Evaluation completed:")
        print(f"  Mean Reward: {evaluation_metrics['mean_reward']:.2f} ± {evaluation_metrics['std_reward']:.2f}")
        print(f"  Mean Return: {evaluation_metrics['mean_return']:.2%}")
        print(f"  Mean Sharpe Ratio: {evaluation_metrics['mean_sharpe']:.2f}")
        
        return evaluation_metrics
    
    def plot_training_results(self, evaluation_metrics: Dict[str, Any]) -> None:
        """Plot training results and performance metrics"""
        print("Creating training plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PPO Trading Agent Training Results', fontsize=16)
        
        # Plot 1: Episode rewards
        axes[0, 0].plot(evaluation_metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: Episode returns
        if evaluation_metrics['episode_returns']:
            axes[0, 1].plot(evaluation_metrics['episode_returns'])
            axes[0, 1].set_title('Episode Returns')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Return')
            axes[0, 1].grid(True)
        
        # Plot 3: Sharpe ratios
        if evaluation_metrics['episode_sharpe_ratios']:
            axes[1, 0].plot(evaluation_metrics['episode_sharpe_ratios'])
            axes[1, 0].set_title('Episode Sharpe Ratios')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].grid(True)
        
        # Plot 4: Performance summary
        metrics = ['Mean Reward', 'Mean Return', 'Mean Sharpe']
        values = [
            evaluation_metrics['mean_reward'],
            evaluation_metrics['mean_return'] * 100,  # Convert to percentage
            evaluation_metrics['mean_sharpe']
        ]
        
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/training_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training plots saved to {self.output_dir}/plots/")
    
    def save_training_results(self, evaluation_metrics: Dict[str, Any]) -> str:
        """Save training results to JSON file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'train_samples': len(self.train_data),
                'val_samples': len(self.val_data),
                'test_samples': len(self.test_data)
            },
            'evaluation_metrics': evaluation_metrics,
            'model_path': f"{self.output_dir}/models/final_model.zip"
        }
        
        results_path = f"{self.output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Training results saved to {results_path}")
        return results_path
    
    def run_full_pipeline(self, total_timesteps: int = 100000, 
                         n_eval_episodes: int = 10) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        print("Starting full training pipeline...")
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Create environments
        self.create_environments()
        
        # Step 3: Create agent
        self.create_agent()
        
        # Step 4: Train agent
        training_results = self.train_agent(total_timesteps=total_timesteps)
        
        # Step 5: Evaluate agent
        evaluation_metrics = self.evaluate_agent(n_episodes=n_eval_episodes)
        
        # Step 6: Plot results
        self.plot_training_results(evaluation_metrics)
        
        # Step 7: Save results
        results_path = self.save_training_results(evaluation_metrics)
        
        print("Full training pipeline completed successfully!")
        
        return {
            'training_results': training_results,
            'evaluation_metrics': evaluation_metrics,
            'results_path': results_path
        }


def main():
    """Main function to run training pipeline"""
    # Initialize pipeline
    pipeline = TrainingPipeline(
        data_path="Gold_Metals_M1.csv",
        initial_balance=10000.0,
        output_dir="training_output"
    )
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        total_timesteps=50000,  # Reduced for initial testing
        n_eval_episodes=5
    )
    
    print("Training pipeline completed!")
    print(f"Results saved to: {results['results_path']}")


if __name__ == "__main__":
    main()
