"""
PPO Agent for Gold Metals Trading
Uses stable-baselines3 with custom policy and callbacks
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Any, List
import os


class TradingCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for trading data
    Processes time series data with convolutional layers
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Input shape: (batch_size, lookback_window, n_features)
        self.lookback_window = observation_space.shape[0]
        self.n_features = observation_space.shape[1]
        
        # Reshape for CNN: (batch_size, n_features, lookback_window)
        self.conv1 = nn.Conv1d(self.n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        
        # Final layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, features_dim)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
        # Initialize weights properly
        self._initialize_weights()
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape: (batch_size, lookback_window, n_features) -> (batch_size, n_features, lookback_window)
        x = observations.transpose(1, 2)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        
        # Reshape for LSTM: (batch_size, lookback_window, 256)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Final layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize weights to prevent NaN values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)


class TradingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress
    """
    
    def __init__(self, eval_env, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode info
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
        
        # Evaluate model periodically
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = self._evaluate_model()
            
            if self.verbose > 0:
                print(f"Evaluation at step {self.n_calls}")
                print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best model with mean reward: {mean_reward:.2f}")
                self.model.save("best_trading_model")
        
        return True
    
    def _evaluate_model(self) -> tuple:
        """Evaluate model on evaluation environment"""
        obs = self.eval_env.reset()
        episode_rewards = []
        
        for _ in range(10):  # Run 10 episodes
            episode_reward = 0
            done = False
            obs = self.eval_env.reset()
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)


class PPOTradingAgent:
    """
    PPO Trading Agent with custom configuration
    """
    
    def __init__(self, env, model_path: str = None, learning_rate: float = 3e-4,
                 n_steps: int = 2048, batch_size: int = 64, n_epochs: int = 10,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_range: float = 0.2):
        
        self.env = env
        self.model_path = model_path
        
        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        
        # Initialize model
        self.model = None
        self._create_model()
        
    def _create_model(self):
        """Create PPO model with custom policy"""
        policy_kwargs = {
            "features_extractor_class": TradingCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[256, 128], vf=[256, 128]),
            "activation_fn": torch.nn.ReLU,
        }
        
        self.model = PPO(
            "MlpPolicy",  # Will be overridden by custom policy
            self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            max_grad_norm=0.5,  # Add gradient clipping
            ent_coef=0.01,  # Add entropy coefficient
            vf_coef=0.5,  # Value function coefficient
            target_kl=0.01,  # Target KL divergence
            device='cpu'  # Force CPU usage to avoid GPU issues
        )
        
        # Load existing model if path provided
        if self.model_path and os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path, env=self.env)
            print(f"Loaded model from {self.model_path}")
    
    def train(self, total_timesteps: int = 100000, eval_env=None, 
              save_freq: int = 10000, log_interval: int = 1):
        """Train the PPO agent"""
        
        # Setup callbacks
        callbacks = []
        if eval_env is not None:
            callback = TradingCallback(eval_env, eval_freq=save_freq)
            callbacks.append(callback)
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval
        )
        
        # Save final model
        self.model.save("final_trading_model")
        print("Training completed. Model saved as 'final_trading_model'")
    
    def predict(self, obs, deterministic: bool = True):
        """Make prediction using trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(obs, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")
    
    def load(self, path: str):
        """Load a trained model"""
        if os.path.exists(path):
            self.model = PPO.load(path, env=self.env)
            print(f"Model loaded from {path}")
        else:
            print(f"Model file {path} not found")


def create_trading_policy(observation_space, action_space, lr_schedule, **kwargs):
    """
    Custom policy factory for trading
    """
    from stable_baselines3.common.policies import ActorCriticPolicy
    
    class TradingPolicy(ActorCriticPolicy):
        def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
            super().__init__(observation_space, action_space, lr_schedule, **kwargs)
    
    return TradingPolicy
