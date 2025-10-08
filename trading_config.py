"""
Trading Account Configuration Management
Handles credentials and trading parameters
"""
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    # Account credentials
    account: int
    password: str
    server: str
    
    # Trading parameters
    symbol: str = "XAUUSD"
    lot_size: float = 0.01
    max_risk_per_trade: float = 0.02  # 2% of account balance
    max_daily_trades: int = 10
    max_position_size: float = 0.1  # 10% of account balance
    
    # Risk management
    stop_loss_pips: int = 20
    take_profit_pips: int = 40
    max_drawdown: float = 0.05  # 5% max drawdown
    
    # Model parameters
    model_path: str = "best_trading_model.zip"
    confidence_threshold: float = 0.7
    
    # Trading hours (UTC)
    trading_start_hour: int = 0
    trading_end_hour: int = 23
    
    # Trading fees (optional)
    trading_fees: dict = None
    
    # Margin trading (optional)
    margin_trading: dict = None
    
    # Data fetching (optional)
    data_fetching: dict = None
    
    # Risk management (optional)
    risk_management: dict = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'account': self.account,
            'password': self.password,
            'server': self.server,
            'symbol': self.symbol,
            'lot_size': self.lot_size,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_daily_trades': self.max_daily_trades,
            'max_position_size': self.max_position_size,
            'stop_loss_pips': self.stop_loss_pips,
            'take_profit_pips': self.take_profit_pips,
            'max_drawdown': self.max_drawdown,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'trading_start_hour': self.trading_start_hour,
            'trading_end_hour': self.trading_end_hour
        }
        
        # Add optional parameters if they exist
        if self.trading_fees is not None:
            result['trading_fees'] = self.trading_fees
        if self.margin_trading is not None:
            result['margin_trading'] = self.margin_trading
        if self.data_fetching is not None:
            result['data_fetching'] = self.data_fetching
        if self.risk_management is not None:
            result['risk_management'] = self.risk_management
            
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TradingConfig':
        """Create from dictionary"""
        return cls(**config_dict)


class ConfigManager:
    """Manages trading configuration and credentials"""
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config_file = config_file
        self.config = None
        
    def load_config(self) -> TradingConfig:
        """Load configuration from file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        
        try:
            with open(self.config_file, 'r') as f:
                config_dict = json.load(f)
            
            self.config = TradingConfig.from_dict(config_dict)
            print(f"Configuration loaded from {self.config_file}")
            return self.config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def save_config(self, config: TradingConfig) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=4)
            
            print(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")
    
    def create_default_config(self) -> TradingConfig:
        """Create default configuration template"""
        default_config = TradingConfig(
            account=0,  # To be filled by user
            password="",  # To be filled by user
            server=""  # To be filled by user
        )
        
        self.save_config(default_config)
        print(f"Default configuration template created at {self.config_file}")
        print("Please edit the configuration file with your trading account details")
        
        return default_config
    
    def validate_config(self, config: TradingConfig) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Check required fields
        if config.account == 0:
            errors.append("Account number must be set")
        if not config.password:
            errors.append("Password must be set")
        if not config.server:
            errors.append("Server must be set")
        
        # Check trading parameters
        if config.lot_size <= 0:
            errors.append("Lot size must be positive")
        if not 0 < config.max_risk_per_trade <= 1:
            errors.append("Max risk per trade must be between 0 and 1")
        if config.max_daily_trades <= 0:
            errors.append("Max daily trades must be positive")
        if not 0 < config.max_position_size <= 1:
            errors.append("Max position size must be between 0 and 1")
        
        # Check risk management
        if config.stop_loss_pips <= 0:
            errors.append("Stop loss pips must be positive")
        if config.take_profit_pips <= 0:
            errors.append("Take profit pips must be positive")
        if not 0 < config.max_drawdown <= 1:
            errors.append("Max drawdown must be between 0 and 1")
        
        # Check model parameters
        if not 0 < config.confidence_threshold <= 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        # Check trading hours
        if not 0 <= config.trading_start_hour <= 23:
            errors.append("Trading start hour must be between 0 and 23")
        if not 0 <= config.trading_end_hour <= 23:
            errors.append("Trading end hour must be between 0 and 23")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation passed")
        return True
    
    def get_config(self) -> Optional[TradingConfig]:
        """Get current configuration"""
        return self.config


def create_config_template():
    """Create a configuration template file"""
    config_manager = ConfigManager()
    return config_manager.create_default_config()


def load_trading_config(config_file: str = "trading_config.json") -> TradingConfig:
    """Load trading configuration from file"""
    config_manager = ConfigManager(config_file)
    config = config_manager.load_config()
    
    # Validate configuration
    if not config_manager.validate_config(config):
        raise ValueError("Configuration validation failed")
    
    return config


# Example configuration template
EXAMPLE_CONFIG = {
    "account": 12345678,
    "password": "your_password_here",
    "server": "YourBroker-Server",
    "symbol": "XAUUSD",
    "lot_size": 0.01,
    "max_risk_per_trade": 0.02,
    "max_daily_trades": 10,
    "max_position_size": 0.1,
    "stop_loss_pips": 20,
    "take_profit_pips": 40,
    "max_drawdown": 0.05,
    "model_path": "best_trading_model.zip",
    "confidence_threshold": 0.7,
    "trading_start_hour": 0,
    "trading_end_hour": 23
}


if __name__ == "__main__":
    # Create configuration template
    create_config_template()
    print("Configuration template created. Please edit trading_config.json with your details.")
