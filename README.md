# PPO Trading Agent for Gold Metals

A reinforcement learning trading agent using Proximal Policy Optimization (PPO) for Gold Metals trading. The agent learns to make trading decisions based on market data and technical indicators.

## Project Structure

```
ppo/
├── main.py                    # Main execution script
├── trading_env.py            # Custom trading environment
├── ppo_agent.py              # PPO agent implementation
├── data_preprocessor.py      # Data preprocessing and feature engineering
├── training_pipeline.py      # Complete training pipeline
├── trading_config.py         # Trading account configuration
├── requirements.txt          # Python dependencies
├── Gold_Metals_M1.csv        # Gold Metals dataset
└── README.md                 # This file
```

## Features

- **Custom Trading Environment**: Gymnasium-compatible environment with realistic trading mechanics
- **Advanced PPO Agent**: Custom CNN-LSTM architecture with attention mechanism
- **Comprehensive Data Processing**: Technical indicators, metals correlations, and feature engineering
- **Risk Management**: Built-in position sizing, stop-loss, and drawdown controls
- **Performance Tracking**: Detailed metrics and visualization
- **Configuration Management**: Secure credential handling and parameter management

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ppo
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Validate Setup
```bash
python main.py --mode validate
```

### 2. Create Configuration
```bash
python main.py --mode create-config
```
Edit `trading_config.json` with your trading account details.

### 3. Train the Model
```bash
python main.py --mode train --timesteps 100000
```

### 4. Evaluate the Model
```bash
python main.py --mode evaluate --model training_output/models/final_model.zip
```

## Usage Examples

### Training with Custom Parameters
```bash
python main.py --mode train --timesteps 200000 --output my_training
```

### Evaluation with Multiple Episodes
```bash
python main.py --mode evaluate --model best_model.zip --episodes 20
```

### Using Different Dataset
```bash
python main.py --mode train --data my_data.csv --timesteps 50000
```

## Configuration

The `trading_config.json` file contains all trading parameters:

```json
{
    "account": 12345678,
    "password": "your_password",
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
```

## Trading Environment

The trading environment provides:

- **State Space**: 20 timesteps × 12 features (OHLCV + technical indicators + position info)
- **Action Space**: 4 actions (Hold, Buy, Sell, Close Position)
- **Reward Function**: Based on realized/unrealized P&L, risk management, and market timing

### State Features
- OHLCV data (normalized)
- Technical indicators (RSI, MACD, Moving Averages)
- Position information (current position, balance ratio)
- Unrealized P&L

### Actions
- **0**: Hold (no action)
- **1**: Buy (open long position)
- **2**: Sell (open short position)
- **3**: Close Position

## PPO Agent Architecture

The agent uses a custom CNN-LSTM architecture:

- **CNN Layers**: 1D convolutions for feature extraction
- **LSTM Layer**: Temporal dependency modeling
- **Attention Mechanism**: Multi-head attention for important timesteps
- **Actor-Critic**: Separate networks for policy and value estimation

## Data Preprocessing

The system includes comprehensive data preprocessing:

### Technical Indicators
- RSI, MACD, Bollinger Bands
- Moving Averages (5, 10, 20, 50 periods)
- ATR, Support/Resistance levels
- Volume indicators

### Metals Correlations
- Cross-correlation with Silver (XAGUSD)
- Cross-correlation with Platinum (XPTUSD)
- Cross-correlation with Palladium (XPDUSD)

### Time Features
- Market session indicators (London, NY, Asian)
- Hour of day, day of week
- Weekend indicators

## Training Pipeline

The training pipeline includes:

1. **Data Preprocessing**: Feature engineering and normalization
2. **Environment Creation**: Training, validation, and test environments
3. **Model Training**: PPO with custom callbacks and monitoring
4. **Evaluation**: Performance metrics and visualization
5. **Results Saving**: Model checkpoints and training logs

## Performance Metrics

The system tracks multiple performance metrics:

- **Episode Rewards**: Cumulative rewards per episode
- **Returns**: Percentage returns on initial balance
- **Sharpe Ratio**: Risk-adjusted returns
- **Drawdown**: Maximum drawdown during episodes
- **Win Rate**: Percentage of profitable trades

## Output Files

Training generates several output files:

```
training_output/
├── models/
│   ├── final_model.zip          # Final trained model
│   └── best_trading_model.zip    # Best performing model
├── logs/
│   ├── train/                    # Training logs
│   ├── val/                      # Validation logs
│   └── test/                     # Test logs
├── plots/
│   └── training_results.png      # Performance plots
├── processed_data.csv            # Preprocessed dataset
└── training_results.json         # Training metrics
```

## Risk Management

Built-in risk management features:

- **Position Sizing**: Maximum 10% of account balance per trade
- **Stop Loss**: Configurable stop-loss levels
- **Take Profit**: Configurable take-profit levels
- **Daily Limits**: Maximum number of trades per day
- **Drawdown Control**: Maximum drawdown limits

## Troubleshooting

### Common Issues

1. **Missing Dataset**: Ensure `Gold_Metals_M1.csv` is in the project directory
2. **Configuration Errors**: Run `python main.py --mode create-config` to create template
3. **Memory Issues**: Reduce batch size or lookback window in environment
4. **Training Slow**: Reduce timesteps for initial testing

### Performance Tips

1. **GPU Acceleration**: Ensure PyTorch is installed with CUDA support
2. **Data Quality**: Ensure dataset has no missing values or outliers
3. **Hyperparameters**: Tune learning rate and batch size for your data
4. **Feature Engineering**: Add domain-specific indicators for better performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always test thoroughly before using with real money.
