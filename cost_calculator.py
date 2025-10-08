"""
Trading Cost Calculator
Handles all trading costs including commissions, spreads, slippage, and margin requirements
"""
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class TradingCosts:
    """Trading costs breakdown"""
    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    swap: float = 0.0
    total: float = 0.0


class CostCalculator:
    """
    Calculates all trading costs for margin trading
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_fees = config.get('trading_fees', {})
        self.margin_trading = config.get('margin_trading', {})
        
        # Extract cost parameters
        self.commission_per_lot = self.trading_fees.get('commission_per_lot', 5.0)
        self.spread_pips = self.trading_fees.get('spread_pips', 2.0)
        self.slippage_pips = self.trading_fees.get('slippage_pips', 1.0)
        self.swap_long = self.trading_fees.get('swap_long', 0.0)
        self.swap_short = self.trading_fees.get('swap_short', 0.0)
        
        # Margin parameters
        self.leverage = self.margin_trading.get('leverage', 100)
        self.margin_requirement = self.margin_trading.get('margin_requirement', 0.01)
        self.margin_call_level = self.margin_trading.get('margin_call_level', 0.5)
        self.stop_out_level = self.margin_trading.get('stop_out_level', 0.3)
    
    def calculate_trade_costs(self, lot_size: float, current_price: float, 
                            action: str = "open") -> TradingCosts:
        """
        Calculate all costs for a trade
        
        Args:
            lot_size: Size of the trade in lots
            current_price: Current market price
            action: "open" or "close"
        
        Returns:
            TradingCosts object with cost breakdown
        """
        costs = TradingCosts()
        
        # Commission cost (per lot)
        costs.commission = self.commission_per_lot * lot_size
        
        # Spread cost (in pips)
        spread_cost_pips = self.spread_pips * 0.0001
        costs.spread = spread_cost_pips * current_price * lot_size * 100
        
        # Slippage cost (in pips)
        slippage_cost_pips = self.slippage_pips * 0.0001
        costs.slippage = slippage_cost_pips * current_price * lot_size * 100
        
        # Swap cost (for overnight positions)
        if action == "open":
            # Use long swap for buy, short swap for sell
            swap_rate = self.swap_long if action == "buy" else self.swap_short
            costs.swap = swap_rate * lot_size
        
        # Total cost
        costs.total = costs.commission + costs.spread + costs.slippage + costs.swap
        
        return costs
    
    def calculate_margin_requirement(self, lot_size: float, current_price: float) -> float:
        """
        Calculate margin requirement for a trade
        
        Args:
            lot_size: Size of the trade in lots
            current_price: Current market price
        
        Returns:
            Required margin in account currency
        """
        # Calculate position value
        position_value = current_price * lot_size * 100  # 100 units per lot
        
        # Calculate margin requirement
        margin_required = position_value * self.margin_requirement
        
        return margin_required
    
    def calculate_position_value(self, lot_size: float, current_price: float) -> float:
        """
        Calculate total position value
        
        Args:
            lot_size: Size of the trade in lots
            current_price: Current market price
        
        Returns:
            Total position value
        """
        return current_price * lot_size * 100
    
    def calculate_pnl(self, lot_size: float, entry_price: float, 
                     current_price: float, position_type: str) -> float:
        """
        Calculate profit/loss for a position
        
        Args:
            lot_size: Size of the trade in lots
            entry_price: Entry price of the position
            current_price: Current market price
            position_type: "long" or "short"
        
        Returns:
            Profit/loss amount
        """
        if position_type == "long":
            pnl = (current_price - entry_price) * lot_size * 100
        else:  # short
            pnl = (entry_price - current_price) * lot_size * 100
        
        return pnl
    
    def calculate_breakeven_price(self, lot_size: float, entry_price: float, 
                                 position_type: str) -> float:
        """
        Calculate breakeven price including costs
        
        Args:
            lot_size: Size of the trade in lots
            entry_price: Entry price of the position
            position_type: "long" or "short"
        
        Returns:
            Breakeven price
        """
        # Calculate total costs
        costs = self.calculate_trade_costs(lot_size, entry_price, "open")
        total_costs = costs.total
        
        # Calculate breakeven price
        if position_type == "long":
            breakeven = entry_price + (total_costs / (lot_size * 100))
        else:  # short
            breakeven = entry_price - (total_costs / (lot_size * 100))
        
        return breakeven
    
    def calculate_risk_reward_ratio(self, lot_size: float, entry_price: float,
                                   stop_loss_price: float, take_profit_price: float,
                                   position_type: str) -> Tuple[float, float, float]:
        """
        Calculate risk-reward ratio for a trade
        
        Args:
            lot_size: Size of the trade in lots
            entry_price: Entry price of the position
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            position_type: "long" or "short"
        
        Returns:
            Tuple of (risk_amount, reward_amount, risk_reward_ratio)
        """
        # Calculate risk (distance to stop loss)
        if position_type == "long":
            risk_distance = entry_price - stop_loss_price
            reward_distance = take_profit_price - entry_price
        else:  # short
            risk_distance = stop_loss_price - entry_price
            reward_distance = entry_price - take_profit_price
        
        # Calculate risk and reward amounts
        risk_amount = risk_distance * lot_size * 100
        reward_amount = reward_distance * lot_size * 100
        
        # Calculate risk-reward ratio
        if risk_amount > 0:
            risk_reward_ratio = reward_amount / risk_amount
        else:
            risk_reward_ratio = 0.0
        
        return risk_amount, reward_amount, risk_reward_ratio
    
    def calculate_max_position_size(self, account_balance: float, 
                                  current_price: float, risk_percent: float = 0.02) -> float:
        """
        Calculate maximum position size based on risk management
        
        Args:
            account_balance: Current account balance
            current_price: Current market price
            risk_percent: Maximum risk per trade (as decimal)
        
        Returns:
            Maximum position size in lots
        """
        # Calculate maximum risk amount
        max_risk_amount = account_balance * risk_percent
        
        # Calculate maximum position size based on risk
        # Assuming 20 pip stop loss
        stop_loss_pips = 20
        stop_loss_distance = stop_loss_pips * 0.0001 * current_price
        
        max_position_size = max_risk_amount / (stop_loss_distance * 100)
        
        # Also consider margin requirements
        margin_required = self.calculate_margin_requirement(max_position_size, current_price)
        if margin_required > account_balance * 0.1:  # Don't use more than 10% of balance for margin
            max_position_size = (account_balance * 0.1) / (current_price * 100 * self.margin_requirement)
        
        return max_position_size
    
    def calculate_daily_costs(self, lot_size: float, current_price: float, 
                          position_type: str, days_held: int = 1) -> float:
        """
        Calculate daily costs for holding a position
        
        Args:
            lot_size: Size of the trade in lots
            current_price: Current market price
            position_type: "long" or "short"
            days_held: Number of days position is held
        
        Returns:
            Total daily costs
        """
        # Swap costs
        swap_rate = self.swap_long if position_type == "long" else self.swap_short
        daily_swap_cost = swap_rate * lot_size * days_held
        
        # Other daily costs (spread, slippage) are one-time
        return daily_swap_cost
    
    def get_cost_summary(self, lot_size: float, current_price: float) -> Dict[str, Any]:
        """
        Get comprehensive cost summary for a trade
        
        Args:
            lot_size: Size of the trade in lots
            current_price: Current market price
        
        Returns:
            Dictionary with cost breakdown
        """
        # Calculate all costs
        open_costs = self.calculate_trade_costs(lot_size, current_price, "open")
        close_costs = self.calculate_trade_costs(lot_size, current_price, "close")
        
        # Calculate position value and margin
        position_value = self.calculate_position_value(lot_size, current_price)
        margin_required = self.calculate_margin_requirement(lot_size, current_price)
        
        # Calculate maximum position size
        max_position_size = self.calculate_max_position_size(10000, current_price)  # Assuming $10k balance
        
        return {
            'lot_size': lot_size,
            'current_price': current_price,
            'position_value': position_value,
            'margin_required': margin_required,
            'open_costs': {
                'commission': open_costs.commission,
                'spread': open_costs.spread,
                'slippage': open_costs.slippage,
                'swap': open_costs.swap,
                'total': open_costs.total
            },
            'close_costs': {
                'commission': close_costs.commission,
                'spread': close_costs.spread,
                'slippage': close_costs.slippage,
                'swap': close_costs.swap,
                'total': close_costs.total
            },
            'total_costs': open_costs.total + close_costs.total,
            'max_position_size': max_position_size,
            'cost_percentage': (open_costs.total + close_costs.total) / position_value * 100
        }


def main():
    """Test the cost calculator"""
    # Example configuration
    config = {
        'trading_fees': {
            'commission_per_lot': 5.0,
            'spread_pips': 2.0,
            'slippage_pips': 1.0,
            'swap_long': 0.0,
            'swap_short': 0.0
        },
        'margin_trading': {
            'leverage': 100,
            'margin_requirement': 0.01,
            'margin_call_level': 0.5,
            'stop_out_level': 0.3
        }
    }
    
    # Create calculator
    calculator = CostCalculator(config)
    
    # Test parameters
    lot_size = 0.35
    current_price = 2000.0
    
    # Calculate costs
    costs = calculator.calculate_trade_costs(lot_size, current_price, "open")
    print(f"Open costs: ${costs.total:.2f}")
    print(f"  Commission: ${costs.commission:.2f}")
    print(f"  Spread: ${costs.spread:.2f}")
    print(f"  Slippage: ${costs.slippage:.2f}")
    print(f"  Swap: ${costs.swap:.2f}")
    
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


if __name__ == "__main__":
    main()
