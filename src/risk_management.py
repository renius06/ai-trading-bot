import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class RiskMetrics:
    portfolio_value: float
    daily_var: float  # Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    position_concentration: float


@dataclass
class RiskLimits:
    max_position_size: float  # % of portfolio
    max_sector_exposure: float  # % of portfolio
    max_daily_loss: float  # % of portfolio
    max_drawdown: float  # % of portfolio
    min_liquidity_ratio: float  # % of portfolio
    max_leverage: float
    var_confidence: float  # For VaR calculation
    var_timeframe: int  # Days


class RiskManager:
    def __init__(self, risk_limits: RiskLimits):
        self.limits = risk_limits
        self.logger = logging.getLogger('RiskManager')
        self.portfolio_history = []
        self.daily_returns = []
        
    def calculate_position_size(self, 
                              signal_confidence: float,
                              portfolio_value: float,
                              price: float,
                              volatility: float = None) -> float:
        """Calculate optimal position size using Kelly Criterion and risk limits"""
        
        # Base position size from risk limit
        max_position_value = portfolio_value * self.limits.max_position_size
        max_shares = max_position_value / price
        
        # Kelly Criterion sizing
        if volatility is not None:
            # Simplified Kelly: f = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            win_prob = signal_confidence
            loss_prob = 1 - win_prob
            odds = 1.0  # Simplified odds
            
            kelly_fraction = (odds * win_prob - loss_prob) / odds
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            kelly_shares = (portfolio_value * kelly_fraction) / price
            max_shares = min(max_shares, kelly_shares)
        
        return max_shares
    
    def check_position_limits(self, positions: Dict, new_symbol: str, new_size: float, portfolio_value: float) -> bool:
        """Check if new position violates risk limits"""
        
        # Check maximum position size
        new_position_value = new_size * self.get_current_price(new_symbol)
        if new_position_value > portfolio_value * self.limits.max_position_size:
            self.logger.warning(f"Position size exceeds limit for {new_symbol}")
            return False
        
        # Check portfolio concentration
        total_value = sum(pos['quantity'] * pos['current_price'] for pos in positions.values())
        total_with_new = total_value + new_position_value
        
        if len(positions) > 0 and new_position_value / total_with_new > self.limits.max_position_size:
            self.logger.warning(f"Portfolio concentration too high for {new_symbol}")
            return False
        
        return True
    
    def calculate_var(self, returns: List[float], confidence: float = None, timeframe: int = None) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0.0
        
        confidence = confidence or self.limits.var_confidence
        timeframe = timeframe or self.limits.var_timeframe
        
        # Historical VaR
        returns_array = np.array(returns)
        var_percentile = (1 - confidence) * 100
        
        var_daily = np.percentile(returns_array, var_percentile)
        
        # Scale to timeframe
        var_timeframe = var_daily * np.sqrt(timeframe)
        
        return abs(var_timeframe)
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def check_risk_limits(self, portfolio_value: float, positions: Dict) -> Tuple[bool, List[str]]:
        """Check if current portfolio violates any risk limits"""
        violations = []
        
        # Check daily loss limit
        if self.daily_returns:
            daily_return = self.daily_returns[-1]
            if daily_return < -self.limits.max_daily_loss:
                violations.append(f"Daily loss {daily_return:.2%} exceeds limit {self.limits.max_daily_loss:.2%}")
        
        # Check maximum drawdown
        if len(self.portfolio_history) > 1:
            max_dd = self.calculate_max_drawdown(self.portfolio_history)
            if max_dd > self.limits.max_drawdown:
                violations.append(f"Max drawdown {max_dd:.2%} exceeds limit {self.limits.max_drawdown:.2%}")
        
        # Check position concentration
        for symbol, pos in positions.items():
            position_value = pos['quantity'] * pos['current_price']
            concentration = position_value / portfolio_value
            if concentration > self.limits.max_position_size:
                violations.append(f"Position {symbol} concentration {concentration:.2%} exceeds limit")
        
        # Check leverage
        total_position_value = sum(pos['quantity'] * pos['current_price'] for pos in positions.values())
        if portfolio_value > 0:
            leverage = total_position_value / portfolio_value
            if leverage > self.limits.max_leverage:
                violations.append(f"Leverage {leverage:.2f} exceeds limit {self.limits.max_leverage:.2f}")
        
        return len(violations) == 0, violations
    
    def update_portfolio_history(self, portfolio_value: float):
        """Update portfolio history for risk calculations"""
        self.portfolio_history.append(portfolio_value)
        
        # Calculate daily return
        if len(self.portfolio_history) > 1:
            daily_return = (portfolio_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
            self.daily_returns.append(daily_return)
        
        # Keep only last 252 trading days (1 year)
        if len(self.portfolio_history) > 252:
            self.portfolio_history = self.portfolio_history[-252:]
            self.daily_returns = self.daily_returns[-252:]
    
    def calculate_risk_metrics(self, portfolio_value: float, positions: Dict, benchmark_returns: List[float] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Value at Risk
        var = self.calculate_var(self.daily_returns)
        
        # Maximum Drawdown
        max_dd = self.calculate_max_drawdown(self.portfolio_history)
        
        # Sharpe Ratio
        sharpe = self.calculate_sharpe_ratio(self.daily_returns)
        
        # Volatility
        volatility = np.std(self.daily_returns) * np.sqrt(252) if self.daily_returns else 0.0
        
        # Beta (relative to benchmark)
        beta = 1.0  # Default
        if benchmark_returns and len(self.daily_returns) == len(benchmark_returns):
            if np.std(benchmark_returns) != 0:
                covariance = np.cov(self.daily_returns, benchmark_returns)[0][1]
                beta = covariance / np.var(benchmark_returns)
        
        # Position Concentration
        total_value = sum(pos['quantity'] * pos['current_price'] for pos in positions.values())
        max_concentration = 0.0
        for pos in positions.values():
            position_value = pos['quantity'] * pos['current_price']
            concentration = position_value / total_value if total_value > 0 else 0
            max_concentration = max(max_concentration, concentration)
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            daily_var=var,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            volatility=volatility,
            beta=beta,
            position_concentration=max_concentration
        )
    
    def should_stop_trading(self, portfolio_value: float, positions: Dict) -> Tuple[bool, str]:
        """Determine if trading should be stopped due to risk limits"""
        within_limits, violations = self.check_risk_limits(portfolio_value, positions)
        
        if not within_limits:
            return True, f"Risk limit violations: {', '.join(violations)}"
        
        # Check for extreme market conditions
        if len(self.daily_returns) >= 5:
            recent_volatility = np.std(self.daily_returns[-5:])
            if recent_volatility > 0.05:  # 5% daily volatility threshold
                return True, "Extreme market volatility detected"
        
        return False, ""
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol (placeholder)"""
        # This would integrate with actual market data
        return 100.0  # Placeholder
    
    def generate_risk_report(self, portfolio_value: float, positions: Dict) -> Dict:
        """Generate comprehensive risk report"""
        metrics = self.calculate_risk_metrics(portfolio_value, positions)
        within_limits, violations = self.check_risk_limits(portfolio_value, positions)
        
        return {
            'timestamp': datetime.now(),
            'risk_metrics': {
                'portfolio_value': metrics.portfolio_value,
                'daily_var_95': metrics.daily_var,
                'max_drawdown': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'annual_volatility': metrics.volatility,
                'beta': metrics.beta,
                'max_position_concentration': metrics.position_concentration
            },
            'risk_limits': {
                'max_position_size': self.limits.max_position_size,
                'max_daily_loss': self.limits.max_daily_loss,
                'max_drawdown': self.limits.max_drawdown,
                'max_leverage': self.limits.max_leverage
            },
            'violations': violations,
            'within_limits': within_limits,
            'recommendations': self._generate_recommendations(metrics, violations)
        }
    
    def _generate_recommendations(self, metrics: RiskMetrics, violations: List[str]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if metrics.max_drawdown > 0.1:
            recommendations.append("Consider reducing position sizes due to high drawdown")
        
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Portfolio risk-adjusted returns are low, consider strategy review")
        
        if metrics.position_concentration > 0.3:
            recommendations.append("Diversify portfolio to reduce concentration risk")
        
        if metrics.volatility > 0.3:
            recommendations.append("High portfolio volatility detected, consider hedging")
        
        if violations:
            recommendations.append("Immediate action required: Address risk limit violations")
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    risk_limits = RiskLimits(
        max_position_size=0.1,  # 10% per position
        max_sector_exposure=0.3,  # 30% per sector
        max_daily_loss=0.05,  # 5% daily loss
        max_drawdown=0.2,  # 20% max drawdown
        min_liquidity_ratio=0.1,  # 10% cash minimum
        max_leverage=2.0,  # 2x leverage max
        var_confidence=0.95,  # 95% VaR
        var_timeframe=1  # 1-day VaR
    )
    
    risk_manager = RiskManager(risk_limits)
    
    # Simulate some portfolio history
    portfolio_values = [100000, 102000, 101000, 103000, 99000, 97000]
    for value in portfolio_values:
        risk_manager.update_portfolio_history(value)
    
    # Check risk limits
    positions = {
        'AAPL': {'quantity': 100, 'current_price': 150},
        'GOOGL': {'quantity': 50, 'current_price': 2500}
    }
    
    within_limits, violations = risk_manager.check_risk_limits(97000, positions)
    print(f"Within limits: {within_limits}")
    print(f"Violations: {violations}")
    
    # Generate risk report
    report = risk_manager.generate_risk_report(97000, positions)
    print(f"Risk Report: {report}")
