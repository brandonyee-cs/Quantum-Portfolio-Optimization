import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats

class RiskMeasures:
    """
    Risk measurement functions for portfolio optimization.
    """
    
    @staticmethod
    def portfolio_variance(weights: np.ndarray, 
                          covariance_matrix: np.ndarray) -> float:
        """
        Calculate portfolio variance.
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix of asset returns
            
        Returns:
            Portfolio variance
        """
        return np.dot(weights.T, np.dot(covariance_matrix, weights))
    
    @staticmethod
    def portfolio_volatility(weights: np.ndarray, 
                            covariance_matrix: np.ndarray) -> float:
        """
        Calculate portfolio volatility (standard deviation).
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix of asset returns
            
        Returns:
            Portfolio volatility
        """
        return np.sqrt(RiskMeasures.portfolio_variance(weights, covariance_matrix))
    
    @staticmethod
    def sharpe_ratio(weights: np.ndarray, 
                    expected_returns: np.ndarray, 
                    covariance_matrix: np.ndarray,
                    risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            weights: Portfolio weights
            expected_returns: Vector of expected returns
            covariance_matrix: Covariance matrix of asset returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = RiskMeasures.portfolio_volatility(weights, covariance_matrix)
        
        return (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    @staticmethod
    def value_at_risk(weights: np.ndarray, 
                     returns_data: np.ndarray, 
                     confidence_level: float = 0.95) -> float:
        """
        Calculate Value-at-Risk (VaR) using historical method.
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns data (assets in columns)
            confidence_level: Confidence level for VaR
            
        Returns:
            Value-at-Risk
        """
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_data, weights)
        
        # Calculate VaR
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        return var
    
    @staticmethod
    def conditional_value_at_risk(weights: np.ndarray, 
                                 returns_data: np.ndarray, 
                                 confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value-at-Risk (CVaR) using historical method.
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns data (assets in columns)
            confidence_level: Confidence level for CVaR
            
        Returns:
            Conditional Value-at-Risk
        """
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_data, weights)
        
        # Calculate VaR
        var = RiskMeasures.value_at_risk(weights, returns_data, confidence_level)
        
        # Calculate CVaR
        cvar = -np.mean(portfolio_returns[portfolio_returns <= -var])
        
        return cvar
    
    @staticmethod
    def drawdown(weights: np.ndarray, 
                returns_data: np.ndarray) -> Dict:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns data (assets in columns)
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_data, weights)
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown_values = (cumulative_returns - running_max) / running_max
        
        # Calculate maximum drawdown
        max_drawdown = np.min(drawdown_values)
        max_drawdown_idx = np.argmin(drawdown_values)
        
        # Find peak and valley
        peak_idx = np.argmax(cumulative_returns[:max_drawdown_idx])
        
        return {
            'max_drawdown': max_drawdown,
            'drawdown_series': drawdown_values,
            'peak_idx': peak_idx,
            'valley_idx': max_drawdown_idx
        }
    
    @staticmethod
    def sortino_ratio(weights: np.ndarray, 
                     returns_data: np.ndarray,
                     expected_returns: np.ndarray,
                     risk_free_rate: float = 0.0,
                     target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns data (assets in columns)
            expected_returns: Vector of expected returns
            risk_free_rate: Risk-free rate
            target_return: Target return for downside deviation calculation
            
        Returns:
            Sortino ratio
        """
        # Calculate portfolio return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_data, weights)
        
        # Calculate downside returns
        downside_returns = portfolio_returns[portfolio_returns < target_return]
        
        # Calculate downside deviation
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
        
        # Calculate Sortino ratio
        sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        return sortino_ratio