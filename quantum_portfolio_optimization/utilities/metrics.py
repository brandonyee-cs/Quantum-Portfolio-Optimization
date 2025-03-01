import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from scipy import stats

class PerformanceMetrics:
    """
    Performance metrics calculation utilities.
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray,
                              risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray,
                               risk_free_rate: float = 0.0,
                               target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate
            target_return: Target return
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < target_return]
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
        
        return np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
    
    @staticmethod
    def calculate_maximum_drawdown(returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown
        """
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Calculate maximum drawdown
        max_drawdown = np.min(drawdown)
        
        return max_drawdown
    
    @staticmethod
    def calculate_var(returns: np.ndarray,
                     confidence_level: float = 0.95) -> float:
        """
        Calculate Value-at-Risk (VaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level
            
        Returns:
            Value-at-Risk
        """
        return -np.percentile(returns, 100 * (1 - confidence_level))
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray,
                      confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value-at-Risk (CVaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level
            
        Returns:
            Conditional Value-at-Risk
        """
        var = PerformanceMetrics.calculate_var(returns, confidence_level)
        return -np.mean(returns[returns <= -var])
    
    @staticmethod
    def calculate_information_ratio(returns: np.ndarray,
                                   benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns
            
        Returns:
            Information Ratio
        """
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns)
        
        return np.mean(active_returns) / tracking_error if tracking_error > 0 else 0
    
    @staticmethod
    def calculate_performance_metrics(returns: np.ndarray,
                                     benchmark_returns: Optional[np.ndarray] = None,
                                     risk_free_rate: float = 0.0) -> Dict:
        """
        Calculate various performance metrics.
        
        Args:
            returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['mean_return'] = np.mean(returns)
        metrics['volatility'] = np.std(returns)
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        
        # Risk metrics
        metrics['sharpe_ratio'] = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
        metrics['sortino_ratio'] = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate)
        metrics['maximum_drawdown'] = PerformanceMetrics.calculate_maximum_drawdown(returns)
        metrics['var_95'] = PerformanceMetrics.calculate_var(returns, 0.95)
        metrics['cvar_95'] = PerformanceMetrics.calculate_cvar(returns, 0.95)
        
        # Benchmark-related metrics
        if benchmark_returns is not None:
            metrics['information_ratio'] = PerformanceMetrics.calculate_information_ratio(returns, benchmark_returns)
            metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            metrics['alpha'] = np.mean(returns) - risk_free_rate - metrics['beta'] * (np.mean(benchmark_returns) - risk_free_rate)
            metrics['r_squared'] = np.corrcoef(returns, benchmark_returns)[0, 1] ** 2
        
        return metrics
    
    @staticmethod
    def compare_portfolios(portfolio_a_weights: np.ndarray,
                          portfolio_b_weights: np.ndarray,
                          asset_returns: np.ndarray,
                          asset_names: Optional[List[str]] = None) -> Dict:
        """
        Compare two portfolios.
        
        Args:
            portfolio_a_weights: Weights of portfolio A
            portfolio_b_weights: Weights of portfolio B
            asset_returns: Asset returns data (assets in columns)
            asset_names: Names of assets
            
        Returns:
            Dictionary of comparison metrics
        """
        n_assets = len(portfolio_a_weights)
        
        # Create default asset names if not provided
        if asset_names is None:
            asset_names = [f'Asset {i+1}' for i in range(n_assets)]
        
        # Calculate portfolio returns
        portfolio_a_returns = np.dot(asset_returns, portfolio_a_weights)
        portfolio_b_returns = np.dot(asset_returns, portfolio_b_weights)
        
        # Calculate performance metrics
        portfolio_a_metrics = PerformanceMetrics.calculate_performance_metrics(portfolio_a_returns)
        portfolio_b_metrics = PerformanceMetrics.calculate_performance_metrics(portfolio_b_returns)
        
        # Calculate weight differences
        weight_diff = portfolio_a_weights - portfolio_b_weights
        
        # Create comparison dictionary
        comparison = {
            'weight_difference': {asset_names[i]: weight_diff[i] for i in range(n_assets)},
            'turnover': np.sum(np.abs(weight_diff)) / 2,
            'metrics_a': portfolio_a_metrics,
            'metrics_b': portfolio_b_metrics,
            'return_correlation': np.corrcoef(portfolio_a_returns, portfolio_b_returns)[0, 1]
        }
        
        return comparison