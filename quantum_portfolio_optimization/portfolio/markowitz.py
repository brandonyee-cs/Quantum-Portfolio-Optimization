import numpy as np
from typing import Dict, List, Tuple, Optional
import scipy.optimize as optimize

class MarkowitzOptimizer:
    """
    Classical Markowitz mean-variance portfolio optimization.
    """
    
    def __init__(self, 
                 covariance_matrix: np.ndarray, 
                 expected_returns: np.ndarray):
        """
        Initialize Markowitz optimizer.
        
        Args:
            covariance_matrix: N x N covariance matrix of asset returns
            expected_returns: Vector of expected returns
        """
        self.covariance_matrix = covariance_matrix
        self.expected_returns = expected_returns
        self.n_assets = len(expected_returns)
        
    def optimize(self, 
                target_return: Optional[float] = None,
                allow_short_selling: bool = False) -> Dict:
        """
        Perform Markowitz mean-variance optimization.
        
        Args:
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            allow_short_selling: Whether to allow short selling
            
        Returns:
            Dictionary with optimization results
        """
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Budget constraint: weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append(
                {'type': 'eq', 'fun': lambda x: np.dot(x, self.expected_returns) - target_return}
            )
        
        # Define bounds
        if allow_short_selling:
            bounds = None
        else:
            bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Define objective function (portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        
        # Initial guess (equal weights)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Run optimization
        result = optimize.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, self.expected_returns)
        portfolio_risk = np.sqrt(objective(optimal_weights))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'optimal_weights': optimal_weights,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success,
            'message': result.message
        }
    
    def optimize_sharpe(self, 
                       risk_free_rate: float = 0.0,
                       allow_short_selling: bool = False) -> Dict:
        """
        Maximize Sharpe ratio.
        
        Args:
            risk_free_rate: Risk-free rate
            allow_short_selling: Whether to allow short selling
            
        Returns:
            Dictionary with optimization results
        """
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Budget constraint: weights sum to 1
        ]
        
        # Define bounds
        if allow_short_selling:
            bounds = None
        else:
            bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            return -(portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Initial guess (equal weights)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Run optimization
        result = optimize.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, self.expected_returns)
        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(self.covariance_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'optimal_weights': optimal_weights,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success,
            'message': result.message
        }
    
    def efficient_frontier(self, 
                          n_points: int = 20,
                          allow_short_selling: bool = False) -> Dict:
        """
        Calculate the efficient frontier.
        
        Args:
            n_points: Number of points on the efficient frontier
            allow_short_selling: Whether to allow short selling
            
        Returns:
            Dictionary with efficient frontier points
        """
        # Find minimum and maximum returns
        min_return = np.min(self.expected_returns)
        max_return = np.max(self.expected_returns)
        
        # Create range of target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        # Calculate optimal portfolios for each target return
        frontier_returns = []
        frontier_risks = []
        frontier_weights = []
        
        for target_return in target_returns:
            result = self.optimize(target_return, allow_short_selling)
            
            if result['success']:
                frontier_returns.append(result['portfolio_return'])
                frontier_risks.append(result['portfolio_risk'])
                frontier_weights.append(result['optimal_weights'])
        
        return {
            'frontier_returns': frontier_returns,
            'frontier_risks': frontier_risks,
            'frontier_weights': frontier_weights
        }