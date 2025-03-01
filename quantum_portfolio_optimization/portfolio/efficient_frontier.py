import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import time

class EfficientFrontier:
    """
    Efficient frontier mapping for portfolio optimization.
    """
    
    def __init__(self, 
                 covariance_matrix: np.ndarray, 
                 expected_returns: np.ndarray,
                 solver: Callable,
                 allow_short_selling: bool = False):
        """
        Initialize efficient frontier mapper.
        
        Args:
            covariance_matrix: N x N covariance matrix of asset returns
            expected_returns: Vector of expected returns
            solver: Optimization solver function
            allow_short_selling: Whether to allow short selling
        """
        self.covariance_matrix = covariance_matrix
        self.expected_returns = expected_returns
        self.n_assets = len(expected_returns)
        self.solver = solver
        self.allow_short_selling = allow_short_selling
        
    def compute_frontier(self, 
                         n_points: int = 20,
                         min_return: Optional[float] = None,
                         max_return: Optional[float] = None,
                         constraints: Optional[Dict] = None) -> Dict:
        """
        Compute the efficient frontier.
        
        Args:
            n_points: Number of points on the efficient frontier
            min_return: Minimum target return (default: min of expected returns)
            max_return: Maximum target return (default: max of expected returns)
            constraints: Additional constraints for the solver
            
        Returns:
            Dictionary with efficient frontier data
        """
        # Determine min and max returns if not provided
        if min_return is None:
            min_return = np.min(self.expected_returns)
        
        if max_return is None:
            max_return = np.max(self.expected_returns)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        # Initialize results
        frontier_returns = []
        frontier_risks = []
        frontier_weights = []
        computation_times = []
        
        # Compute optimal portfolios for each target return
        for target_return in target_returns:
            start_time = time.time()
            
            # Call solver with target return
            result = self.solver(
                self.covariance_matrix,
                self.expected_returns,
                target_return,
                self.allow_short_selling,
                constraints
            )
            
            computation_time = time.time() - start_time
            
            # Store results if optimization was successful
            if 'optimal_weights' in result:
                weights = result['optimal_weights']
                
                # Calculate portfolio return and risk
                portfolio_return = np.dot(weights, self.expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
                
                frontier_returns.append(portfolio_return)
                frontier_risks.append(portfolio_risk)
                frontier_weights.append(weights)
                computation_times.append(computation_time)
        
        return {
            'frontier_returns': frontier_returns,
            'frontier_risks': frontier_risks,
            'frontier_weights': frontier_weights,
            'computation_times': computation_times,
            'target_returns': target_returns
        }
    
    def find_optimal_portfolio(self, 
                              criterion: str = 'sharpe',
                              risk_free_rate: float = 0.0,
                              constraints: Optional[Dict] = None) -> Dict:
        """
        Find the optimal portfolio based on a given criterion.
        
        Args:
            criterion: Optimization criterion ('sharpe', 'min_risk', 'target_risk')
            risk_free_rate: Risk-free rate (for Sharpe ratio)
            constraints: Additional constraints for the solver
            
        Returns:
            Dictionary with optimal portfolio data
        """
        if criterion == 'sharpe':
            # Maximize Sharpe ratio
            
            def objective(weights):
                portfolio_return = np.dot(weights, self.expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
                sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                return -sharpe  # Negative for minimization
            
            # Call solver with Sharpe objective
            result = self.solver(
                self.covariance_matrix,
                self.expected_returns,
                None,  # No target return
                self.allow_short_selling,
                constraints,
                objective
            )
            
        elif criterion == 'min_risk':
            # Minimize risk
            
            # Call solver with minimum return constraint
            min_return = np.min(self.expected_returns)
            result = self.solver(
                self.covariance_matrix,
                self.expected_returns,
                min_return,
                self.allow_short_selling,
                constraints
            )
            
        elif criterion.startswith('target_risk_'):
            # Find portfolio with target risk
            target_risk = float(criterion.split('_')[-1])
            
            # Compute efficient frontier
            frontier = self.compute_frontier(n_points=50, constraints=constraints)
            
            # Find portfolio closest to target risk
            risks = np.array(frontier['frontier_risks'])
            idx = np.argmin(np.abs(risks - target_risk))
            
            result = {
                'optimal_weights': frontier['frontier_weights'][idx],
                'portfolio_return': frontier['frontier_returns'][idx],
                'portfolio_risk': frontier['frontier_risks'][idx]
            }
            
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        # Extract and return optimal portfolio
        weights = result['optimal_weights']
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'optimal_weights': weights,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'criterion': criterion
        }