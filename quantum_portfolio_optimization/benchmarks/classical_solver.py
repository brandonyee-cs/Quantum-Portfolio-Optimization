import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import scipy.optimize as optimize
import time
import cvxpy as cp

class ClassicalPortfolioSolvers:
    """
    Classical optimization methods for portfolio optimization.
    """
    
    @staticmethod
    def solve_markowitz(covariance_matrix: np.ndarray, 
                       expected_returns: np.ndarray,
                       target_return: Optional[float] = None,
                       allow_short_selling: bool = False) -> Dict:
        """
        Solve Markowitz portfolio optimization using quadratic programming.
        
        Args:
            covariance_matrix: Covariance matrix
            expected_returns: Expected returns
            target_return: Target return (if None, maximize Sharpe ratio)
            allow_short_selling: Whether to allow short selling
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        n_assets = len(expected_returns)
        
        # Define variables
        w = cp.Variable(n_assets)
        
        # Define objective (minimize risk)
        risk = cp.quad_form(w, covariance_matrix)
        objective = cp.Minimize(risk)
        
        # Define constraints
        constraints = [cp.sum(w) == 1]  # Budget constraint
        
        if target_return is not None:
            constraints.append(expected_returns @ w >= target_return)  # Return constraint
        
        if not allow_short_selling:
            constraints.append(w >= 0)  # No short selling
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Extract results
        if problem.status == 'optimal':
            weights = w.value
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            result = {
                'status': 'optimal',
                'weights': weights,
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
                'computation_time': time.time() - start_time
            }
        else:
            result = {
                'status': problem.status,
                'weights': None,
                'portfolio_return': None,
                'portfolio_risk': None,
                'sharpe_ratio': None,
                'computation_time': time.time() - start_time
            }
        
        return result
    
    @staticmethod
    def solve_max_sharpe(covariance_matrix: np.ndarray, 
                        expected_returns: np.ndarray,
                        risk_free_rate: float = 0.0,
                        allow_short_selling: bool = False) -> Dict:
        """
        Solve for maximum Sharpe ratio portfolio.
        
        Args:
            covariance_matrix: Covariance matrix
            expected_returns: Expected returns
            risk_free_rate: Risk-free rate
            allow_short_selling: Whether to allow short selling
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        n_assets = len(expected_returns)
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            return -sharpe  # Negative because we're minimizing
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Budget constraint
        ]
        
        # Define bounds
        if allow_short_selling:
            bounds = None
        else:
            bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Solve problem
        result = optimize.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract results
        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            result_dict = {
                'status': 'optimal',
                'weights': weights,
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'computation_time': time.time() - start_time
            }
        else:
            result_dict = {
                'status': 'failed',
                'weights': None,
                'portfolio_return': None,
                'portfolio_risk': None,
                'sharpe_ratio': None,
                'computation_time': time.time() - start_time,
                'message': result.message
            }
        
        return result_dict
    
    @staticmethod
    def solve_with_cardinality_constraint(covariance_matrix: np.ndarray, 
                                         expected_returns: np.ndarray,
                                         target_return: float,
                                         max_assets: int) -> Dict:
        """
        Solve portfolio optimization with cardinality constraint.
        
        Args:
            covariance_matrix: Covariance matrix
            expected_returns: Expected returns
            target_return: Target return
            max_assets: Maximum number of assets to include
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        n_assets = len(expected_returns)
        
        # Approach: Use heuristic (greedy algorithm with local search)
        
        # Sort assets by Sharpe ratio
        sharpe_ratios = expected_returns / np.sqrt(np.diag(covariance_matrix))
        sorted_indices = np.argsort(sharpe_ratios)[::-1]
        
        # Start with the top max_assets assets
        selected_indices = sorted_indices[:max_assets]
        
        # Local optimization
        def optimize_selected(indices):
            # Extract submatrices
            sub_cov = covariance_matrix[np.ix_(indices, indices)]
            sub_ret = expected_returns[indices]
            
            # Solve Markowitz for selected assets
            try:
                # Define variables
                w_sub = cp.Variable(len(indices))
                
                # Define objective
                risk = cp.quad_form(w_sub, sub_cov)
                objective = cp.Minimize(risk)
                
                # Define constraints
                constraints = [
                    cp.sum(w_sub) == 1,
                    w_sub >= 0,
                    sub_ret @ w_sub >= target_return
                ]
                
                # Solve problem
                problem = cp.Problem(objective, constraints)
                problem.solve()
                
                if problem.status == 'optimal':
                    sub_weights = w_sub.value
                    
                    # Convert to full weight vector
                    weights = np.zeros(n_assets)
                    for i, idx in enumerate(indices):
                        weights[idx] = sub_weights[i]
                    
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                    
                    return weights, portfolio_return, portfolio_risk
                else:
                    return None, None, None
            except:
                return None, None, None
        
        # Optimize with initial selection
        weights, portfolio_return, portfolio_risk = optimize_selected(selected_indices)
        
        # Local search iterations
        for _ in range(100):
            # Randomly select an asset to replace
            if weights is None or len(selected_indices) == 0:
                break
                
            asset_to_replace = np.random.choice(selected_indices)
            
            # Try to replace with each unselected asset
            unselected_indices = [i for i in range(n_assets) if i not in selected_indices]
            
            best_risk = float('inf') if portfolio_risk is None else portfolio_risk
            best_weights = weights
            best_return = portfolio_return
            best_indices = selected_indices
            
            for new_asset in unselected_indices:
                new_indices = [i for i in selected_indices if i != asset_to_replace] + [new_asset]
                
                new_weights, new_return, new_risk = optimize_selected(new_indices)
                
                if new_weights is not None and new_return >= target_return and new_risk < best_risk:
                    best_risk = new_risk
                    best_weights = new_weights
                    best_return = new_return
                    best_indices = new_indices
            
            # Update if improvement found
            if best_indices != selected_indices:
                selected_indices = best_indices
                weights = best_weights
                portfolio_return = best_return
                portfolio_risk = best_risk
            else:
                # No improvement found, stop searching
                break
        
        # Return results
        if weights is not None:
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            result = {
                'status': 'optimal',
                'weights': weights,
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'selected_assets': selected_indices.tolist(),
                'computation_time': time.time() - start_time
            }
        else:
            result = {
                'status': 'failed',
                'weights': None,
                'portfolio_return': None,
                'portfolio_risk': None,
                'sharpe_ratio': None,
                'selected_assets': [],
                'computation_time': time.time() - start_time
            }
        
        return result