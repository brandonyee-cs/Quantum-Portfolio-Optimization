import numpy as np
from typing import Dict, List, Tuple, Optional

class QuboFormulation:
    """
    Quadratic Unconstrained Binary Optimization (QUBO) formulation
    for portfolio optimization problems.
    """
    
    def __init__(self, 
                 covariance_matrix: np.ndarray, 
                 expected_returns: np.ndarray, 
                 target_return: float,
                 penalty_coefficients: Dict[str, float] = None):
        """
        Initialize QUBO formulation with problem parameters.
        
        Args:
            covariance_matrix: N x N covariance matrix of asset returns
            expected_returns: Vector of expected returns
            target_return: Target portfolio return
            penalty_coefficients: Dictionary of penalty coefficients for constraints
        """
        self.covariance_matrix = covariance_matrix
        self.expected_returns = expected_returns
        self.target_return = target_return
        self.n_assets = len(expected_returns)
        
        # Default penalty coefficients
        self.penalty_coefficients = {
            'return': 10.0,
            'budget': 10.0,
            'cardinality': 5.0
        }
        
        if penalty_coefficients:
            self.penalty_coefficients.update(penalty_coefficients)
            
    def encode_weights(self, n_bits: int) -> np.ndarray:
        """
        Create binary encoding scheme for portfolio weights.
        
        Args:
            n_bits: Number of bits to use for encoding each weight
            
        Returns:
            Binary encoding matrix
        """
        self.n_bits = n_bits
        self.n_variables = self.n_assets * n_bits
        encoding_matrix = np.zeros((self.n_assets, self.n_variables))
        
        for i in range(self.n_assets):
            for j in range(n_bits):
                encoding_matrix[i, i*n_bits + j] = 2**j
                
        return encoding_matrix
    
    def build_qubo_matrix(self, 
                          encoding_matrix: np.ndarray,
                          cardinality_constraint: Optional[int] = None) -> np.ndarray:
        """
        Build the QUBO matrix Q for the portfolio optimization problem.
        
        Args:
            encoding_matrix: Binary encoding matrix for weights
            cardinality_constraint: Maximum number of assets to include
            
        Returns:
            QUBO matrix Q
        """
        # Initialize Q matrix
        n_variables = encoding_matrix.shape[1]
        Q = np.zeros((n_variables, n_variables))
        
        # Risk term: w^T Σ w
        for i in range(n_variables):
            for j in range(n_variables):
                sum_val = 0
                for k in range(self.n_assets):
                    for l in range(self.n_assets):
                        sum_val += encoding_matrix[k, i] * self.covariance_matrix[k, l] * encoding_matrix[l, j]
                Q[i, j] += sum_val
        
        # Return constraint penalty: max(0, r_target - w^T μ)^2
        A = self.penalty_coefficients['return']
        return_vector = np.zeros(n_variables)
        for i in range(n_variables):
            for k in range(self.n_assets):
                return_vector[i] += encoding_matrix[k, i] * self.expected_returns[k]
        
        for i in range(n_variables):
            for j in range(n_variables):
                Q[i, j] -= A * return_vector[i] * return_vector[j]
            Q[i, i] += 2 * A * self.target_return * return_vector[i]
        
        # Budget constraint penalty: (w^T 1 - 1)^2
        B = self.penalty_coefficients['budget']
        budget_vector = np.zeros(n_variables)
        for i in range(n_variables):
            for k in range(self.n_assets):
                budget_vector[i] += encoding_matrix[k, i]
        
        for i in range(n_variables):
            for j in range(n_variables):
                Q[i, j] += B * budget_vector[i] * budget_vector[j]
            Q[i, i] -= 2 * B * budget_vector[i]
        
        # Cardinality constraint penalty if specified
        if cardinality_constraint is not None:
            C = self.penalty_coefficients['cardinality']
            asset_indicators = np.zeros((self.n_assets, n_variables))
            
            # Identify variables corresponding to each asset
            for i in range(self.n_assets):
                for j in range(self.n_bits):
                    asset_indicators[i, i*self.n_bits + j] = 1
            
            # Add penalty for exceeding cardinality constraint
            for i in range(n_variables):
                for j in range(n_variables):
                    for k in range(self.n_assets):
                        for l in range(self.n_assets):
                            if k != l:  # Only penalize cross-asset terms
                                Q[i, j] += C * asset_indicators[k, i] * asset_indicators[l, j]
        
        return Q
    
    def extract_portfolio_weights(self, binary_solution: np.ndarray) -> np.ndarray:
        """
        Extract portfolio weights from binary solution vector.
        
        Args:
            binary_solution: Binary solution vector
            
        Returns:
            Portfolio weights
        """
        weights = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            for j in range(self.n_bits):
                if binary_solution[i*self.n_bits + j] == 1:
                    weights[i] += 2**j
        
        # Normalize weights to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            
        return weights
    
    def evaluate_portfolio(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Evaluate portfolio performance metrics.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Calculate expected return
        expected_return = np.dot(weights, self.expected_returns)
        
        # Calculate risk (variance)
        risk = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = expected_return / np.sqrt(risk) if risk > 0 else 0
        
        # Count number of assets with non-zero allocation
        num_assets = np.sum(weights > 1e-6)
        
        return {
            'expected_return': expected_return,
            'risk': risk,
            'sharpe_ratio': sharpe_ratio,
            'num_assets': num_assets
        }