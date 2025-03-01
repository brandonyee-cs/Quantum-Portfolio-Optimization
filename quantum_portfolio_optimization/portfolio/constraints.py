import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import scipy.optimize as optimize

class PortfolioConstraints:
    """
    Portfolio constraint implementations for optimization problems.
    """
    
    @staticmethod
    def budget_constraint(weights: np.ndarray) -> float:
        """
        Budget constraint: weights sum to 1.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Constraint violation (0 if satisfied)
        """
        return np.sum(weights) - 1
    
    @staticmethod
    def return_constraint(weights: np.ndarray, 
                         expected_returns: np.ndarray, 
                         target_return: float) -> float:
        """
        Return constraint: portfolio return equals target.
        
        Args:
            weights: Portfolio weights
            expected_returns: Vector of expected returns
            target_return: Target portfolio return
            
        Returns:
            Constraint violation (0 if satisfied)
        """
        return np.dot(weights, expected_returns) - target_return
    
    @staticmethod
    def cardinality_constraint(weights: np.ndarray, 
                              max_assets: int,
                              threshold: float = 1e-6) -> float:
        """
        Cardinality constraint: limit number of assets in portfolio.
        
        Args:
            weights: Portfolio weights
            max_assets: Maximum number of assets to include
            threshold: Threshold for considering weight as non-zero
            
        Returns:
            Constraint violation (0 if satisfied, positive otherwise)
        """
        num_assets = np.sum(weights > threshold)
        return max(0, num_assets - max_assets)
    
    @staticmethod
    def minimum_investment_constraint(weights: np.ndarray, 
                                     min_values: np.ndarray,
                                     threshold: float = 1e-6) -> np.ndarray:
        """
        Minimum investment constraint: minimum allocation for selected assets.
        
        Args:
            weights: Portfolio weights
            min_values: Minimum weight values for each asset
            threshold: Threshold for considering weight as selected
            
        Returns:
            Array of constraint violations (0 if satisfied)
        """
        violations = np.zeros_like(weights)
        for i in range(len(weights)):
            if weights[i] > threshold and weights[i] < min_values[i]:
                violations[i] = min_values[i] - weights[i]
        return violations
    
    @staticmethod
    def maximum_investment_constraint(weights: np.ndarray, 
                                     max_values: np.ndarray) -> np.ndarray:
        """
        Maximum investment constraint: maximum allocation for each asset.
        
        Args:
            weights: Portfolio weights
            max_values: Maximum weight values for each asset
            
        Returns:
            Array of constraint violations (0 if satisfied)
        """
        return np.maximum(0, weights - max_values)
    
    @staticmethod
    def sector_constraint(weights: np.ndarray, 
                         sector_mapper: Dict[int, int],
                         min_sector: Dict[int, float] = None,
                         max_sector: Dict[int, float] = None) -> np.ndarray:
        """
        Sector constraint: limit exposure to specific sectors.
        
        Args:
            weights: Portfolio weights
            sector_mapper: Mapping from asset index to sector index
            min_sector: Minimum allocation per sector
            max_sector: Maximum allocation per sector
            
        Returns:
            Array of constraint violations (0 if satisfied)
        """
        n_assets = len(weights)
        
        # Calculate sector weights
        unique_sectors = set(sector_mapper.values())
        sector_weights = {sector: 0.0 for sector in unique_sectors}
        
        for i in range(n_assets):
            sector = sector_mapper.get(i)
            if sector is not None:
                sector_weights[sector] += weights[i]
        
        # Check constraints
        violations = []
        
        if min_sector is not None:
            for sector, min_weight in min_sector.items():
                if sector in sector_weights and sector_weights[sector] < min_weight:
                    violations.append(min_weight - sector_weights[sector])
        
        if max_sector is not None:
            for sector, max_weight in max_sector.items():
                if sector in sector_weights and sector_weights[sector] > max_weight:
                    violations.append(sector_weights[sector] - max_weight)
        
        return np.array(violations)
    
    @staticmethod
    def get_inequality_constraints(covariance_matrix: np.ndarray,
                                 expected_returns: np.ndarray,
                                 max_assets: Optional[int] = None,
                                 min_weights: Optional[np.ndarray] = None,
                                 max_weights: Optional[np.ndarray] = None,
                                 sector_mapper: Optional[Dict[int, int]] = None,
                                 min_sector: Optional[Dict[int, float]] = None,
                                 max_sector: Optional[Dict[int, float]] = None) -> List[Dict]:
        """
        Generate inequality constraints for scipy.optimize.
        
        Args:
            covariance_matrix: N x N covariance matrix of asset returns
            expected_returns: Vector of expected returns
            max_assets: Maximum number of assets to include
            min_weights: Minimum weight values for each asset
            max_weights: Maximum weight values for each asset
            sector_mapper: Mapping from asset index to sector index
            min_sector: Minimum allocation per sector
            max_sector: Maximum allocation per sector
            
        Returns:
            List of inequality constraints for scipy.optimize
        """
        n_assets = len(expected_returns)
        constraints = []
        
        # Non-negative weights constraint is handled by bounds
        
        # Maximum number of assets constraint
        if max_assets is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_assets - np.sum(x > 1e-6)
            })
        
        # Minimum investment constraint
        if min_weights is not None:
            for i in range(n_assets):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[i] - min_weights[i] if x[i] > 1e-6 else 0
                })
        
        # Maximum investment constraint
        if max_weights is not None:
            for i in range(n_assets):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: max_weights[i] - x[i]
                })
        
        # Sector constraints
        if sector_mapper is not None:
            unique_sectors = set(sector_mapper.values())
            
            # Minimum sector allocation
            if min_sector is not None:
                for sector, min_weight in min_sector.items():
                    if sector in unique_sectors:
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda x, s=sector: sum(x[i] for i in range(n_assets) 
                                                         if sector_mapper.get(i) == s) - min_weight
                        })
            
            # Maximum sector allocation
            if max_sector is not None:
                for sector, max_weight in max_sector.items():
                    if sector in unique_sectors:
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda x, s=sector: max_weight - sum(x[i] for i in range(n_assets) 
                                                                      if sector_mapper.get(i) == s)
                        })
        
        return constraints