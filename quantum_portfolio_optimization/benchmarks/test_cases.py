import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from utilities.data_preparation import DataPreparation

class TestCases:
    """
    Standard test portfolios for benchmarking.
    """
    
    @staticmethod
    def generate_random_portfolio(n_assets: int, 
                                 seed: Optional[int] = None) -> Dict:
        """
        Generate random portfolio test case.
        
        Args:
            n_assets: Number of assets
            seed: Random seed
            
        Returns:
            Dictionary with test case data
        """
        # Generate random data
        data = DataPreparation.generate_random_data(n_assets, 252, seed)
        
        # Add metadata
        data['name'] = f'Random Portfolio ({n_assets} assets)'
        data['description'] = f'Randomly generated portfolio with {n_assets} assets'
        data['n_assets'] = n_assets
        
        return data
    
    @staticmethod
    def balanced_portfolio() -> Dict:
        """
        Generate balanced portfolio test case.
        
        Returns:
            Dictionary with test case data
        """
        n_assets = 5
        
        # Set expected returns
        expected_returns = np.array([0.08, 0.10, 0.12, 0.07, 0.15])
        
        # Set correlation matrix
        correlation = np.array([
            [1.00, 0.25, 0.18, 0.10, 0.25],
            [0.25, 1.00, 0.36, 0.20, 0.15],
            [0.18, 0.36, 1.00, 0.30, 0.38],
            [0.10, 0.20, 0.30, 1.00, 0.30],
            [0.25, 0.15, 0.38, 0.30, 1.00]
        ])
        
        # Set volatilities
        volatilities = np.array([0.15, 0.20, 0.25, 0.10, 0.35])
        
        # Calculate covariance matrix
        covariance_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                covariance_matrix[i, j] = correlation[i, j] * volatilities[i] * volatilities[j]
        
        # Create asset names
        asset_names = ['Stock', 'Bond', 'Real Estate', 'Gold', 'Crypto']
        
        return {
            'name': 'Balanced Portfolio',
            'description': 'Diversified portfolio with stocks, bonds, real estate, gold, and cryptocurrency',
            'n_assets': n_assets,
            'asset_names': asset_names,
            'expected_returns': pd.Series(expected_returns, index=asset_names),
            'covariance_matrix': pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names),
            'volatilities': pd.Series(volatilities, index=asset_names)
        }
    
    @staticmethod
    def sector_portfolio() -> Dict:
        """
        Generate sector-based portfolio test case.
        
        Returns:
            Dictionary with test case data
        """
        n_assets = 10
        
        # Define sectors
        sectors = ['Technology', 'Technology', 'Technology', 'Finance', 'Finance', 
                  'Healthcare', 'Healthcare', 'Energy', 'Energy', 'Consumer']
        
        # Set asset names
        asset_names = ['Tech1', 'Tech2', 'Tech3', 'Fin1', 'Fin2', 
                      'Health1', 'Health2', 'Energy1', 'Energy2', 'Cons1']
        
        # Create sector mapping
        sector_mapping = {i: sectors.index(sectors[i]) for i in range(n_assets)}
        
        # Set expected returns
        expected_returns = np.array([0.15, 0.18, 0.12, 0.10, 0.08, 0.14, 0.11, 0.09, 0.07, 0.13])
        
        # Set volatilities
        volatilities = np.array([0.30, 0.35, 0.25, 0.20, 0.18, 0.22, 0.19, 0.25, 0.22, 0.24])
        
        # Set correlation matrix (higher within sectors)
        correlation = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                if sectors[i] == sectors[j]:
                    # Within sector correlation (higher)
                    correlation[i, j] = 0.75
                else:
                    # Across sector correlation (lower)
                    correlation[i, j] = 0.20
                
                correlation[j, i] = correlation[i, j]
        
        # Calculate covariance matrix
        covariance_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                covariance_matrix[i, j] = correlation[i, j] * volatilities[i] * volatilities[j]
        
        # Add sector constraints
        sector_constraints = {
            'min_sector': {0: 0.20, 1: 0.15, 2: 0.10, 3: 0.05, 4: 0.05},  # Min allocation per sector
            'max_sector': {0: 0.40, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15}   # Max allocation per sector
        }
        
        return {
            'name': 'Sector Portfolio',
            'description': 'Portfolio with sector allocation constraints',
            'n_assets': n_assets,
            'asset_names': asset_names,
            'sectors': sectors,
            'sector_mapping': sector_mapping,
            'expected_returns': pd.Series(expected_returns, index=asset_names),
            'covariance_matrix': pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names),
            'volatilities': pd.Series(volatilities, index=asset_names),
            'sector_constraints': sector_constraints
        }
    
    @staticmethod
    def high_dimensional_portfolio(n_assets: int = 50, 
                                  seed: Optional[int] = None) -> Dict:
        """
        Generate high-dimensional portfolio test case.
        
        Args:
            n_assets: Number of assets
            seed: Random seed
            
        Returns:
            Dictionary with test case data
        """
        # Generate random data
        data = DataPreparation.generate_random_data(n_assets, 252, seed)
        
        # Add metadata
        data['name'] = f'High-Dimensional Portfolio ({n_assets} assets)'
        data['description'] = f'High-dimensional portfolio optimization test case with {n_assets} assets'
        data['n_assets'] = n_assets
        
        # Add cardinality constraint
        data['cardinality_constraint'] = 10  # Select only 10 assets
        
        return data