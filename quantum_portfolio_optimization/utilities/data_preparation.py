import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
import datetime as dt

class DataPreparation:
    """
    Financial data preparation utilities.
    """
    
    @staticmethod
    def calculate_returns(prices: pd.DataFrame,
                         method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: DataFrame of asset prices
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            DataFrame of returns
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        elif method == 'simple':
            returns = prices.pct_change()
        else:
            raise ValueError(f"Unknown return calculation method: {method}")
        
        return returns.dropna()
    
    @staticmethod
    def calculate_covariance_matrix(returns: pd.DataFrame,
                                   annualization_factor: float = 252) -> pd.DataFrame:
        """
        Calculate covariance matrix from returns data.
        
        Args:
            returns: DataFrame of asset returns
            annualization_factor: Annualization factor (252 for daily data)
            
        Returns:
            Covariance matrix
        """
        return returns.cov() * annualization_factor
    
    @staticmethod
    def calculate_expected_returns(returns: pd.DataFrame,
                                  method: str = 'mean',
                                  annualization_factor: float = 252) -> pd.Series:
        """
        Calculate expected returns.
        
        Args:
            returns: DataFrame of asset returns
            method: Expected return calculation method ('mean', 'capm', 'black_litterman')
            annualization_factor: Annualization factor (252 for daily data)
            
        Returns:
            Series of expected returns
        """
        if method == 'mean':
            # Simple mean return
            expected_returns = returns.mean() * annualization_factor
            
        elif method == 'ewma':
            # Exponentially weighted moving average
            expected_returns = returns.ewm(span=60).mean().iloc[-1] * annualization_factor
            
        else:
            raise ValueError(f"Unknown expected return calculation method: {method}")
        
        return expected_returns
    
    @staticmethod
    def clean_data(data: pd.DataFrame,
                  fill_method: str = 'ffill',
                  handle_outliers: bool = True,
                  outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        Clean financial data.
        
        Args:
            data: DataFrame of financial data
            fill_method: Method for filling missing values ('ffill', 'bfill', 'interpolate')
            handle_outliers: Whether to handle outliers
            outlier_threshold: Threshold for outlier detection (number of standard deviations)
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        cleaned_data = data.copy()
        
        # Fill missing values
        if fill_method == 'ffill':
            cleaned_data = cleaned_data.fillna(method='ffill')
        elif fill_method == 'bfill':
            cleaned_data = cleaned_data.fillna(method='bfill')
        elif fill_method == 'interpolate':
            cleaned_data = cleaned_data.interpolate()
        else:
            raise ValueError(f"Unknown fill method: {fill_method}")
        
        # Handle outliers
        if handle_outliers:
            for column in cleaned_data.columns:
                series = cleaned_data[column]
                mean = series.mean()
                std = series.std()
                
                # Replace outliers with NaN
                outliers = (series - mean).abs() > outlier_threshold * std
                cleaned_data.loc[outliers, column] = np.nan
                
                # Fill outliers
                if fill_method == 'ffill':
                    cleaned_data[column] = cleaned_data[column].fillna(method='ffill')
                elif fill_method == 'bfill':
                    cleaned_data[column] = cleaned_data[column].fillna(method='bfill')
                elif fill_method == 'interpolate':
                    cleaned_data[column] = cleaned_data[column].interpolate()
        
        return cleaned_data
    
    @staticmethod
    def generate_random_data(n_assets: int,
                            n_periods: int,
                            seed: Optional[int] = None) -> Dict:
        """
        Generate random financial data for testing.
        
        Args:
            n_assets: Number of assets
            n_periods: Number of time periods
            seed: Random seed
            
        Returns:
            Dictionary with generated data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random correlation matrix
        A = np.random.normal(0, 1, (n_assets, n_assets))
        correlation = np.corrcoef(A)
        
        # Generate random volatilities
        volatilities = np.random.uniform(0.1, 0.5, n_assets)
        
        # Generate covariance matrix
        covariance = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                covariance[i, j] = correlation[i, j] * volatilities[i] * volatilities[j]
        
        # Generate expected returns
        expected_returns = np.random.uniform(0.03, 0.15, n_assets)
        
        # Generate price data
        start_date = dt.datetime(2020, 1, 1)
        dates = [start_date + dt.timedelta(days=i) for i in range(n_periods)]
        
        # Generate return data from multivariate normal distribution
        daily_returns = np.random.multivariate_normal(
            expected_returns / 252,  # Daily expected returns
            covariance / 252,  # Daily covariance
            n_periods
        )
        
        # Convert to prices
        prices = np.cumprod(1 + daily_returns, axis=0)
        
        # Create price DataFrame
        price_data = pd.DataFrame(
            prices,
            index=dates,
            columns=[f'Asset_{i+1}' for i in range(n_assets)]
        )
        
        return {
            'price_data': price_data,
            'covariance_matrix': pd.DataFrame(
                covariance,
                index=[f'Asset_{i+1}' for i in range(n_assets)],
                columns=[f'Asset_{i+1}' for i in range(n_assets)]
            ),
            'expected_returns': pd.Series(
                expected_returns,
                index=[f'Asset_{i+1}' for i in range(n_assets)]
            ),
            'volatilities': pd.Series(
                volatilities,
                index=[f'Asset_{i+1}' for i in range(n_assets)]
            )
        }