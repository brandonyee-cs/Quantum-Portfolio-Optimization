import numpy as np
from typing import Dict, List, Tuple, Optional

class BinaryEncoding:
    """
    Binary encoding schemes for portfolio weights.
    """
    
    @staticmethod
    def direct_binary_encoding(n_assets: int, 
                              n_bits: int = 4) -> np.ndarray:
        """
        Direct binary encoding of portfolio weights.
        
        Args:
            n_assets: Number of assets
            n_bits: Number of bits per weight
            
        Returns:
            Binary encoding matrix
        """
        n_variables = n_assets * n_bits
        encoding_matrix = np.zeros((n_assets, n_variables))
        
        for i in range(n_assets):
            for j in range(n_bits):
                encoding_matrix[i, i*n_bits + j] = 2**j
        
        return encoding_matrix
    
    @staticmethod
    def threshold_binary_encoding(n_assets: int) -> np.ndarray:
        """
        Threshold binary encoding for asset selection.
        
        Args:
            n_assets: Number of assets
            
        Returns:
            Binary encoding matrix
        """
        # One binary variable per asset (0 = exclude, 1 = include)
        encoding_matrix = np.eye(n_assets)
        
        return encoding_matrix
    
    @staticmethod
    def one_hot_encoding(n_assets: int, 
                        n_levels: int = 10) -> np.ndarray:
        """
        One-hot encoding for discrete weight levels.
        
        Args:
            n_assets: Number of assets
            n_levels: Number of weight levels per asset
            
        Returns:
            One-hot encoding matrix
        """
        n_variables = n_assets * n_levels
        encoding_matrix = np.zeros((n_assets, n_variables))
        
        for i in range(n_assets):
            for j in range(n_levels):
                encoding_matrix[i, i*n_levels + j] = (j + 1) / n_levels
        
        return encoding_matrix
    
    @staticmethod
    def decode_binary_weights(binary_solution: np.ndarray, 
                             encoding_matrix: np.ndarray) -> np.ndarray:
        """
        Decode binary solution vector to portfolio weights.
        
        Args:
            binary_solution: Binary solution vector
            encoding_matrix: Binary encoding matrix
            
        Returns:
            Portfolio weights
        """
        n_assets = encoding_matrix.shape[0]
        weights = np.zeros(n_assets)
        
        for i in range(n_assets):
            for j in range(len(binary_solution)):
                if binary_solution[j] == 1:
                    weights[i] += encoding_matrix[i, j]
        
        # Normalize weights to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        return weights
    
    @staticmethod
    def encode_discrete_weights(weights: np.ndarray, 
                               n_levels: int = 10) -> np.ndarray:
        """
        Encode continuous weights as discrete levels.
        
        Args:
            weights: Continuous portfolio weights
            n_levels: Number of discrete levels
            
        Returns:
            Discretized weights
        """
        # Discretize weights to n_levels
        discrete_weights = np.zeros_like(weights)
        
        for i in range(len(weights)):
            # Map to level [0, n_levels-1]
            level = min(int(weights[i] * n_levels), n_levels - 1)
            # Convert back to weight
            discrete_weights[i] = level / n_levels
        
        # Normalize to sum to 1
        if np.sum(discrete_weights) > 0:
            discrete_weights = discrete_weights / np.sum(discrete_weights)
        
        return discrete_weights