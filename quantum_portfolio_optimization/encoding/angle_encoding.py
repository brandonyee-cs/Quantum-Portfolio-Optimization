import numpy as np
from typing import Dict, List, Tuple, Optional

class AngleEncoding:
    """
    Angle-based encoding schemes for quantum circuits.
    """
    
    @staticmethod
    def weights_to_angles(weights: np.ndarray) -> np.ndarray:
        """
        Convert portfolio weights to rotation angles.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Rotation angles
        """
        # Normalize weights to sum to 1
        normalized_weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        # Convert to angles
        angles = 2 * np.arccos(np.sqrt(normalized_weights))
        
        return angles
    
    @staticmethod
    def angles_to_weights(angles: np.ndarray) -> np.ndarray:
        """
        Convert rotation angles to portfolio weights.
        
        Args:
            angles: Rotation angles
            
        Returns:
            Portfolio weights
        """
        # Convert angles to weights
        weights = np.cos(angles / 2) ** 2
        
        # Normalize weights to sum to 1
        normalized_weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        return normalized_weights
    
    @staticmethod
    def portfolio_unitary(angles: np.ndarray) -> np.ndarray:
        """
        Create unitary matrix for portfolio state preparation.
        
        Args:
            angles: Rotation angles
            
        Returns:
            Unitary matrix
        """
        n_assets = len(angles)
        
        # Create identity matrix
        unitary = np.eye(2**n_assets, dtype=complex)
        
        # Apply rotations
        for i in range(n_assets):
            # Create Ry rotation matrix for qubit i
            ry = np.array([
                [np.cos(angles[i]/2), -np.sin(angles[i]/2)],
                [np.sin(angles[i]/2), np.cos(angles[i]/2)]
            ])
            
            # Apply to unitary (this is a simplified implementation)
            # In practice, would use tensor product and matrix multiplication
            # for the correct qubit
            
            # Placeholder for actual implementation
            pass
        
        return unitary
    
    @staticmethod
    def create_qaoa_parameters(n_assets: int, 
                              p: int = 1) -> np.ndarray:
        """
        Create initial QAOA parameters for portfolio optimization.
        
        Args:
            n_assets: Number of assets
            p: Number of QAOA layers
            
        Returns:
            Initial QAOA parameters
        """
        # Create random initial parameters
        # Gammas in [0, 2π], betas in [0, π]
        gammas = np.random.uniform(0, 2*np.pi, p)
        betas = np.random.uniform(0, np.pi, p)
        
        # Combine parameters
        parameters = np.concatenate([gammas, betas])
        
        return parameters
    
    @staticmethod
    def create_vqe_parameters(n_assets: int, 
                             layers: int = 2) -> np.ndarray:
        """
        Create initial VQE parameters for portfolio optimization.
        
        Args:
            n_assets: Number of assets
            layers: Number of variational layers
            
        Returns:
            Initial VQE parameters
        """
        # For a hardware-efficient ansatz with layers
        # Each layer has:
        # - n_assets rotation angles for Ry
        # - n_assets rotation angles for Rz
        # - Entangling layer (not parameterized)
        n_params = 2 * n_assets * layers
        
        # Create random initial parameters
        parameters = np.random.uniform(0, 2*np.pi, n_params)
        
        return parameters