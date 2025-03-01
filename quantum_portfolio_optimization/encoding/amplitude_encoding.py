import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.linalg import sqrtm

class AmplitudeEncoding:
    """
    Amplitude encoding techniques for quantum portfolio optimization.
    """
    
    @staticmethod
    def encode_returns_distribution(returns: np.ndarray,
                                   n_qubits: int = None) -> np.ndarray:
        """
        Encode asset returns distribution as quantum amplitudes.
        
        Args:
            returns: Asset returns data
            n_qubits: Number of qubits (default: log2 of returns length)
            
        Returns:
            Amplitude vector
        """
        n_samples = returns.shape[0]
        
        # Determine number of qubits if not provided
        if n_qubits is None:
            n_qubits = int(np.ceil(np.log2(n_samples)))
        
        # Calculate required size (power of 2)
        required_size = 2**n_qubits
        
        # Pad returns if necessary
        if n_samples < required_size:
            padded_returns = np.pad(returns, ((0, required_size - n_samples), (0, 0)))
        else:
            padded_returns = returns[:required_size]
        
        # Flatten and normalize
        flattened = padded_returns.flatten()
        norm = np.linalg.norm(flattened)
        
        if norm > 0:
            amplitudes = flattened / norm
        else:
            amplitudes = np.zeros_like(flattened)
        
        return amplitudes
    
    @staticmethod
    def encode_covariance_matrix(covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Encode covariance matrix for quantum algorithm.
        
        Args:
            covariance_matrix: Asset covariance matrix
            
        Returns:
            Encoded matrix for quantum processing
        """
        # Ensure matrix is positive semidefinite
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Reconstruct positive semidefinite matrix
        psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Calculate square root of matrix
        matrix_sqrt = sqrtm(psd_matrix)
        
        return matrix_sqrt
    
    @staticmethod
    def create_state_preparation_circuit(amplitudes: np.ndarray) -> str:
        """
        Create quantum circuit for amplitude encoding.
        
        Args:
            amplitudes: Amplitude vector
            
        Returns:
            Circuit description (placeholder)
        """
        n_amplitudes = len(amplitudes)
        n_qubits = int(np.log2(n_amplitudes))
        
        # Check if number of amplitudes is power of 2
        if 2**n_qubits != n_amplitudes:
            raise ValueError("Number of amplitudes must be a power of 2")
        
        # Placeholder for state preparation circuit
        # In practice, would create a proper quantum circuit
        # with rotation gates and controlled operations
        
        circuit = f"State preparation circuit for {n_qubits} qubits"
        
        return circuit
    
    @staticmethod
    def load_balance_distribution(weights: np.ndarray, 
                                 n_qubits: int) -> Dict:
        """
        Create load-balanced distribution of portfolio weights.
        
        Args:
            weights: Portfolio weights
            n_qubits: Number of qubits
            
        Returns:
            Dictionary with load-balanced encoding
        """
        n_assets = len(weights)
        
        # Maximum number of assets per qubit
        max_assets_per_qubit = int(np.ceil(n_assets / n_qubits))
        
        # Assign assets to qubits
        qubit_assignments = {}
        asset_idx = 0
        
        for qubit in range(n_qubits):
            qubit_assets = []
            qubit_weights = []
            
            for _ in range(max_assets_per_qubit):
                if asset_idx < n_assets:
                    qubit_assets.append(asset_idx)
                    qubit_weights.append(weights[asset_idx])
                    asset_idx += 1
            
            if qubit_assets:
                qubit_assignments[qubit] = {
                    'assets': qubit_assets,
                    'weights': qubit_weights
                }
        
        return qubit_assignments