import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

class NoiseModels:
    """
    Quantum hardware noise models for portfolio optimization simulations.
    """
    
    @staticmethod
    def apply_depolarizing_error(density_matrix: np.ndarray, 
                                error_probability: float) -> np.ndarray:
        """
        Apply depolarizing error to density matrix.
        
        Args:
            density_matrix: Quantum state density matrix
            error_probability: Probability of error
            
        Returns:
            Density matrix after error
        """
        dim = density_matrix.shape[0]
        identity = np.eye(dim) / dim
        
        # Apply depolarizing channel: ρ -> (1-p)ρ + p I/d
        noisy_density_matrix = (1 - error_probability) * density_matrix + error_probability * identity
        
        return noisy_density_matrix
    
    @staticmethod
    def apply_amplitude_damping(statevector: np.ndarray, 
                               damping_parameter: float) -> np.ndarray:
        """
        Apply amplitude damping to statevector.
        
        Args:
            statevector: Quantum state vector
            damping_parameter: Damping parameter
            
        Returns:
            Density matrix after damping
        """
        # Convert statevector to density matrix
        density_matrix = np.outer(statevector, np.conj(statevector))
        
        # Dimension of the system
        dim = density_matrix.shape[0]
        n_qubits = int(np.log2(dim))
        
        # Apply amplitude damping to each qubit
        for qubit in range(n_qubits):
            # Kraus operators for amplitude damping
            K0 = np.eye(2, dtype=complex)
            K0[1, 1] = np.sqrt(1 - damping_parameter)
            
            K1 = np.zeros((2, 2), dtype=complex)
            K1[0, 1] = np.sqrt(damping_parameter)
            
            # Apply to specific qubit
            K0_full = np.eye(1, dtype=complex)
            K1_full = np.eye(1, dtype=complex)
            
            for i in range(n_qubits):
                if i == qubit:
                    K0_full = np.kron(K0_full, K0)
                    K1_full = np.kron(K1_full, K1)
                else:
                    K0_full = np.kron(K0_full, np.eye(2, dtype=complex))
                    K1_full = np.kron(K1_full, np.eye(2, dtype=complex))
            
            # Apply Kraus operators
            density_matrix = (K0_full @ density_matrix @ K0_full.conj().T + 
                             K1_full @ density_matrix @ K1_full.conj().T)
        
        return density_matrix
    
    @staticmethod
    def apply_bit_flip(statevector: np.ndarray, 
                      flip_probability: float) -> Dict[str, float]:
        """
        Apply bit flip errors to measurement outcomes.
        
        Args:
            statevector: Quantum state vector
            flip_probability: Probability of bit flip
            
        Returns:
            Dictionary of bitstrings and their probabilities
        """
        # Calculate ideal probabilities
        probabilities = np.abs(statevector) ** 2
        
        # Dimension of the system
        dim = len(statevector)
        n_qubits = int(np.log2(dim))
        
        # Apply bit flip errors
        noisy_probabilities = {}
        
        for i in range(dim):
            ideal_bitstring = format(i, f'0{n_qubits}b')
            ideal_prob = probabilities[i]
            
            # Consider all possible error patterns
            for error_pattern in range(2**n_qubits):
                # Number of bit flips in this pattern
                num_flips = bin(error_pattern).count('1')
                
                # Probability of this error pattern
                error_prob = (flip_probability ** num_flips) * ((1 - flip_probability) ** (n_qubits - num_flips))
                
                # Resulting bitstring after error
                noisy_bitstring = format(i ^ error_pattern, f'0{n_qubits}b')
                
                # Add to noisy probabilities
                noisy_probabilities[noisy_bitstring] = noisy_probabilities.get(noisy_bitstring, 0) + ideal_prob * error_prob
        
        return noisy_probabilities
    
    @staticmethod
    def readout_error_matrix(n_qubits: int, 
                            p_0_given_1: float = 0.01, 
                            p_1_given_0: float = 0.01) -> np.ndarray:
        """
        Create readout error matrix.
        
        Args:
            n_qubits: Number of qubits
            p_0_given_1: Probability of reading 0 given 1
            p_1_given_0: Probability of reading 1 given 0
            
        Returns:
            Readout error matrix
        """
        # Single-qubit readout error matrix
        single_qubit_matrix = np.array([
            [1 - p_1_given_0, p_0_given_1],
            [p_1_given_0, 1 - p_0_given_1]
        ])
        
        # Full readout error matrix (tensor product)
        readout_matrix = np.copy(single_qubit_matrix)
        
        for _ in range(n_qubits - 1):
            readout_matrix = np.kron(readout_matrix, single_qubit_matrix)
        
        return readout_matrix
    
    @staticmethod
    def apply_readout_error(ideal_counts: Dict[str, int], 
                           readout_matrix: np.ndarray) -> Dict[str, float]:
        """
        Apply readout error to measurement counts.
        
        Args:
            ideal_counts: Dictionary of ideal measurement counts
            readout_matrix: Readout error matrix
            
        Returns:
            Dictionary of noisy measurement probabilities
        """
        n_qubits = len(next(iter(ideal_counts)))
        dim = 2**n_qubits
        
        # Convert counts to vector
        ideal_vector = np.zeros(dim)
        for bitstring, count in ideal_counts.items():
            idx = int(bitstring, 2)
            ideal_vector[idx] = count
        
        # Normalize
        ideal_vector = ideal_vector / np.sum(ideal_vector)
        
        # Apply readout matrix
        noisy_vector = np.dot(readout_matrix, ideal_vector)
        
        # Convert back to dictionary
        noisy_counts = {}
        for i in range(dim):
            bitstring = format(i, f'0{n_qubits}b')
            if noisy_vector[i] > 0:
                noisy_counts[bitstring] = noisy_vector[i]
        
        return noisy_counts