import numpy as np
from typing import Dict, List, Tuple, Optional
import json

class MatrixConversion:
    """
    Matrix format conversion utilities for quantum algorithms.
    """
    
    @staticmethod
    def convert_to_qubo(covariance_matrix: np.ndarray, 
                       expected_returns: np.ndarray, 
                       target_return: float,
                       penalty_coefficients: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Convert portfolio optimization problem to QUBO form.
        
        Args:
            covariance_matrix: Covariance matrix
            expected_returns: Expected returns
            target_return: Target return
            penalty_coefficients: Penalty coefficients for constraints
            
        Returns:
            QUBO matrix
        """
        n_assets = len(expected_returns)
        
        # Default penalty coefficients
        if penalty_coefficients is None:
            penalty_coefficients = {
                'return': 10.0,
                'budget': 10.0
            }
        
        # Initialize QUBO matrix
        Q = np.zeros((n_assets, n_assets))
        
        # Risk term: w^T Σ w
        Q += covariance_matrix
        
        # Return constraint penalty: A * (w^T μ - r_target)^2
        # = A * (w^T μ)^2 - 2A*r_target*(w^T μ) + A*r_target^2
        A = penalty_coefficients['return']
        
        # (w^T μ)^2 term
        for i in range(n_assets):
            for j in range(n_assets):
                Q[i, j] += A * expected_returns[i] * expected_returns[j]
        
        # -2A*r_target*(w^T μ) term
        for i in range(n_assets):
            Q[i, i] -= 2 * A * target_return * expected_returns[i]
        
        # Budget constraint penalty: B * (w^T 1 - 1)^2
        # = B * (w^T 1)^2 - 2B*(w^T 1) + B
        B = penalty_coefficients['budget']
        
        # (w^T 1)^2 term
        Q += B
        
        # -2B*(w^T 1) term
        for i in range(n_assets):
            Q[i, i] -= 2 * B
        
        return Q
    
    @staticmethod
    def convert_to_ising(qubo_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert QUBO matrix to Ising model.
        
        Args:
            qubo_matrix: QUBO matrix
            
        Returns:
            Tuple of (Ising coupling matrix, offset)
        """
        n = qubo_matrix.shape[0]
        
        # Initialize Ising coupling matrix
        J = np.zeros((n, n))
        h = np.zeros(n)
        
        # Convert QUBO to Ising
        # Q_{ij} x_i x_j = J_{ij} s_i s_j + ... (where s_i = 2*x_i - 1)
        
        offset = 0
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements contribute to h
                    h[i] += qubo_matrix[i, i] / 2
                    offset += qubo_matrix[i, i] / 2
                else:
                    # Off-diagonal elements contribute to J
                    J[i, j] = qubo_matrix[i, j] / 4
                    
                    # and also to h and offset
                    h[i] += qubo_matrix[i, j] / 2
                    offset += qubo_matrix[i, j] / 4
        
        return J, h, offset
    
    @staticmethod
    def convert_to_hamiltonian(J: np.ndarray, 
                              h: np.ndarray) -> np.ndarray:
        """
        Convert Ising model to Hamiltonian matrix.
        
        Args:
            J: Ising coupling matrix
            h: Ising local field vector
            
        Returns:
            Hamiltonian matrix
        """
        n = len(h)
        N = 2**n
        
        # Initialize Hamiltonian
        H = np.zeros((N, N))
        
        # Iterate over all computational basis states
        for i in range(N):
            # Convert index to bit string
            bitstring = format(i, f'0{n}b')
            
            # Convert bit string to spin configuration
            spins = [2 * int(bit) - 1 for bit in bitstring]
            
            # Calculate energy
            energy = 0
            
            # Local field term
            for j in range(n):
                energy += h[j] * spins[j]
            
            # Coupling term
            for j in range(n):
                for k in range(j+1, n):
                    energy += J[j, k] * spins[j] * spins[k]
            
            # Set diagonal element
            H[i, i] = energy
        
        return H
    
    @staticmethod
    def convert_to_binary_polynomial(covariance_matrix: np.ndarray,
                                    expected_returns: np.ndarray,
                                    target_return: float,
                                    n_bits: int = 3) -> Dict:
        """
        Convert to binary polynomial for direct quantum circuit implementation.
        
        Args:
            covariance_matrix: Covariance matrix
            expected_returns: Expected returns
            target_return: Target return
            n_bits: Number of bits for binary expansion
            
        Returns:
            Dictionary with binary polynomial coefficients
        """
        n_assets = len(expected_returns)
        
        # Total number of binary variables
        n_variables = n_assets * n_bits
        
        # Initialize coefficient dictionaries
        linear_terms = {}    # {var_idx: coefficient}
        quadratic_terms = {} # {(var_i, var_j): coefficient}
        
        # Create binary expansion matrix
        binary_values = np.array([2**j for j in range(n_bits)])
        
        # Calculate variable indices
        var_indices = np.zeros((n_assets, n_bits), dtype=int)
        for i in range(n_assets):
            for j in range(n_bits):
                var_indices[i, j] = i * n_bits + j
        
        # Risk term: w^T Σ w
        for i in range(n_assets):
            for j in range(n_assets):
                cov_ij = covariance_matrix[i, j]
                
                for bi in range(n_bits):
                    for bj in range(n_bits):
                        idx_i = var_indices[i, bi]
                        idx_j = var_indices[j, bj]
                        
                        coef = cov_ij * binary_values[bi] * binary_values[bj]
                        
                        if idx_i == idx_j:
                            # Diagonal term
                            linear_terms[idx_i] = linear_terms.get(idx_i, 0) + coef
                        else:
                            # Off-diagonal term
                            key = (min(idx_i, idx_j), max(idx_i, idx_j))
                            quadratic_terms[key] = quadratic_terms.get(key, 0) + coef
        
        # Return constraint: w^T μ >= r_target
        # Implemented as penalty: A * max(0, r_target - w^T μ)^2
        A = 10.0  # Penalty coefficient
        
        # Calculate binary representation of target return
        for i in range(n_assets):
            ret_i = expected_returns[i]
            
            for bi in range(n_bits):
                idx_i = var_indices[i, bi]
                
                coef = -2 * A * target_return * ret_i * binary_values[bi]
                linear_terms[idx_i] = linear_terms.get(idx_i, 0) + coef
                
                for j in range(n_assets):
                    ret_j = expected_returns[j]
                    
                    for bj in range(n_bits):
                        idx_j = var_indices[j, bj]
                        
                        coef = A * ret_i * ret_j * binary_values[bi] * binary_values[bj]
                        
                        if idx_i == idx_j:
                            # Diagonal term
                            linear_terms[idx_i] = linear_terms.get(idx_i, 0) + coef
                        else:
                            # Off-diagonal term
                            key = (min(idx_i, idx_j), max(idx_i, idx_j))
                            quadratic_terms[key] = quadratic_terms.get(key, 0) + coef
        
        # Budget constraint: sum(w) = 1
        # Implemented as penalty: B * (sum(w) - 1)^2
        B = 10.0  # Penalty coefficient
        
        # Add constant term from (sum(w) - 1)^2
        constant = B
        
        # Add linear and quadratic terms
        for i in range(n_assets):
            for bi in range(n_bits):
                idx_i = var_indices[i, bi]
                
                # Linear term: -2B * binary_values[bi]
                coef = -2 * B * binary_values[bi]
                linear_terms[idx_i] = linear_terms.get(idx_i, 0) + coef
                
                for j in range(n_assets):
                    for bj in range(n_bits):
                        idx_j = var_indices[j, bj]
                        
                        # Quadratic term: B * binary_values[bi] * binary_values[bj]
                        coef = B * binary_values[bi] * binary_values[bj]
                        
                        if idx_i == idx_j:
                            # Diagonal term
                            linear_terms[idx_i] = linear_terms.get(idx_i, 0) + coef
                        else:
                            # Off-diagonal term
                            key = (min(idx_i, idx_j), max(idx_i, idx_j))
                            quadratic_terms[key] = quadratic_terms.get(key, 0) + coef
        
        return {
            'n_variables': n_variables,
            'n_assets': n_assets,
            'n_bits': n_bits,
            'var_indices': var_indices.tolist(),
            'linear_terms': {str(k): v for k, v in linear_terms.items()},
            'quadratic_terms': {f"{k[0]},{k[1]}": v for k, v in quadratic_terms.items()},
            'constant': constant
        }