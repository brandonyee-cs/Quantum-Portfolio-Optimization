import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

class StateVectorSimulator:
    """
    Exact statevector simulator for quantum portfolio optimization.
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize statevector simulator.
        
        Args:
            n_qubits: Number of qubits in the system
        """
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.reset()
        
    def reset(self):
        """Reset the statevector to |0...0>."""
        self.statevector = np.zeros(self.dim, dtype=complex)
        self.statevector[0] = 1.0
        
    def _get_unitary(self, gate: str, params: List[float] = None) -> np.ndarray:
        """
        Get unitary matrix for gate operation.
        
        Args:
            gate: Gate name
            params: Gate parameters
            
        Returns:
            Unitary matrix
        """
        if gate == 'I':
            return np.eye(2, dtype=complex)
        
        elif gate == 'X':
            return np.array([[0, 1], [1, 0]], dtype=complex)
        
        elif gate == 'Y':
            return np.array([[0, -1j], [1j, 0]], dtype=complex)
        
        elif gate == 'Z':
            return np.array([[1, 0], [0, -1]], dtype=complex)
        
        elif gate == 'H':
            return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        elif gate == 'Rx':
            theta = params[0]
            return np.array([
                [np.cos(theta/2), -1j*np.sin(theta/2)],
                [-1j*np.sin(theta/2), np.cos(theta/2)]
            ], dtype=complex)
        
        elif gate == 'Ry':
            theta = params[0]
            return np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ], dtype=complex)
        
        elif gate == 'Rz':
            theta = params[0]
            return np.array([
                [np.exp(-1j*theta/2), 0],
                [0, np.exp(1j*theta/2)]
            ], dtype=complex)
        
        else:
            raise ValueError(f"Unknown gate: {gate}")
    
    def apply_gate(self, gate: str, target: int, params: List[float] = None):
        """
        Apply single-qubit gate to the statevector.
        
        Args:
            gate: Gate name
            target: Target qubit
            params: Gate parameters
        """
        # Get gate unitary
        unitary = self._get_unitary(gate, params)
        
        # Create full unitary by tensor product with identity operators
        full_unitary = np.eye(1, dtype=complex)
        
        for i in range(self.n_qubits):
            if i == target:
                full_unitary = np.kron(full_unitary, unitary)
            else:
                full_unitary = np.kron(full_unitary, np.eye(2, dtype=complex))
        
        # Apply unitary
        self.statevector = np.dot(full_unitary, self.statevector)
    
    def apply_controlled_gate(self, gate: str, control: int, target: int, params: List[float] = None):
        """
        Apply controlled single-qubit gate to the statevector.
        
        Args:
            gate: Gate name
            control: Control qubit
            target: Target qubit
            params: Gate parameters
        """
        # Get gate unitary
        unitary = self._get_unitary(gate, params)
        
        # Create full unitary for controlled operation
        full_unitary = np.eye(self.dim, dtype=complex)
        
        # For each basis state where control qubit is |1>
        for i in range(self.dim):
            if (i >> control) & 1:  # Control is 1
                # Apply gate on target qubit
                new_i = i
                target_val = (i >> target) & 1
                
                # Apply unitary on target qubit
                if target_val == 0:
                    new_target_val = unitary[0, 0] * 0 + unitary[0, 1] * 1
                    if abs(new_target_val) > 0:
                        new_i ^= (1 << target)  # Flip target bit
                        full_unitary[new_i, i] = new_target_val
                else:  # target_val == 1
                    new_target_val = unitary[1, 0] * 0 + unitary[1, 1] * 1
                    if abs(new_target_val) > 0:
                        new_i ^= (1 << target)  # Flip target bit
                        full_unitary[new_i, i] = new_target_val
        
        # Apply unitary
        self.statevector = np.dot(full_unitary, self.statevector)
    
    def measure(self, n_shots: int = 1024) -> Dict[str, int]:
        """
        Measure all qubits in the computational basis.
        
        Args:
            n_shots: Number of measurement shots
            
        Returns:
            Dictionary of bitstrings and their counts
        """
        # Calculate probabilities
        probabilities = np.abs(self.statevector) ** 2
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample from distribution
        samples = np.random.choice(self.dim, size=n_shots, p=probabilities)
        
        # Convert to bitstrings
        bitstrings = {}
        for sample in samples:
            bitstring = format(sample, f'0{self.n_qubits}b')
            bitstrings[bitstring] = bitstrings.get(bitstring, 0) + 1
        
        return bitstrings
    
    def expectation_value(self, observable: np.ndarray) -> float:
        """
        Calculate expectation value of observable.
        
        Args:
            observable: Observable matrix
            
        Returns:
            Expectation value <ψ|O|ψ>
        """
        # Calculate <ψ|O|ψ>
        expectation = np.vdot(self.statevector, np.dot(observable, self.statevector))
        
        return expectation.real