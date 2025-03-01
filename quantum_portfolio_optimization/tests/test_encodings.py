import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoding.binary_encoding import BinaryEncoding
from encoding.angle_encoding import AngleEncoding
from encoding.amplitude_encoding import AmplitudeEncoding

class TestBinaryEncoding(unittest.TestCase):
    """
    Test cases for binary encoding.
    """
    
    def test_direct_binary_encoding(self):
        """Test direct binary encoding."""
        n_assets = 3
        n_bits = 2
        
        encoding_matrix = BinaryEncoding.direct_binary_encoding(n_assets, n_bits)
        
        # Check shape
        self.assertEqual(encoding_matrix.shape, (n_assets, n_assets * n_bits))
        
        # Check values
        self.assertEqual(encoding_matrix[0, 0], 1)  # 2^0
        self.assertEqual(encoding_matrix[0, 1], 2)  # 2^1
        self.assertEqual(encoding_matrix[1, 2], 1)  # 2^0
        self.assertEqual(encoding_matrix[1, 3], 2)  # 2^1
        self.assertEqual(encoding_matrix[2, 4], 1)  # 2^0
        self.assertEqual(encoding_matrix[2, 5], 2)  # 2^1
    
    def test_threshold_binary_encoding(self):
        """Test threshold binary encoding."""
        n_assets = 3
        
        encoding_matrix = BinaryEncoding.threshold_binary_encoding(n_assets)
        
        # Check shape
        self.assertEqual(encoding_matrix.shape, (n_assets, n_assets))
        
        # Check values (identity matrix)
        np.testing.assert_array_equal(encoding_matrix, np.eye(n_assets))
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding."""
        n_assets = 2
        n_levels = 3
        
        encoding_matrix = BinaryEncoding.one_hot_encoding(n_assets, n_levels)
        
        # Check shape
        self.assertEqual(encoding_matrix.shape, (n_assets, n_assets * n_levels))
        
        # Check values
        self.assertEqual(encoding_matrix[0, 0], 1/3)  # Level 1
        self.assertEqual(encoding_matrix[0, 1], 2/3)  # Level 2
        self.assertEqual(encoding_matrix[0, 2], 3/3)  # Level 3
        self.assertEqual(encoding_matrix[1, 3], 1/3)  # Level 1
        self.assertEqual(encoding_matrix[1, 4], 2/3)  # Level 2
        self.assertEqual(encoding_matrix[1, 5], 3/3)  # Level 3
    
    def test_decode_binary_weights(self):
        """Test decoding binary solution to portfolio weights."""
        # Create encoding matrix
        encoding_matrix = np.array([
            [1, 2, 0, 0],
            [0, 0, 1, 2]
        ])
        
        # Create binary solution
        binary_solution = np.array([1, 0, 0, 1])
        
        # Decode weights
        weights = BinaryEncoding.decode_binary_weights(binary_solution, encoding_matrix)
        
        # Expected weights: [1, 2] -> normalized to [1/3, 2/3]
        expected_weights = np.array([1/3, 2/3])
        np.testing.assert_array_almost_equal(weights, expected_weights)
    
    def test_encode_discrete_weights(self):
        """Test encoding continuous weights as discrete levels."""
        # Create continuous weights
        weights = np.array([0.4, 0.6])
        
        # Encode with 5 levels
        discrete_weights = BinaryEncoding.encode_discrete_weights(weights, n_levels=5)
        
        # Expected discrete weights: [0.4, 0.6] -> [2/5, 3/5] -> normalized to [0.4, 0.6]
        expected_weights = np.array([0.4, 0.6])
        np.testing.assert_array_almost_equal(discrete_weights, expected_weights)

class TestAngleEncoding(unittest.TestCase):
    """
    Test cases for angle encoding.
    """
    
    def test_weights_to_angles(self):
        """Test conversion of weights to angles."""
        # Create weights
        weights = np.array([0.2, 0.8])
        
        # Convert to angles
        angles = AngleEncoding.weights_to_angles(weights)
        
        # Expected angles: arccos(sqrt(0.2)) and arccos(sqrt(0.8))
        expected_angles = 2 * np.arccos(np.sqrt(weights))
        np.testing.assert_array_almost_equal(angles, expected_angles)
    
    def test_angles_to_weights(self):
        """Test conversion of angles to weights."""
        # Create angles
        angles = np.array([np.pi/2, np.pi/4])
        
        # Convert to weights
        weights = AngleEncoding.angles_to_weights(angles)
        
        # Expected weights: cos(pi/4)^2 and cos(pi/8)^2, normalized
        expected_weights = np.array([np.cos(np.pi/4)**2, np.cos(np.pi/8)**2])
        expected_weights = expected_weights / np.sum(expected_weights)
        np.testing.assert_array_almost_equal(weights, expected_weights)
    
    def test_create_qaoa_parameters(self):
        """Test creation of QAOA parameters."""
        n_assets = 3
        p = 2
        
        parameters = AngleEncoding.create_qaoa_parameters(n_assets, p)
        
        # Check length
        self.assertEqual(len(parameters), 2 * p)
        
        # Check ranges
        gammas = parameters[:p]
        betas = parameters[p:]
        
        for gamma in gammas:
            self.assertGreaterEqual(gamma, 0)
            self.assertLessEqual(gamma, 2 * np.pi)
        
        for beta in betas:
            self.assertGreaterEqual(beta, 0)
            self.assertLessEqual(beta, np.pi)
    
    def test_create_vqe_parameters(self):
        """Test creation of VQE parameters."""
        n_assets = 3
        layers = 2
        
        parameters = AngleEncoding.create_vqe_parameters(n_assets, layers)
        
        # Check length (2 * n_assets * layers)
        self.assertEqual(len(parameters), 2 * n_assets * layers)
        
        # Check ranges
        for param in parameters:
            self.assertGreaterEqual(param, 0)
            self.assertLessEqual(param, 2 * np.pi)

class TestAmplitudeEncoding(unittest.TestCase):
    """
    Test cases for amplitude encoding.
    """
    
    def test_encode_returns_distribution(self):
        """Test encoding of returns distribution."""
        # Create returns data
        returns = np.array([
            [0.01, 0.02, 0.03],
            [0.02, 0.03, 0.04],
            [0.03, 0.04, 0.05]
        ])
        
        # Encode with 2 qubits
        amplitudes = AmplitudeEncoding.encode_returns_distribution(returns, n_qubits=2)
        
        # Check length (2^n_qubits)
        self.assertEqual(len(amplitudes), 2**2)
        
        # Check normalization
        self.assertAlmostEqual(np.sum(np.abs(amplitudes)**2), 1.0)
    
    def test_encode_covariance_matrix(self):
        """Test encoding of covariance matrix."""
        # Create covariance matrix
        covariance_matrix = np.array([
            [0.04, 0.02],
            [0.02, 0.09]
        ])
        
        # Encode covariance matrix
        encoded_matrix = AmplitudeEncoding.encode_covariance_matrix(covariance_matrix)
        
        # Check shape
        self.assertEqual(encoded_matrix.shape, covariance_matrix.shape)
        
        # Check that encoded_matrix @ encoded_matrix.T ≈ covariance_matrix
        reconstructed = encoded_matrix @ encoded_matrix.T
        np.testing.assert_array_almost_equal(reconstructed, covariance_matrix)
    
    def test_load_balance_distribution(self):
        """Test load balancing of portfolio weights."""
        # Create weights
        weights = np.array([0.2, 0.3, 0.1, 0.4])
        
        # Load balance with 2 qubits
        qubit_assignments = AmplitudeEncoding.load_balance_distribution(weights, n_qubits=2)
        
        # Check number of qubits
        self.assertEqual(len(qubit_assignments), 2)
        
        # Check that all assets are assigned
        assigned_assets = []
        for qubit, data in qubit_assignments.items():
            assigned_assets.extend(data['assets'])
        
        self.assertEqual(len(assigned_assets), len(weights))
        for i in range(len(weights)):
            self.assertIn(i, assigned_assets)

if __name__ == '__main__':
    unittest.main()