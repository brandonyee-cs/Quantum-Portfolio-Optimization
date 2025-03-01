import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.qubo import QuboFormulation

class TestQuboFormulation(unittest.TestCase):
    """
    Test cases for QUBO formulation.
    """
    
    def setUp(self):
        """Set up test cases."""
        # Small test portfolio
        self.cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])
        self.exp_returns = np.array([0.10, 0.15, 0.20])
        self.target_return = 0.15
        
        # Create QUBO formulation
        self.qubo = QuboFormulation(self.cov_matrix, self.exp_returns, self.target_return)
    
    def test_initialization(self):
        """Test initialization of QuboFormulation."""
        self.assertEqual(self.qubo.n_assets, 3)
        self.assertEqual(self.qubo.target_return, 0.15)
        np.testing.assert_array_equal(self.qubo.covariance_matrix, self.cov_matrix)
        np.testing.assert_array_equal(self.qubo.expected_returns, self.exp_returns)
        
        # Check default penalty coefficients
        self.assertEqual(self.qubo.penalty_coefficients['return'], 10.0)
        self.assertEqual(self.qubo.penalty_coefficients['budget'], 10.0)
        self.assertEqual(self.qubo.penalty_coefficients['cardinality'], 5.0)
    
    def test_encode_weights(self):
        """Test binary encoding of weights."""
        # Test with 2 bits
        encoding_matrix = self.qubo.encode_weights(2)
        
        # Check shape
        self.assertEqual(encoding_matrix.shape, (3, 6))  # 3 assets * 2 bits = 6 variables
        
        # Check values for first asset, first bit
        self.assertEqual(encoding_matrix[0, 0], 1)  # 2^0
        self.assertEqual(encoding_matrix[0, 1], 2)  # 2^1
        
        # Check values for second asset, first bit
        self.assertEqual(encoding_matrix[1, 2], 1)  # 2^0
        self.assertEqual(encoding_matrix[1, 3], 2)  # 2^1
        
        # Check values for third asset, first bit
        self.assertEqual(encoding_matrix[2, 4], 1)  # 2^0
        self.assertEqual(encoding_matrix[2, 5], 2)  # 2^1
    
    def test_build_qubo_matrix(self):
        """Test QUBO matrix construction."""
        # Test with 1 bit for simplicity
        encoding_matrix = self.qubo.encode_weights(1)
        qubo_matrix = self.qubo.build_qubo_matrix(encoding_matrix)
        
        # Check shape
        self.assertEqual(qubo_matrix.shape, (3, 3))  # 3 assets * 1 bit = 3 variables
        
        # Check symmetry
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(qubo_matrix[i, j], qubo_matrix[j, i])
        
        # Check diagonal elements are non-zero
        for i in range(3):
            self.assertNotEqual(qubo_matrix[i, i], 0)
    
    def test_build_qubo_with_cardinality(self):
        """Test QUBO matrix construction with cardinality constraint."""
        # Test with 1 bit and cardinality constraint
        encoding_matrix = self.qubo.encode_weights(1)
        qubo_matrix = self.qubo.build_qubo_matrix(encoding_matrix, cardinality_constraint=2)
        
        # Cardinality constraint should increase off-diagonal elements
        qubo_without_cardinality = self.qubo.build_qubo_matrix(encoding_matrix)
        
        # At least one off-diagonal element should be different
        different_elements = False
        for i in range(3):
            for j in range(3):
                if i != j and not np.isclose(qubo_matrix[i, j], qubo_without_cardinality[i, j]):
                    different_elements = True
                    break
        
        self.assertTrue(different_elements)
    
    def test_extract_portfolio_weights(self):
        """Test extraction of portfolio weights from binary solution."""
        # Create binary solution [1, 0, 1]
        binary_solution = np.array([1, 0, 1])
        
        # Set bits
        self.qubo.n_variables = 3
        self.qubo.n_bits = 1
        
        # Extract weights
        weights = self.qubo.extract_portfolio_weights(binary_solution)
        
        # Check weights
        self.assertEqual(len(weights), 3)
        self.assertEqual(weights[0], 1)  # First bit is 1
        self.assertEqual(weights[1], 0)  # Second bit is 0
        self.assertEqual(weights[2], 1)  # Third bit is 1
    
    def test_evaluate_portfolio(self):
        """Test portfolio evaluation."""
        # Create test weights
        weights = np.array([0.4, 0.3, 0.3])
        
        # Evaluate portfolio
        result = self.qubo.evaluate_portfolio(weights)
        
        # Check expected return
        expected_return = 0.4 * 0.10 + 0.3 * 0.15 + 0.3 * 0.20
        self.assertAlmostEqual(result['expected_return'], expected_return)
        
        # Check risk (variance)
        risk = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        self.assertAlmostEqual(result['risk'], risk)
        
        # Check Sharpe ratio
        sharpe = expected_return / np.sqrt(risk)
        self.assertAlmostEqual(result['sharpe_ratio'], sharpe)
        
        # Check number of assets
        self.assertEqual(result['num_assets'], 3)

if __name__ == '__main__':
    unittest.main()