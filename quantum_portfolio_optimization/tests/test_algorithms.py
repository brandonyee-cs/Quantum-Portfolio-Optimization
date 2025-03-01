import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.qubo import QuboFormulation
from algorithms.vqe import VariationalQuantumEigensolver
from algorithms.qaoa import QAOA
from algorithms.hybrid_solvers import HybridSolver

class TestQAOA(unittest.TestCase):
    """
    Test cases for QAOA algorithm.
    """
    
    def setUp(self):
        """Set up test cases."""
        # Create small QUBO matrix
        self.qubo_matrix = np.array([
            [1.0, -0.5, 0.25],
            [-0.5, 2.0, -0.75],
            [0.25, -0.75, 1.5]
        ])
        
        # Initialize QAOA
        self.p = 1
        self.qaoa = QAOA(self.qubo_matrix, p=self.p)
    
    def test_initialization(self):
        """Test initialization of QAOA."""
        self.assertEqual(self.qaoa.n_variables, 3)
        self.assertEqual(self.qaoa.p, 1)
        np.testing.assert_array_equal(self.qaoa.qubo_matrix, self.qubo_matrix)
    
    def test_create_qaoa_circuit(self):
        """Test QAOA circuit creation."""
        # Create parameters
        parameters = np.array([0.1, 0.2])  # 1 gamma, 1 beta
        
        # Create circuit
        circuit = self.qaoa.create_qaoa_circuit(parameters)
        
        # In the actual implementation, this would check the circuit structure
        # For this test, we just check that a circuit is returned
        self.assertIsNotNone(circuit)
    
    def test_optimize(self):
        """Test QAOA optimization."""
        # Run optimization with max_iterations=1 for speed
        result = self.qaoa.optimize(max_iterations=1)
        
        # Check result structure
        self.assertIn('optimal_parameters', result)
        self.assertIn('binary_solution', result)
        self.assertIn('objective_value', result)
        self.assertIn('success', result)
        
        # Check binary solution shape
        self.assertEqual(len(result['binary_solution']), self.qaoa.n_variables)
        
        # Check that binary solution contains only 0s and 1s
        for bit in result['binary_solution']:
            self.assertIn(bit, [0, 1])
    
    def test_sample_bitstrings(self):
        """Test sampling bitstrings from QAOA circuit."""
        # Create parameters
        parameters = np.array([0.1, 0.2])  # 1 gamma, 1 beta
        
        # Sample bitstrings
        bitstrings = self.qaoa.sample_bitstrings(parameters, n_samples=10)
        
        # Check that bitstrings are returned
        self.assertIsNotNone(bitstrings)
        self.assertGreater(len(bitstrings), 0)
        
        # Check that bitstrings have correct length
        for bitstring in bitstrings:
            self.assertEqual(len(bitstring), self.qaoa.n_variables)
            
            # Check that bitstrings contain only 0s and 1s
            for bit in bitstring:
                self.assertIn(bit, ['0', '1'])

class TestVQE(unittest.TestCase):
    """
    Test cases for VQE algorithm.
    """
    
    def setUp(self):
        """Set up test cases."""
        # Create small covariance matrix and expected returns
        self.covariance_matrix = np.array([
            [0.04, 0.02],
            [0.02, 0.09]
        ])
        self.expected_returns = np.array([0.10, 0.15])
        
        # Create circuit factory function
        def circuit_factory(parameters, n_assets):
            return {'parameters': parameters, 'n_assets': n_assets}
        
        # Initialize VQE
        self.vqe = VariationalQuantumEigensolver(
            self.covariance_matrix,
            self.expected_returns,
            circuit_factory
        )
    
    def test_initialization(self):
        """Test initialization of VQE."""
        self.assertEqual(self.vqe.n_assets, 2)
        np.testing.assert_array_equal(self.vqe.covariance_matrix, self.covariance_matrix)
        np.testing.assert_array_equal(self.vqe.expected_returns, self.expected_returns)
    
    def test_construct_hamiltonian(self):
        """Test Hamiltonian construction."""
        target_return = 0.12
        lambda_val = 1.0
        
        hamiltonian = self.vqe.construct_hamiltonian(target_return, lambda_val)
        
        # Check shape
        self.assertEqual(hamiltonian.shape, (2**self.vqe.n_assets, 2**self.vqe.n_assets))
        
        # Check that hamiltonian is Hermitian
        np.testing.assert_array_almost_equal(hamiltonian, hamiltonian.conj().T)
    
    def test_expectation_value(self):
        """Test expectation value calculation."""
        # Create simple Hamiltonian
        hamiltonian = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0]
        ])
        
        # Calculate expectation value
        parameters = np.array([0.1, 0.2])
        expectation = self.vqe.expectation_value(parameters, hamiltonian)
        
        # In the actual implementation, this would compute a real expectation value
        # For this test, we just check that a value is returned
        self.assertIsNotNone(expectation)

class TestHybridSolver(unittest.TestCase):
    """
    Test cases for HybridSolver.
    """
    
    def setUp(self):
        """Set up test cases."""
        # Create small covariance matrix and expected returns
        self.covariance_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])
        self.expected_returns = np.array([0.10, 0.15, 0.20])
        
        # Create mock solvers
        def quantum_solver(cov, ret, target):
            return {
                'optimal_portfolio': np.array([0.3, 0.3, 0.4]),
                'portfolio_return': np.dot(np.array([0.3, 0.3, 0.4]), ret),
                'portfolio_risk': np.sqrt(np.dot(np.array([0.3, 0.3, 0.4]).T, np.dot(cov, np.array([0.3, 0.3, 0.4]))))
            }
        
        def classical_solver(cov, ret, target, constraint=None):
            return {
                'optimal_portfolio': np.array([0.2, 0.3, 0.5]),
                'portfolio_return': np.dot(np.array([0.2, 0.3, 0.5]), ret),
                'portfolio_risk': np.sqrt(np.dot(np.array([0.2, 0.3, 0.5]).T, np.dot(cov, np.array([0.2, 0.3, 0.5]))))
            }
        
        # Initialize HybridSolver
        self.hybrid_solver = HybridSolver(
            self.covariance_matrix,
            self.expected_returns,
            quantum_solver,
            classical_solver
        )
    
    def test_initialization(self):
        """Test initialization of HybridSolver."""
        self.assertEqual(self.hybrid_solver.n_assets, 3)
        np.testing.assert_array_equal(self.hybrid_solver.covariance_matrix, self.covariance_matrix)
        np.testing.assert_array_equal(self.hybrid_solver.expected_returns, self.expected_returns)
    
    def test_asset_clustering(self):
        """Test asset clustering."""
        n_clusters = 2
        
        cluster_info = self.hybrid_solver.asset_clustering(n_clusters)
        
        # Check that cluster info is returned
        self.assertIn('clusters', cluster_info)
        self.assertIn('n_clusters', cluster_info)
        self.assertIn('representatives', cluster_info)
        
        # Check that number of clusters is correct
        self.assertEqual(cluster_info['n_clusters'], n_clusters)
        
        # Check that all assets are assigned to a cluster
        self.assertEqual(len(cluster_info['clusters']), self.hybrid_solver.n_assets)
        
        # Check that representatives are valid asset indices
        for rep in cluster_info['representatives']:
            self.assertGreaterEqual(rep, 0)
            self.assertLess(rep, self.hybrid_solver.n_assets)
    
    def test_solve_reduced_problem(self):
        """Test solving reduced problem."""
        # Create cluster info
        cluster_info = {
            'representatives': [0, 2],
            'representative_weights': [0.5, 0.5]
        }
        
        target_return = 0.15
        
        result = self.hybrid_solver.solve_reduced_problem(cluster_info, target_return)
        
        # Check that result is returned
        self.assertIn('optimal_portfolio', result)
        self.assertIn('portfolio_return', result)
        self.assertIn('portfolio_risk', result)
        
        # Check that portfolio weights have correct shape
        self.assertEqual(len(result['optimal_portfolio']), self.hybrid_solver.n_assets)
    
    def test_adaptive_solver(self):
        """Test adaptive solver."""
        target_return = 0.15
        
        result = self.hybrid_solver.adaptive_solver(target_return)
        
        # Check that result is returned
        self.assertIn('optimal_portfolio', result)
        self.assertIn('portfolio_return', result)
        self.assertIn('portfolio_risk', result)
        self.assertIn('solver_type', result)
        
        # Check that portfolio weights have correct shape
        self.assertEqual(len(result['optimal_portfolio']), self.hybrid_solver.n_assets)
        
        # Check that solver type is one of 'classical' or 'hybrid'
        self.assertIn(result['solver_type'], ['classical', 'hybrid'])

if __name__ == '__main__':
    unittest.main()