import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.constraints import PortfolioConstraints

class TestPortfolioConstraints(unittest.TestCase):
    """
    Test cases for portfolio constraints.
    """
    
    def test_budget_constraint(self):
        """Test budget constraint."""
        # Test with weights summing to 1
        weights = np.array([0.2, 0.3, 0.5])
        result = PortfolioConstraints.budget_constraint(weights)
        self.assertAlmostEqual(result, 0.0)
        
        # Test with weights summing to more than 1
        weights = np.array([0.3, 0.4, 0.4])
        result = PortfolioConstraints.budget_constraint(weights)
        self.assertAlmostEqual(result, 0.1)
        
        # Test with weights summing to less than 1
        weights = np.array([0.2, 0.3, 0.4])
        result = PortfolioConstraints.budget_constraint(weights)
        self.assertAlmostEqual(result, -0.1)
    
    def test_return_constraint(self):
        """Test return constraint."""
        weights = np.array([0.2, 0.3, 0.5])
        expected_returns = np.array([0.1, 0.15, 0.2])
        target_return = 0.16
        
        # Calculate expected portfolio return
        portfolio_return = np.dot(weights, expected_returns)  # 0.02 + 0.045 + 0.1 = 0.165
        
        result = PortfolioConstraints.return_constraint(weights, expected_returns, target_return)
        self.assertAlmostEqual(result, portfolio_return - target_return)
        self.assertAlmostEqual(result, 0.005)
    
    def test_cardinality_constraint(self):
        """Test cardinality constraint."""
        # Test with 3 assets, max 3
        weights = np.array([0.2, 0.3, 0.5])
        max_assets = 3
        result = PortfolioConstraints.cardinality_constraint(weights, max_assets)
        self.assertEqual(result, 0.0)
        
        # Test with 3 assets, max 2
        weights = np.array([0.2, 0.3, 0.5])
        max_assets = 2
        result = PortfolioConstraints.cardinality_constraint(weights, max_assets)
        self.assertEqual(result, 1.0)
        
        # Test with 3 assets (1 zero), max 2
        weights = np.array([0.0, 0.4, 0.6])
        max_assets = 2
        result = PortfolioConstraints.cardinality_constraint(weights, max_assets, threshold=1e-6)
        self.assertEqual(result, 0.0)
    
    def test_minimum_investment_constraint(self):
        """Test minimum investment constraint."""
        # Test with weights above minimum
        weights = np.array([0.2, 0.3, 0.5])
        min_values = np.array([0.1, 0.1, 0.1])
        result = PortfolioConstraints.minimum_investment_constraint(weights, min_values)
        np.testing.assert_array_equal(result, np.zeros(3))
        
        # Test with some weights below minimum
        weights = np.array([0.05, 0.3, 0.5])
        min_values = np.array([0.1, 0.1, 0.1])
        result = PortfolioConstraints.minimum_investment_constraint(weights, min_values)
        expected = np.array([0.05, 0.0, 0.0])  # Violation for first asset
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with zero weight (should not be considered selected)
        weights = np.array([0.0, 0.4, 0.6])
        min_values = np.array([0.1, 0.1, 0.1])
        result = PortfolioConstraints.minimum_investment_constraint(weights, min_values, threshold=1e-6)
        np.testing.assert_array_equal(result, np.zeros(3))
    
    def test_maximum_investment_constraint(self):
        """Test maximum investment constraint."""
        # Test with weights below maximum
        weights = np.array([0.2, 0.3, 0.5])
        max_values = np.array([0.3, 0.4, 0.6])
        result = PortfolioConstraints.maximum_investment_constraint(weights, max_values)
        np.testing.assert_array_equal(result, np.zeros(3))
        
        # Test with some weights above maximum
        weights = np.array([0.4, 0.3, 0.3])
        max_values = np.array([0.3, 0.4, 0.6])
        result = PortfolioConstraints.maximum_investment_constraint(weights, max_values)
        expected = np.array([0.1, 0.0, 0.0])  # Violation for first asset
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sector_constraint(self):
        """Test sector constraint."""
        # Create test scenario
        weights = np.array([0.2, 0.3, 0.2, 0.3])
        sector_mapper = {0: 0, 1: 0, 2: 1, 3: 1}  # Assets 0,1 in sector 0, assets 2,3 in sector 1
        min_sector = {0: 0.4, 1: 0.4}
        max_sector = {0: 0.6, 1: 0.6}
        
        # Calculate sector weights
        # Sector 0: 0.2 + 0.3 = 0.5
        # Sector 1: 0.2 + 0.3 = 0.5
        
        # Test with constraints met
        result = PortfolioConstraints.sector_constraint(weights, sector_mapper, min_sector, max_sector)
        np.testing.assert_array_equal(result, np.array([]))
        
        # Test with minimum constraint violated
        min_sector = {0: 0.6, 1: 0.4}  # Increased minimum for sector 0
        result = PortfolioConstraints.sector_constraint(weights, sector_mapper, min_sector, max_sector)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.1)  # Violation is 0.6 - 0.5 = 0.1
        
        # Test with maximum constraint violated
        min_sector = {0: 0.4, 1: 0.4}
        max_sector = {0: 0.4, 1: 0.6}  # Decreased maximum for sector 0
        result = PortfolioConstraints.sector_constraint(weights, sector_mapper, min_sector, max_sector)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.1)  # Violation is 0.5 - 0.4 = 0.1
    
    def test_get_inequality_constraints(self):
        """Test generation of inequality constraints for scipy.optimize."""
        # Create test scenario
        covariance_matrix = np.eye(3)
        expected_returns = np.array([0.1, 0.15, 0.2])
        max_assets = 2
        min_weights = np.array([0.1, 0.1, 0.1])
        max_weights = np.array([0.5, 0.5, 0.5])
        sector_mapper = {0: 0, 1: 0, 2: 1}
        min_sector = {0: 0.4, 1: 0.2}
        max_sector = {0: 0.7, 1: 0.6}
        
        # Get constraints
        constraints = PortfolioConstraints.get_inequality_constraints(
            covariance_matrix,
            expected_returns,
            max_assets,
            min_weights,
            max_weights,
            sector_mapper,
            min_sector,
            max_sector
        )
        
        # Check that constraints are returned
        self.assertIsNotNone(constraints)
        self.assertGreater(len(constraints), 0)
        
        # Check that each constraint has the required keys
        for constraint in constraints:
            self.assertIn('type', constraint)
            self.assertIn('fun', constraint)
            self.assertEqual(constraint['type'], 'ineq')

if __name__ == '__main__':
    unittest.main()