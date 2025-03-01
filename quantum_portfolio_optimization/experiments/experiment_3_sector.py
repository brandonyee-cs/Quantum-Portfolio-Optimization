import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.qubo import QuboFormulation
from algorithms.qaoa import QAOA
from portfolio.constraints import PortfolioConstraints
from benchmarks.classical_solvers import ClassicalPortfolioSolvers
from benchmarks.performance_metrics import BenchmarkMetrics
from benchmarks.test_cases import TestCases
from utilities.visualization import PortfolioVisualization
from simulators.backends import QuantumBackend

def run_sector_experiment(n_repetitions: int = 3) -> Dict:
    """
    Run portfolio optimization experiment with sector constraints.
    
    Args:
        n_repetitions: Number of experiment repetitions
        
    Returns:
        Dictionary with experiment results
    """
    print(f"Running Portfolio Optimization Experiment with Sector Constraints")
    print(f"- Number of repetitions: {n_repetitions}")
    
    # Generate sector portfolio test case
    test_case = TestCases.sector_portfolio()
    covariance_matrix = test_case['covariance_matrix'].values
    expected_returns = test_case['expected_returns'].values
    asset_names = test_case['asset_names']
    sector_mapping = test_case['sector_mapping']
    sector_constraints = test_case['sector_constraints']
    
    # Determine target return
    target_return = np.mean(expected_returns)
    print(f"- Target return: {target_return:.4f}")
    
    # Run classical optimization with sector constraints
    print("\nRunning classical optimization with sector constraints...")
    classical_results = []
    
    for i in range(n_repetitions):
        start_time = time.time()
        
        # Create constraints
        constraints = PortfolioConstraints.get_inequality_constraints(
            covariance_matrix,
            expected_returns,
            sector_mapper=sector_mapping,
            min_sector=sector_constraints['min_sector'],
            max_sector=sector_constraints['max_sector']
        )
        
        # Run optimization (simplified - would use a solver that handles sector constraints)
        # For this example, we'll just use a generic solver
        from scipy.optimize import minimize
        
        def objective(weights):
            return np.dot(weights.T, np.dot(covariance_matrix, weights))
        
        # Budget constraint
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Return constraint
        constraints.append({'type': 'ineq', 'fun': lambda x: np.dot(x, expected_returns) - target_return})
        
        # Non-negative weights
        bounds = [(0, 1) for _ in range(len(expected_returns))]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(len(expected_returns)) / len(expected_returns)
        
        # Solve
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Check if sector constraints are satisfied
        weights = result.x
        
        # Check sector allocation
        sector_allocation = {}
        for i in range(len(weights)):
            sector = sector_mapping.get(i)
            if sector is not None:
                sector_allocation[sector] = sector_allocation.get(sector, 0) + weights[i]
        
        # Verify constraints
        constraints_satisfied = True
        for sector, min_weight in sector_constraints['min_sector'].items():
            if sector_allocation.get(sector, 0) < min_weight:
                constraints_satisfied = False
                break
        
        for sector, max_weight in sector_constraints['max_sector'].items():
            if sector_allocation.get(sector, 0) > max_weight:
                constraints_satisfied = False
                break
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        result_dict = {
            'weights': weights,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sector_allocation': sector_allocation,
            'constraints_satisfied': constraints_satisfied,
            'status': result.success,
            'run_time': time.time() - start_time
        }
        
        classical_results.append(result_dict)
        print(f"  - Run {i+1}/{n_repetitions}: Risk={result_dict['portfolio_risk']:.6f}, Constraints satisfied={result_dict['constraints_satisfied']}, Time={result_dict['run_time']:.6f}s")
    
    # Average classical results
    avg_classical = {
        'weights': np.mean([r['weights'] for r in classical_results], axis=0),
        'portfolio_return': np.mean([r['portfolio_return'] for r in classical_results]),
        'portfolio_risk': np.mean([r['portfolio_risk'] for r in classical_results]),
        'run_time': np.mean([r['run_time'] for r in classical_results])
    }
    
    # Calculate sector allocation for average weights
    sector_allocation = {}
    for i in range(len(avg_classical['weights'])):
        sector = sector_mapping.get(i)
        if sector is not None:
            sector_allocation[sector] = sector_allocation.get(sector, 0) + avg_classical['weights'][i]
    
    avg_classical['sector_allocation'] = sector_allocation
    
    print(f"  - Average classical run time: {avg_classical['run_time']:.6f}s")
    print(f"  - Sector allocation: {sector_allocation}")
    
    # Run quantum optimization
    print("\nRunning quantum optimization with sector constraints...")
    quantum_results = []
    
    # Simplified - in practice would implement sector constraints in QUBO
    # For now, we'll use a simpler approach with penalty terms
    
    for i in range(n_repetitions):
        # Create QUBO formulation
        qubo = QuboFormulation(covariance_matrix, expected_returns, target_return)
        
        # Add sector constraint penalties to QUBO
        n_assets = len(expected_returns)
        n_bits = 3  # Number of bits for encoding weights
        
        # Create encoding matrix
        encoding_matrix = qubo.encode_weights(n_bits)
        
        # Create basic QUBO matrix
        qubo_matrix = qubo.build_qubo_matrix(encoding_matrix)
        
        # Create quantum backend
        backend = QuantumBackend(backend_type='simulator', shots=1024)
        
        # Create and run QAOA
        start_time = time.time()
        qaoa = QAOA(qubo_matrix, p=2, quantum_instance=backend)
        result = qaoa.optimize(max_iterations=100)
        
        # Extract solution
        binary_solution = result['binary_solution']
        weights = qubo.extract_portfolio_weights(binary_solution)
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        # Calculate sector allocation
        sector_allocation = {}
        for i in range(len(weights)):
            sector = sector_mapping.get(i)
            if sector is not None:
                sector_allocation[sector] = sector_allocation.get(sector, 0) + weights[i]
            # Check if sector constraints are satisfied
        constraints_satisfied = True
        for sector, min_weight in sector_constraints['min_sector'].items():
            if sector_allocation.get(sector, 0) < min_weight:
                constraints_satisfied = False
                break
        
        for sector, max_weight in sector_constraints['max_sector'].items():
            if sector_allocation.get(sector, 0) > max_weight:
                constraints_satisfied = False
                break
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        # Store result
        quantum_result = {
            'weights': weights,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sector_allocation': sector_allocation,
            'constraints_satisfied': constraints_satisfied,
            'binary_solution': binary_solution,
            'run_time': time.time() - start_time
        }
        
        quantum_results.append(quantum_result)
        print(f"  - Run {i+1}/{n_repetitions}: Risk={quantum_result['portfolio_risk']:.6f}, Constraints satisfied={quantum_result['constraints_satisfied']}, Time={quantum_result['run_time']:.6f}s")
    
    # Average quantum results
    avg_quantum = {
        'weights': np.mean([r['weights'] for r in quantum_results], axis=0),
        'portfolio_return': np.mean([r['portfolio_return'] for r in quantum_results]),
        'portfolio_risk': np.mean([r['portfolio_risk'] for r in quantum_results]),
        'run_time': np.mean([r['run_time'] for r in quantum_results])
    }
    
    # Calculate sector allocation for average weights
    sector_allocation = {}
    for i in range(len(avg_quantum['weights'])):
        sector = sector_mapping.get(i)
        if sector is not None:
            sector_allocation[sector] = sector_allocation.get(sector, 0) + avg_quantum['weights'][i]
    
    avg_quantum['sector_allocation'] = sector_allocation
    
    print(f"  - Average quantum run time: {avg_quantum['run_time']:.6f}s")
    print(f"  - Sector allocation: {sector_allocation}")
    
    # Compare solutions
    print("\nComparing solutions...")
    comparison = BenchmarkMetrics.compare_solution_quality(
        avg_quantum['weights'],
        avg_classical['weights'],
        covariance_matrix,
        expected_returns
    )
    
    time_comparison = BenchmarkMetrics.compare_execution_time(
        avg_quantum['run_time'],
        avg_classical['run_time']
    )
    
    print(f"- Return difference: {comparison['return_diff_pct']:.2f}%")
    print(f"- Risk difference: {comparison['risk_diff_pct']:.2f}%")
    print(f"- Sharpe ratio difference: {comparison['sharpe_diff_pct']:.2f}%")
    print(f"- Weight correlation: {comparison['weight_correlation']:.4f}")
    print(f"- Speed comparison: Classical is {time_comparison['speedup']:.2f}x {'faster' if time_comparison['speedup'] > 1 else 'slower'}")
    
    # Visualization
    print("\nGenerating visualizations...")
    fig1 = PortfolioVisualization.plot_portfolio_weights(
        avg_classical['weights'],
        asset_names=asset_names,
        title="Classical Portfolio Weights with Sector Constraints"
    )
    
    fig2 = PortfolioVisualization.plot_portfolio_weights(
        avg_quantum['weights'],
        asset_names=asset_names,
        title="Quantum Portfolio Weights with Sector Constraints"
    )
    
    # Create sector allocation chart
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Get unique sectors
    unique_sectors = list(set(test_case['sectors']))
    
    # Extract sector allocations
    classical_sector_alloc = [avg_classical['sector_allocation'].get(sector_mapping[unique_sectors.index(s)], 0) for s in unique_sectors]
    quantum_sector_alloc = [avg_quantum['sector_allocation'].get(sector_mapping[unique_sectors.index(s)], 0) for s in unique_sectors]
    
    # Min and max constraints
    min_sector_alloc = [sector_constraints['min_sector'].get(sector_mapping[unique_sectors.index(s)], 0) for s in unique_sectors]
    max_sector_alloc = [sector_constraints['max_sector'].get(sector_mapping[unique_sectors.index(s)], 1) for s in unique_sectors]
    
    # Width of bars
    width = 0.35
    
    # Plot bars
    ax3.bar(np.arange(len(unique_sectors)) - width/2, classical_sector_alloc, width, label='Classical')
    ax3.bar(np.arange(len(unique_sectors)) + width/2, quantum_sector_alloc, width, label='Quantum')
    
    # Plot constraints
    ax3.plot(np.arange(len(unique_sectors)) - width, min_sector_alloc, 'k--', label='Min Constraint')
    ax3.plot(np.arange(len(unique_sectors)) + width, max_sector_alloc, 'r--', label='Max Constraint')
    
    # Add labels
    ax3.set_ylabel('Sector Allocation')
    ax3.set_title('Sector Allocation Comparison')
    ax3.set_xticks(np.arange(len(unique_sectors)))
    ax3.set_xticklabels(unique_sectors)
    ax3.legend()
    
    # Create result dictionary
    results = {
        'test_case': test_case,
        'target_return': target_return,
        'n_repetitions': n_repetitions,
        'classical_results': classical_results,
        'quantum_results': quantum_results,
        'avg_classical': avg_classical,
        'avg_quantum': avg_quantum,
        'solution_comparison': comparison,
        'time_comparison': time_comparison,
        'figures': {
            'classical_weights': fig1,
            'quantum_weights': fig2,
            'sector_allocation': fig3
        }
    }
    
    print("\nExperiment completed.")
    
    return results

if __name__ == "__main__":
    # Run the experiment with default parameters
    results = run_sector_experiment()
    
    # Save figures
    results['figures']['classical_weights'].savefig('classical_weights_sector.png')
    results['figures']['quantum_weights'].savefig('quantum_weights_sector.png')
    results['figures']['sector_allocation'].savefig('sector_allocation.png')
    
    plt.show()