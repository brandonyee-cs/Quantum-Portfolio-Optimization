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
from portfolio.markowitz import MarkowitzOptimizer
from benchmarks.classical_solvers import ClassicalPortfolioSolvers
from benchmarks.performance_metrics import BenchmarkMetrics
from benchmarks.test_cases import TestCases
from utilities.visualization import PortfolioVisualization
from simulators.backends import QuantumBackend

def run_basic_experiment(n_assets: int = 5, 
                         target_return: Optional[float] = None,
                         n_bits: int = 3,
                         n_repetitions: int = 3) -> Dict:
    """
    Run basic portfolio optimization experiment comparing classical vs. quantum.
    
    Args:
        n_assets: Number of assets
        target_return: Target portfolio return
        n_bits: Number of bits for encoding weights
        n_repetitions: Number of experiment repetitions
        
    Returns:
        Dictionary with experiment results
    """
    print(f"Running Basic Portfolio Optimization Experiment")
    print(f"- Number of assets: {n_assets}")
    print(f"- Target return: {target_return}")
    print(f"- Number of bits for encoding: {n_bits}")
    print(f"- Number of repetitions: {n_repetitions}")
    
    # Generate test case
    test_case = TestCases.generate_random_portfolio(n_assets, seed=42)
    covariance_matrix = test_case['covariance_matrix'].values
    expected_returns = test_case['expected_returns'].values
    
    # Determine target return if not provided
    if target_return is None:
        target_return = np.mean(expected_returns)
    
    print(f"- Determined target return: {target_return:.4f}")
    
    # Run classical optimization
    print("\nRunning classical optimization...")
    classical_results = []
    
    for i in range(n_repetitions):
        start_time = time.time()
        result = ClassicalPortfolioSolvers.solve_markowitz(
            covariance_matrix,
            expected_returns,
            target_return,
            allow_short_selling=False
        )
        result['run_time'] = time.time() - start_time
        classical_results.append(result)
        print(f"  - Run {i+1}/{n_repetitions}: Status={result['status']}, Risk={result.get('portfolio_risk', 'N/A'):.6f}, Time={result['run_time']:.6f}s")
    
    # Average classical results
    avg_classical = {
        'weights': np.mean([r['weights'] for r in classical_results if r['weights'] is not None], axis=0),
        'portfolio_return': np.mean([r['portfolio_return'] for r in classical_results if r['portfolio_return'] is not None]),
        'portfolio_risk': np.mean([r['portfolio_risk'] for r in classical_results if r['portfolio_risk'] is not None]),
        'run_time': np.mean([r['run_time'] for r in classical_results])
    }
    
    print(f"  - Average classical run time: {avg_classical['run_time']:.6f}s")
    
    # Run quantum optimization
    print("\nRunning quantum optimization...")
    quantum_results = []
    
    for i in range(n_repetitions):
        # Create QUBO formulation
        qubo = QuboFormulation(covariance_matrix, expected_returns, target_return)
        encoding_matrix = qubo.encode_weights(n_bits)
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
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        # Store result
        quantum_result = {
            'weights': weights,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'binary_solution': binary_solution,
            'run_time': time.time() - start_time
        }
        
        quantum_results.append(quantum_result)
        print(f"  - Run {i+1}/{n_repetitions}: Risk={quantum_result['portfolio_risk']:.6f}, Time={quantum_result['run_time']:.6f}s")
    
    # Average quantum results
    avg_quantum = {
        'weights': np.mean([r['weights'] for r in quantum_results], axis=0),
        'portfolio_return': np.mean([r['portfolio_return'] for r in quantum_results]),
        'portfolio_risk': np.mean([r['portfolio_risk'] for r in quantum_results]),
        'run_time': np.mean([r['run_time'] for r in quantum_results])
    }
    
    print(f"  - Average quantum run time: {avg_quantum['run_time']:.6f}s")
    
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
        title="Classical Portfolio Weights"
    )
    
    fig2 = PortfolioVisualization.plot_portfolio_weights(
        avg_quantum['weights'],
        title="Quantum Portfolio Weights"
    )
    
    # Create result dictionary
    results = {
        'test_case': test_case,
        'target_return': target_return,
        'n_bits': n_bits,
        'n_repetitions': n_repetitions,
        'classical_results': classical_results,
        'quantum_results': quantum_results,
        'avg_classical': avg_classical,
        'avg_quantum': avg_quantum,
        'solution_comparison': comparison,
        'time_comparison': time_comparison,
        'figures': {
            'classical_weights': fig1,
            'quantum_weights': fig2
        }
    }
    
    print("\nExperiment completed.")
    
    return results

if __name__ == "__main__":
    # Run the experiment with default parameters
    results = run_basic_experiment()
    
    # Save figures
    results['figures']['classical_weights'].savefig('classical_weights.png')
    results['figures']['quantum_weights'].savefig('quantum_weights.png')
    
    plt.show()