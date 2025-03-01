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
from algorithms.hybrid_solvers import HybridSolver
from portfolio.markowitz import MarkowitzOptimizer
from benchmarks.classical_solvers import ClassicalPortfolioSolvers
from benchmarks.performance_metrics import BenchmarkMetrics
from benchmarks.test_cases import TestCases
from utilities.visualization import PortfolioVisualization
from simulators.backends import QuantumBackend

def run_scaling_experiment(asset_range: List[int] = [5, 10, 15, 20, 25, 30],
                          n_bits: int = 3,
                          n_repetitions: int = 2) -> Dict:
    """
    Run scaling experiment comparing classical vs. quantum vs. hybrid approaches.
    
    Args:
        asset_range: List of asset counts to test
        n_bits: Number of bits for encoding weights
        n_repetitions: Number of repetitions per asset count
        
    Returns:
        Dictionary with experiment results
    """
    print(f"Running Scaling Experiment")
    print(f"- Asset range: {asset_range}")
    print(f"- Number of bits for encoding: {n_bits}")
    print(f"- Number of repetitions: {n_repetitions}")
    
    # Initialize result storage
    classical_times = []
    quantum_times = []
    hybrid_times = []
    
    classical_risks = []
    quantum_risks = []
    hybrid_risks = []
    
    for n_assets in asset_range:
        print(f"\nRunning experiments for {n_assets} assets...")
        
        # Generate test case
        test_case = TestCases.generate_random_portfolio(n_assets, seed=42)
        covariance_matrix = test_case['covariance_matrix'].values
        expected_returns = test_case['expected_returns'].values
        
        # Determine target return
        target_return = np.mean(expected_returns)
        
        # Initialize arrays for repetitions
        classical_rep_times = []
        quantum_rep_times = []
        hybrid_rep_times = []
        
        classical_rep_risks = []
        quantum_rep_risks = []
        hybrid_rep_risks = []
        
        for rep in range(n_repetitions):
            print(f"  - Running repetition {rep+1}/{n_repetitions}...")
            
            # Run classical optimization
            print(f"    - Classical optimization...")
            start_time = time.time()
            classical_result = ClassicalPortfolioSolvers.solve_markowitz(
                covariance_matrix,
                expected_returns,
                target_return,
                allow_short_selling=False
            )
            classical_time = time.time() - start_time
            
            if classical_result['status'] == 'optimal':
                classical_rep_times.append(classical_time)
                classical_rep_risks.append(classical_result['portfolio_risk'])
                print(f"      - Completed in {classical_time:.6f}s with risk {classical_result['portfolio_risk']:.6f}")
            else:
                print(f"      - Failed with status {classical_result['status']}")
            
            # Run quantum optimization
            print(f"    - Quantum optimization...")
            start_time = time.time()
            
            # Create QUBO formulation
            qubo = QuboFormulation(covariance_matrix, expected_returns, target_return)
            encoding_matrix = qubo.encode_weights(n_bits)
            qubo_matrix = qubo.build_qubo_matrix(encoding_matrix)
            
            # Create quantum backend
            backend = QuantumBackend(backend_type='simulator', shots=1024)
            
            # Create and run QAOA
            qaoa = QAOA(qubo_matrix, p=1, quantum_instance=backend)
            result = qaoa.optimize(max_iterations=50)
            
            # Extract solution
            binary_solution = result['binary_solution']
            weights = qubo.extract_portfolio_weights(binary_solution)
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            quantum_time = time.time() - start_time
            
            quantum_rep_times.append(quantum_time)
            quantum_rep_risks.append(portfolio_risk)
            print(f"      - Completed in {quantum_time:.6f}s with risk {portfolio_risk:.6f}")
            
            # Run hybrid optimization
            print(f"    - Hybrid optimization...")
            start_time = time.time()
            
            # Create hybrid solver
            def quantum_solver(cov, ret, target):
                # Simplified QAOA solver
                qubo = QuboFormulation(cov, ret, target)
                encoding_matrix = qubo.encode_weights(n_bits)
                qubo_matrix = qubo.build_qubo_matrix(encoding_matrix)
                
                qaoa = QAOA(qubo_matrix, p=1, quantum_instance=backend)
                result = qaoa.optimize(max_iterations=50)
                
                binary_solution = result['binary_solution']
                weights = qubo.extract_portfolio_weights(binary_solution)
                
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                
                return {'optimal_portfolio': weights}
            
            def classical_solver(cov, ret, target, constraint=None):
                # Simplified Markowitz solver
                result = ClassicalPortfolioSolvers.solve_markowitz(
                    cov, ret, target, allow_short_selling=False
                )
                
                if result['status'] == 'optimal':
                    return {'optimal_portfolio': result['weights']}
                else:
                    return {'optimal_portfolio': np.ones(len(ret)) / len(ret)}
            
            hybrid_solver = HybridSolver(
                covariance_matrix, 
                expected_returns,
                quantum_solver,
                classical_solver
            )
            
            # Use adaptive solver that chooses the best approach
            result = hybrid_solver.adaptive_solver(
                target_return,
                time_threshold=5.0  # Use classical if it's faster than 5s
            )
            
            hybrid_time = time.time() - start_time
            
            hybrid_rep_times.append(hybrid_time)
            hybrid_rep_risks.append(result['portfolio_risk'])
            print(f"      - Completed in {hybrid_time:.6f}s with risk {result['portfolio_risk']:.6f} using {result['solver_type']} solver")
        
        # Calculate averages for this asset count
        if classical_rep_times:
            classical_times.append(np.mean(classical_rep_times))
            classical_risks.append(np.mean(classical_rep_risks))
        else:
            classical_times.append(np.nan)
            classical_risks.append(np.nan)
        
        if quantum_rep_times:
            quantum_times.append(np.mean(quantum_rep_times))
            quantum_risks.append(np.mean(quantum_rep_risks))
        else:
            quantum_times.append(np.nan)
            quantum_risks.append(np.nan)
        
        if hybrid_rep_times:
            hybrid_times.append(np.mean(hybrid_rep_times))
            hybrid_risks.append(np.mean(hybrid_rep_risks))
        else:
            hybrid_times.append(np.nan)
            hybrid_risks.append(np.nan)
    
    # Perform scaling analysis
    print("\nAnalyzing scaling behavior...")
    classical_quantum_scaling = BenchmarkMetrics.compare_scaling(
        asset_range,
        quantum_times,
        classical_times
    )
    
    hybrid_scaling_data = pd.DataFrame({
        'asset_count': asset_range,
        'classical_time': classical_times,
        'quantum_time': quantum_times,
        'hybrid_time': hybrid_times
    })
    
    # Create scaling plots
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(asset_range, classical_times, 'o-', label='Classical')
    ax1.plot(asset_range, quantum_times, 's-', label='Quantum')
    ax1.plot(asset_range, hybrid_times, '^-', label='Hybrid')
    ax1.set_xlabel('Number of Assets')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Scaling Comparison - Linear Scale')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.loglog(asset_range, classical_times, 'o-', label='Classical')
    ax2.loglog(asset_range, quantum_times, 's-', label='Quantum')
    ax2.loglog(asset_range, hybrid_times, '^-', label='Hybrid')
    ax2.set_xlabel('Number of Assets')
    ax2.set_ylabel('Execution Time (s)')
    ax2.set_title('Scaling Comparison - Log-Log Scale')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Create risk comparison plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(asset_range, classical_risks, 'o-', label='Classical')
    ax3.plot(asset_range, quantum_risks, 's-', label='Quantum')
    ax3.plot(asset_range, hybrid_risks, '^-', label='Hybrid')
    ax3.set_xlabel('Number of Assets')
    ax3.set_ylabel('Portfolio Risk')
    ax3.set_title('Risk Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Create result dictionary
    results = {
        'asset_range': asset_range,
        'n_bits': n_bits,
        'n_repetitions': n_repetitions,
        'classical_times': classical_times,
        'quantum_times': quantum_times,
        'hybrid_times': hybrid_times,
        'classical_risks': classical_risks,
        'quantum_risks': quantum_risks,
        'hybrid_risks': hybrid_risks,
        'scaling_analysis': classical_quantum_scaling,
        'hybrid_scaling_data': hybrid_scaling_data,
        'figures': {
            'linear_scaling': fig1,
            'log_scaling': fig2,
            'risk_comparison': fig3
        }
    }
    
    print("\nExperiment completed.")
    
    return results

if __name__ == "__main__":
    # Run the experiment with default parameters
    results = run_scaling_experiment()
    
    # Save figures
    results['figures']['linear_scaling'].savefig('linear_scaling.png')
    results['figures']['log_scaling'].savefig('log_scaling.png')
    results['figures']['risk_comparison'].savefig('risk_comparison.png')
    
    plt.show()