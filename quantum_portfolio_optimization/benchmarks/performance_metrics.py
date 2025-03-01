import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import pandas as pd
import matplotlib.pyplot as plt

class BenchmarkMetrics:
    """
    Performance metrics for comparing quantum and classical optimization.
    """
    
    @staticmethod
    def compare_solution_quality(quantum_weights: np.ndarray,
                                classical_weights: np.ndarray,
                                covariance_matrix: np.ndarray,
                                expected_returns: np.ndarray) -> Dict:
        """
        Compare solution quality between quantum and classical methods.
        
        Args:
            quantum_weights: Portfolio weights from quantum optimization
            classical_weights: Portfolio weights from classical optimization
            covariance_matrix: Covariance matrix
            expected_returns: Expected returns
            
        Returns:
            Dictionary with comparison metrics
        """
        # Calculate portfolio returns
        quantum_return = np.dot(quantum_weights, expected_returns)
        classical_return = np.dot(classical_weights, expected_returns)
        
        # Calculate portfolio risks
        quantum_risk = np.sqrt(np.dot(quantum_weights.T, np.dot(covariance_matrix, quantum_weights)))
        classical_risk = np.sqrt(np.dot(classical_weights.T, np.dot(covariance_matrix, classical_weights)))
        
        # Calculate Sharpe ratios
        quantum_sharpe = quantum_return / quantum_risk if quantum_risk > 0 else 0
        classical_sharpe = classical_return / classical_risk if classical_risk > 0 else 0
        
        # Calculate weight differences
        weight_diff = np.linalg.norm(quantum_weights - classical_weights)
        weight_corr = np.corrcoef(quantum_weights, classical_weights)[0, 1] if not np.isnan(np.sum(quantum_weights * classical_weights)) else np.nan
        
        # Calculate relative performance
        return_diff_pct = (quantum_return - classical_return) / abs(classical_return) * 100 if classical_return != 0 else np.inf
        risk_diff_pct = (quantum_risk - classical_risk) / classical_risk * 100 if classical_risk != 0 else np.inf
        sharpe_diff_pct = (quantum_sharpe - classical_sharpe) / classical_sharpe * 100 if classical_sharpe != 0 else np.inf
        
        return {
            'quantum_return': quantum_return,
            'classical_return': classical_return,
            'quantum_risk': quantum_risk,
            'classical_risk': classical_risk,
            'quantum_sharpe': quantum_sharpe,
            'classical_sharpe': classical_sharpe,
            'weight_difference': weight_diff,
            'weight_correlation': weight_corr,
            'return_diff_pct': return_diff_pct,
            'risk_diff_pct': risk_diff_pct,
            'sharpe_diff_pct': sharpe_diff_pct
        }
    
    @staticmethod
    def compare_execution_time(quantum_time: float,
                              classical_time: float,
                              quantum_setup_time: Optional[float] = 0) -> Dict:
        """
        Compare execution time between quantum and classical methods.
        
        Args:
            quantum_time: Quantum execution time
            classical_time: Classical execution time
            quantum_setup_time: Quantum setup time (circuit preparation, etc.)
            
        Returns:
            Dictionary with comparison metrics
        """
        total_quantum_time = quantum_time + quantum_setup_time
        speedup = classical_time / total_quantum_time if total_quantum_time > 0 else float('inf')
        
        return {
            'quantum_execution_time': quantum_time,
            'quantum_setup_time': quantum_setup_time,
            'total_quantum_time': total_quantum_time,
            'classical_time': classical_time,
            'speedup': speedup,
            'speedup_potential': classical_time / quantum_time if quantum_time > 0 else float('inf')
        }
    
    @staticmethod
    def compare_scaling(asset_counts: List[int],
                       quantum_times: List[float],
                       classical_times: List[float]) -> Dict:
        """
        Compare scaling behavior between quantum and classical methods.
        
        Args:
            asset_counts: List of asset counts
            quantum_times: List of quantum execution times
            classical_times: List of classical execution times
            
        Returns:
            Dictionary with scaling analysis
        """
        # Create DataFrame for analysis
        scaling_df = pd.DataFrame({
            'asset_count': asset_counts,
            'quantum_time': quantum_times,
            'classical_time': classical_times
        })
        
        # Calculate speedup at each scale
        scaling_df['speedup'] = scaling_df['classical_time'] / scaling_df['quantum_time']
        
        # Fit polynomial models to estimate scaling
        quantum_fit = np.polyfit(np.log(asset_counts), np.log(quantum_times), 1)
        classical_fit = np.polyfit(np.log(asset_counts), np.log(classical_times), 1)
        
        # Calculate scaling exponents
        quantum_exponent = quantum_fit[0]
        classical_exponent = classical_fit[0]
        
        # Crossover point (asset count where quantum becomes faster)
        # Solve: c_q * n^q_exp = c_c * n^c_exp
        try:
            if quantum_exponent < classical_exponent:
                # Find coefficients
                c_q = np.exp(quantum_fit[1])
                c_c = np.exp(classical_fit[1])
                
                # Calculate crossover point
                crossover_n = np.exp((np.log(c_q) - np.log(c_c)) / (classical_exponent - quantum_exponent))
                
                # If crossover point is less than the max asset count we've tested, 
                # set it to the actual crossover from the data
                if crossover_n <= max(asset_counts):
                    crossovers = scaling_df[scaling_df['speedup'] > 1]
                    if not crossovers.empty:
                        crossover_n = crossovers.iloc[0]['asset_count']
                
            else:
                crossover_n = float('inf')
        except:
            crossover_n = float('inf')
        
        # Create scaling plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(asset_counts, quantum_times, 'o-', label=f'Quantum Time (O(n^{quantum_exponent:.2f}))')
        ax.loglog(asset_counts, classical_times, 's-', label=f'Classical Time (O(n^{classical_exponent:.2f}))')
        ax.set_xlabel('Number of Assets')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Scaling Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return {
            'scaling_data': scaling_df,
            'quantum_exponent': quantum_exponent,
            'classical_exponent': classical_exponent,
            'quantum_coefficient': np.exp(quantum_fit[1]),
            'classical_coefficient': np.exp(classical_fit[1]),
            'crossover_point': crossover_n,
            'scaling_plot': fig
        }