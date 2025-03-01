import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
import seaborn as sns

class PortfolioVisualization:
    """
    Visualization tools for portfolio optimization results.
    """
    
    @staticmethod
    def plot_efficient_frontier(frontier_data: Dict, 
                               comparison_data: Optional[Dict] = None,
                               title: str = 'Efficient Frontier') -> plt.Figure:
        """
        Plot the efficient frontier.
        
        Args:
            frontier_data: Dictionary with frontier data
            comparison_data: Optional comparison data
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot efficient frontier
        ax.plot(
            frontier_data['frontier_risks'], 
            frontier_data['frontier_returns'], 
            'o-', 
            label='Efficient Frontier'
        )
        
        # Plot comparison data if provided
        if comparison_data is not None:
            ax.plot(
                comparison_data['frontier_risks'], 
                comparison_data['frontier_returns'], 
                's--', 
                label=comparison_data.get('label', 'Comparison')
            )
        
        # Add labels and title
        ax.set_xlabel('Portfolio Risk (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    @staticmethod
    def plot_portfolio_weights(weights: np.ndarray, 
                              asset_names: Optional[List[str]] = None,
                              title: str = 'Portfolio Weights') -> plt.Figure:
        """
        Plot portfolio weights.
        
        Args:
            weights: Portfolio weights
            asset_names: List of asset names
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_assets = len(weights)
        
        # Create default asset names if not provided
        if asset_names is None:
            asset_names = [f'Asset {i+1}' for i in range(n_assets)]
        
        # Sort weights for better visualization
        sorted_indices = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_indices]
        sorted_names = [asset_names[i] for i in sorted_indices]
        
        # Plot weights
        bars = ax.bar(sorted_names, sorted_weights)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # Add labels and title
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weight')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if there are many assets
        if n_assets > 10:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_performance_comparison(classical_results: Dict, 
                                   quantum_results: Dict,
                                   metric: str = 'computational_time',
                                   title: str = 'Performance Comparison') -> plt.Figure:
        """
        Plot performance comparison between classical and quantum methods.
        
        Args:
            classical_results: Dictionary with classical results
            quantum_results: Dictionary with quantum results
            metric: Performance metric to compare
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract metrics
        if metric in classical_results and metric in quantum_results:
            labels = ['Classical', 'Quantum']
            values = [classical_results[metric], quantum_results[metric]]
            
            # Plot comparison
            ax.bar(labels, values)
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.4f}', ha='center')
            
        else:
            # Create comparison dataframe
            comparison_data = []
            
            for method, result in [('Classical', classical_results), ('Quantum', quantum_results)]:
                for key, value in result.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        comparison_data.append({
                            'Method': method,
                            'Metric': key,
                            'Value': value
                        })
            
            df = pd.DataFrame(comparison_data)
            
            # Plot all metrics
            sns.barplot(x='Metric', y='Value', hue='Method', data=df, ax=ax)
        
        # Add labels and title
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_quantum_circuit_statistics(circuit_data: Dict,
                                       title: str = 'Quantum Circuit Statistics') -> plt.Figure:
        """
        Plot quantum circuit statistics.
        
        Args:
            circuit_data: Dictionary with circuit data
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract gate counts
        gate_counts = circuit_data.get('gate_counts', {})
        
        if gate_counts:
            # Plot gate counts
            gates = list(gate_counts.keys())
            counts = list(gate_counts.values())
            
            bars = ax.bar(gates, counts)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # Add labels and title
            ax.set_xlabel('Gate Type')
            ax.set_ylabel('Count')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')
            
        else:
            # Plot other circuit statistics
            metrics = []
            values = []
            
            for key, value in circuit_data.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics.append(key)
                    values.append(value)
            
            if metrics:
                bars = ax.bar(metrics, values)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
                
                # Add labels and title
                ax.set_title(title)
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_optimization_convergence(iterations: List[int],
                                     objective_values: List[float],
                                     title: str = 'Optimization Convergence') -> plt.Figure:
        """
        Plot optimization convergence.
        
        Args:
            iterations: List of iteration numbers
            objective_values: List of objective function values
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot convergence
        ax.plot(iterations, objective_values, 'o-')
        
        # Add labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add best value annotation
        best_iter = np.argmin(objective_values)
        best_value = objective_values[best_iter]
        
        ax.annotate(f'Best: {best_value:.4f}',
                   xy=(iterations[best_iter], best_value),
                   xytext=(10, -10),
                   textcoords="offset points",
                   arrowprops=dict(arrowstyle="->"))
        
        plt.tight_layout()
        
        return fig