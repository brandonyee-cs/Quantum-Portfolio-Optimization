import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import time

class HybridSolver:
    """
    Hybrid classical-quantum algorithms for portfolio optimization.
    """
    
    def __init__(self, 
                 covariance_matrix: np.ndarray, 
                 expected_returns: np.ndarray,
                 quantum_solver: Callable,
                 classical_solver: Callable):
        """
        Initialize hybrid solver for portfolio optimization.
        
        Args:
            covariance_matrix: N x N covariance matrix of asset returns
            expected_returns: Vector of expected returns
            quantum_solver: Quantum optimization function
            classical_solver: Classical optimization function
        """
        self.covariance_matrix = covariance_matrix
        self.expected_returns = expected_returns
        self.n_assets = len(expected_returns)
        self.quantum_solver = quantum_solver
        self.classical_solver = classical_solver
        
    def asset_clustering(self, 
                         n_clusters: int = None) -> Dict:
        """
        Perform hierarchical clustering to group similar assets.
        
        Args:
            n_clusters: Number of clusters (default: sqrt(n_assets))
            
        Returns:
            Dictionary with clustering results
        """
        if n_clusters is None:
            n_clusters = int(np.sqrt(self.n_assets))
        
        # Calculate correlation matrix from covariance matrix
        std_devs = np.sqrt(np.diag(self.covariance_matrix))
        correlation_matrix = np.zeros_like(self.covariance_matrix)
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if std_devs[i] > 0 and std_devs[j] > 0:
                    correlation_matrix[i, j] = self.covariance_matrix[i, j] / (std_devs[i] * std_devs[j])
                else:
                    correlation_matrix[i, j] = 0
        
        # Distance matrix based on correlations
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
        
        # Hierarchical clustering using distance matrix
        # In practice, would use scipy.cluster.hierarchy
        # For demo purposes, assigning random clusters
        clusters = np.random.randint(0, n_clusters, self.n_assets)
        
        # Create representative assets for each cluster
        cluster_representatives = []
        cluster_weights = []
        
        for i in range(n_clusters):
            cluster_members = np.where(clusters == i)[0]
            if len(cluster_members) > 0:
                # Select member with highest Sharpe ratio as representative
                sharpe_ratios = np.zeros(len(cluster_members))
                for j, asset_idx in enumerate(cluster_members):
                    sharpe_ratios[j] = self.expected_returns[asset_idx] / std_devs[asset_idx] if std_devs[asset_idx] > 0 else 0
                
                rep_idx = cluster_members[np.argmax(sharpe_ratios)]
                cluster_representatives.append(rep_idx)
                cluster_weights.append(len(cluster_members) / self.n_assets)
        
        return {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'representatives': cluster_representatives,
            'representative_weights': cluster_weights
        }
    
    def solve_reduced_problem(self, 
                             cluster_info: Dict,
                             target_return: float) -> Dict:
        """
        Solve reduced portfolio optimization problem with representative assets.
        
        Args:
            cluster_info: Clustering information
            target_return: Target portfolio return
            
        Returns:
            Dictionary with optimization results
        """
        # Extract representative assets
        representatives = cluster_info['representatives']
        n_representatives = len(representatives)
        
        # Create reduced problem
        reduced_covariance = np.zeros((n_representatives, n_representatives))
        reduced_returns = np.zeros(n_representatives)
        
        for i in range(n_representatives):
            reduced_returns[i] = self.expected_returns[representatives[i]]
            for j in range(n_representatives):
                reduced_covariance[i, j] = self.covariance_matrix[representatives[i], representatives[j]]
        
        # Scale target return based on representative weights
        initial_weights = np.array(cluster_info['representative_weights'])
        initial_return = np.dot(initial_weights, reduced_returns)
        scaled_target = target_return * (initial_return / np.mean(self.expected_returns))
        
        # Solve reduced problem using quantum solver
        start_time = time.time()
        result = self.quantum_solver(
            reduced_covariance, 
            reduced_returns, 
            scaled_target
        )
        quantum_time = time.time() - start_time
        
        # Extract reduced portfolio weights
        reduced_weights = result['optimal_portfolio']
        
        # Map back to full portfolio
        full_weights = np.zeros(self.n_assets)
        for i, rep_idx in enumerate(representatives):
            # Assign weight to representative asset
            full_weights[rep_idx] = reduced_weights[i]
        
        # Normalize weights
        if np.sum(full_weights) > 0:
            full_weights = full_weights / np.sum(full_weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(full_weights, self.expected_returns)
        portfolio_risk = np.dot(full_weights.T, np.dot(self.covariance_matrix, full_weights))
        
        return {
            'optimal_portfolio': full_weights,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'computational_time': quantum_time,
            'reduced_problem_size': n_representatives,
            'original_problem_size': self.n_assets
        }
    
    def adaptive_solver(self, 
                       target_return: float,
                       cardinality_constraint: Optional[int] = None,
                       time_threshold: float = 5.0) -> Dict:
        """
        Adaptive solver that switches between classical and quantum approaches.
        
        Args:
            target_return: Target portfolio return
            cardinality_constraint: Maximum number of assets to include
            time_threshold: Time threshold for switching in seconds
            
        Returns:
            Dictionary with optimization results
        """
        # Try classical solver first
        start_time = time.time()
        
        classical_result = self.classical_solver(
            self.covariance_matrix,
            self.expected_returns,
            target_return,
            cardinality_constraint
        )
        
        classical_time = time.time() - start_time
        
        # If classical solver is fast enough, return its result
        if classical_time < time_threshold:
            classical_result['solver_type'] = 'classical'
            classical_result['computational_time'] = classical_time
            return classical_result
        
        # Otherwise, try hybrid approach with clustering
        cluster_info = self.asset_clustering()
        
        hybrid_result = self.solve_reduced_problem(
            cluster_info,
            target_return
        )
        
        hybrid_result['solver_type'] = 'hybrid'
        
        return hybrid_result