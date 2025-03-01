import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
from typing import Dict, List, Optional

from experiments.experiment_1_basic import run_basic_experiment
from experiments.experiment_2_cardinality import run_cardinality_experiment
from experiments.experiment_3_sector import run_sector_experiment
from experiments.experiment_4_scaling import run_scaling_experiment

def main():
    """
    Main entry point for quantum portfolio optimization experiments.
    """
    parser = argparse.ArgumentParser(description='Quantum Portfolio Optimization')
    
    parser.add_argument('--experiment', type=str, default='basic',
                        choices=['basic', 'cardinality', 'sector', 'scaling'],
                        help='Type of experiment to run')
    
    parser.add_argument('--n_assets', type=int, default=5,
                        help='Number of assets for the portfolio')
    
    parser.add_argument('--max_assets', type=int, default=3,
                        help='Maximum number of assets to include (for cardinality experiment)')
    
    parser.add_argument('--n_bits', type=int, default=3,
                        help='Number of bits for encoding weights')
    
    parser.add_argument('--n_repetitions', type=int, default=3,
                        help='Number of experiment repetitions')
    
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run appropriate experiment
    if args.experiment == 'basic':
        print(f"Running basic portfolio optimization experiment with {args.n_assets} assets")
        results = run_basic_experiment(
            n_assets=args.n_assets,
            n_bits=args.n_bits,
            n_repetitions=args.n_repetitions
        )
        
        # Save results
        if not args.no_plots:
            results['figures']['classical_weights'].savefig(
                os.path.join(args.save_dir, 'classical_weights.png')
            )
            results['figures']['quantum_weights'].savefig(
                os.path.join(args.save_dir, 'quantum_weights.png')
            )
        
    elif args.experiment == 'cardinality':
        print(f"Running cardinality-constrained portfolio optimization with {args.n_assets} assets, max {args.max_assets}")
        results = run_cardinality_experiment(
            n_assets=args.n_assets,
            max_assets=args.max_assets,
            n_bits=args.n_bits,
            n_repetitions=args.n_repetitions
        )
        
        # Save results
        if not args.no_plots:
            results['figures']['classical_weights'].savefig(
                os.path.join(args.save_dir, 'classical_weights_cardinality.png')
            )
            results['figures']['quantum_weights'].savefig(
                os.path.join(args.save_dir, 'quantum_weights_cardinality.png')
            )
        
    elif args.experiment == 'sector':
        print("Running sector-constrained portfolio optimization")
        results = run_sector_experiment(
            n_repetitions=args.n_repetitions
        )
        
        # Save results
        if not args.no_plots:
            results['figures']['classical_weights'].savefig(
                os.path.join(args.save_dir, 'classical_weights_sector.png')
            )
            results['figures']['quantum_weights'].savefig(
                os.path.join(args.save_dir, 'quantum_weights_sector.png')
            )
            results['figures']['sector_allocation'].savefig(
                os.path.join(args.save_dir, 'sector_allocation.png')
            )
        
    elif args.experiment == 'scaling':
        print("Running scaling experiment")
        asset_range = [5, 10, 15, 20, 25] if args.n_assets <= 5 else list(range(5, args.n_assets + 1, 5))
        results = run_scaling_experiment(
            asset_range=asset_range,
            n_bits=args.n_bits,
            n_repetitions=args.n_repetitions
        )
        
        # Save results
        if not args.no_plots:
            results['figures']['linear_scaling'].savefig(
                os.path.join(args.save_dir, 'linear_scaling.png')
            )
            results['figures']['log_scaling'].savefig(
                os.path.join(args.save_dir, 'log_scaling.png')
            )
            results['figures']['risk_comparison'].savefig(
                os.path.join(args.save_dir, 'risk_comparison.png')
            )
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"- Experiment type: {args.experiment}")
    print(f"- Results saved to: {args.save_dir}")
    
    if not args.no_plots:
        plt.show()

if __name__ == "__main__":
    main()