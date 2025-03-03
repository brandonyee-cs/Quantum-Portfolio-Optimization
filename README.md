# Quantum Portfolio Optimization

A comprehensive framework for implementing and evaluating quantum computing algorithms for portfolio optimization problems. This project compares quantum approaches (QUBO, VQE, QAOA) against classical methods, with a focus on identifying scenarios where quantum computing may offer advantages.

## Features

- **Multiple Quantum Algorithms**: QUBO, VQE, QAOA implementations 
- **Encoding Strategies**: Binary, angle, and amplitude encoding
- **Hybrid Approaches**: Quantum-classical solvers leveraging the strengths of both paradigms
- **Constraint Handling**: Cardinality and sector constraints
- **Performance Analysis**: Comparative evaluation framework

## Installation

```bash
# Clone the repository
git clone https://github.com/brandonyee-cs/Quantum-Portfolio-Optimization.git
cd Quantum-Portfolio-Optimization

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
scipy>=1.5.0
seaborn>=0.11.0
cvxpy>=1.1.0
pytest>=6.0.0
jupyter>=1.0.0
```

## Project Structure

```
quantum_portfolio_optimization/
├── algorithms/           # Quantum algorithm implementations
│   ├── qubo.py          # QUBO formulation
│   ├── vqe.py           # Variational Quantum Eigensolver
│   ├── qaoa.py          # Quantum Approximate Optimization Algorithm
│   └── hybrid_solvers.py # Hybrid quantum-classical approaches
├── encoding/            # Encoding strategies
├── portfolio/           # Classical portfolio optimization
├── simulators/          # Quantum simulators
├── benchmarks/          # Performance comparison utilities
├── utilities/           # Utility functions
├── experiments/         # Standardized experiment scripts
└── data/                # Test data and results
```

## Usage

### Basic Portfolio Optimization

```python
from quantum_portfolio_optimization.experiments.experiment_1_basic import run_basic_experiment

# Run basic portfolio optimization experiment
results = run_basic_experiment(
    n_assets=5,           # Number of assets
    n_repetitions=3,      # Number of experiment repetitions
    n_bits=3              # Number of bits for encoding weights
)

# Access results
classical_results = results['avg_classical']
quantum_results = results['avg_quantum']
comparison = results['solution_comparison']
```

### Cardinality-Constrained Optimization

```python
from quantum_portfolio_optimization.experiments.experiment_2_cardinality import run_cardinality_experiment

# Run cardinality-constrained portfolio optimization
results = run_cardinality_experiment(
    n_assets=10,        # Number of assets
    max_assets=3,       # Maximum number of assets to include
    n_repetitions=2     # Number of experiment repetitions
)
```

### Scaling Analysis

```python
from quantum_portfolio_optimization.experiments.experiment_4_scaling import run_scaling_experiment

# Analyze scaling behavior
results = run_scaling_experiment(
    asset_range=[5, 10, 15, 20, 25, 30],  # Asset counts to test
    n_repetitions=2                       # Number of repetitions per asset count
)
```

## Key Findings

- Quantum approaches show promise for portfolios with discrete constraints (especially cardinality constraints)
- Quantum methods could provide advantages for large portfolios (>40 assets) and non-convex objective functions
- Classical methods currently outperform for small, unconstrained portfolios
- Hybrid quantum-classical approaches offer the most practical path forward

## Examples

See `notebooks/` directory for Jupyter notebooks demonstrating:
1. Problem formulation
2. Classical optimization approaches
3. QUBO formulation
4. VQE implementation
5. Results analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Brandon Yee

## Citation

If you use this project in your research, please cite:
```
@misc{yee2025quantum,
  author = {Yee, Brandon},
  title = {Quantum-Portfolio-Optimization},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/brandon-yee/Quantum-Portfolio-Optimization}}
}
```