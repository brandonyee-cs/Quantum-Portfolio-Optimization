import numpy as np
from algorithms.qubo import QuboFormulation
from algorithms.qaoa import QAOA
from simulators.backends import QuantumBackend

# Create QUBO formulation
covariance_matrix = np.array([[0.2, 0.1], [0.1, 0.3]])
expected_returns = np.array([0.1, 0.2])
target_return = 0.15

qubo = QuboFormulation(covariance_matrix, expected_returns, target_return)
encoding_matrix = qubo.encode_weights(n_bits=3)
qubo_matrix = qubo.build_qubo_matrix(encoding_matrix)

# Create and run QAOA
backend = QuantumBackend(backend_type='simulator')
qaoa = QAOA(qubo_matrix, p=2, quantum_instance=backend)
result = qaoa.optimize()

# Extract solution
binary_solution = result['binary_solution']
weights = qubo.extract_portfolio_weights(binary_solution)