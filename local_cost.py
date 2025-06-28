import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define number of qubits
n_qubits = 4  # Example number of qubits
dev = qml.device("default.qubit", wires=n_qubits)

# --- 1. Define a Global Cost Function ---
# This cost function depends on an observable that acts on all qubits.
# Example: The parity of all qubits, Z_0 * Z_1 *... * Z_{n-1}
global_observable = qml.PauliZ(0)
for i in range(1, n_qubits):
    global_observable @= qml.PauliZ(i)

@qml.qnode(dev)
def global_cost_circuit(params):
    # Assuming 2 layers for StronglyEntanglingLayers based on shape calculation later
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(global_observable)

# --- 2. Define a Local Cost Function ---
# This cost function is a sum of observables that each act on only one qubit.
local_observables = [qml.PauliZ(i) for i in range(n_qubits)]

@qml.qnode(dev)
def local_cost_circuit(params):
    # Assuming 2 layers for StronglyEntanglingLayers
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    # The cost is the sum of local expectation values
    # The coefficients for the Hamiltonian are typically 1.0 for each local observable
    coeffs = np.ones(n_qubits)
    hamiltonian = qml.Hamiltonian(coeffs, local_observables)
    return qml.expval(hamiltonian)


# --- 3. Compare Gradient Variance (Conceptual) ---
# We can calculate the gradient for both. In practice, for large n_qubits,
# the gradient of the global cost function would be exponentially small.
# Define number of layers for the specific shape calculation
n_layers_for_circuit = 2 # This should match what's used in the qnodes
shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers_for_circuit, n_wires=n_qubits)
params = np.random.uniform(0, 2 * np.pi, size=shape)

grad_global_fn = qml.grad(global_cost_circuit)
grad_local_fn = qml.grad(local_cost_circuit)

grad_global = grad_global_fn(params)
grad_local = grad_local_fn(params)

print("--- Gradient Comparison ---")
print(f"Number of qubits: {n_qubits}")
print(f"Number of layers in StronglyEntanglingLayers: {n_layers_for_circuit}")
print(f"Shape of parameters: {shape}")
print(f"Norm of gradient for GLOBAL cost function: {np.linalg.norm(grad_global):.6f}")
print(f"Norm of gradient for LOCAL cost function:  {np.linalg.norm(grad_local):.6f}")
print("\nFor a large number of qubits, the global gradient norm would approach zero,")
print("while the local gradient norm would remain significant, avoiding a barren plateau.")

# Visualize the circuit (local_cost_circuit as an example)
fig, ax = qml.draw_mpl(local_cost_circuit)(params)
plt.suptitle("Circuit with Local Observables for Cost Function")
plt.show()