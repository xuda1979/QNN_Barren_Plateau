# --- 1. Define a Global Cost Function ---
# This cost function depends on an observable that acts on all qubits.
# Example: The parity of all qubits, Z_0 * Z_1 *... * Z_{n-1}
global_observable = qml.PauliZ(0)
for i in range(1, n_qubits):
    global_observable @= qml.PauliZ(i)

@qml.qnode(dev)
def global_cost_circuit(params):
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(global_observable)

# --- 2. Define a Local Cost Function ---
# This cost function is a sum of observables that each act on only one qubit.
local_observables = [qml.PauliZ(i) for i in range(n_qubits)]

@qml.qnode(dev)
def local_cost_circuit(params):
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    # The cost is the sum of local expectation values
    return qml.expval(qml.Hamiltonian(np.ones(n_qubits), local_observables))


# --- 3. Compare Gradient Variance (Conceptual) ---
# We can calculate the gradient for both. In practice, for large n_qubits,
# the gradient of the global cost function would be exponentially small.
shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits)
params = np.random.uniform(0, 2 * np.pi, size=shape)

grad_global_fn = qml.grad(global_cost_circuit)
grad_local_fn = qml.grad(local_cost_circuit)

grad_global = grad_global_fn(params)
grad_local = grad_local_fn(params)

print("--- Gradient Comparison ---")
print(f"Number of qubits: {n_qubits}")
print(f"Norm of gradient for GLOBAL cost function: {np.linalg.norm(grad_global):.6f}")
print(f"Norm of gradient for LOCAL cost function:  {np.linalg.norm(grad_local):.6f}")
print("\nFor a large number of qubits, the global gradient norm would approach zero,")
print("while the local gradient norm would remain significant, avoiding a barren plateau.")

# Visualize the circuit
fig, ax = qml.draw_mpl(local_cost_circuit)(params)
plt.suptitle("Circuit with Local Observables for Cost Function")
plt.show()