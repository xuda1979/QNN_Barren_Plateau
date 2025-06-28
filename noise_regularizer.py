import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define number of qubits
n_qubits = 4  # Example number of qubits, assuming 3 CNOTs as in the original snippet means at least 3 qubits.
              # The CNOT sequence implies wires 0,1,2,3 or a chain like 0-1, 1-2, 2-3.
              # Let's assume 4 qubits for a chain CNOT[0,1], CNOT[1,2], CNOT[2,3]
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def noisy_qnn(params, noise_strength):
    """A simple QNN with depolarizing noise."""
    # Layer of initial rotations
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)

    # Layer of entangling gates
    # Assuming a chain of CNOTs for n_qubits
    # Corrected CNOT wires and added depolarizing channels consistently
    if n_qubits >= 2:
        qml.CNOT(wires=[0, 1])
        qml.DepolarizingChannel(noise_strength, wires=1) # Noise on target qubit
    if n_qubits >= 3:
        qml.CNOT(wires=[1, 2])
        qml.DepolarizingChannel(noise_strength, wires=2) # Noise on target qubit
    if n_qubits >= 4: # As per original example CNOT(wires=[2,3])
        qml.CNOT(wires=[2, 3])
        qml.DepolarizingChannel(noise_strength, wires=3) # Noise on target qubit
    # Add more if n_qubits is larger, or adjust logic for general n_qubits

    return qml.expval(qml.PauliZ(n_qubits - 1)) # Measure last qubit

# --- Treat noise as a hyperparameter ---
# Parameters for RY rotations, one for each qubit
params = np.random.uniform(0, 2 * np.pi, n_qubits)

# Case 1: Noiseless execution
noise_level_1 = 0.0
result_noiseless = noisy_qnn(params, noise_level_1)

# Case 2: Execution with some noise
noise_level_2 = 0.05 # 5% depolarizing probability
result_noisy = noisy_qnn(params, noise_level_2)

print("--- Noise as a Hyperparameter ---")
print(f"Number of qubits: {n_qubits}")
print(f"Output with noise_strength = {noise_level_1}: {result_noiseless:.4f}")
print(f"Output with noise_strength = {noise_level_2}: {result_noisy:.4f}")
print("\nBy tuning 'noise_strength' as a hyperparameter, we can find a level")
print("that improves generalization on a validation set.")

# Visualize the circuit with noise channels
fig, ax = qml.draw_mpl(noisy_qnn)(params, noise_level_2)
plt.suptitle("QNN with Noise Channels for Regularization")
plt.show()
