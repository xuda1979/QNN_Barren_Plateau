import pennylane as qml
from pennylane import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# --- 1. Generate a synthetic classical dataset ---
# We create a dataset where features have underlying correlations.
n_features = 4
X, y = make_classification(n_samples=100, n_features=n_features, n_informative=2, n_redundant=2, random_state=42)

print("--- Classical Pre-processing Stage ---")
# --- 2. Use PCA to find the principal components of the data ---
# This step learns the most important directions of variance in the data.
pca = PCA(n_components=2)
pca.fit(X)
principal_components = pca.components_
print(f"Discovered Principal Components (shape: {principal_components.shape}):\n", principal_components)

# --- 3. Design a mapping from classical features to quantum parameters ---
# Innovation: Use the principal components to initialize different layers,
# creating a structured, data-informed starting point.
# We scale the components to be suitable as rotation angles (e.g., in [0, pi]).
initial_params_layer1 = np.pi * (principal_components + 1) / 2
initial_params_layer2 = np.pi * (principal_components[1] + 1) / 2

print("\n--- Quantum Circuit Initialization ---")
print("Initial parameters for Layer 1 (from PCA comp 1):\n", initial_params_layer1)
print("Initial parameters for Layer 2 (from PCA comp 2):\n", initial_params_layer2)

# --- 4. Define and Initialize the Quantum Circuit ---
n_qubits = n_features
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def data_driven_qnn(x, params_layer1, params_layer2):
    """A simple QNN with data-driven initialization."""
    # Layer 1: Initial parameters are set by PCA component 1
    for i in range(n_qubits):
        qml.RY(params_layer1[i], wires=i)

    # Data encoding layer
    for i in range(n_qubits):
        qml.RX(x[i], wires=i)

    # Layer 2: Initial parameters are set by PCA component 2
    for i in range(n_qubits):
        qml.RY(params_layer2[i], wires=i)

    # Entangling block
    qml.CNOT(wires=)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])

    return qml.expval(qml.PauliZ(0))

# --- 5. Verify Initialization ---
# The initial parameters for the trainable layers are now set by our PCA analysis
# instead of being chosen randomly.
print("\n--- Verification ---")
print("Circuit initialized with data-driven parameters.")
# Use the first data point for demonstration
print("Cost for first data point with these parameters:", data_driven_qnn(X, initial_params_layer1, initial_params_layer2))

# Visualize the circuit with initial parameters
fig, ax = qml.draw_mpl(data_driven_qnn, style='solarized_dark')(X, initial_params_layer1, initial_params_layer2)
fig.suptitle("QNN with Data-Driven Initial Parameters", fontsize=16)
plt.show()