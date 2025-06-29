import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define the number of qubits
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# --- Define QCNN components ---

def conv_layer(params, wires):
    """Applies a parameterized two-qubit gate to adjacent pairs of qubits."""
    # Expects params to be a list/array of parameters for the unitary
    # For simplicity, using a generic qml.RandomLayers or a fixed ansatz
    # For a specific 2-qubit unitary, one might use qml.Rot then CNOTs, etc.
    # Here, let's use a simple structure: RZ, RY, RZ on each qubit, then CNOT
    # params should have 3 parameters per qubit in the conv layer + CNOTs don't need params

    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])


def pool_layer(wires_to_keep, wire_to_measure):
    """
    Applies a controlled rotation, measures the control qubit, and traces out.
    For simplicity in this barren plateau context, 'pooling' often means reducing
    qubit count. A common way is to measure a qubit and use its outcome for a
    classically controlled operation on another, then discard the measured qubit.
    Or, more simply, just discard (trace out) qubits.
    The description mentions "trace out (discard) half the qubits".
    Let's implement a version that measures one qubit and effectively discards it.
    The 'control' aspect for pooling is often simplified in QCNN examples to just
    reducing dimensionality by discarding qubits after a unitary.
    """
    # Here, we are not conditioning on measurement for simplicity as per "trace out"
    # We will ensure the circuit is defined on fewer qubits in the next stage.
    # This function's role in the QCNN definition below will be more about
    # selecting which qubits pass to the next layer.
    # A true pooling might involve measurement and conditional operations.
    # For now, its effect is implicit in how we wire subsequent layers.
    pass # Actual discarding happens by not using those wires in the next layer.


@qml.qnode(dev)
def qcnn_circuit(params):
    """Implements a 4-qubit QCNN reducing to 1 qubit."""

    # Parameters are structured for each conv layer
    # params[0] for conv1_1 (wires 0,1) - 6 params
    # params[1] for conv1_2 (wires 2,3) - 6 params
    # params[2] for conv2_1 (wires 0,2) - 6 params (after re-mapping)

    # Layer 1: Convolution
    # Apply conv to (0,1) and (2,3)
    conv_layer(params[0], wires=[0, 1])
    conv_layer(params[1], wires=[2, 3])

    # Layer 1: Pooling
    # We 'pool' by selecting which qubits go to the next layer.
    # Qubits 1 and 3 will be "discarded" (not used). Qubits 0 and 2 proceed.
    # No explicit qml.pool_layer call needed if we just re-wire.
    # The original description: "trace out (discard) half the qubits"
    # Effective qubits are now 0 and 2.

    # Layer 2: Convolution
    # Acting on effective qubits 0 and 2.
    # For the qml.device, these are still wires 0 and 2.
    conv_layer(params[2], wires=[0, 2]) # Apply to the "surviving" qubits

    # Layer 2: Pooling
    # Pool qubit 2, keeping qubit 0.
    # Effective qubit is now 0.

    # Final measurement on the single remaining effective qubit
    return qml.expval(qml.PauliZ(0))


# --- Define parameters and test ---
# Each conv_layer takes 6 parameters as defined.
# L1: (0,1), (2,3) -> 2 conv_layers * 6 params/layer = 12 params
# L2: (0,2) -> 1 conv_layer * 6 params/layer = 6 params
# Total params = 12 + 6 = 18
params_shape = (3, 6) # 3 convolutional blocks, 6 params each
params = np.random.uniform(0, 2 * np.pi, size=params_shape)

print("--- QCNN Circuit Definition ---")
print(f"Device: {dev.name}, Wires: {dev.num_wires}")
print(f"Parameters shape: {params.shape}")

# Test the circuit
print("\n--- Circuit Output (example) ---")
result = qcnn_circuit(params)
print(f"Expectation value: {result}")

# Visualize the circuit
fig, ax = qml.draw_mpl(qcnn_circuit)(params)
plt.suptitle("Quantum Convolutional Neural Network (QCNN)")
plt.show()

print("\nNote: The 'pooling' is implemented by selective application of subsequent layers")
print("on a reduced set of qubits, effectively 'tracing out' the others.")
