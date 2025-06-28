import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define number of qubits and layers
n_qubits = 4
n_layers = 3 # As per problem description
dev = qml.device("default.qubit", wires=n_qubits)

# Define a single layer of the circuit
def circuit_layer(params, layer_idx):
    """A single layer of the variational circuit."""
    offset = layer_idx * n_qubits * 2 # Each layer has n_qubits * 2 parameters
    for i in range(n_qubits):
        qml.RY(params[offset + i], wires=i)
        qml.RZ(params[offset + n_qubits + i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev)
def multilayer_circuit(params):
    """A circuit with multiple layers."""
    for i in range(n_layers):
        circuit_layer(params, i)
    return qml.expval(qml.PauliZ(0))

def layer_wise_training():
    """Demonstrates the logic of layer-wise training."""
    # Initialize all parameters randomly
    total_params_per_layer = n_qubits * 2
    total_params = n_layers * total_params_per_layer
    # Ensure params is a PennyLane NumPy array for requires_grad tracking
    params = np.array(np.random.uniform(0, 2 * np.pi, total_params), requires_grad=True)


    print("--- Layer-wise Training Simulation ---")

    for layer_num in range(n_layers):
        print(f"\n--- Training Layer {layer_num + 1} ---")

        # Create a new params tensor for each layer's training
        # to manage requires_grad correctly.
        current_params = np.array(params.tolist(), requires_grad=True)

        # Define which parameters are trainable for this stage
        # All parameters start as non-trainable for this optimization step
        for i in range(total_params):
            current_params.requires_grad_(False) # New way to set for individual elements if needed

        # Set parameters for the current layer to be trainable
        start_idx = layer_num * total_params_per_layer
        end_idx = (layer_num + 1) * total_params_per_layer

        # We need to be careful here. PennyLane's numpy array might not support
        # item assignment for requires_grad in a straightforward way for slices.
        # A common approach is to split params or use a mask if the optimizer supports it.
        # For AdamOptimizer, it will only update parameters where gradient is not None.
        # Let's recreate params with updated trainability for clarity for the optimizer.

        trainable_indices = list(range(start_idx, end_idx))

        def cost_fn_wrapper(p_all):
            # This wrapper ensures that only the correct part of 'params' is used by multilayer_circuit
            # if the optimizer tries to pass a subsection of params.
            # However, AdamOptimizer usually passes the whole parameter array.
            return multilayer_circuit(p_all)

        print(f"Optimizing parameters from index {start_idx} to {end_idx-1}.")

        # Create an optimizer (only trainable params will be updated)
        # The requires_grad flags on the `params` tensor itself will handle trainability.
        opt = qml.AdamOptimizer(stepsize=0.1)

        # Create a fresh params_for_opt for each layer to ensure correct grad status
        params_for_opt = np.array(params.tolist(), requires_grad=True)
        for i in range(total_params):
            if not (start_idx <= i < end_idx):
                params_for_opt.requires_grad_(False) # Mark non-current layer params as non-trainable


        # Dummy training loop for demonstration
        for step in range(20): # Number of optimization steps for the current layer
            # The optimizer will only compute gradients for parameters where requires_grad is True
            params_for_opt, cost = opt.step_and_cost(lambda p: multilayer_circuit(p), params_for_opt)
            if (step + 1) % 10 == 0:
                print(f"Step {step+1:2d}: Layer {layer_num+1} Cost = {cost:.4f}")

        # Update the main 'params' array with the optimized parameters for the current layer
        for i in range(start_idx, end_idx):
            params[i] = params_for_opt[i]


    print("\n--- Final Trained Parameters ---")
    # pennylane.numpy array prints directly
    print(params)
    return params

# Run the demonstration
final_params = layer_wise_training()

# Visualize the full circuit
fig, ax = qml.draw_mpl(multilayer_circuit)(final_params)
plt.suptitle("Full Circuit after Layer-wise Training")
plt.show()