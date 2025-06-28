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
    params = np.random.uniform(0, 2 * np.pi, total_params, requires_grad=True)

    print("--- Layer-wise Training Simulation ---")

    for layer_num in range(n_layers):
        print(f"\n--- Training Layer {layer_num + 1} ---")

        # Define which parameters are trainable for this stage
        # All parameters start as non-trainable
        for p in params:
            p.requires_grad = False

        # Set parameters for the current layer to be trainable
        start_idx = layer_num * total_params_per_layer
        end_idx = (layer_num + 1) * total_params_per_layer
        for i in range(start_idx, end_idx):
            params[i].requires_grad = True

        print(f"Optimizing parameters from index {start_idx} to {end_idx-1}.")

        # Create an optimizer (only trainable params will be updated)
        opt = qml.AdamOptimizer(stepsize=0.1)

        # Dummy training loop for demonstration
        for step in range(20):
            params, cost = opt.step_and_cost(lambda p: multilayer_circuit(p), params)
            if (step + 1) % 10 == 0:
                print(f"Step {step+1:2d}: Cost = {cost:.4f}")

    print("\n--- Final Trained Parameters ---")
    print(params)
    return params

# Run the demonstration
final_params = layer_wise_training()

# Visualize the full circuit
fig, ax = qml.draw_mpl(multilayer_circuit)(final_params)
plt.suptitle("Full Circuit after Layer-wise Training")
plt.show()