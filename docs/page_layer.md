# Layers

The `Layer` class is a core component in a neural network architecture, representing an individual computational layer within a larger model. Each layer is responsible for managing its weights, biases, and activation functions while orchestrating forward and backward propagation through its connections with other layers.

This class supports various initialization strategies for weights and biases, dynamic layer connectivity, and parallel computation via threading to optimize training efficiency. Additionally, it ensures flexibility through customizable activation functions and optimization strategies. The class also provides utilities for managing input/output connections, detecting cyclic dependencies in the network, and deep-copying layers for advanced use cases.

---

### Attributes
- **`id`**: A unique identifier for the layer.
- **`model`**: A reference to the model the layer belongs to, enabling inter-layer operations.
- **`input_size`**: The size (number of features) of the input expected by the layer.
- **`neurons`**: The number of neurons in the layer.
- **`act_fun`**: The activation function applied to the layer’s output.
- **`is_ending`**: A flag indicating if the layer is the last in the network.
- **`is_starting`**: A flag indicating if the layer is the first in the network.
- **`input_layers_ids`**: List of IDs of layers connected as inputs to this layer.
- **`output_layers_ids`**: List of IDs of layers receiving outputs from this layer.
- **`f_input`**: Stores the accumulated inputs for forward propagation.
- **`b_input`**: Stores the accumulated inputs for backward propagation.
- **`reshspers`**: A dictionary for reshaping parameters during propagation.
- **`optimizer`**: The default optimizer for parameter updates (e.g., `SGDOptimizer`).
- **`optimizer_W`**: A specific optimizer instance for weight updates.
- **`optimizer_B`**: A specific optimizer instance for bias updates.
- **`connections`**: A dictionary tracking connections between layers.

---

### Initialization
The `__init__` method initializes a layer with the following parameters:

- **`_id`**: A unique identifier for the layer.
- **`_model`**: The model the layer is part of.
- **`input_size`**: Size of the input vector.
- **`neurons`**: Number of neurons in the layer.
- **`act_fun`**: The activation function applied during forward propagation.
- **`layer_type`** (optional): Specifies the type of layer initialization (e.g., `RANDOM`, `EYE`, `ZERO`). Defaults to `Layer_Type.RANDOM`.
- **`_optimizer`** (optional): The optimizer for training weights and biases. Defaults to an instance of `SGDOptimizer`.

---

### Core Functionalities

#### Layer Connectivity
- **`set_as_ending()`**  
  Marks the layer as the final layer in the network.

- **`set_as_starting()`**  
  Marks the layer as the first layer in the network.

- **`connect_input(layer_id)`**  
  Connects another layer as an input to this layer.

- **`connect_output(layer_id)`**  
  Connects this layer as an output to another layer.

- **`disconnect(to_remove_layer_id)`**  
  Removes the connection between this layer and a specified layer.

---

#### Propagation
- **`forward_prop(X, sender_id, depth=0)`**  
  Executes forward propagation through the layer. Accumulates input data from connected input layers, computes the activation values, and forwards them to connected output layers.

- **`back_prop(E, m, alpha)`**  
  Executes backpropagation to compute and propagate errors. Updates weights and biases using gradients calculated from the error signal.

---

#### Parameter Updates
- **`update_params(alpha)`**  
  Updates the layer’s weights (`W`) and biases (`B`) using the optimizer.

---

#### Static Utility Methods
- **`update_weights_shape(W, input_size)`**  
  Adjusts the shape of the weight matrix to match the expected input size.

- **`calcuale_Z(W, I, B)`**  
  Computes the weighted input (`Z`) for the layer.

- **`calcuale_dW(m, dZ, I)`**  
  Calculates the gradient of weights (`dW`).

- **`calcuale_dB(m, dZ, B)`**  
  Calculates the gradient of biases (`dB`).

---

#### Thread Management
- **`should_thread_forward()`**  
  Determines if forward propagation should be executed in a separate thread based on active thread count and input readiness.

- **`should_thread_backward()`**  
  Determines if backpropagation should be executed in a separate thread.

---

#### Additional Utilities
- **`get_all_childrens_connections(deepth=0)`**  
  Retrieves all downstream connections from this layer.

- **`is_cyclic(visited, additional_pair, depth=0)`**  
  Checks for cycles in the network starting from this layer.

- **`deepcopy()`**  
  Creates a deep copy of the layer, duplicating all attributes except connections to the model.

- **`get_weights_summary()`**  
  Returns a summary of the layer’s weights and biases.

- **`get_paint_label()`**  
  Provides a formatted string label for the layer, displaying its ID and connection dimensions.

---

### String Representation
- **`__str__()`**  
  Returns a string representation of the layer, including its ID, model reference, and connection counts.

---

### Example Usage

```python
# Creating a layer
layer = Layer(
    _id=1, 
    _model=model_instance, 
    input_size=128, 
    neurons=64, 
    act_fun=ReLU(), 
    layer_type=Layer_Type.RANDOM, 
    _optimizer=AdamOptimizer()
)

# Connecting layers
layer.connect_input(0)
layer.connect_output(2)

# Forward propagation
input_data = np.random.rand(128, 1)
layer.forward_prop(input_data, sender_id=0)

# Backpropagation
error = np.random.rand(64, 1)
layer.back_prop(E=error, m=1, alpha=0.01)

# Checking for cycles
if layer.is_cyclic([], additional_pair=(0, 2)):
    print("Cycle detected!")
```

--- 

Let me know if you need further refinements or additional examples!