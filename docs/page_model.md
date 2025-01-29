# Model Class Documentation

### Overview

The `Model` class implements a flexible neural network model with support for both feed-forward and convolutional architectures. It is designed to allow for dynamic creation of layers, forward and backward propagation, and the training process using gradient descent. The model supports multiple types of layers, including regular layers and convolutional layers, and provides various functionalities such as layer management, loss calculation, and accuracy evaluation.

### Constructor

```python
class Model:
    def __init__(self, input_size, hidden_size, output_size, loss_function = Loss.multiclass_cross_entropy, activation_fun = Activations.Sigmoid, input_paths = 1, _optimizer = SGDOptimizer()):
```

#### Parameters:
- `input_size`: Integer specifying the size of the input layer.
- `hidden_size`: Integer specifying the size of hidden layers.
- `output_size`: Integer specifying the size of the output layer.
- `loss_function`: The loss function used for training. Default is `Loss.multiclass_cross_entropy`.
- `activation_fun`: The activation function used in the layers. Default is `Activations.Sigmoid`.
- `input_paths`: Number of input paths. Default is `1`.
- `_optimizer`: The optimizer used for training. Default is `SGDOptimizer()`.

### Key Attributes:
- `batch_size`: The batch size used during training (default is 128).
- `loss_function`: The loss function used for training.
- `input_size`: The size of the input layer.
- `hidden_size`: The size of the hidden layers.
- `output_size`: The size of the output layer.
- `hidden_layers`: A list of hidden layers in the model.
- `avaible_id`: An identifier used for new layers.
- `activation_fun`: The activation function used in layers.
- `input_layers`: A list of input layers.
- `optimizer`: The optimizer used for training.
- `output_layer`: The output layer of the model.

---

### Methods

#### `set_convolution_mode(input_shape, kernel_size, depth)`

Sets the model to use convolutional layers.

- `input_shape`: Shape of the input for convolutional layers.
- `kernel_size`: The size of the convolutional kernels.
- `depth`: The number of convolutional kernels.

This method converts the input layers to convolutional layers and adjusts their connections.

#### `add_res_layer(layer_from_id, layer_to_id, layer_type = Layer_Type.ZERO)`

Adds a residual layer between two existing layers. This helps to improve training by allowing the model to learn residual functions.

- `layer_from_id`: The ID of the starting layer.
- `layer_to_id`: The ID of the ending layer.
- `layer_type`: Type of the residual layer (default is `Layer_Type.ZERO`).

Returns the ID of the new residual layer.

#### `add_norm_layer(layer_from_id, layer_to_id, layer_type = Layer_Type.RANDOM)`

Adds a normalization layer between two layers. This helps with the stability and performance of the training process.

- `layer_from_id`: The ID of the starting layer.
- `layer_to_id`: The ID of the ending layer.
- `layer_type`: Type of normalization layer (default is `Layer_Type.RANDOM`).

Returns the ID of the new normalization layer.

#### `add_sequential_output_Layer()`

Adds a sequential output layer to the model. This is typically used when the model has multiple output paths.

Returns the ID of the newly added sequential output layer.

#### `add_conv_norm_layer(layer_from_id, layer_to_id, layer_type = Layer_Type.RANDOM)`

Adds a convolutional normalization layer between two layers.

- `layer_from_id`: The ID of the starting layer.
- `layer_to_id`: The ID of the ending layer.
- `layer_type`: Type of the normalization layer (default is `Layer_Type.RANDOM`).

Returns the ID of the newly added convolutional normalization layer.

#### `add_conv_res_layer(layer_from_id, layer_to_id, layer_type = Layer_Type.ZERO)`

Adds a convolutional residual layer between two layers.

- `layer_from_id`: The ID of the starting layer.
- `layer_to_id`: The ID of the ending layer.
- `layer_type`: Type of the residual layer (default is `Layer_Type.ZERO`).

Returns the ID of the newly added convolutional residual layer.

#### `add_connection(layer_from_id, layer_to_id)`

Adds a connection between two layers.

- `layer_from_id`: The ID of the starting layer.
- `layer_to_id`: The ID of the ending layer.

#### `get_layer(id)`

Returns the layer with the specified ID.

#### `get_predictions(A2)`

Returns the predicted class for each sample.

- `A2`: The output of the model's forward pass.

#### `get_accuracy(predictions, Y)`

Calculates the accuracy of the model by comparing the predictions to the true values.

- `predictions`: The predicted output.
- `Y`: The true output.

Returns the accuracy as a float.

#### `forward_prop(input)`

Performs forward propagation through the network, computing the output for the given input.

- `input`: The input data to pass through the network.

Returns the output of the network after forward propagation.

#### `back_prop(E, m, alpha)`

Performs backpropagation to adjust the weights based on the error.

- `E`: The error signal.
- `m`: The number of samples in the batch.
- `alpha`: The learning rate.

#### `gradient_descent(X, Y, iterations, lr_scheduler, quiet = False, one_hot_needed = True, path=".")`

Trains the model using gradient descent.

- `X`: The input data.
- `Y`: The true output.
- `iterations`: The number of iterations for training.
- `lr_scheduler`: A scheduler for adjusting the learning rate.
- `quiet`: Whether to suppress the output during training (default is `False`).
- `one_hot_needed`: Whether the output is one-hot encoded (default is `True`).
- `path`: The path where the model should be saved.

Returns the accuracy history for the last iteration.

#### `evaluate(x, y)`

Evaluates the performance of the model on the given dataset.

- `x`: The input data.
- `y`: The true output.

Returns the accuracy of the model.

#### `remove_layer(layer_id, preserve_flow = True)`

Removes a layer from the model and adjusts the connections accordingly.

- `layer_id`: The ID of the layer to remove.
- `preserve_flow`: Whether to preserve the flow of connections (default is `True`).

#### `get_all_childrens_connections()`

Returns a list of all connections in the model.

#### `deepcopy()`

Creates a deep copy of the model.

Returns the copied model.

#### `is_cyclic(additional_pair)`

Checks if adding a new connection would create a cycle in the network.

- `additional_pair`: A tuple representing the new connection.

Returns `True` if a cycle is detected, otherwise `False`.

#### `get_sequence_connection()`

Returns a list of all layer connections in sequence.

#### `get_state_representation()`

Returns a string representation of the model's current state.

#### `show_connection_table()`

Prints a table showing the layers and their connections.

#### `get_connection_table()`

Returns a string table showing the layers and their connections.

---

### Example Usage

```python
# Initialize the model
model = Model(input_size=28*28, hidden_size=128, output_size=10)

# Train the model
history = model.gradient_descent(X_train, Y_train, iterations=10, lr_scheduler=my_lr_scheduler)

# Evaluate the model
accuracy = model.evaluate(X_test, Y_test)
```

---

### Notes

- The model supports both feed-forward and convolutional architectures.
- It supports dynamic layer creation, allowing for easy experimentation with different network structures.
- Various utility functions for training, evaluation, and model management are provided.
