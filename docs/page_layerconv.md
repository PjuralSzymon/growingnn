# Convolutional Layers

## Overview

The `Conv` class represents a convolutional layer in a neural network, responsible for extracting spatial features from input data using convolution operations. It supports multiple activation functions, various weight initialization distributions, and backpropagation for training.

## Class Definition

```python
class Conv(Layer):
```

This class inherits from `Layer` and implements forward and backward propagation for convolutional operations.

## Initialization

```python
__init__(self, _id, _model, input_shape, kernel_size, depth, act_fun, _optimizer=SGDOptimizer())
```

### Parameters:

- `_id`: Unique identifier for the layer.
- `_model`: Reference to the neural network model.
- `input_shape`: Tuple specifying the shape of the input (height, width, depth).
- `kernel_size`: Size of the convolution kernel.
- `depth`: Number of filters in the layer.
- `act_fun`: Activation function used after convolution.
- `_optimizer`: Optimization algorithm for updating weights (default: `SGDOptimizer`).

### Attributes:

- `kernels`: Weight matrices for convolution.
- `biases`: Bias terms for each filter.
- `output_shape`: Shape of the output feature map.
- `input_layers_ids`: List of input layer IDs.
- `output_layers_ids`: List of output layer IDs.
- `optimizer`: Optimization function for weight updates.

## Forward Propagation

```python
forward_prop(self, X, sender_id, deepth=0)
```

Performs convolution on the input `X` and applies the activation function.

### Steps:

1. Receives input from connected layers.
2. Performs 2D convolution using `correlate2d`.
3. Applies activation function to the output.
4. Sends transformed output to the next layers.

## Backward Propagation

```python
back_prop(self, E, m, alpha)
```

Computes gradients and updates weights during training.

### Steps:

1. Reshapes error gradient `E` to match output shape.
2. Computes the gradient of the activation function.
3. Computes weight updates using the convolution operation.
4. Propagates errors to previous layers.
5. Updates kernels and biases using the optimizer.

## Weight Initialization

Weights and biases can be initialized using different distributions:

- **Uniform**: Values between -1 and 1.
- **Normal**: Mean 0, standard deviation 1/3.
- **Gamma**: Gamma distribution values.
- **Reversed Gaussian**: Custom reversed normal distribution.

```python
if WEIGHT_DISTRIBUTION_MODE == DistributionMode.UNIFORM:
    self.kernels = np.random.uniform(-1.0, 1.0, self.kernels_shape) - 0.5
```

## Utility Methods

- `get_output_size()`: Returns the flattened output size.
- `deepcopy()`: Creates a deep copy of the layer.
- `update_params(alpha)`: Updates kernel weights and biases.
- `get_weights_summary()`: Returns mean values of kernels and biases.

## Example Usage

```python
conv_layer = Conv(
    _id=1,
    _model=model,
    input_shape=(28, 28, 1),
    kernel_size=3,
    depth=16,
    act_fun=ReLU(),
    _optimizer=AdamOptimizer()
)
```

This creates a convolutional layer with 16 filters of size 3x3, using ReLU activation and Adam optimization.

## Summary

The `Conv` class implements a convolutional layer with customizable initialization, activation functions, and learning algorithms. It integrates into a neural network model, supporting both forward and backward propagation for training.

