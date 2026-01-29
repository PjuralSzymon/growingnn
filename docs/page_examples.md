# Examples

This page provides examples of how to use the `growingnn` library for training and evaluating neural network models. The examples progress from simple to more complex use cases.

---

## Basic Examples

### 1. Basic Model Creation and Training with SGD

The simplest way to create and train a model using Stochastic Gradient Descent (SGD) is as follows:

```python
import growingnn as gnn
import numpy as np

# Define model parameters
input_size = 20
output_size = 2
hidden_layers = 1

# Create a model
model = gnn.structure.Model(
    input_size, input_size, output_size,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    hidden_layers,
    gnn.optimizers.SGDOptimizer()
)

# Generate random data
x_train = np.random.rand(input_size, input_size)
y_train = np.random.randint(2, size=(input_size,))

# Set learning rate scheduler
lr_scheduler = gnn.structure.LearningRateScheduler(
    gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8
)

# Train model
accuracy, _ = model.gradient_descent(x_train, y_train, epochs=5, lr_scheduler=lr_scheduler)

# Print accuracy
print(f"Training accuracy: {accuracy}")
```

---

### 2. Training with Adam Optimizer

The following example trains a model using the Adam optimizer:

```python
import growingnn as gnn
import numpy as np

# Create a model with Adam optimizer
model = gnn.structure.Model(
    20, 20, 2,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1,
    gnn.optimizers.AdamOptimizer()
)

# Generate training data
x_train = np.random.rand(20, 20)
y_train = np.random.randint(2, size=(20,))

# Define learning rate scheduler
lr_scheduler = gnn.structure.LearningRateScheduler(
    gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8
)

# Train model
accuracy, _ = model.gradient_descent(x_train, y_train, epochs=5, lr_scheduler=lr_scheduler)

# Print accuracy
print(f"Training accuracy: {accuracy}")
```

---

### 3. Adding and Removing Residual Layers

The following example demonstrates how to add and remove residual layers dynamically:

```python
import growingnn as gnn
import numpy as np

# Create a model
model = gnn.structure.Model(
    20, 20, 2,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1,
    gnn.optimizers.SGDOptimizer()
)

# Add a residual layer
layer_id = model.add_res_layer('init_0', 1)
print(f"Added residual layer with ID: {layer_id}")

# Remove the layer
model.remove_layer(layer_id)
print("Residual layer removed successfully.")
```

---

### 4. Forward Propagation Example

Forward propagation can be tested using a simple example:

```python
import growingnn as gnn
import numpy as np

# Create a model
model = gnn.structure.Model(
    10, 10, 3,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1,
    gnn.optimizers.SGDOptimizer()
)

# Generate input data
input_data = np.random.rand(10, 10)

# Perform forward propagation
output = model.forward_prop(input_data)

# Print output shape
print(f"Output shape: {output.shape}")
```

---

### 5. Training a Convolutional Model with SGD

If using convolutional networks, you can set up and train the model like this:

```python
import growingnn as gnn
import numpy as np

# Create a convolutional model
model = gnn.structure.Model(
    20, 20, 20,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1,
    gnn.optimizers.SGDOptimizer()
)

# Set convolution mode
model.set_convolution_mode((20, 20, 1), 20, 1)
model.add_res_layer('init_0', 1)

# Generate input data
x_train = np.random.random((20, 20, 20, 1))

# Forward propagation test
output = model.forward_prop(x_train)
print(f"Output shape: {output.shape}")
```

---

### 6. Comparing Adam vs SGD Performance

This example trains models using both Adam and SGD optimizers and compares their accuracy:

```python
import growingnn as gnn
import numpy as np

# Generate dataset
x_train = np.random.random((10, 20))
y_train = np.random.randint(3, size=(20,))

# Train model with Adam optimizer
adam_model = gnn.structure.Model(10, 10, 3,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1,
    gnn.optimizers.AdamOptimizer()
)
adam_model.gradient_descent(x_train, y_train, epochs=5)
acc_adam = gnn.Model.get_accuracy(gnn.Model.get_predictions(adam_model.forward_prop(x_train)), y_train)

# Train model with SGD optimizer
sgd_model = gnn.structure.Model(10, 10, 3,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1,
    gnn.optimizers.SGDOptimizer()
)
sgd_model.gradient_descent(x_train, y_train, epochs=5)
acc_sgd = gnn.Model.get_accuracy(gnn.Model.get_predictions(sgd_model.forward_prop(x_train)), y_train)

# Compare accuracy
print(f"Adam accuracy: {acc_adam}, SGD accuracy: {acc_sgd}")
```

---

### 7. Saving and Loading Models

The following example demonstrates how to save and load models:

```python
import growingnn as gnn
import numpy as np

# Create a model
model = gnn.structure.Model(
    3, 3, 1,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1
)

# Generate data
x_train = np.random.rand(3, 3)

# Perform forward propagation
output_before = model.forward_prop(x_train)

# Save the model
gnn.Storage.saveModel(model, "model.json")

# Load the model
loaded_model = gnn.Storage.loadModel("model.json")

# Perform forward propagation again
output_after = loaded_model.forward_prop(x_train)

# Ensure outputs are identical
assert np.allclose(output_before, output_after), "Model outputs differ after loading."
print("Model successfully saved and loaded.")
```

---

These examples cover the fundamental usage of `growingnn`, progressing from simple training to more advanced features like convolution, residual layers, and model storage.


## Examples with training a model with trainer


### 1. Training a Dense Network

This example demonstrates training a dense network with a small dataset:

```python
import numpy as np

# Generate synthetic data
x_train = np.random.random((10, 20))
y_train = np.random.randint(3, size=(20,))
x_test = np.random.random((10, 10))
y_test = np.random.randint(3, size=(10,))

# Train model
trained_model = gnn.trainer.train(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    labels=range(3),
    epochs=5,
    generations=3,
    input_size=10,
    hidden_size=20,
    output_size=3,
    optimizer=optimizer
)
```

This example trains a simple dense network for 5 epochs with 3 generations of evolution using the selected optimizer.

---

### 2. Training a Convolutional Neural Network (CNN)

For image-like data, a convolutional neural network can be used:

```python
# Generate synthetic image data
x_conv_train = np.random.random((20, 10, 10, 1))
y_conv_train = np.random.randint(3, size=(20,))
x_conv_test = np.random.random((10, 10, 10, 1))
y_conv_test = np.random.randint(3, size=(10,))

# Train convolutional model
trained_cnn = gnn.trainer.train(
    x_train=x_conv_train,
    y_train=y_conv_train,
    x_test=x_conv_test,
    y_test=y_conv_test,
    labels=range(3),
    input_size=10,
    hidden_size=10,
    output_size=3,
    input_shape=(10, 10, 1),
    kernel_size=3,
    optimizer=optimizer,
    epochs=5,
    generations=3
)
```

---

### 3. Training with Monte Carlo Simulation

```python
# Configure Monte Carlo Simulation
simulation_scheduler = gnn.structure.SimulationScheduler(
    mode=gnn.structure.SimulationScheduler.PROGRESS_CHECK,
    simulation_time=10,
    simulation_epochs=2
)

# Train with Monte Carlo simulation
trained_model = gnn.trainer.train(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    labels=range(3),
    epochs=5,
    generations=3,
    input_size=10,
    hidden_size=20,
    output_size=3,
    optimizer=optimizer,
    simulation_scheduler=simulation_scheduler,
    simulation_alg=gnn.montecarlo_alg
)
```

This example enables Monte Carlo simulation during training to explore different evolutionary paths.

---

## Comprehensive Examples with Detailed Arguments

### Example 1: MNIST Classification

Complete example for training on MNIST dataset with detailed argument explanations:

```python
import growingnn as gnn
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Reshape for convolutional input (optional, can use flattened)
# For dense network, keep flattened: x_train shape is (n_samples, 784)
# For conv network, reshape: x_train.reshape(-1, 28, 28, 1)

# Train the model
M = gnn.trainer.train(
    # Data arguments
    x_train=x_train,                    # Training features: (n_samples, 784) for MNIST
    y_train=y_train,                    # Training labels: (n_samples,) integer labels
    x_test=x_test,                      # Test features for evaluation
    y_test=y_test,                      # Test labels for evaluation
    labels=list(range(10)),             # List of class labels [0, 1, ..., 9]
    
    # Model structure arguments
    input_paths=1,                       # Number of input paths (usually 1)
    input_size=784,                      # Flattened input size: 28*28 = 784
    hidden_size=30,                      # Initial hidden layer size (will evolve)
    output_size=10,                      # Number of output classes
    input_shape=(28, 28, 1),            # Shape for convolutional layers (height, width, channels)
    kernel_size=3,                       # Convolution kernel size (3x3)
    deepth=1,                           # Depth of convolutional layers (number of filters)
    
    # Training arguments
    epochs=50,                          # Number of training epochs per generation
    generations=20,                    # Number of generations (structure evolution cycles)
    batch_size=128,                     # Batch size for gradient descent
    
    # Path and naming
    path="./results/mnist/",            # Directory to save model and results
    model_name="mnist_model",           # Base name for saved models
    
    # Learning rate scheduler
    lr_scheduler=gnn.LearningRateScheduler(
        gnn.LearningRateScheduler.PROGRESIVE,  # Progressive schedule (starts low after changes)
        0.005,                          # Maximum learning rate
        0.8                            # Decay factor
    ),
    
    # Simulation scheduler
    simulation_scheduler=gnn.SimulationScheduler(
        gnn.SimulationScheduler.CONSTANT,      # Run simulation every generation
        simulation_time=300,                   # Max time (seconds) for simulation per generation
        simulation_epochs=20                    # Epochs to train candidate models during simulation
    ),
    
    # Optimizer
    optimizer=gnn.AdamOptimizer(),      # Adam optimizer (alternative: SGDOptimizer())
    
    # Simulation scoring
    simulation_score=gnn.Simulation_score(
        weight_acc=1.0,                 # Weight for accuracy in scoring (always 1.0)
        weight_countW=0.1               # Weight for parameter count (low = focus on accuracy)
    ),
    
    # Simulation algorithm
    simulation_alg=gnn.montecarlo_alg,  # Monte Carlo Tree Search algorithm
    
    # Stopper
    stopper=gnn.AccuracyStopper(0.9)    # Stop training when accuracy reaches 0.9
)

# Evaluate final model
final_accuracy = M.evaluate(x_test, y_test)
param_count = M.get_parametr_count()
print(f"Final accuracy: {final_accuracy:.4f}")
print(f"Parameter count: {param_count:,}")
```

### Example 2: CIFAR-10 Classification

Example for CIFAR-10 with convolutional layers:

```python
import growingnn as gnn
import numpy as np
from tensorflow import keras

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Train the model
M = gnn.trainer.train(
    # Data arguments
    x_train=x_train,                    # Shape: (50000, 32, 32, 3)
    y_train=y_train,                    # Shape: (50000,)
    x_test=x_test,                      # Shape: (10000, 32, 32, 3)
    y_test=y_test,                      # Shape: (10000,)
    labels=list(range(10)),             # CIFAR-10 has 10 classes
    
    # Model structure arguments
    input_paths=1,
    input_size=32 * 32 * 3,             # 3072 flattened pixels
    hidden_size=50,                     # Initial hidden size
    output_size=10,                     # 10 classes
    input_shape=(32, 32, 3),            # CIFAR-10 image shape
    kernel_size=3,                     # 3x3 convolution kernels
    deepth=2,                          # 2 filters per conv layer
    
    # Training arguments
    epochs=50,
    generations=20,
    batch_size=128,
    
    # Path and naming
    path="./results/cifar10/",
    model_name="cifar10_model",
    
    # Learning rate scheduler
    lr_scheduler=gnn.LearningRateScheduler(
        gnn.LearningRateScheduler.PROGRESIVE,
        0.005,
        0.8
    ),
    
    # Simulation scheduler
    simulation_scheduler=gnn.SimulationScheduler(
        gnn.SimulationScheduler.CONSTANT,
        simulation_time=300,
        simulation_epochs=20
    ),
    
    # Optimizer
    optimizer=gnn.AdamOptimizer(),
    
    # Simulation scoring
    simulation_score=gnn.Simulation_score(
        weight_acc=1.0,
        weight_countW=0.1
    ),
    
    # Simulation algorithm
    simulation_alg=gnn.montecarlo_alg,
    
    # Stopper
    stopper=gnn.AccuracyStopper(0.7)    # Lower threshold for CIFAR-10
)
```

### Example 3: Regression Task

Example for regression (predicting continuous values):

```python
import growingnn as gnn
import numpy as np
from sklearn.model_selection import train_test_split

# Generate or load regression data
# Example: predict house prices from features
np.random.seed(42)
n_samples = 1000
n_features = 10

X = np.random.randn(n_samples, n_features)
# Create target: linear combination with some noise
y = (X @ np.random.randn(n_features)) + np.random.randn(n_samples) * 0.1

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train regression model
M = gnn.trainer.train(
    # Data arguments
    x_train=x_train,                    # Training features
    y_train=y_train,                    # Training targets (continuous values)
    x_test=x_test,
    y_test=y_test,
    labels=['target'],                  # Single output for regression
    
    # Model structure arguments
    input_paths=1,
    input_size=n_features,              # Number of input features
    hidden_size=20,                     # Hidden layer size
    output_size=1,                      # Single output for regression
    input_shape=None,                   # No convolutional layers
    kernel_size=None,                   # Not used for dense networks
    deepth=None,                        # Not used for dense networks
    
    # Training arguments
    epochs=100,
    generations=5,
    batch_size=32,
    
    # Path and naming
    path="./results/regression/",
    model_name="regression_model",
    
    # Loss and activation functions (CRITICAL for regression)
    loss_function=gnn.Loss.MSE,         # Mean Squared Error for regression
    activation_fun=gnn.Activations.Linear,      # Linear activation
    output_activation_fun=gnn.Activations.Linear,  # Linear output
    
    # Learning rate scheduler
    lr_scheduler=gnn.LearningRateScheduler(
        gnn.LearningRateScheduler.CONSTANT,    # Constant LR for regression
        0.001                                 # Lower learning rate
    ),
    
    # Simulation scheduler
    simulation_scheduler=gnn.SimulationScheduler(
        gnn.SimulationScheduler.CONSTANT,
        simulation_time=60,              # Shorter simulation for regression
        simulation_epochs=10
    ),
    
    # Optimizer
    optimizer=gnn.SGDOptimizer(),        # SGD often works well for regression
    
    # Simulation scoring (focus on loss, not accuracy)
    simulation_score=gnn.Simulation_score(
        weight_acc=0.0,                  # No accuracy for regression
        weight_loss=1.0,                 # Focus on minimizing loss
        weight_countW=0.1                # Slight preference for smaller models
    ),
    
    # Simulation algorithm
    simulation_alg=gnn.montecarlo_alg,
    
    # No stopper for regression (or use custom loss-based stopper)
    stopper=gnn.EmptyStopper()
)

# Evaluate regression model
predictions = M.forward_prop(x_test)
mse = np.mean((predictions.flatten() - y_test) ** 2)
print(f"Test MSE: {mse:.4f}")
```

## Argument Reference

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `x_train` | array | Training features, shape: `(n_samples, features)` or `(n_samples, H, W, C)` for images |
| `y_train` | array | Training labels/targets, shape: `(n_samples,)` |
| `x_test` | array | Test features for evaluation |
| `y_test` | array | Test labels/targets |
| `labels` | list | List of class names or `['target']` for regression |
| `path` | str | Directory path to save models and results |
| `model_name` | str | Base name for saved model files |
| `epochs` | int | Number of training epochs per generation |
| `generations` | int | Number of structure evolution cycles |
| `input_size` | int | Flattened input size (e.g., 784 for MNIST, 3072 for CIFAR-10) |
| `hidden_size` | int | Initial hidden layer size (will evolve during training) |
| `output_size` | int | Number of output classes (or 1 for regression) |
| `input_shape` | tuple | Image shape `(H, W, C)` for conv layers, `None` for dense only |
| `kernel_size` | int | Convolution kernel size (e.g., 3 for 3x3), `None` for dense |
| `deepth` | int | Number of convolution filters, `None` for dense |

### Optional Arguments with Defaults

| Argument | Default | Description |
|----------|---------|-------------|
| `input_paths` | `1` | Number of input paths (usually 1) |
| `batch_size` | `128` | Batch size for gradient descent |
| `simulation_set_size` | `20` | Number of samples used in simulation |
| `simulation_alg` | `montecarlo_alg` | Algorithm for action selection |
| `simulation_scheduler` | `PROGRESS_CHECK, 60s, 20 epochs` | When to run simulations |
| `lr_scheduler` | `PROGRESIVE, 0.03, 0.8` | Learning rate schedule |
| `loss_function` | `multiclass_cross_entropy` | Loss function (`MSE` for regression) |
| `activation_fun` | `Sigmoid` | Hidden layer activation (`Linear` for regression) |
| `output_activation_fun` | `SoftMax` | Output activation (`Linear` for regression) |
| `optimizer` | `SGDOptimizer()` | Optimizer (`AdamOptimizer()` recommended) |
| `simulation_score` | `weight_acc=1.0, weight_countW=0.5` | Scoring weights |
| `stopper` | `EmptyStopper()` | Early stopping condition |
| `quiet` | `False` | Suppress training output |

### Key Configuration Tips
1. **For Classification**: Use `SoftMax` output, `multiclass_cross_entropy` loss, `Sigmoid`/`ReLU` activations
2. **For Regression**: Use `Linear` activation/output, `MSE`/`MAE` loss, `weight_loss=1.0` in scoring
3. **For Images**: Set `input_shape`, `kernel_size`, `deepth` to enable convolutional layers
4. **For Speed**: Reduce `simulation_time`, `epochs`, or use `PROGRESS_CHECK` instead of `CONSTANT`
5. **For Accuracy**: Increase `epochs`, `generations`, use `AdamOptimizer()`, higher `weight_acc`
6. **For Parameter Reduction**: Use two-phase training with higher `weight_countW` in phase 2
