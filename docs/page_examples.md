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