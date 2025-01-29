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


## Examples with training a model

### **1. Training a Model with Stochastic Gradient Descent (SGD)**  

###### **Description**  
This example demonstrates how to train a simple neural network using **SGDOptimizer** with a **multiclass cross-entropy loss function** and **Sigmoid activation**.  

###### **Example Code**  
```python
import numpy as np
import growingnn as gnn

# Define model parameters
shape = 20
epochs = 5

# Create a model
model = gnn.structure.Model(
    input_size=shape,
    hidden_size=shape,
    output_size=2,
    loss_function=gnn.structure.Loss.multiclass_cross_entropy,
    activation_function=gnn.structure.Activations.Sigmoid,
    deepth=1,
    optimizer=gnn.optimizers.SGDOptimizer()
)

# Generate random training data
x_train = np.random.rand(shape, shape)
y_train = np.random.randint(2, size=(shape,))

# Define learning rate scheduler
lr_scheduler = gnn.structure.LearningRateScheduler(
    gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8
)

# Train the model
accuracy, _ = model.gradient_descent(x_train, y_train, epochs, lr_scheduler)

# Print the final accuracy
print(f"Final Training Accuracy: {accuracy:.2f}")
```

---

### **2. Training a Model with Adam Optimizer**  

###### **Description**  
This example shows how to train a model using **AdamOptimizer**, which typically results in faster convergence than SGD.  

###### **Example Code**  
```python
import numpy as np
import growingnn as gnn

shape = 20
epochs = 5

model = gnn.structure.Model(
    shape, shape, 2,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1, gnn.optimizers.AdamOptimizer()
)

x_train = np.random.rand(shape, shape)
y_train = np.random.randint(2, size=(shape,))

lr_scheduler = gnn.structure.LearningRateScheduler(
    gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8
)

accuracy, _ = model.gradient_descent(x_train, y_train, epochs, lr_scheduler)
print(f"Final Training Accuracy: {accuracy:.2f}")
```

---

### **3. Adding a Residual Layer to a Model**  

###### **Description**  
This example demonstrates how to add a **residual layer** to a neural network model, which can help with training deeper networks.  

###### **Example Code**  
```python
import numpy as np
import growingnn as gnn

shape = 20

model = gnn.structure.Model(
    shape, shape, 2,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1, gnn.optimizers.SGDOptimizer()
)

# Add a residual layer
model.add_res_layer('init_0', 1)

# Generate random input data
x = np.random.rand(shape, shape)

# Perform forward propagation multiple times
for _ in range(10):
    output = model.forward_prop(x * float(np.random.rand(1)))

print("Residual Layer Output Shape:", output.shape)
```

---

### **4. Training a Convolutional Model with SGD**  

###### **Description**  
This example shows how to define and train a **convolutional model** using **SGDOptimizer**.  

###### **Example Code**  
```python
import numpy as np
import growingnn as gnn

shape = 20

# Create a convolutional model
model = gnn.structure.Model(
    shape, shape, shape,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1, gnn.optimizers.SGDOptimizer()
)

# Enable convolution mode
model.set_convolution_mode((shape, shape, 1), shape, 1)

# Add a residual layer
model.add_res_layer('init_0', 1)

# Generate random input data
x = np.random.random((shape, shape, shape, 1))

# Perform forward propagation multiple times
for _ in range(10):
    output = model.forward_prop(x * float(np.random.rand(1)))

print("Convolutional Model Output Shape:", output.shape)
```

---

### **5. Saving and Loading a Model**  

###### **Description**  
This example shows how to **save** a trained model and **load** it later for inference.  

###### **Example Code**  
```python
import numpy as np
import growingnn as gnn

# Create a model
model = gnn.structure.Model(
    3, 3, 1,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1
)

# Generate random input data
x = np.random.rand(3, 3)

# Forward propagate to get initial output
output1 = model.forward_prop(x)

# Save the model
gnn.Storage.saveModel(model, "model.json")

# Load the model
loaded_model = gnn.Storage.loadModel("model.json")

# Forward propagate using the loaded model
output2 = loaded_model.forward_prop(x)

# Check if outputs are the same
print("Difference between original and loaded model outputs:", np.sum(output1 - output2))
```

---

### **6. Comparing Adam vs. SGD Optimizer Performance**  

###### **Description**  
This example trains the same model using both **AdamOptimizer** and **SGDOptimizer** and compares their accuracy.  

###### **Example Code**  
```python
import numpy as np
import growingnn as gnn

shape = 20
epochs = 5

# Train with Adam
model_adam = gnn.structure.Model(
    shape, shape, 2,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1, gnn.optimizers.AdamOptimizer()
)

x_train = np.random.rand(shape, shape)
y_train = np.random.randint(2, size=(shape,))

accuracy_adam, _ = model_adam.gradient_descent(x_train, y_train, epochs)
print(f"Adam Optimizer Accuracy: {accuracy_adam:.2f}")

# Train with SGD
model_sgd = gnn.structure.Model(
    shape, shape, 2,
    gnn.structure.Loss.multiclass_cross_entropy,
    gnn.structure.Activations.Sigmoid,
    1, gnn.optimizers.SGDOptimizer()
)

accuracy_sgd, _ = model_sgd.gradient_descent(x_train, y_train, epochs)
print(f"SGD Optimizer Accuracy: {accuracy_sgd:.2f}")

# Compare performance
if accuracy_adam > accuracy_sgd:
    print("Adam performs better.")
else:
    print("SGD performs better.")
```

---
