# Training Function

## Overview

The `train` function is responsible for training a neural network model using gradient descent while incorporating simulation-based optimizations. The function handles data preprocessing, model initialization, training, and simulation-driven improvements.

## Function Signature

```python
train(x_train, x_test, y_train, y_test, labels, path, model_name, epochs, generations, input_size, hidden_size, output_size, input_shape, kernel_size, deepth, batch_size=128, simulation_set_size=20, simulation_alg=montecarlo_alg, sim_set_generator=create_simulation_set_SAMLE, simulation_scheduler=SimulationScheduler(SimulationScheduler.PROGRESS_CHECK, simulation_time=60, simulation_epochs=20), lr_scheduler=LearningRateScheduler(LearningRateScheduler.PROGRESIVE, 0.03, 0.8), loss_function=Loss.multiclass_cross_entropy, activation_fun=Activations.Sigmoid, input_paths=1, sample_sub_generator=None, simulation_score=Simulation_score(), optimizer=SGDOptimizer())
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x_train` | array-like | Training feature data |
| `x_test` | array-like | Testing feature data |
| `y_train` | array-like | Training labels |
| `y_test` | array-like | Testing labels |
| `labels` | list | List of label names |
| `path` | str | Directory path for saving model and history |
| `model_name` | str | Name of the model to be saved |
| `epochs` | int | Number of epochs for each training phase |
| `generations` | int | Number of training iterations with simulations |
| `input_size` | int | Number of input neurons |
| `hidden_size` | int | Number of hidden neurons |
| `output_size` | int | Number of output neurons |
| `input_shape` | tuple or None | Shape of input data (for convolutional mode) |
| `kernel_size` | int | Size of convolution kernel |
| `deepth` | int | Depth of convolution layers |
| `batch_size` | int | Batch size for training (default: 128) |
| `simulation_set_size` | int | Number of samples used in simulation (default: 20) |
| `simulation_alg` | object | Algorithm used for simulations (default: `montecarlo_alg`) |
| `sim_set_generator` | function | Function for generating simulation set |
| `simulation_scheduler` | object | Scheduler controlling simulation frequency |
| `lr_scheduler` | object | Learning rate scheduler |
| `loss_function` | function | Loss function used during training |
| `activation_fun` | function | Activation function used in the model |
| `input_paths` | int | Number of input paths for model |
| `sample_sub_generator` | function or None | Function for generating sample subsets (default: None) |
| `simulation_score` | object | Scoring function for simulations |
| `optimizer` | object | Optimizer used for gradient descent (default: `SGDOptimizer`) |

## Returns

- A trained `Model` instance after applying training and simulation steps.

## Example Usage

### Basic Training Example

```python
from training_module import train

# Sample data (replace with actual dataset)
x_train = [[0.1, 0.2], [0.3, 0.4]]
x_test = [[0.5, 0.6]]
y_train = [0, 1]
y_test = [1]
labels = ['class_0', 'class_1']

# Define parameters
path = "./model_output/"
model_name = "neural_net"
epochs = 10
generations = 5
input_size = 2
hidden_size = 4
output_size = 2
input_shape = None
kernel_size = 3
deepth = 2

# Train model
model = train(x_train, x_test, y_train, y_test, labels, path, model_name, epochs, generations, input_size, hidden_size, output_size, input_shape, kernel_size, deepth)
```

### Training with Custom Learning Rate Scheduler

```python
from training_module import train, LearningRateScheduler

lr_scheduler = LearningRateScheduler(LearningRateScheduler.EXPONENTIAL, 0.05, 0.9)

model = train(x_train, x_test, y_train, y_test, labels, path, model_name, epochs, generations, input_size, hidden_size, output_size, input_shape, kernel_size, deepth, lr_scheduler=lr_scheduler)
```

### Training with Simulation Algorithm

```python
from training_module import train, montecarlo_alg, create_simulation_set_SAMLE

sim_alg = montecarlo_alg
sim_set_gen = create_simulation_set_SAMLE

model = train(x_train, x_test, y_train, y_test, labels, path, model_name, epochs, generations, input_size, hidden_size, output_size, input_shape, kernel_size, deepth, simulation_alg=sim_alg, sim_set_generator=sim_set_gen)
```

## Regression Training

The growingnn library supports regression tasks. Regression models predict continuous values instead of discrete classes.

### Key Components for Regression

1. **Loss Functions**: 
   - `Loss.MSE` - Mean Squared Error (most common for regression)
   - `Loss.MAE` - Mean Absolute Error (robust to outliers)

2. **Activation Functions**:
   - `Activations.Linear` - Identity function (no transformation)

3. **Model Configuration**:
   - `output_size = 1` for single-value regression
   - `one_hot_needed = False` in gradient descent

4. **Trainer Configuration**:
   - Use `gnn.trainer.train` with regression-specific parameters
   - Set `loss_function=gnn.Loss.MSE` or `gnn.Loss.MAE`
   - Configure `activation_fun=gnn.Activations.Linear` and `output_activation_fun=gnn.Activations.Linear`
   - Use smaller learning rates (0.001-0.01) for stable convergence
   - Set `simulation_score` with `weight_acc=0.0, weight_loss=1.0` to focus on loss minimization

### Simple Regression Training Example

```python
import growingnn as gnn
import numpy as np

# Generate simple linear regression data: y = 3x
x_train = np.arange(1, 5)
y_train = x_train * 3

# Create regression model
model = gnn.Model(
    input_size=1,
    hidden_size=20,
    output_size=1,
    loss_function=gnn.Loss.MSE,
    activation_fun=gnn.Activations.Linear,
    output_activation_fun=gnn.Activations.Linear,
    input_paths=1,
    _optimizer=gnn.SGDOptimizer()
)

# Configure learning rate
lr_scheduler = gnn.LearningRateScheduler(gnn.LearningRateScheduler.CONSTANT, 0.001)

# Train the model
final_loss, history = model.gradient_descent(
    X=x_train,
    Y=y_train,
    iterations=100,
    lr_scheduler=lr_scheduler,
    quiet=True,
    one_hot_needed=False  # Important for regression!
)

print(f"Final loss: {final_loss}")
```

### Complex Regression Training Example

```python
import growingnn as gnn
import numpy as np
import tempfile

# Generate quadratic regression data: y = x²
x_train = np.arange(1, 5)
y_train = x_train ** 2

# Use the trainer for more advanced training with simulations
temp_dir = tempfile.mkdtemp()
model = gnn.trainer.train(
    x_train=x_train,
    y_train=y_train,
    x_test=x_train,  # Using same data for simplicity
    y_test=y_train,
    labels=['Y'],
    input_paths=1,
    path=temp_dir,
    model_name="quadratic_regression",
    epochs=200,
    generations=2,
    input_size=1,
    hidden_size=5,
    output_size=1,
    input_shape=None,
    kernel_size=None,
    batch_size=10,
    activation_fun=gnn.Activations.Linear,
    output_activation_fun=gnn.Activations.Linear,
    loss_function=gnn.Loss.MSE,
    lr_scheduler=gnn.LearningRateScheduler(
        gnn.LearningRateScheduler.CONSTANT, 
        0.001, 
        0.5
    ),
    simulation_scheduler=gnn.SimulationScheduler(
        gnn.SimulationScheduler.CONSTANT, 
        simulation_time=5, 
        simulation_epochs=100
    ),
    simulation_score=gnn.Simulation_score(weight_acc=0.0, weight_loss=1.0),
    deepth=None,
    quiet=True,
    simulation_alg=gnn.montecarlo_alg,
    optimizer=gnn.SGDOptimizer()
)
```
