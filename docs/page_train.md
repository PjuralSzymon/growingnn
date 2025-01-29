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

## Notes
- Ensure that the dataset is properly formatted before passing it to the `train` function.
- The function includes a simulation-based improvement mechanism that optimizes model performance through iterative refinements.
- Various parameters such as `simulation_alg`, `lr_scheduler`, and `optimizer` allow customization of the training process.

## References
- **Simulation Scheduler**: Used to control the timing and execution of simulations.
- **Learning Rate Scheduler**: Defines the learning rate adaptation strategy.
- **Simulation Algorithms**: Improve model performance through reinforcement learning-based exploration.

