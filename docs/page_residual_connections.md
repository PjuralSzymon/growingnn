# Matrix-Extended Residual Connections

## Overview

GrowingNN uses a novel approach to residual connections called **matrix-extended residual connections** with matrix extension. Unlike traditional ResNet architectures that use summation points for residual connections, our method uses direct connections between neurons, which makes structural modifications more effective.

## Algorithm Foundation

The described algorithm is based on Stochastic Gradient Descent (SGD) and Adam optimizer. It operates with a model represented as a directed acyclic graph of layers. Each layer is an independent node that manages its own incoming and outgoing connections to other layers.

In this algorithm, the training procedure is divided into generations and then further into epochs. One epoch consists of one forward and backward propagation.

A generation consists of a training phase with multiple epochs, followed by a structure modification phase. After training, all possible changes to the current structure are generated. We refer to these changes as actions. The simulation provides information on which action is the best, and the best action is then applied to the structure.

## Action Types

We modify the architecture using three types of actions:

1. **Sequential Layer Addition**: Adding a sequential layer between two directly connected layers
2. **Residual Layer Addition**: Adding a layer with residual connections; a layer is added between two not directly connected layers through skip connections
3. **Layer Removal**: Actions that remove an existing layer

In each generation, the algorithm generates all possible actions for the current state and evaluates them using a Monte Carlo simulation to determine and execute the best action.

The graph starts from the smallest possible setup: an input and output layer linked by a single connection and evolves over generations by adding or removing layers. Architecture changes in a given generation only if there is no improvement in accuracy, which means that if a given structure is capable of learning a given dataset, training continues without a change in the structure.

## Matrix-Extended Residual Connections

New residual connections with matrix extension are based on reshaping matrices. This helps the neural network adjust to the new connections without losing information.

In the first epoch, after adding a new connection, the neural network will return exactly the same output, but the number of all trainable weights will increase. This means that the layer will not forget any data it has already learned, but it will have a larger number of weights that are initially set to zero.

### Mathematical Formulation

**Before adding new connection:**

```
z^[l] = W^[l] · a^[0] + b^[l]
```

**After adding n connections with classical ResNet approach:**

```
z^[l] = W^[l] · Σ(i=0 to n) a^[i] + b^[l]
```

**After adding n connections with Matrix-extended ResNet:**

```
z^[l] = [W^[l]     ]   [a^[0]  ...  a^[n]]   + b^[l]
        [  ...     ]
        [   0      ]
```

### Key Concepts

In the presented formulas:

- **a** represents the data forwarded from the preceding layer
- **W** is the weight matrix
- **b** is the bias
- **z^[l]** denotes the output of a layer l, which is subsequently processed by an activation function and passed to the next layers

To handle multiple input signals without data loss, we extend the weight matrix by adding rows filled with zero values and combining all incoming signals into a single matrix in a column-wise manner. The output remains the same as before during the first epoch after introducing this change (i.e., after adding the new connections). However, in subsequent backpropagation steps, these newly added zero weights become trainable, enabling additional flexibility for the layer.

This approach allows us to add or remove a specific number of layers without losing data or functionality.

## Advantages Over Traditional ResNet

### Backpropagation Efficiency

With our approach, the backpropagated error through multiple residual connections does not need to be duplicated. Instead, only the error related to the specific part of the weight matrix corresponding to a given layer's input is propagated. This ensures that each layer receives a backpropagated error that is directly calculated for it.

### Increased Flexibility

Our new approach can be interpreted as increasing the number of connections but not the number of neurons, which brings a new field of possibilities to the algorithms that change the structure while training. Adding those connections can not only bring changes to the structure but also increase the number of trainable weights inside the layer without causing any loss in network memory.

### Neuron Removal Benefits

This design makes neuron removal more effective, as removing neurons directly affects all connected layers. Unlike traditional ResNet where residual connections are merged using summation points, our matrix-extended residual connections are formed through direct connections between neurons. This means that when neurons are removed, the algorithm can directly identify and adjust the corresponding connections in all affected layers.

## Connection to Neuron Removal

Removing neurons in one layer also affects the next connected layers by rescaling all connections related to the removed neurons. In those structures, it is possible to have many residual connections.

For each layer **ℓ'** receiving input from layer **ℓ**, the algorithm identifies the column range **[s, e]** in **W_ℓ'** that corresponds to the output of layer **ℓ**.

The weight matrix is partitioned as:

```
W_ℓ' = [W_before | W_middle | W_after]
```

where **W_middle** (size: n_ℓ' × n) corresponds to layer **ℓ**'s contribution. Only **W_middle** is rescaled:

```
W'_ℓ' = [W_before | W_middle · Q_I | W_after]
```

In layer $\ell$, the weight matrix rescaling was performed to reduce the number of neurons in a given layer. But in layer $\ell'$ rescaling aims to reduce the number of connections, effectively reducing the total number of parameters in the network. The matrices $W_{\text{before}}$ or $W_{\text{after}}$ may be empty.

## Summary

Matrix-extended residual connections provide a more flexible and efficient approach to handling skip connections in dynamically evolving neural networks. By using direct neuron connections rather than summation points, the method enables more effective structural modifications, particularly for neuron removal operations, while preserving learned information during architectural changes.

