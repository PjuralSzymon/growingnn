# Simulation-based Structural Optimization

## Overview

Optimizing already trained neural networks is one of the core problems in the domain of Artificial Intelligence. In this paper, we present a new approach capable of optimizing the structure of an already trained neural network. We present a new way of removing neurons from layers that is based on matrix scaling, which truly decreases the number of parameters in a model rather than zeroing weights. We present a simulation-based approach for optimizing the structure of a neural network that can select the best change to the network without causing data loss. In experiment section we demonstrate that GrowingNN can optimize already trained neural networks, achieving up to 85% parameter reduction while maintaining approximately 95% training accuracy.

From a high-level point of view, the proposed method aims to modify the structure of the neural network to improve the accuracy that takes place while the network is training. Another viable scenario for the proposed method is to take an already-trained neural network or a pre-trained neural network and use our method as a fine-tuning tool.

## Simulation Description

The decision-making process for choosing the best action uses a variant of Monte Carlo Tree Search (MCTS), adapted for neural network structure optimization.

Training is divided into generations. In each generation, the simulation starts from the current model **M** and expands it by considering all possible actions **a ∈ A(M)**. Each action produces a new candidate model **M_a**.

We then evaluate **M_a** using a short training step and assign it a score **S(M_a)** based on the weighted sum described earlier. In the expansion phase, every action **a** is applied to the current model **M**, producing a new child node with model **M_a = a(M)**.

From each expanded model **M_a**, we simulate a random sequence of further actions until a maximum depth **d** is reached. During the rollout phase, each intermediate model is briefly trained and its performance is graded. The final score from the rollout is used to estimate the long-term value of action **a**. To balance exploration and exploitation, we use the UCB1 criterion:

```
UCB1(n) = v(n) + c · √(ln(N) / n)
```

### Simulation Score Function

The score of **M_a** is defined as a weighted combination of accuracy and parameter count:

```
S(M_a) = (w_acc · S_acc(M_a) + w_params · (1 / (1 + β · p_a))) / (w_acc + w_params)
```

where:

- **S_acc(M_a)** denotes the short-run training accuracy of the model
- **w_acc** is fixed at 1.0 in all experiments, serving as a reference since accuracy is the main optimization objective
- **p_a** represents the total number of parameters in M_a, including weights, biases, and filter parameters across all layers
- **β** (currently set to 0.001) adjusts how the grade for the number of parameters is computed
- **w_params** controls how strongly parameter count affects the simulation score and is a key focus of our experiments

Since models are compared from the same initial structure, those with similar accuracy are preferred when they have fewer parameters.

## Structural Modifications: Neuron Removal and Addition

The simulation process selects the best action in each generation. Each action can modify the structure of the neural network in a predefined way. In our previous papers, we introduced actions that could add or remove layers.

The contribution presented in this work concerns a much more difficult optimization task, which requires a wider set of possibilities for the simulation to choose from. We introduce new actions that can **add or remove neurons**. The removal of neurons from a layer works differently than in currently available approaches. This entails that we introduce a **novel mathematical representation for the removal action**. Instead of setting the weights to zero, we fully remove neurons and reduce the size of the corresponding weight matrices, which truly decreases the number of parameters in a model.

A key objective of this process is to **preserve previously learned information**. From an algebraic point of view, a dense layer can be seen as a linear transformation followed by an activation function. Therefore, we reduce the layer size by applying a scaling operation to the weight and bias matrix that allows us to control how much information is lost. Larger scaling factors, which correspond to the removal of more neurons, may result in greater loss of information.

### Mathematical Formulation

The neuron removal action in some layer **ℓ** reduces the number of neurons from **n** to **n'** by scaling the weight matrix **W_ℓ** (size: n × d) and bias vector **b_ℓ** (size: n) using a quasi-identity matrix **Q** (size: n' × n):

```
W'_ℓ = Q_I · W_ℓ
b'_ℓ = Q_I · b_ℓ
```

A quasi-identity matrix **Q** is constructed by resizing an identity matrix, preserving information through optimal linear projection rather than truncation or averaging. This approach is very beneficial because if the reduction ratio is small, for example **r = 0.1**, the matrix **Q** is almost identical to the identity matrix. From the formulas above, we can then assume that **W'_ℓ ≈ W_ℓ**.

It is important to note that **Q_I** can be used to remove neurons or connections to neurons.

### Reduction Ratios

Using this neuron removal procedure has limitations. The simulation mechanism cannot generate all possible ways to reduce neurons. When generating neuron reduction actions, the simulation iterates over each hidden layer and creates an action that reduces that layer by a fixed percentage. Due to the large number of possible actions, generating too many of them would make the simulation inefficient and difficult to evaluate. Therefore, the number of generated actions must be limited.

After extensive experimentation, we chose to allow the simulation to generate three types of neuron reduction: **10%**, **50%**, and **90%**.

Similarly, to generate new neurons, we use the same approach and allow neuron expansion by **150%** and **200%**. This means that if a model has **n** hidden layers, at each simulation step we add an additional **5n** possible actions. Due to this increase in action space, a time limit on the simulation was extended to 12 minutes in our experiments.

### Propagation to Connected Layers

Removing neurons in one layer also affects the next connected layers by rescaling all connections related to the removed neurons. In those structures, it is possible to have many residual connections. In the original ResNet paper, residual connections are merged using summation points. In contrast, our method uses our custom **matrix-extended residual connections**, where residual connections are formed through direct connections between neurons rather than summation points.

This design makes neuron removal more effective, as removing neurons directly affects all connected layers. To apply this correctly, the algorithm must identify which connections should be removed in all layers connected to the layer from which neurons were removed.

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

## Two-Phase Optimization Approach

The method can be used in two phases:

1. **Phase 1: Accuracy Optimization**: Focus on achieving high accuracy with `weight_countW` set to a low value (e.g., 0.1)
2. **Phase 2: Parameter Reduction**: Optimize for parameter reduction while maintaining accuracy with `weight_countW` set to a higher value (e.g., 4.0)

### Example: Two-Phase Training

```python
import growingnn as gnn
import numpy as np

# Load your dataset
x_train, x_test, y_train, y_test = load_and_preprocess_dataset(...)

# Phase 1: Train for accuracy
M = gnn.trainer.train(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    labels=list(range(output_size)),
    input_paths=1,
    path="./results/",
    model_name="GNN_model_phase1",
    epochs=50,
    generations=20,
    input_size=input_size,
    hidden_size=30,
    output_size=output_size,
    input_shape=input_shape,
    kernel_size=3,
    deepth=1,
    lr_scheduler=gnn.LearningRateScheduler(
        gnn.LearningRateScheduler.PROGRESIVE, 0.005, 0.8
    ),
    simulation_scheduler=gnn.SimulationScheduler(
        gnn.SimulationScheduler.CONSTANT,
        simulation_time=300,
        simulation_epochs=20
    ),
    optimizer=gnn.AdamOptimizer(),
    simulation_score=gnn.Simulation_score(
        weight_acc=1.0,
        weight_countW=0.1  # Low weight: focus on accuracy
    ),
    simulation_alg=gnn.montecarlo_alg,
    stopper=gnn.AccuracyStopper(0.9)
)

# Evaluate Phase 1
accuracy_before = M.evaluate(x_train, y_train)
param_count_before = M.get_parametr_count()

# Phase 2: Optimize for parameter reduction
M = gnn.trainer.train_continue(
    M=M,  # Continue from Phase 1 model
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    labels=list(range(output_size)),
    input_paths=1,
    path="./results/",
    model_name="GNN_model_phase2",
    epochs=50,
    generations=20,
    input_size=input_size,
    hidden_size=30,
    output_size=output_size,
    input_shape=input_shape,
    kernel_size=3,
    deepth=1,
    lr_scheduler=gnn.LearningRateScheduler(
        gnn.LearningRateScheduler.PROGRESIVE, 0.005, 0.8
    ),
    simulation_scheduler=gnn.SimulationScheduler(
        gnn.SimulationScheduler.CONSTANT,
        simulation_time=300,
        simulation_epochs=20
    ),
    optimizer=gnn.AdamOptimizer(),
    simulation_score=gnn.Simulation_score(
        weight_acc=1.0,
        weight_countW=4.0  # High weight: focus on parameter reduction
    ),
    simulation_alg=gnn.montecarlo_alg,
    stopper=gnn.AccuracyAndReductionStopper(0.8, 0.1)
)

# Evaluate Phase 2
accuracy_after = M.evaluate(x_train, y_train)
param_count_after = M.get_parametr_count()

# Calculate improvements
param_reduction = (1 - param_count_after / param_count_before) * 100
acc_change = (accuracy_after - accuracy_before) * 100

print(f"Phase 1 - Accuracy: {accuracy_before:.4f}, Parameters: {param_count_before:,}")
print(f"Phase 2 - Accuracy: {accuracy_after:.4f}, Parameters: {param_count_after:,}")
print(f"Parameter reduction: {param_reduction:.1f}%")
print(f"Accuracy change: {acc_change:+.2f}%")
```

## Key Benefits

- **Information Preservation**: Quasi-identity matrices preserve learned information during neuron removal
- **Parameter Reduction**: Can achieve significant parameter reduction (up to 85%) while maintaining accuracy
- **Flexible Optimization**: Two-phase approach allows balancing accuracy and efficiency
- **Effective with Residual Connections**: Works particularly well with matrix-extended residual connections

## Summary

The simulation-based structural optimization approach provides a powerful method for optimizing already trained neural networks. By using Monte Carlo Tree Search to select optimal structural changes and quasi-identity matrices for neuron removal, the method can effectively reduce parameters in trained networks (up to 85% reduction) while maintaining high accuracy (approximately 95% training accuracy). This makes it an effective tool for fine-tuning and optimizing pre-trained neural networks without causing data loss.

