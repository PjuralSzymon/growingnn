# Method Overview

## Core Philosophy

GrowingNN is fundamentally built upon **Stochastic Gradient Descent (SGD)** and relies on basic methods, avoiding advanced tools commonly used in training neural networks. This simplicity makes it ideal for **educational settings**, focusing on fundamental machine learning principles.

## Model Architecture

### Graph-Based Structure

The model is the main structure that stores layers as nodes in a **directed acyclic graph (DAG)**. Key characteristics:

- **Layer Identifiers**: The system operates on layer identifiers rather than sequential positions
- **Independent Layers**: Each layer is an independent structure that manages its own:
  - Incoming connections (input layers)
  - Outgoing connections (output layers)
- **Default Structure**: Starts with the smallest possible setup:
  - One input layer
  - One output layer
  - A single connection between them

### Evolution Process

As training progresses through generations:

1. **Layer Addition**: New layers may be added between existing layers
2. **Layer Removal**: Existing layers may be removed (with automatic reconnection)
3. **Connection Growth**: As structure grows, each layer gains more incoming and outgoing connections
4. **Neuron Modification**: Neurons can be added or removed within layers

## Training Process

### Generation-Based Training

Training is divided into **generations**, where each generation consists of:

1. **Training Phase**: Multiple epochs of gradient descent
2. **Structure Modification Phase**: Potential architectural changes based on simulation

### Propagation Mechanism

During forward propagation:

1. **Signal Collection**: Each layer waits until it receives signals from **all** input layers
2. **Signal Averaging**: Once all signals are received, they are averaged together
3. **Processing**: The averaged signal is processed through the layer's transformation (weights, activation)
4. **Propagation**: The processed signal is propagated through all outgoing connections

This mechanism allows the network to handle:
- Multiple input paths
- Residual connections
- Complex graph topologies

## Simulation Orchestrator

### Purpose

The simulation orchestrator (implemented as `canSimulate()`) determines **when** a simulation should be executed during the learning process. This timing is critical for maintaining balance between:

- **Exploration**: Trying new structures
- **Exploitation**: Fully training current structures

### The Balance Problem

- **Too Frequent Changes**: May prevent a particular structure from full training
- **Too Infrequent Changes**: May lead to:
  - Continuously falling into local minima
  - Excessively long learning time
  - Inefficient method

### Progress Check Method

The default approach is **progress check**:

1. After each generation, check if there has been improvement in the model's learning
2. If **no improvement** is detected, run the simulation
3. If improvement is detected, continue training without structural changes

This means: **If a given structure is capable of learning the dataset, training continues without changes to the structure.**

### Implementation

```python
simulation_scheduler = gnn.SimulationScheduler(
    gnn.SimulationScheduler.PROGRESS_CHECK,  # Progress-based triggering
    simulation_time=300,
    simulation_epochs=20,
    progres_delta=0.01  # Minimum improvement threshold
)
```

## Progressive Learning Rate

### Custom Implementation

The algorithm implements a custom modification of the progressive learning rate based on Schaul et al. This learning rate schedule plays a crucial role in minimizing the impact of structural changes.

### Behavior Pattern

Within a single generation, the learning rate follows this pattern:

1. **Initial Epoch (after structure change)**: Learning rate is very close to zero
2. **Growth Phase**: Learning rate grows to a constant maximum value
3. **Constant Phase**: Maintains maximum value
4. **Decay Phase**: Learning rate slowly decreases before the next potential action

### Mathematical Description

The learning rate $\alpha(t)$ within a generation of $T$ epochs:

- **Early epochs**: $\alpha(t) \approx 0$ (minimal learning)
- **Middle epochs**: $\alpha(t)$ increases to $\alpha_{\max}$
- **Late epochs**: $\alpha(t)$ decreases from $\alpha_{\max}$

### Purpose

This pattern **minimizes the negative impact of introduced changes on already learned information** in the network. By starting with a very low learning rate after structural modifications, the network can:

1. Preserve existing knowledge
2. Gradually adapt to the new structure
3. Avoid catastrophic forgetting

### Implementation

```python
lr_scheduler = gnn.LearningRateScheduler(
    gnn.LearningRateScheduler.PROGRESIVE,  # Progressive schedule
    initial_lr=0.005,                      # Maximum learning rate
    decay_factor=0.8                       # Decay rate
)
```

## Connection Types

### Sequential Connections

**Definition**: Direct connections between layers in sequence.

**Action**: `Add_Seq_Layer` adds a layer between two directly connected layers, breaking the direct connection and inserting the new layer.

### Matrix-Extended Residual Connections

**Definition**: Skip connections that bypass one or more layers, but unlike traditional ResNet, these use **direct connections between neurons** rather than summation points.

**Key Difference from ResNet**:
- **Traditional ResNet**: Residual connections merge using summation points: $y = f(x) + x$
- **GrowingNN**: Matrix-extended residual connections use direct neuron-to-neuron connections

**Benefits**:
- More granular control over information flow
- Makes neuron removal more effective
- Directly affects all connected layers when neurons are removed

**Action**: `Add_Res_Layer` adds a layer with residual connections between non-directly connected layers.

### Layer Removal

**Action**: `Del_Layer` removes an existing layer and automatically reconnects:
- Input layers of the removed layer → Output layers of the removed layer
- Maintains graph connectivity
- Prevents cycles in the structural graph

## Action Types

The algorithm modifies architecture using three main types of actions:

1. **Sequential Layer Addition**: Add layer between directly connected layers
2. **Residual Layer Addition**: Add layer with skip connections
3. **Layer Removal**: Remove existing layer with automatic reconnection

Additionally, the system supports:
4. **Neuron Removal**: Reduce neurons in a layer (10%, 50%, 90%)
5. **Neuron Addition**: Expand neurons in a layer (150%, 200%)

## Algorithm Flow

```
Initialization:
    SimSet ← Create simulation set
    Model ← Create model with basic structure (input → output)

For each generation:
    1. GradientDescent(Model, Dataset, epochs)
       - Train current structure
       - Use progressive learning rate
       
    2. If canSimulate():
       - Check if learning has improved
       - If no improvement:
           a. Generate all possible actions
           b. Action ← MCTS.simulation(Model)
           c. Model ← Action.execute(Model)
           d. Adjust learning rate (start near zero)
```

## Use Cases

### 1. Training from Scratch

The primary use case: train a network that dynamically evolves its structure during training.

**Characteristics**:
- Starts from minimal structure
- Grows based on learning needs
- Structure changes only when learning stalls
- Focus on accuracy improvement

### 2. Optimizing Existing Networks

Secondary use case: take an already-trained network and optimize it for parameter efficiency.

**Characteristics**:
- Starts from pre-trained model
- Focus on parameter reduction
- Maintains accuracy while reducing size
- Two-phase approach recommended

## Educational Value

The simplicity of the approach makes it valuable for:

- **Understanding SGD**: Core training mechanism
- **Learning Neural Networks**: Basic forward/backward propagation
- **Architecture Search**: How structures can evolve
- **Graph-Based Models**: Understanding DAG representations
- **No Black Boxes**: Avoids advanced frameworks, focuses on fundamentals

## References

- Stochastic Gradient Descent: Core optimization method
- Schaul et al.: Progressive learning rate inspiration
- Monte Carlo Tree Search: Action selection mechanism
- ResNet: Inspiration for residual connections (extended in this work)

