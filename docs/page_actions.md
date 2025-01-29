# Action Module Documentation

## Overview

The `Action` module is responsible for managing structural modifications to a neural network model. It defines different actions that can be performed on a model, such as adding or removing layers. This module enables dynamic modifications to the network structure, enhancing the adaptability of the learning process.

In our approach, we utilize a structured search method where different actions are generated and evaluated. Each action affects the model's topology, and by executing these actions in a controlled manner, we enable adaptive learning. Inspired by Schaul et al. \cite{DBLP\:journals/corr/Smith15a}, we allow progressive modifications to the structure, ensuring that drastic changes do not disrupt the learning process.

## Action Class Hierarchy

### `Action`

The base class for all actions. Each action must implement:

- `execute(Model)`: Applies the action to the given model.
- `can_be_influenced(by_action)`: Determines if an action is affected by another action.
- `generate_all_actions(Model)`: Generates all possible instances of the action for a given model.

### `Add_Seq_Layer`

Adds a sequential layer between two existing layers. The layer type is determined based on the model's current state.

- **Execution:** `Model.add_norm_layer(layer1, layer2, layer_type)`
- **Influence:** A delete action on either layer1 or layer2 affects this action.
- **Generation:** Iterates over sequence connections to propose new layers.

#### Example:

```python
action = Add_Seq_Layer([layer_1, layer_2, Layer_Type.EYE])
action.execute(model)
```

### `Add_Res_Layer`

Adds a residual connection between two layers, allowing gradient flow across multiple layers.

- **Execution:** `Model.add_res_layer(layer1, layer2, layer_type)`
- **Influence:** A delete action on either layer1 or layer2 affects this action.
- **Generation:** Identifies child-parent connections and proposes residual links.

#### Example:

```python
action = Add_Res_Layer([layer_1, layer_2, Layer_Type.RANDOM])
action.execute(model)
```

### `Del_Layer`

Removes a layer from the model, updating all associated connections.

- **Execution:** `Model.remove_layer(layer_id)`
- **Influence:** Does not get influenced by other actions.
- **Generation:** Proposes deletion for each hidden layer.

#### Example:

```python
action = Del_Layer(layer_id)
action.execute(model)
```

### `Add_Seq_Conv_Layer`

Adds a sequential convolutional layer between two existing layers.

- **Execution:** `Model.add_conv_norm_layer(layer1, layer2)`
- **Influence:** A delete action on either layer1 or layer2 affects this action.
- **Generation:** Identifies potential convolutional connections.

#### Example:

```python
action = Add_Seq_Conv_Layer([conv_layer_1, conv_layer_2])
action.execute(model)
```

### `Add_Res_Conv_Layer`

Adds a residual convolutional layer between two existing layers.

- **Execution:** `Model.add_conv_res_layer(layer1, layer2)`
- **Influence:** A delete action on either layer1 or layer2 affects this action.
- **Generation:** Identifies convolutional child-parent connections.

#### Example:

```python
action = Add_Res_Conv_Layer([conv_layer_1, conv_layer_2])
action.execute(model)
```

### `Empty`

A placeholder action that does nothing, useful for maintaining structure.

- **Execution:** No effect on the model.
- **Influence:** Not influenced by any action.
- **Generation:** Always returns a single empty action.

#### Example:

```python
action = Empty(None)
action.execute(model)
```

## Summary

This module enables dynamic modifications to a neural network structure, supporting structural search and optimization techniques. The approach is designed to maintain stability while iteratively refining the network topology for improved learning performance.

