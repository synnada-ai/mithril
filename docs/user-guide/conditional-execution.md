# Conditional Execution

Conditional execution in Mithril allows you to create models with dynamic behavior based on runtime conditions. This guide covers how to implement and use conditional paths in your models.

## Basic Conditional Execution

### The `If` Model

The most basic form of conditional execution in Mithril is the `If` model, which allows you to choose between two execution paths based on a condition:

```python
from mithril.models import Model, Linear, Dropout, Relu, If, Identity

model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")

# Add dropout during training, but not during inference
model += If(
    condition="is_training",
    then_branch=Dropout(p=0.5)(input="hidden", output="hidden_processed"),
    else_branch=Identity()(input="hidden", output="hidden_processed")
)

model += Linear(dimension=10)(input="hidden_processed", output="output")
```

In this example:
- `condition` is a string that will be evaluated at runtime
- `then_branch` is the model to execute if the condition is True
- `else_branch` is the model to execute if the condition is False

### Passing Conditions at Runtime

When evaluating the model, you can pass the condition value as part of the inputs:

```python
# Training mode
outputs_train = compiled_model.evaluate(
    params,
    {"input": data, "is_training": True}
)

# Inference mode
outputs_infer = compiled_model.evaluate(
    params,
    {"input": data, "is_training": False}
)
```

## Advanced Conditional Execution

### Multiple Conditions

You can create models with multiple conditions:

```python
model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")

# Condition 1: Training mode
model += If(
    condition="is_training",
    then_branch=Dropout(p=0.5)(input="hidden", output="hidden_dropout"),
    else_branch=Identity()(input="hidden", output="hidden_dropout")
)

# Condition 2: Feature normalization
model += If(
    condition="normalize_features",
    then_branch=BatchNorm1d(num_features=64)(input="hidden_dropout", output="hidden_processed"),
    else_branch=Identity()(input="hidden_dropout", output="hidden_processed")
)

model += Linear(dimension=10)(input="hidden_processed", output="output")
```

### Nested Conditions

You can nest conditions to create more complex conditional paths:

```python
# Create nested conditional model
nested_condition = Model()
nested_condition |= If(
    condition="use_batch_norm",
    then_branch=BatchNorm1d(num_features=64)(input="input", output="normalized"),
    else_branch=Identity()(input="input", output="normalized")
)
nested_condition += Relu()(input="normalized", output="output")

# Main model with nested conditions
model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")

# Outer condition based on training mode
model += If(
    condition="is_training", 
    then_branch=Model()
        |= Dropout(p=0.5)(input="input", output="dropout")
        += nested_condition(input="dropout", output="output"),
    else_branch=nested_condition(input="input", output="output")
)(input="hidden", output="hidden_processed")

model += Linear(dimension=10)(input="hidden_processed", output="output")
```

### Computational Expressions in Conditions

You can use computational expressions in conditions:

```python
model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")

# Condition based on comparison
model += If(
    condition="hidden_norm > threshold",
    then_branch=LayerNorm(normalized_shape=64)(input="hidden", output="hidden_processed"),
    else_branch=Identity()(input="hidden", output="hidden_processed")
)

model += Linear(dimension=10)(input="hidden_processed", output="output")
```

When evaluating, you need to compute the condition inputs:

```python
# Compute hidden_norm
hidden = model_partial.evaluate(params, {"input": data})["hidden"]
hidden_norm = backend.sum(backend.square(hidden))

# Complete model evaluation with the condition
outputs = compiled_model.evaluate(
    params,
    {"input": data, "hidden_norm": hidden_norm, "threshold": 10.0}
)
```

## Conditional Routing Models

### Switch Model

The `Switch` model allows you to select one of multiple execution paths based on an index:

```python
from mithril.models import Switch

model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")

# Route through different activation functions based on index
model += Switch(
    index="activation_index",
    branches={
        0: Relu()(input="input", output="output"),
        1: LeakyRelu(negative_slope=0.1)(input="input", output="output"),
        2: Tanh()(input="input", output="output"),
        3: Sigmoid()(input="input", output="output")
    },
    default=Identity()(input="input", output="output")
)(input="hidden", output="hidden_act")

model += Linear(dimension=10)(input="hidden_act", output="output")
```

When evaluating:

```python
# Use ReLU activation (index 0)
outputs_relu = compiled_model.evaluate(
    params,
    {"input": data, "activation_index": 0}
)

# Use Tanh activation (index 2)
outputs_tanh = compiled_model.evaluate(
    params,
    {"input": data, "activation_index": 2}
)
```

### Case Model

The `Case` model is similar to a switch statement but uses predicates for conditions:

```python
from mithril.models import Case

model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")

# Case-based routing
model += Case(
    cases=[
        ("error_rate < 0.1", Dropout(p=0.1)(input="input", output="output")),
        ("error_rate < 0.3", Dropout(p=0.3)(input="input", output="output")),
        ("error_rate < 0.5", Dropout(p=0.5)(input="input", output="output"))
    ],
    default=Dropout(p=0.7)(input="input", output="output")
)(input="hidden", output="hidden_dropout")

model += Linear(dimension=10)(input="hidden_dropout", output="output")
```

## Loop Constructs

### While Loop

The `While` model allows you to create loops that execute until a condition is false:

```python
from mithril.models import While

# Create a recurrent block
recurrent_block = Model()
recurrent_block |= Linear(dimension=64)(input="state", output="new_state_pre")
recurrent_block += Tanh()(input="new_state_pre", output="new_state")
recurrent_block += Add()(left="new_state", right="input", output="output")

# Create a model with a while loop
model = Model()
model |= Linear(dimension=64)(input="input", output="initial_state")
model += While(
    condition="step < max_steps",
    body=recurrent_block,
    initial_state={"state": "initial_state", "step": 0},
    loop_vars={"state": "new_state", "step": "step + 1"},
    output_vars={"final_state": "state"}
)(output="final_state")
model += Linear(dimension=10)(input="final_state", output="output")
```

When evaluating:

```python
outputs = compiled_model.evaluate(
    params,
    {"input": data, "max_steps": 5}
)
```

### For Loop

The `For` model allows you to create loops with a fixed number of iterations:

```python
from mithril.models import For

# Create a recurrent block
recurrent_block = Model()
recurrent_block |= Linear(dimension=64)(input="state", output="new_state_pre")
recurrent_block += Tanh()(input="new_state_pre", output="new_state")

# Create a model with a for loop
model = Model()
model |= Linear(dimension=64)(input="input", output="initial_state")
model += For(
    iterations=5,
    body=recurrent_block,
    initial_state={"state": "initial_state"},
    loop_vars={"state": "new_state"},
    output_vars={"final_state": "state"}
)(output="final_state")
model += Linear(dimension=10)(input="final_state", output="output")
```

## Conditional Branching in Training

### Different Behavior During Training and Inference

One common use case for conditional execution is to have different behavior during training and inference:

```python
# Model with different behavior for training and inference
model = Model()

# Input embedding
model |= Embedding(num_embeddings=10000, embedding_dim=512)(input="input_ids", output="embeddings")

# Dropout during training only
model += If(
    condition="is_training",
    then_branch=Dropout(p=0.1)(input="embeddings", output="embeddings_processed"),
    else_branch=Identity()(input="embeddings", output="embeddings_processed")
)

# Main model body
model += Transformer(d_model=512, nhead=8, num_layers=6)(
    input="embeddings_processed", output="transformed"
)

# Output projection
model += Linear(dimension=10000)(input="transformed", output="logits")

# Teacher forcing during training, greedy decoding during inference
model += If(
    condition="is_training",
    then_branch=Identity()(input="logits", output="output"),
    else_branch=ArgMax(dim=-1)(input="logits", output="output")
)
```

### Conditional Gradient Computation

You can also use conditions to control gradient computation:

```python
# Model with conditional gradient computation
model = Model()
model |= Linear(dimension=64)(input="input", output="hidden1")
model += Linear(dimension=32)(input="hidden1", output="hidden2")

# Multiple output paths with different gradients
model += Linear(dimension=10)(input="hidden2", output="classification_output")
model += If(
    condition="compute_regression",
    then_branch=Linear(dimension=1)(input="hidden2", output="regression_output"),
    else_branch=Constant(value=0.0)(output="regression_output")  # No gradient path
)
```

When evaluating with gradients:

```python
# Compute gradients for both outputs
outputs, gradients = compiled_model.evaluate(
    params,
    {"input": data, "compute_regression": True},
    output_gradients={
        "classification_output": classification_loss_grad,
        "regression_output": regression_loss_grad
    }
)

# Compute gradients for classification only
outputs, gradients = compiled_model.evaluate(
    params,
    {"input": data, "compute_regression": False},
    output_gradients={
        "classification_output": classification_loss_grad,
        "regression_output": backend.zeros(batch_size, 1)  # Zero gradients
    }
)
```

## Implementation Details

### Backend Support

Conditional execution support varies by backend:

- **JAX Backend**: Fully supported through JAX's control flow primitives
- **PyTorch Backend**: Fully supported through PyTorch's dynamic control flow
- **NumPy Backend**: Basic support without gradient computation
- **MLX Backend**: Full support through MLX's control flow operators

### Performance Considerations

Conditional execution can impact performance in several ways:

1. **Compilation Overhead**: Models with many conditions may take longer to compile
2. **Runtime Performance**: Dynamic control flow can prevent some optimizations
3. **Memory Usage**: All potential execution paths may need to be stored

To optimize performance:

```python
# Optimize for the most common condition
compiled_model = ml.compile(
    model=model,
    backend=backend,
    optimize_condition_values={"is_training": False}  # Optimize for inference
)
```

### Tracing vs. Dynamic Execution

Some backends (like JAX) use tracing for compilation, which can limit dynamic behavior. For fully dynamic execution:

```python
# Use dynamic mode for JAX backend
compiled_model = ml.compile(
    model=model,
    backend=ml.JaxBackend(jit=False),  # Disable JIT for fully dynamic behavior
    dynamic_control_flow=True
)
```

## Best Practices

1. **Keep Conditions Simple**: Use simple, scalar conditions when possible
2. **Balance Static and Dynamic**: Use conditions only where necessary
3. **Optimize Common Paths**: Optimize for the most common execution path
4. **Test All Paths**: Ensure all conditional paths work correctly
5. **Document Conditions**: Clearly document expected condition inputs
6. **Consider Compilation Time**: Complex conditional models may take longer to compile

## Examples

### Mixture of Experts

```python
from mithril.models import Model, Linear, Softmax, Switch, Multiply, Add

# Create expert models
def create_expert(input_dim, hidden_dim, output_dim):
    expert = Model()
    expert |= Linear(dimension=hidden_dim)(input="input", output="hidden")
    expert += Relu()(input="hidden", output="hidden_act")
    expert += Linear(dimension=output_dim)(input="hidden_act", output="output")
    return expert

# Create a gating network
gate = Model()
gate |= Linear(dimension=4)(input="input", output="gate_logits")
gate += Softmax(dim=-1)(input="gate_logits", output="gate_weights")

# Create experts
experts = [create_expert(input_dim=10, hidden_dim=64, output_dim=20) for _ in range(4)]

# Create mixture of experts model
moe = Model()
moe |= gate(input="input", output="gate_weights")

# Expert outputs
for i, expert in enumerate(experts):
    moe += expert(input="input", output=f"expert_{i}_output")

# Create weighted combination of expert outputs
weighted_outputs = []
for i in range(4):
    # Extract the i-th gate weight
    moe += SliceAt(index=i, dim=-1)(input="gate_weights", output=f"gate_weight_{i}")
    # Reshape to broadcastable shape
    moe += Reshape(shape=(-1, 1))(input=f"gate_weight_{i}", output=f"gate_weight_{i}_reshaped")
    # Weight the expert output
    moe += Multiply()(
        left=f"gate_weight_{i}_reshaped", 
        right=f"expert_{i}_output", 
        output=f"weighted_expert_{i}"
    )
    weighted_outputs.append(f"weighted_expert_{i}")

# Sum the weighted outputs
moe += Add(inputs=weighted_outputs, output="output")
```

### Dynamic Graph Neural Network

```python
from mithril.models import Model, Linear, Relu, If, Identity, GATConv

# Create a dynamic GNN that adapts to graph structure
model = Model()

# Node feature embedding
model |= Linear(dimension=64)(input="node_features", output="node_embeddings")

# Apply GAT convolution if using graph structure, otherwise use MLP
model += If(
    condition="use_graph_structure",
    then_branch=GATConv(in_channels=64, out_channels=64, heads=8)(
        input="node_embeddings", 
        edge_index="edge_index",
        output="conv_output"
    ),
    else_branch=Linear(dimension=64)(
        input="node_embeddings", 
        output="conv_output"
    )
)

# Add a residual connection based on the embedding size
model += If(
    condition="use_residual",
    then_branch=Add()(
        left="node_embeddings", 
        right="conv_output", 
        output="layer_output"
    ),
    else_branch=Identity()(
        input="conv_output", 
        output="layer_output"
    )
)

# Final classification layer
model += Linear(dimension=10)(input="layer_output", output="output")
```