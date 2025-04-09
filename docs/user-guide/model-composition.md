# Model Composition

Mithril's powerful model composition system allows you to build complex architectures by combining simpler components. This guide explains the different ways to compose models and best practices for creating modular, reusable architectures.

## Composition Operators

Mithril provides several operators for model composition:

### Pipe-Assign (`|=`)

The pipe-assign operator (`|=`) adds a model as the first component in a chain and connects it to the input.

```python
from mithril.models import Model, Linear, Relu

model = Model()
model |= Linear(dimension=64)  # First layer connects to input
```

### Add-Assign (`+=`)

The add-assign operator (`+=`) adds a model to the end of the chain, connecting it to the previous output.

```python
model = Model()
model |= Linear(dimension=64)  # First layer connects to input
model += Relu()                # Connects to the output of Linear
model += Linear(dimension=32)  # Connects to the output of Relu
```

### Explicit Connections

For more complex architectures, you can use explicit connections by naming terminals:

```python
model = Model()
model |= Linear(dimension=64)(input="input", output="hidden1")
model += Relu()(input="hidden1", output="hidden1_act")
model += Linear(dimension=32)(input="hidden1_act", output="output")
```

## Sequential Composition

Sequential composition is the simplest form of model composition, where outputs of one layer are connected to inputs of the next layer.

### Linear Chain

```python
from mithril.models import Model, Linear, Relu, Dropout

# Create a simple sequential model
model = Model()
model |= Linear(dimension=128)
model += Relu()
model += Dropout(p=0.2)
model += Linear(dimension=64)
model += Relu()
model += Linear(dimension=10)
```

### Named Terminals in Sequential Models

Even in sequential models, you can use named terminals for clarity:

```python
model = Model()
model |= Linear(dimension=128)(input="input", output="hidden1")
model += Relu()(input="hidden1", output="hidden1_act")
model += Linear(dimension=64)(input="hidden1_act", output="hidden2")
model += Relu()(input="hidden2", output="hidden2_act")
model += Linear(dimension=10)(input="hidden2_act", output="output")
```

## Branched Composition

Branched composition allows for multiple paths through the model.

### Simple Branching

```python
from mithril.models import Model, Linear, Relu, Add, Concat

# Create a model with two branches
model = Model()
model |= Linear(dimension=64)(input="input", output="branch1")
model += Linear(dimension=64)(input="input", output="branch2")
model += Relu()(input="branch1", output="branch1_act")
model += Relu()(input="branch2", output="branch2_act")

# Merge branches by addition
model += Add()(left="branch1_act", right="branch2_act", output="merged")

# Final layer
model += Linear(dimension=10)(input="merged", output="output")
```

### Multiple Inputs and Outputs

Models can have multiple inputs and outputs:

```python
# Create a model with two inputs and two outputs
model = Model()
model |= Linear(dimension=64)(input="image_input", output="image_hidden")
model += Linear(dimension=32)(input="text_input", output="text_hidden")
model += Concat()(inputs=["image_hidden", "text_hidden"], output="combined")
model += Linear(dimension=64)(input="combined", output="hidden")
model += Linear(dimension=10)(input="hidden", output="class_output")
model += Linear(dimension=1)(input="hidden", output="regression_output")
```

## Nested Composition

Nested composition involves using entire models as components within other models.

### Using Models as Components

```python
# Define a feature extractor
feature_extractor = Model()
feature_extractor |= Linear(dimension=256)
feature_extractor += Relu()
feature_extractor += Linear(dimension=128)
feature_extractor += Relu()

# Define a classifier
classifier = Model()
classifier |= Linear(dimension=64)
classifier += Relu()
classifier += Linear(dimension=10)

# Combine them in a full model
model = Model()
model |= feature_extractor(output="features")
model += classifier(input="features", output="output")
```

### Reusing the Same Component Multiple Times

You can reuse the same component multiple times with different connections:

```python
# Define a shared block
shared_block = Model()
shared_block |= Linear(dimension=64)
shared_block += Relu()

# Use it in multiple places
model = Model()
model |= Linear(dimension=128)(input="input", output="hidden1")
model += shared_block(input="hidden1", output="path1")
model += shared_block(input="hidden1", output="path2")  # Same block, different connections
model += Add()(left="path1", right="path2", output="merged")
model += Linear(dimension=10)(input="merged", output="output")
```

## Advanced Composition Patterns

### Residual Connections

Create models with skip connections:

```python
from mithril.models import Model, Linear, Relu, Add

# Create a residual block
def residual_block(dimension):
    block = Model()
    block |= Linear(dimension=dimension)(input="input", output="hidden")
    block += Relu()(input="hidden", output="hidden_act")
    block += Linear(dimension=dimension)(input="hidden_act", output="block_output")
    block += Add()(left="input", right="block_output", output="output")
    return block

# Create a model with residual blocks
model = Model()
model |= Linear(dimension=64)(input="input", output="layer1")
model += residual_block(dimension=64)(input="layer1", output="layer2")
model += residual_block(dimension=64)(input="layer2", output="layer3")
model += Linear(dimension=10)(input="layer3", output="output")
```

### Inception-style Modules

Create models with parallel paths of different scales:

```python
def inception_module(input_channels, output_channels):
    module = Model()
    
    # 1x1 convolution branch
    module |= Conv2d(in_channels=input_channels, out_channels=output_channels//4, 
                    kernel_size=1)(input="input", output="branch1")
    
    # 1x1 -> 3x3 convolution branch
    module += Conv2d(in_channels=input_channels, out_channels=output_channels//4, 
                    kernel_size=1)(input="input", output="branch2_proj")
    module += Relu()(input="branch2_proj", output="branch2_act")
    module += Conv2d(in_channels=output_channels//4, out_channels=output_channels//4, 
                    kernel_size=3, padding=1)(input="branch2_act", output="branch2")
    
    # 1x1 -> 5x5 convolution branch
    module += Conv2d(in_channels=input_channels, out_channels=output_channels//4, 
                    kernel_size=1)(input="input", output="branch3_proj")
    module += Relu()(input="branch3_proj", output="branch3_act1")
    module += Conv2d(in_channels=output_channels//4, out_channels=output_channels//4, 
                    kernel_size=5, padding=2)(input="branch3_act1", output="branch3")
    
    # Pool -> 1x1 convolution branch
    module += MaxPool2d(kernel_size=3, stride=1, padding=1)(input="input", output="branch4_pool")
    module += Conv2d(in_channels=input_channels, out_channels=output_channels//4, 
                    kernel_size=1)(input="branch4_pool", output="branch4")
    
    # Concatenate all branches
    module += Concat(dim=1)(inputs=["branch1", "branch2", "branch3", "branch4"], 
                          output="output")
    
    return module
```

### Dynamic Composition

You can create models that are composed dynamically based on parameters:

```python
def create_dynamic_network(input_size, hidden_sizes, output_size, activation="relu"):
    model = Model()
    
    # Select activation function
    if activation == "relu":
        act_fn = Relu
    elif activation == "tanh":
        act_fn = Tanh
    elif activation == "sigmoid":
        act_fn = Sigmoid
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    # First layer
    model |= Linear(dimension=hidden_sizes[0])(input="input", output="hidden_0")
    
    # Hidden layers with chosen activation
    for i in range(len(hidden_sizes) - 1):
        model += act_fn()(input=f"hidden_{i}", output=f"act_{i}")
        model += Linear(dimension=hidden_sizes[i+1])(input=f"act_{i}", output=f"hidden_{i+1}")
    
    # Output layer
    last_idx = len(hidden_sizes) - 1
    model += act_fn()(input=f"hidden_{last_idx}", output=f"act_{last_idx}")
    model += Linear(dimension=output_size)(input=f"act_{last_idx}", output="output")
    
    return model
```

## Conditional Composition

You can create models with conditional paths:

```python
from mithril.models import Model, Linear, Relu, If, Identity

model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")
model += If(
    condition="is_training",
    then_branch=Dropout(p=0.5)(input="hidden", output="hidden_processed"),
    else_branch=Identity()(input="hidden", output="hidden_processed")
)
model += Linear(dimension=10)(input="hidden_processed", output="output")
```

## Best Practices for Model Composition

### 1. Use Clear Terminal Names

Choose descriptive terminal names that indicate what the data represents:

```python
# Good terminal naming
model |= Conv2d(in_channels=3, out_channels=64, kernel_size=3)(
    input="image", output="conv1_features"
)
```

### 2. Keep Components Modular

Design components to be self-contained and reusable:

```python
def create_block(in_channels, out_channels):
    block = Model()
    block |= Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
    block += BatchNorm2d(num_features=out_channels)
    block += Relu()
    return block
```

### 3. Use Functions to Create Parameterized Architectures

Create functions that generate models based on parameters:

```python
def create_resnet(num_blocks, num_classes):
    model = Model()
    # Input stem
    model |= Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)(
        input="input", output="stem_conv"
    )
    model += BatchNorm2d(num_features=64)(input="stem_conv", output="stem_bn")
    model += Relu()(input="stem_bn", output="stem_act")
    model += MaxPool2d(kernel_size=3, stride=2, padding=1)(input="stem_act", output="stem_pool")
    
    # Residual blocks
    current_channels = 64
    prev_output = "stem_pool"
    
    for i, block_count in enumerate(num_blocks):
        channels = 64 * (2 ** i)
        for j in range(block_count):
            stride = 2 if j == 0 and i > 0 else 1
            downsample = stride > 1 or current_channels != channels
            
            block = create_residual_block(
                in_channels=current_channels,
                out_channels=channels,
                stride=stride,
                downsample=downsample
            )
            
            model += block(input=prev_output, output=f"block_{i}_{j}")
            prev_output = f"block_{i}_{j}"
            current_channels = channels
    
    # Classification head
    model += AdaptiveAvgPool2d(output_size=(1, 1))(input=prev_output, output="pool")
    model += Flatten()(input="pool", output="flat")
    model += Linear(dimension=num_classes)(input="flat", output="output")
    
    return model

# Create ResNet-18
resnet18 = create_resnet(num_blocks=[2, 2, 2, 2], num_classes=1000)

# Create ResNet-50
resnet50 = create_resnet(num_blocks=[3, 4, 6, 3], num_classes=1000)
```

### 4. Validate Models Early

Test your model structure before compilation:

```python
# Print model summary to check structure
print(model.summary())

# Test compilation with sample shapes
test_compiled = ml.compile(
    model=model,
    backend=ml.NumpyBackend(dtype=ml.float32),
    shapes={"input": [1, 3, 224, 224]},
    verbose=True
)
```

### 5. Document Component Interfaces

Document the expected inputs and outputs of your components:

```python
def attention_block(query_dim, key_dim, value_dim, num_heads):
    """
    Multi-head attention block.
    
    Inputs:
        - "query": Query tensor of shape [batch_size, seq_len_q, query_dim]
        - "key": Key tensor of shape [batch_size, seq_len_k, key_dim]
        - "value": Value tensor of shape [batch_size, seq_len_k, value_dim]
        - "mask": Optional attention mask of shape [batch_size, seq_len_q, seq_len_k]
    
    Outputs:
        - "output": Attention output of shape [batch_size, seq_len_q, value_dim]
    """
    # Implementation...
```

## Common Pitfalls and Solutions

### Disconnected Components

If your model has disconnected components, you might see an error during compilation.

```python
# Incorrect - branch2 is disconnected
model = Model()
model |= Linear(dimension=64)(input="input", output="branch1")
model += Linear(dimension=64)  # Missing terminal connections
model += Add()(left="branch1", right="branch2", output="merged")  # "branch2" doesn't exist

# Correct
model = Model()
model |= Linear(dimension=64)(input="input", output="branch1")
model += Linear(dimension=64)(input="input", output="branch2")  # Proper connection
model += Add()(left="branch1", right="branch2", output="merged")
```

### Mismatched Dimensions

Be careful when connecting components that might have incompatible dimensions:

```python
# Use explicit reshaping or projections when dimensions don't match
model = Model()
model |= Linear(dimension=64)(input="input", output="branch1")
model += Linear(dimension=32)(input="input", output="branch2")
# Can't add directly because dimensions don't match
model += Linear(dimension=64)(input="branch2", output="branch2_proj")  # Project to match
model += Add()(left="branch1", right="branch2_proj", output="merged")
```

### Cyclical Dependencies

Avoid creating cycles in your model graph:

```python
# Incorrect - creates a cycle
model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")
model += Linear(dimension=64)(input="output", output="hidden")  # Cycle: output -> hidden -> output

# Correct - use different terminal names
model = Model()
model |= Linear(dimension=64)(input="input", output="hidden1")
model += Linear(dimension=64)(input="hidden1", output="hidden2")
model += Linear(dimension=10)(input="hidden2", output="output")
```