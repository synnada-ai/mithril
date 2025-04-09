# Logical Models

Logical models in Mithril define the architecture of your machine learning components without specifying implementation details. This guide covers how to create, compose, and work with logical models.

## Basic Model Creation

### Creating Empty Models

You can start with an empty model and add components to it:

```python
from mithril.models import Model

# Create an empty model
model = Model()
```

### Using Predefined Models

Mithril provides many predefined models that you can use directly:

```python
from mithril.models import Linear, Conv2d, LSTM, Transformer

# Create a linear layer
linear = Linear(dimension=64)

# Create a convolutional layer
conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)

# Create an LSTM
lstm = LSTM(input_size=128, hidden_size=256)

# Create a transformer block
transformer = Transformer(d_model=512, nhead=8)
```

## Model Composition

### Sequential Composition

For simple sequential models, you can use the `|=` and `+=` operators:

```python
from mithril.models import Model, Linear, Relu, Dropout

# Create a sequential model
model = Model()
model |= Linear(dimension=64)  # First layer connects to input
model += Relu()                # Each += connects to previous output
model += Dropout(p=0.1)
model += Linear(dimension=10)
```

### Explicit Connections

For more complex architectures, you can use explicit connections:

```python
from mithril.models import Model, Linear, Add, Tanh

# Create a model with explicit connections
model = Model()
model |= Linear(dimension=32)(input="input", output="hidden1")
model += Linear(dimension=32)(input="input", output="hidden2")
model += Tanh()(input="hidden1", output="tanh_out")
model += Add()(left="tanh_out", right="hidden2", output="output")
```

### Reusing Models as Components

Models can be used as components in other models:

```python
# Create a residual block
residual_block = Model()
residual_block |= Linear(dimension=64)(output="hidden")
residual_block += Relu()(input="hidden", output="hidden_act")
residual_block += Linear(dimension=64)(input="hidden_act", output="block_output")
residual_block += Add()(left="input", right="block_output", output="output")

# Use the residual block in a larger model
model = Model()
model |= Linear(dimension=64)
model += residual_block
model += Relu()
model += Linear(dimension=10)
```

## Custom Models

### Creating Custom Models

You can create custom models by subclassing `Model`:

```python
from mithril.models import Model, Linear, Relu

class MLP(Model):
    def __init__(self, in_features, hidden_dim, out_features, dropout_rate=0.1):
        super().__init__()
        
        self |= Linear(dimension=hidden_dim)(input="input", output="hidden")
        self += Relu()(input="hidden", output="hidden_act")
        self += Dropout(p=dropout_rate)(input="hidden_act", output="hidden_drop")
        self += Linear(dimension=out_features)(input="hidden_drop", output="output")
```

## Working with Models

### Inspecting Models

You can inspect your model structure:

```python
# Print model summary
print(model.summary())

# Get model parameters
params = model.get_params()
```

### Saving and Loading

You can serialize models to dictionaries or JSON:

```python
# Convert model to dictionary
model_dict = model.to_dict()

# Load model from dictionary
loaded_model = Model.from_dict(model_dict)
```

## Advanced Features

### Conditional Execution

You can create models with conditional execution paths:

```python
from mithril.models import Model, If, Linear, Relu

model = Model()
model |= Linear(dimension=64)(output="hidden")
model += If(
    condition="training", 
    then_branch=Dropout(p=0.5)(input="hidden", output="hidden"),
    else_branch=None
)
model += Relu()(input="hidden", output="hidden_act")
model += Linear(dimension=10)(input="hidden_act", output="output")
```

### Dynamic Shapes

Logical models can work with dynamic shapes:

```python
# Define a model that can handle variable batch sizes
model = Model()
model |= Linear(dimension=64)
# ... rest of the model

# When compiled, this model can handle any batch size
```

## Best Practices

1. **Keep models modular**: Design models that encapsulate specific functionality
2. **Use meaningful names**: Name your terminals to make connections clear
3. **Reuse components**: Create components once and reuse them
4. **Validate early**: Test your logical models before complex compilation
5. **Separate architecture from implementation**: Keep your logical models free of backend-specific code