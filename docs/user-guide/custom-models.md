# Custom Models

This guide explains how to create custom models in Mithril, allowing you to encapsulate reusable model architectures and extend the built-in functionality.

## Creating Custom Model Classes

Custom models in Mithril are typically created by subclassing the `Model` class. This approach allows you to encapsulate model architecture and reuse it across your projects.

### Basic Custom Model

```python
from mithril.models import Model, Linear, Relu, Dropout

class MLP(Model):
    """
    A simple multi-layer perceptron with configurable layers and activation.
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.1):
        super().__init__()
        
        # Store constructor parameters for reference
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Input layer
        self |= Linear(dimension=hidden_sizes[0])(input="input", output="hidden_0")
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self += Relu()(input=f"hidden_{i-1}", output=f"act_{i-1}")
            self += Dropout(p=dropout_rate)(input=f"act_{i-1}", output=f"drop_{i-1}")
            self += Linear(dimension=hidden_sizes[i])(input=f"drop_{i-1}", output=f"hidden_{i}")
        
        # Output layer
        last_hidden = len(hidden_sizes) - 1
        self += Relu()(input=f"hidden_{last_hidden}", output=f"act_{last_hidden}")
        self += Dropout(p=dropout_rate)(input=f"act_{last_hidden}", output=f"drop_{last_hidden}")
        self += Linear(dimension=output_size)(input=f"drop_{last_hidden}", output="output")
```

### Using Your Custom Model

Once defined, you can use your custom model just like any built-in model:

```python
# Create an instance of your custom model
model = MLP(
    input_size=784,     # Input features (e.g., MNIST flattened image)
    hidden_sizes=[256, 128, 64],  # Three hidden layers
    output_size=10      # Output classes
)

# Compile the model with a backend
import mithril as ml
backend = ml.TorchBackend(dtype=ml.float32)
compiled_model = ml.compile(model, backend)

# Use the model
params = compiled_model.randomize_params()
inputs = {"input": backend.randn(32, 784)}  # Batch of 32 samples
outputs = compiled_model.evaluate(params, inputs)
```

## Parameterized Model Architectures

You can create flexible model architectures that adapt based on constructor parameters:

```python
from mithril.models import Model, Linear, Relu, Tanh, Sigmoid, Softmax

class ConfigurableNetwork(Model):
    """
    A network with configurable activation functions and normalization.
    """
    def __init__(
        self, 
        input_size, 
        hidden_sizes, 
        output_size, 
        activation="relu", 
        output_activation=None,
        use_batch_norm=False
    ):
        super().__init__()
        
        # Select activation function
        if activation == "relu":
            act_fn = Relu
        elif activation == "tanh":
            act_fn = Tanh
        elif activation == "sigmoid":
            act_fn = Sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Select output activation function
        out_act_fn = None
        if output_activation == "softmax":
            out_act_fn = Softmax
        elif output_activation == "sigmoid":
            out_act_fn = Sigmoid
        elif output_activation == "tanh":
            out_act_fn = Tanh
        
        # Input layer
        self |= Linear(dimension=hidden_sizes[0])(input="input", output="hidden_0")
        
        # Add optional batch normalization
        if use_batch_norm:
            from mithril.models import BatchNorm1d
            self += BatchNorm1d(num_features=hidden_sizes[0])(input="hidden_0", output="bn_0")
            prev_output = "bn_0"
        else:
            prev_output = "hidden_0"
        
        # Add activation
        self += act_fn()(input=prev_output, output="act_0")
        prev_output = "act_0"
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self += Linear(dimension=hidden_sizes[i])(input=prev_output, output=f"hidden_{i}")
            
            if use_batch_norm:
                self += BatchNorm1d(num_features=hidden_sizes[i])(
                    input=f"hidden_{i}", output=f"bn_{i}"
                )
                self += act_fn()(input=f"bn_{i}", output=f"act_{i}")
            else:
                self += act_fn()(input=f"hidden_{i}", output=f"act_{i}")
                
            prev_output = f"act_{i}"
        
        # Output layer
        self += Linear(dimension=output_size)(input=prev_output, output="logits")
        
        # Add output activation if specified
        if out_act_fn:
            self += out_act_fn()(input="logits", output="output")
        else:
            # Rename the terminal if no output activation
            self.rename_terminal("logits", "output")
```

## Creating Reusable Components

Complex architectures can be broken down into reusable components:

```python
from mithril.models import Model, Linear, Relu, Add

# Define a reusable ResNet block
class ResidualBlock(Model):
    """
    A residual block with two linear layers and a skip connection.
    """
    def __init__(self, dimension, activation=Relu):
        super().__init__()
        
        self.dimension = dimension
        
        # First branch: two linear layers with activation
        self |= Linear(dimension=dimension)(input="input", output="hidden1")
        self += activation()(input="hidden1", output="act1")
        self += Linear(dimension=dimension)(input="act1", output="branch1_out")
        
        # Skip connection (second branch is identity)
        self += Add()(left="input", right="branch1_out", output="output")

# Now use the residual block in a larger model
class ResNet(Model):
    """
    A simple ResNet-style model with configurable residual blocks.
    """
    def __init__(self, input_size, hidden_size, output_size, num_blocks=3):
        super().__init__()
        
        # Input projection
        self |= Linear(dimension=hidden_size)(input="input", output="projected")
        
        # Add residual blocks
        prev_output = "projected"
        for i in range(num_blocks):
            self += ResidualBlock(dimension=hidden_size)(
                input=prev_output, output=f"block_{i}_out"
            )
            prev_output = f"block_{i}_out"
        
        # Final activation and output projection
        self += Relu()(input=prev_output, output="final_act")
        self += Linear(dimension=output_size)(input="final_act", output="output")
```

## Meta-Models and Factories

You can create factory functions that generate models based on higher-level specifications:

```python
def create_classifier(
    input_size, 
    output_size, 
    architecture="mlp", 
    hidden_sizes=None,
    activation="relu",
    dropout=0.1
):
    """
    Factory function to create different classifier architectures.
    
    Args:
        input_size: Number of input features
        output_size: Number of output classes
        architecture: 'mlp', 'resnet', or 'transformer'
        hidden_sizes: List of hidden layer sizes
        activation: Activation function to use
        dropout: Dropout rate
        
    Returns:
        A Mithril model
    """
    if hidden_sizes is None:
        hidden_sizes = [128, 64]
        
    if architecture == "mlp":
        return MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout_rate=dropout
        )
    elif architecture == "resnet":
        return ResNet(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            output_size=output_size,
            num_blocks=len(hidden_sizes)
        )
    elif architecture == "transformer":
        # Import transformer components
        from mithril.models import TransformerEncoder, PositionalEncoding
        
        model = Model()
        model |= Linear(dimension=hidden_sizes[0])(input="input", output="embedding")
        model += PositionalEncoding(d_model=hidden_sizes[0])(input="embedding", output="pos_embedding")
        model += TransformerEncoder(
            d_model=hidden_sizes[0],
            nhead=8,
            dim_feedforward=hidden_sizes[1] if len(hidden_sizes) > 1 else hidden_sizes[0] * 4,
            num_layers=len(hidden_sizes),
            dropout=dropout
        )(input="pos_embedding", output="encoded")
        model += Linear(dimension=output_size)(input="encoded", output="output")
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
```

## Custom Operators and Primitives

You can also create custom operators that encapsulate complex operations:

```python
from mithril.models import Model, Primitive
from mithril.framework.logical.operators import register_primitive

# Define a new primitive operation
class GaussianNoise(Primitive):
    """
    Add Gaussian noise to the input during training.
    """
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        
    def __call__(self, **kwargs):
        """
        Apply the operator to connect terminals.
        """
        # Inherit terminal connections from Primitive.__call__
        return super().__call__(**kwargs)

# Register implementation for PyTorch backend    
@register_primitive("GaussianNoise", "torch")
def gaussian_noise_torch(x, std, training):
    import torch
    if training:
        return x + torch.randn_like(x) * std
    return x

# Register implementation for JAX backend
@register_primitive("GaussianNoise", "jax")
def gaussian_noise_jax(x, std, training):
    import jax.numpy as jnp
    import jax.random as random
    if training:
        key = random.PRNGKey(0)  # In practice, you'd want a dynamic key
        return x + random.normal(key, x.shape) * std
    return x

# Usage in a model
class NoiseRegularizedMLP(Model):
    def __init__(self, input_size, hidden_sizes, output_size, noise_std=0.1):
        super().__init__()
        
        self |= Linear(dimension=hidden_sizes[0])(input="input", output="hidden_0")
        self += GaussianNoise(std=noise_std)(input="hidden_0", output="noisy_0")
        self += Relu()(input="noisy_0", output="act_0")
        
        # Rest of the model...
```

## Best Practices for Custom Models

1. **Naming Conventions**:
   - Give your model class a descriptive name
   - Use clear and consistent terminal names
   - Document your model's purpose and parameters

2. **Constructor Parameters**:
   - Store all constructor parameters as attributes
   - Provide sensible defaults for optional parameters
   - Validate parameters to catch errors early

3. **Terminal Management**:
   - Be explicit about terminal connections
   - Use named terminals for complex models
   - Consider the model's interface from a user perspective

4. **Composition**:
   - Break complex models into reusable components
   - Use composition over inheritance when possible
   - Maintain a clear hierarchy of components

5. **Documentation**:
   - Document the purpose of your model
   - Explain the architecture and how to use it
   - Provide examples of typical usage

## Debugging Custom Models

If you encounter issues with your custom model:

1. **Verify Connections**: Check that all terminals are properly connected
   ```python
   print(model.summary())  # Prints model structure and connections
   ```

2. **Inspect Shapes**: Compile with shape inference to check for shape mismatches
   ```python
   compiled_model = ml.compile(model, backend, shapes={"input": [32, 784]}, verbose=True)
   ```

3. **Test Components**: Test each component individually before combining them

4. **Isolate Issues**: Create a minimal reproduction of any issue you encounter