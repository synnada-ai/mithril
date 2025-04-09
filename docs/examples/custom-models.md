# Creating Custom Models in Mithril

This guide demonstrates how to create custom models in Mithril by designing new architectures, extending existing models, and implementing custom operators.

## Understanding Mithril's Model Architecture

Mithril uses a flexible model composition system with:

1. **Logical Models**: High-level abstract representations of computational graphs
2. **Physical Models**: Compiled models optimized for specific backends
3. **Operators**: Building blocks that define computational operations

These components provide a balance between flexibility and performance, allowing you to create custom architectures while leveraging Mithril's backend-agnostic compilation.

## Creating Custom Model Classes

### Basic Model Construction

The core of a custom model is the `Model` class. Here's a simple example of creating a custom MLP:

```python
from mithril.models import Model, Linear, Relu, IOKey

def custom_mlp(input_dim: int, hidden_dims: list[int], output_dim: int):
    model = Model(name="custom_mlp")
    
    # Input layer
    model += Linear(hidden_dims[0])(input="input")
    model += Relu()
    
    # Hidden layers
    for i in range(1, len(hidden_dims)):
        model += Linear(hidden_dims[i])
        model += Relu()
    
    # Output layer
    model += Linear(output_dim)(output=IOKey("output"))
    
    return model
```

This pattern allows you to build up a model layer by layer, with each layer automatically connecting to the previous one's output.

### Using Operator Composition

Mithril's operator composition system (`|=` and `+=` operators) provides intuitive model construction:

```python
def attention_block(dim: int, num_heads: int):
    block = Model(name="attention_block")
    
    # Layer normalization
    block += LayerNorm()(input="input", output="norm_out")
    
    # Multi-head attention
    block |= MultiHeadAttention(dim, num_heads)(
        query="norm_out", 
        key="norm_out", 
        value="norm_out",
        output="attn_out"
    )
    
    # Residual connection
    block |= Add()(left="input", right="attn_out", output=IOKey("output"))
    
    return block
```

- `+=` adds a component and automatically connects it to the previous component's output
- `|=` adds a component with explicit input and output connections

### Creating Reusable Components

For complex architectures, create reusable components that can be composed into larger models:

```python
def resnet_block(channels: int, stride: int = 1):
    block = Model()
    
    # Main path
    block += Convolution2D(channels, kernel_size=3, stride=stride, padding=1)
    block += BatchNorm2D(channels)
    block += Relu()
    block += Convolution2D(channels, kernel_size=3, padding=1)
    block += BatchNorm2D(channels)
    
    # Shortcut path if needed
    if stride > 1:
        shortcut = Model()
        shortcut += Convolution2D(channels, kernel_size=1, stride=stride)
        shortcut += BatchNorm2D(channels)
        block |= shortcut(input=block.cin, output="shortcut")
        block |= Add()(left="shortcut", right=block.cout, output="res")
    else:
        block |= Add()(left=block.cin, right=block.cout, output="res")
    
    block += Relu()(input="res")
    
    return block
```

## Creating Custom Operators

### Implementing Simple Custom Operators

For specialized operations not covered by Mithril's built-in operators, create custom operators:

```python
from mithril.models import Model, Operator, IOKey

class ChannelShuffle(Operator):
    def __init__(self, groups: int, name: str | None = None):
        super().__init__(name=name)
        self.groups = groups
    
    def create_model(self):
        model = Model()
        input = IOKey("input")  # Expected shape: [B, C, H, W]
        
        # Get input shape
        B, C, H, W = input.shape
        
        # Reshape and transpose for channel shuffle
        x = input.reshape(B, self.groups, C // self.groups, H, W)
        x = x.transpose(0, 2, 1, 3, 4)
        x = x.reshape(B, C, H, W)
        
        model |= Buffer()(x, output=IOKey("output"))
        return model
```

### Extending Existing Operators

You can also extend existing operators to add functionality:

```python
from mithril.models import Convolution2D

class DepthwiseSeparableConv(Operator):
    def __init__(
        self, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0,
        name: str | None = None
    ):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def create_model(self):
        model = Model()
        input = IOKey("input")
        in_channels = input.shape[1]
        
        # Depthwise convolution
        model |= Convolution2D(
            out_channels=in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=in_channels,
            name="depthwise"
        )(input=input, output="depthwise_out")
        
        # Pointwise convolution
        model |= Convolution2D(
            out_channels=self.out_channels,
            kernel_size=1,
            name="pointwise"
        )(input="depthwise_out", output=IOKey("output"))
        
        return model
```

## Custom Models with Multiple Inputs and Outputs

### Handling Multiple Inputs

Models that require multiple inputs can be constructed with explicit connection management:

```python
def fusion_model(text_dim: int, image_dim: int, output_dim: int):
    model = Model()
    
    # Define inputs
    text_input = IOKey("text_input", shape=[None, text_dim])
    image_input = IOKey("image_input", shape=[None, image_dim])
    
    # Process text
    model |= Linear(512, name="text_encoder")(
        input=text_input, output="text_features"
    )
    
    # Process image
    model |= Linear(512, name="image_encoder")(
        input=image_input, output="image_features"
    )
    
    # Fusion
    model |= Concat(axis=1)(
        input=["text_features", "image_features"], output="fused_features"
    )
    model |= Linear(output_dim)(
        input="fused_features", output=IOKey("output")
    )
    
    return model
```

### Creating Multi-Output Models

Models can produce multiple outputs for different tasks:

```python
def multi_task_model(input_dim: int):
    model = Model()
    
    # Shared backbone
    model |= Linear(512, name="backbone")(input="input", output="features")
    model |= Relu()(input="features", output="activated_features")
    
    # Task-specific heads
    model |= Linear(1, name="regression_head")(
        input="activated_features", output=IOKey("regression_output")
    )
    
    model |= Linear(10, name="classification_head")(
        input="activated_features", output=IOKey("classification_output") 
    )
    
    return model
```

## Implementing Complex Architectures

### Transformers with Custom Attention

Implement a transformer block with customized attention mechanisms:

```python
def transformer_block(dim: int, num_heads: int, mlp_dim: int, dropout_rate: float = 0.1):
    block = Model()
    
    # Layer normalization and self-attention
    block |= LayerNorm()(input="input", output="norm1")
    
    # Custom attention implementation
    attn = Model(name="self_attention")
    attn |= Linear(dim * 3, name="qkv")(input="norm1", output="qkv")
    attn |= Split(split_size=dim, axis=-1)(input="qkv", output="qkv_split")
    
    # Process query, key, value and compute attention
    # ... (attention implementation)
    
    block |= attn(input="norm1", output="attn_out")
    block |= Add()(left="input", right="attn_out", output="attn_residual")
    
    # MLP block
    block |= LayerNorm()(input="attn_residual", output="norm2")
    block |= Linear(mlp_dim)(input="norm2", output="mlp1")
    block |= Gelu()(input="mlp1", output="mlp_act")
    block |= Linear(dim)(input="mlp_act", output="mlp_out")
    block |= Add()(left="attn_residual", right="mlp_out", output=IOKey("output"))
    
    return block
```

### Recurrent Neural Networks

Implement custom RNN architectures:

```python
def lstm_cell(input_dim: int, hidden_dim: int):
    cell = Model(name="lstm_cell")
    
    # LSTM gates
    input = IOKey("input", shape=[None, input_dim])
    h_prev = IOKey("h_prev", shape=[None, hidden_dim])
    c_prev = IOKey("c_prev", shape=[None, hidden_dim])
    
    # Concatenate input and previous hidden state
    cell |= Concat(axis=1)(input=[input, h_prev], output="combined")
    
    # Calculate gates
    cell |= Linear(hidden_dim * 4, name="gates")(input="combined", output="gates")
    cell |= Split(split_size=hidden_dim, axis=1)(input="gates", output="split_gates")
    
    i = Sigmoid()(cell.split_gates[0])
    f = Sigmoid()(cell.split_gates[1])
    g = Tanh()(cell.split_gates[2])
    o = Sigmoid()(cell.split_gates[3])
    
    # Update cell state and hidden state
    c_new = f * c_prev + i * g
    h_new = o * Tanh()(c_new)
    
    cell |= Buffer()(c_new, output=IOKey("c_new"))
    cell |= Buffer()(h_new, output=IOKey("h_new"))
    
    return cell
```

## Integrating with Existing Frameworks

### Using Pre-trained Weights

Mithril models can utilize pre-trained weights from other frameworks:

```python
import torch
from torchvision.models import resnet18 as torch_resnet18
from mithril.models import resnet18 as ml_resnet18

def load_pretrained_resnet():
    # Create Mithril model
    ml_model = ml_resnet18(num_classes=1000)
    
    # Load PyTorch model and weights
    torch_model = torch_resnet18(pretrained=True)
    torch_state_dict = torch_model.state_dict()
    
    # Compile Mithril model
    backend = ml.TorchBackend()
    compiled_model = ml.compile(
        ml_model, 
        backend=backend,
        shapes={"input": [1, 3, 224, 224]},
        data_keys={"input"}
    )
    
    # Convert and load weights
    ml_params = {}
    for torch_key, torch_param in torch_state_dict.items():
        ml_key = torch_key.replace(".", "_")
        if ml_key in compiled_model.shapes:
            ml_params[ml_key] = backend.array(torch_param.numpy())
    
    return compiled_model, ml_params
```

## Best Practices for Custom Models

1. **Use Meaningful Names**: Name your models and layers for easier debugging
2. **Manage Connections Explicitly**: For complex models, be explicit about connections
3. **Set Input/Output Keys**: Always define clear input and output keys
4. **Document Shape Requirements**: Document expected input shapes for complex models
5. **Test Across Backends**: Verify models work across different backends
6. **Use Type Annotations**: Add proper type annotations for better static analysis
7. **Handle Edge Cases**: Consider edge cases like batch size of 1 or variable-length inputs

By following these patterns, you can create sophisticated custom models that leverage Mithril's flexible architecture while maintaining compatibility with its backend-agnostic compilation system.