# Models API Reference

This page documents the models available in Mithril, including base classes, primitives, and higher-level components.

## Base Classes

### Model

```python
class Model:
    """
    Base class for all models in Mithril.
    
    A Model represents a composable unit of computation. Models have named 
    input and output terminals, and can be connected to form complex architectures.
    """
    
    def __init__(self):
        """
        Initialize a new Model instance.
        """
    
    def __call__(self, **kwargs):
        """
        Specify terminal connections for the model.
        
        Args:
            **kwargs: Mapping from internal terminal names to external terminal names.
                      Examples: input="data", output="predictions"
        
        Returns:
            The model instance for method chaining.
        """
    
    def __or__(self, other):
        """
        Compose with another model using the | operator.
        
        Args:
            other: The model to compose with.
        
        Returns:
            A new model with both models connected.
        """
    
    def __ior__(self, other):
        """
        In-place compose with another model using the |= operator.
        
        Args:
            other: The model to compose with.
        
        Returns:
            Self with the other model added.
        """
    
    def __add__(self, other):
        """
        Compose with another model using the + operator.
        
        Args:
            other: The model to compose with.
        
        Returns:
            A new model with both models connected.
        """
    
    def __iadd__(self, other):
        """
        In-place compose with another model using the += operator.
        
        Args:
            other: The model to compose with.
        
        Returns:
            Self with the other model added.
        """
    
    def summary(self, detailed=False):
        """
        Generate a summary of the model's structure.
        
        Args:
            detailed: Whether to include detailed information.
        
        Returns:
            A string containing the model summary.
        """
    
    def to_dict(self):
        """
        Serialize the model to a dictionary.
        
        Returns:
            A dictionary representation of the model.
        """
    
    @classmethod
    def from_dict(cls, dict_repr):
        """
        Deserialize a model from a dictionary.
        
        Args:
            dict_repr: A dictionary representation of a model.
        
        Returns:
            The reconstructed model.
        """
    
    def add_input_terminal(self, name):
        """
        Add a new input terminal to the model.
        
        Args:
            name: The name of the new input terminal.
        """
    
    def add_output_terminal(self, name):
        """
        Add a new output terminal to the model.
        
        Args:
            name: The name of the new output terminal.
        """
    
    def rename_terminal(self, old_name, new_name):
        """
        Rename a terminal in the model.
        
        Args:
            old_name: The current name of the terminal.
            new_name: The new name for the terminal.
        """
    
    def get_params(self):
        """
        Get the parameters of the model.
        
        Returns:
            A dictionary of model parameters.
        """
    
    def set_shapes(self, shapes):
        """
        Set input shapes for the model.
        
        Args:
            shapes: A dictionary mapping terminal names to shapes.
        """
    
    def get_input_terminals(self):
        """
        Get the input terminals of the model.
        
        Returns:
            A list of input terminal names.
        """
    
    def get_output_terminals(self):
        """
        Get the output terminals of the model.
        
        Returns:
            A list of output terminal names.
        """
```

### Primitive

```python
class Primitive(Model):
    """
    Base class for primitive operations in Mithril.
    
    Primitives represent atomic operations that are directly implemented 
    in each backend. Examples include activation functions, tensor operations, etc.
    """
    
    def __init__(self):
        """
        Initialize a new Primitive instance.
        """
        super().__init__()
```

## Neural Network Layers

### Linear

```python
class Linear(Model):
    """
    A linear transformation: y = xW^T + b
    
    Args:
        dimension: The output dimension of the linear layer.
        use_bias: Whether to include a bias term. Default: True.
        weight_initializer: Function to initialize weights. Default: kaiming_uniform.
        bias_initializer: Function to initialize biases. Default: zeros.
    """
    
    def __init__(self, dimension, use_bias=True, weight_initializer=None, bias_initializer=None):
        """
        Initialize a Linear layer.
        
        Args:
            dimension: The output dimension of the linear layer.
            use_bias: Whether to include a bias term.
            weight_initializer: Function to initialize weights.
            bias_initializer: Function to initialize biases.
        """
        super().__init__()
```

### Conv1d

```python
class Conv1d(Model):
    """
    1D convolution layer.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections from input to output. Default: 1.
        use_bias: Whether to include a bias term. Default: True.
    """
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        use_bias=True
    ):
        """
        Initialize a Conv1d layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output.
            use_bias: Whether to include a bias term.
        """
        super().__init__()
```

### Conv2d

```python
class Conv2d(Model):
    """
    2D convolution layer.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections from input to output. Default: 1.
        use_bias: Whether to include a bias term. Default: True.
    """
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        use_bias=True
    ):
        """
        Initialize a Conv2d layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output.
            use_bias: Whether to include a bias term.
        """
        super().__init__()
```

### BatchNorm1d

```python
class BatchNorm1d(Model):
    """
    Batch normalization for 1D inputs.
    
    Args:
        num_features: Number of features.
        eps: Small constant for numerical stability. Default: 1e-5.
        momentum: Value used for the running_mean and running_var computation. Default: 0.1.
        affine: Whether to use learnable affine parameters. Default: True.
        track_running_stats: Whether to track running statistics. Default: True.
    """
    
    def __init__(
        self, 
        num_features, 
        eps=1e-5, 
        momentum=0.1, 
        affine=True, 
        track_running_stats=True
    ):
        """
        Initialize a BatchNorm1d layer.
        
        Args:
            num_features: Number of features.
            eps: Small constant for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: Whether to use learnable affine parameters.
            track_running_stats: Whether to track running statistics.
        """
        super().__init__()
```

### BatchNorm2d

```python
class BatchNorm2d(Model):
    """
    Batch normalization for 2D inputs.
    
    Args:
        num_features: Number of features.
        eps: Small constant for numerical stability. Default: 1e-5.
        momentum: Value used for the running_mean and running_var computation. Default: 0.1.
        affine: Whether to use learnable affine parameters. Default: True.
        track_running_stats: Whether to track running statistics. Default: True.
    """
    
    def __init__(
        self, 
        num_features, 
        eps=1e-5, 
        momentum=0.1, 
        affine=True, 
        track_running_stats=True
    ):
        """
        Initialize a BatchNorm2d layer.
        
        Args:
            num_features: Number of features.
            eps: Small constant for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: Whether to use learnable affine parameters.
            track_running_stats: Whether to track running statistics.
        """
        super().__init__()
```

### Dropout

```python
class Dropout(Model):
    """
    Randomly zeroes some of the elements of the input tensor with probability p.
    
    Args:
        p: Probability of an element to be zeroed. Default: 0.5.
        inplace: If True, will do this operation in-place. Default: False.
    """
    
    def __init__(self, p=0.5, inplace=False):
        """
        Initialize a Dropout layer.
        
        Args:
            p: Probability of an element to be zeroed.
            inplace: If True, will do this operation in-place.
        """
        super().__init__()
```

### Embedding

```python
class Embedding(Model):
    """
    A lookup table mapping integer indices to dense vectors.
    
    Args:
        num_embeddings: Size of the dictionary of embeddings.
        embedding_dim: Size of each embedding vector.
        padding_idx: If specified, pads the output with zeros whenever it encounters this index.
        max_norm: If given, each embedding vector with norm larger than max_norm is renormalized.
        norm_type: The p of the p-norm to compute for the max_norm option. Default: 2.
        scale_grad_by_freq: If True, scale gradients by the inverse of frequency of the words in the mini-batch.
        sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor.
    """
    
    def __init__(
        self, 
        num_embeddings, 
        embedding_dim, 
        padding_idx=None, 
        max_norm=None, 
        norm_type=2.0, 
        scale_grad_by_freq=False, 
        sparse=False
    ):
        """
        Initialize an Embedding layer.
        
        Args:
            num_embeddings: Size of the dictionary of embeddings.
            embedding_dim: Size of each embedding vector.
            padding_idx: If specified, pads the output with zeros whenever it encounters this index.
            max_norm: If given, each embedding vector with norm larger than max_norm is renormalized.
            norm_type: The p of the p-norm to compute for the max_norm option.
            scale_grad_by_freq: If True, scale gradients by the inverse of frequency of the words in the mini-batch.
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor.
        """
        super().__init__()
```

## Activation Functions

### Relu

```python
class Relu(Primitive):
    """
    Rectified Linear Unit activation function.
    
    Args:
        inplace: If True, will do this operation in-place. Default: False.
    """
    
    def __init__(self, inplace=False):
        """
        Initialize a ReLU activation.
        
        Args:
            inplace: If True, will do this operation in-place.
        """
        super().__init__()
```

### LeakyRelu

```python
class LeakyRelu(Primitive):
    """
    Leaky Rectified Linear Unit activation function.
    
    Args:
        negative_slope: Controls the angle of the negative slope. Default: 0.01.
        inplace: If True, will do this operation in-place. Default: False.
    """
    
    def __init__(self, negative_slope=0.01, inplace=False):
        """
        Initialize a LeakyReLU activation.
        
        Args:
            negative_slope: Controls the angle of the negative slope.
            inplace: If True, will do this operation in-place.
        """
        super().__init__()
```

### Sigmoid

```python
class Sigmoid(Primitive):
    """
    Sigmoid activation function.
    """
    
    def __init__(self):
        """
        Initialize a Sigmoid activation.
        """
        super().__init__()
```

### Tanh

```python
class Tanh(Primitive):
    """
    Hyperbolic tangent activation function.
    """
    
    def __init__(self):
        """
        Initialize a Tanh activation.
        """
        super().__init__()
```

### Softmax

```python
class Softmax(Primitive):
    """
    Softmax activation function.
    
    Args:
        dim: Dimension along which softmax will be computed. Default: None.
    """
    
    def __init__(self, dim=None):
        """
        Initialize a Softmax activation.
        
        Args:
            dim: Dimension along which softmax will be computed.
        """
        super().__init__()
```

## Pooling Layers

### MaxPool1d

```python
class MaxPool1d(Primitive):
    """
    1D max pooling.
    
    Args:
        kernel_size: Size of the window to take a max over.
        stride: Stride of the window. Default: None (same as kernel_size).
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: A parameter that controls the stride of elements in the window. Default: 1.
        return_indices: If True, will return the indices along with the outputs. Default: False.
        ceil_mode: If True, will use ceil instead of floor to compute the output shape. Default: False.
    """
    
    def __init__(
        self, 
        kernel_size, 
        stride=None, 
        padding=0, 
        dilation=1, 
        return_indices=False, 
        ceil_mode=False
    ):
        """
        Initialize a MaxPool1d layer.
        
        Args:
            kernel_size: Size of the window to take a max over.
            stride: Stride of the window.
            padding: Zero-padding added to both sides of the input.
            dilation: A parameter that controls the stride of elements in the window.
            return_indices: If True, will return the indices along with the outputs.
            ceil_mode: If True, will use ceil instead of floor to compute the output shape.
        """
        super().__init__()
```

### MaxPool2d

```python
class MaxPool2d(Primitive):
    """
    2D max pooling.
    
    Args:
        kernel_size: Size of the window to take a max over.
        stride: Stride of the window. Default: None (same as kernel_size).
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: A parameter that controls the stride of elements in the window. Default: 1.
        return_indices: If True, will return the indices along with the outputs. Default: False.
        ceil_mode: If True, will use ceil instead of floor to compute the output shape. Default: False.
    """
    
    def __init__(
        self, 
        kernel_size, 
        stride=None, 
        padding=0, 
        dilation=1, 
        return_indices=False, 
        ceil_mode=False
    ):
        """
        Initialize a MaxPool2d layer.
        
        Args:
            kernel_size: Size of the window to take a max over.
            stride: Stride of the window.
            padding: Zero-padding added to both sides of the input.
            dilation: A parameter that controls the stride of elements in the window.
            return_indices: If True, will return the indices along with the outputs.
            ceil_mode: If True, will use ceil instead of floor to compute the output shape.
        """
        super().__init__()
```

### AvgPool1d

```python
class AvgPool1d(Primitive):
    """
    1D average pooling.
    
    Args:
        kernel_size: Size of the window to take an average over.
        stride: Stride of the window. Default: None (same as kernel_size).
        padding: Zero-padding added to both sides of the input. Default: 0.
        ceil_mode: If True, will use ceil instead of floor to compute the output shape. Default: False.
        count_include_pad: If True, will include the zero-padding in the averaging calculation. Default: True.
    """
    
    def __init__(
        self, 
        kernel_size, 
        stride=None, 
        padding=0, 
        ceil_mode=False, 
        count_include_pad=True
    ):
        """
        Initialize an AvgPool1d layer.
        
        Args:
            kernel_size: Size of the window to take an average over.
            stride: Stride of the window.
            padding: Zero-padding added to both sides of the input.
            ceil_mode: If True, will use ceil instead of floor to compute the output shape.
            count_include_pad: If True, will include the zero-padding in the averaging calculation.
        """
        super().__init__()
```

### AvgPool2d

```python
class AvgPool2d(Primitive):
    """
    2D average pooling.
    
    Args:
        kernel_size: Size of the window to take an average over.
        stride: Stride of the window. Default: None (same as kernel_size).
        padding: Zero-padding added to both sides of the input. Default: 0.
        ceil_mode: If True, will use ceil instead of floor to compute the output shape. Default: False.
        count_include_pad: If True, will include the zero-padding in the averaging calculation. Default: True.
        divisor_override: If specified, it will be used as divisor when computing the average. Default: None.
    """
    
    def __init__(
        self, 
        kernel_size, 
        stride=None, 
        padding=0, 
        ceil_mode=False, 
        count_include_pad=True, 
        divisor_override=None
    ):
        """
        Initialize an AvgPool2d layer.
        
        Args:
            kernel_size: Size of the window to take an average over.
            stride: Stride of the window.
            padding: Zero-padding added to both sides of the input.
            ceil_mode: If True, will use ceil instead of floor to compute the output shape.
            count_include_pad: If True, will include the zero-padding in the averaging calculation.
            divisor_override: If specified, it will be used as divisor when computing the average.
        """
        super().__init__()
```

## Recurrent Layers

### LSTM

```python
class LSTM(Model):
    """
    Long Short-Term Memory (LSTM) layer.
    
    Args:
        input_size: The number of expected features in the input.
        hidden_size: The number of features in the hidden state.
        num_layers: Number of recurrent layers. Default: 1.
        bias: If False, the layer does not use bias weights. Default: True.
        batch_first: If True, input and output tensors are provided as (batch, seq, feature). Default: False.
        dropout: Dropout probability for the output of each layer except the last layer. Default: 0.
        bidirectional: If True, becomes a bidirectional LSTM. Default: False.
    """
    
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_layers=1, 
        bias=True, 
        batch_first=False, 
        dropout=0.0, 
        bidirectional=False
    ):
        """
        Initialize an LSTM layer.
        
        Args:
            input_size: The number of expected features in the input.
            hidden_size: The number of features in the hidden state.
            num_layers: Number of recurrent layers.
            bias: If False, the layer does not use bias weights.
            batch_first: If True, input and output tensors are provided as (batch, seq, feature).
            dropout: Dropout probability for the output of each layer except the last layer.
            bidirectional: If True, becomes a bidirectional LSTM.
        """
        super().__init__()
```

### GRU

```python
class GRU(Model):
    """
    Gated Recurrent Unit (GRU) layer.
    
    Args:
        input_size: The number of expected features in the input.
        hidden_size: The number of features in the hidden state.
        num_layers: Number of recurrent layers. Default: 1.
        bias: If False, the layer does not use bias weights. Default: True.
        batch_first: If True, input and output tensors are provided as (batch, seq, feature). Default: False.
        dropout: Dropout probability for the output of each layer except the last layer. Default: 0.
        bidirectional: If True, becomes a bidirectional GRU. Default: False.
    """
    
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_layers=1, 
        bias=True, 
        batch_first=False, 
        dropout=0.0, 
        bidirectional=False
    ):
        """
        Initialize a GRU layer.
        
        Args:
            input_size: The number of expected features in the input.
            hidden_size: The number of features in the hidden state.
            num_layers: Number of recurrent layers.
            bias: If False, the layer does not use bias weights.
            batch_first: If True, input and output tensors are provided as (batch, seq, feature).
            dropout: Dropout probability for the output of each layer except the last layer.
            bidirectional: If True, becomes a bidirectional GRU.
        """
        super().__init__()
```

## Transformer Components

### MultiheadAttention

```python
class MultiheadAttention(Model):
    """
    Multi-head attention layer.
    
    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights. Default: 0.0.
        bias: If True, adds bias to input/output projections. Default: True.
        add_bias_kv: If True, adds bias to the key and value sequences. Default: False.
        add_zero_attn: If True, adds a new batch of zeros to the key and value sequences. Default: False.
        kdim: Total dimension of the key. Default: None (= embed_dim).
        vdim: Total dimension of the value. Default: None (= embed_dim).
    """
    
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        dropout=0.0, 
        bias=True, 
        add_bias_kv=False, 
        add_zero_attn=False, 
        kdim=None, 
        vdim=None
    ):
        """
        Initialize a MultiheadAttention layer.
        
        Args:
            embed_dim: Total dimension of the model.
            num_heads: Number of parallel attention heads.
            dropout: Dropout probability on attention weights.
            bias: If True, adds bias to input/output projections.
            add_bias_kv: If True, adds bias to the key and value sequences.
            add_zero_attn: If True, adds a new batch of zeros to the key and value sequences.
            kdim: Total dimension of the key.
            vdim: Total dimension of the value.
        """
        super().__init__()
```

### TransformerEncoderLayer

```python
class TransformerEncoderLayer(Model):
    """
    Transformer encoder layer.
    
    Args:
        d_model: The number of expected features in the input.
        nhead: The number of heads in the multiheadattention models.
        dim_feedforward: The dimension of the feedforward network model. Default: 2048.
        dropout: Dropout probability. Default: 0.1.
        activation: The activation function to use. Default: "relu".
        layer_norm_eps: The epsilon value in layer normalization. Default: 1e-5.
        batch_first: If True, input and output tensors are provided as (batch, seq, feature). Default: False.
        norm_first: If True, layer norm is done prior to attention and feedforward operations. Default: False.
    """
    
    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward=2048, 
        dropout=0.1, 
        activation="relu", 
        layer_norm_eps=1e-5, 
        batch_first=False, 
        norm_first=False
    ):
        """
        Initialize a TransformerEncoderLayer.
        
        Args:
            d_model: The number of expected features in the input.
            nhead: The number of heads in the multiheadattention models.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: Dropout probability.
            activation: The activation function to use.
            layer_norm_eps: The epsilon value in layer normalization.
            batch_first: If True, input and output tensors are provided as (batch, seq, feature).
            norm_first: If True, layer norm is done prior to attention and feedforward operations.
        """
        super().__init__()
```

### TransformerEncoder

```python
class TransformerEncoder(Model):
    """
    Transformer encoder.
    
    Args:
        encoder_layer: An instance of the TransformerEncoderLayer class.
        num_layers: The number of sub-encoder-layers in the encoder.
        norm: The normalization layer. Default: None.
    """
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        Initialize a TransformerEncoder.
        
        Args:
            encoder_layer: An instance of the TransformerEncoderLayer class.
            num_layers: The number of sub-encoder-layers in the encoder.
            norm: The normalization layer.
        """
        super().__init__()
```

## Tensor Operations

### Add

```python
class Add(Primitive):
    """
    Element-wise addition of two tensors.
    """
    
    def __init__(self):
        """
        Initialize an Add operation.
        """
        super().__init__()
```

### Multiply

```python
class Multiply(Primitive):
    """
    Element-wise multiplication of two tensors.
    """
    
    def __init__(self):
        """
        Initialize a Multiply operation.
        """
        super().__init__()
```

### Concat

```python
class Concat(Primitive):
    """
    Concatenate tensors along a specified dimension.
    
    Args:
        dim: The dimension along which to concatenate. Default: 0.
    """
    
    def __init__(self, dim=0):
        """
        Initialize a Concat operation.
        
        Args:
            dim: The dimension along which to concatenate.
        """
        super().__init__()
```

### Reshape

```python
class Reshape(Primitive):
    """
    Reshape a tensor to a specified shape.
    
    Args:
        shape: The target shape.
    """
    
    def __init__(self, shape):
        """
        Initialize a Reshape operation.
        
        Args:
            shape: The target shape.
        """
        super().__init__()
```

### Flatten

```python
class Flatten(Primitive):
    """
    Flatten a tensor.
    
    Args:
        start_dim: First dim to flatten. Default: 1.
        end_dim: Last dim to flatten. Default: -1.
    """
    
    def __init__(self, start_dim=1, end_dim=-1):
        """
        Initialize a Flatten operation.
        
        Args:
            start_dim: First dim to flatten.
            end_dim: Last dim to flatten.
        """
        super().__init__()
```

## Control Flow

### If

```python
class If(Model):
    """
    Conditional execution.
    
    Args:
        condition: The condition to evaluate.
        then_branch: The model to execute if the condition is True.
        else_branch: The model to execute if the condition is False.
    """
    
    def __init__(self, condition, then_branch, else_branch):
        """
        Initialize an If operation.
        
        Args:
            condition: The condition to evaluate.
            then_branch: The model to execute if the condition is True.
            else_branch: The model to execute if the condition is False.
        """
        super().__init__()
```

### Identity

```python
class Identity(Primitive):
    """
    Identity function, returns input unchanged.
    """
    
    def __init__(self):
        """
        Initialize an Identity operation.
        """
        super().__init__()
```

## High-Level Models

### MLP

```python
class MLP(Model):
    """
    Multi-layer perceptron.
    
    Args:
        input_size: Size of the input features.
        hidden_sizes: List of hidden layer sizes.
        output_size: Size of the output features.
        activation: Activation function to use between layers. Default: "relu".
        dropout: Dropout probability. Default: 0.0.
        use_batch_norm: Whether to use batch normalization. Default: False.
    """
    
    def __init__(
        self, 
        input_size, 
        hidden_sizes, 
        output_size, 
        activation="relu", 
        dropout=0.0, 
        use_batch_norm=False
    ):
        """
        Initialize an MLP.
        
        Args:
            input_size: Size of the input features.
            hidden_sizes: List of hidden layer sizes.
            output_size: Size of the output features.
            activation: Activation function to use between layers.
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
```

### ResidualBlock

```python
class ResidualBlock(Model):
    """
    Residual block with skip connection.
    
    Args:
        dimension: Number of input/output features.
        activation: Activation function to use. Default: "relu".
        use_batch_norm: Whether to use batch normalization. Default: True.
        dropout: Dropout probability. Default: 0.0.
    """
    
    def __init__(self, dimension, activation="relu", use_batch_norm=True, dropout=0.0):
        """
        Initialize a ResidualBlock.
        
        Args:
            dimension: Number of input/output features.
            activation: Activation function to use.
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
        """
        super().__init__()
```