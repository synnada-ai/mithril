# Custom Primitives in Mithril

Mithril provides a flexible framework for creating custom primitives, allowing you to extend the framework with specialized operations tailored to your specific needs. This guide explains how to create, register, and use custom primitives across different backends.

## Understanding Primitives in Mithril

Primitives are the fundamental building blocks of computation in Mithril. They represent atomic operations that can be composed to create complex models. Each primitive:

1. Has a unique formula key that identifies the operation
2. Defines its behavior across different backends
3. Specifies how shapes, types, and gradients are handled

Mithril's architecture separates the logical representation of primitives from their backend-specific implementations, making it possible to write primitives once and run them on any supported backend.

## Creating Custom Primitives

### Basic Structure

To create a custom primitive in Mithril, you need to:

1. Define the logical operator
2. Implement the backend-specific function
3. Register the primitive with your backends

Let's look at a complete example of creating a custom activation function:

```python
import mithril as ml
from mithril.framework.logical.operator import Operator
from mithril.framework.logical.primitive import PrimitiveModel
from mithril import Tensor

class Swish(PrimitiveModel):
    """Swish activation function: x * sigmoid(x)"""
    class_name = "Swish"
    
    def __init__(self, *, name: str | None = None):
        super().__init__("swish", name=name, input=ml.IOKey("input"), output=ml.IOKey("output"))
```

### Implementing Backend Functions

For each backend you want to support, you need to implement the corresponding function. Let's implement the Swish activation for PyTorch, JAX, and NumPy backends:

```python
# PyTorch implementation
def torch_swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

# JAX implementation
def jax_swish(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)

# NumPy implementation
def numpy_swish(x: np.ndarray) -> np.ndarray:
    return x * (1 / (1 + np.exp(-x)))

# MLX implementation
def mlx_swish(x: mlx.core.array) -> mlx.core.array:
    return x * mlx.nn.sigmoid(x)
```

### Registering the Primitive

After defining the primitive and its implementations, you need to register it with each backend:

```python
def register_swish_primitive():
    # Register with PyTorch backend
    if hasattr(ml, "TorchBackend"):
        ml.TorchBackend.register_primitive("swish", torch_swish)
    
    # Register with JAX backend
    if hasattr(ml, "JaxBackend"):
        ml.JaxBackend.register_primitive("swish", jax_swish)
    
    # Register with NumPy backend
    if hasattr(ml, "NumpyBackend"):
        ml.NumpyBackend.register_primitive("swish", numpy_swish)
    
    # Register with MLX backend
    if hasattr(ml, "MlxBackend"):
        ml.MlxBackend.register_primitive("swish", mlx_swish)

# Call this function to register the primitive
register_swish_primitive()
```

### Using the Custom Primitive

Now that you've defined and registered your custom primitive, you can use it in your models:

```python
# Create a simple model using the Swish activation
def create_mlp_with_swish(input_dim: int, hidden_dim: int, output_dim: int):
    model = ml.models.Model()
    model += ml.models.Linear(hidden_dim)(input="input")
    model += Swish()(input=model.cout)
    model += ml.models.Linear(output_dim)(output="output")
    return model

# Create and compile the model
model = create_mlp_with_swish(100, 256, 10)
backend = ml.TorchBackend()
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 100]},
    data_keys={"input"}
)
```

## Advanced Primitive Features

### 1. Handling Shapes and Types

For primitives that transform shapes or types, you need to define how these transformations occur:

```python
class Reshape(PrimitiveModel):
    """Reshape tensor to a new shape"""
    class_name = "Reshape"
    
    def __init__(self, shape: tuple[int, ...], *, name: str | None = None):
        self.shape = shape
        super().__init__(
            "reshape", 
            name=name, 
            input=ml.IOKey("input"),
            shape=ml.IOKey("shape", value=shape),
            output=ml.IOKey("output")
        )
```

For the backend implementations:

```python
# PyTorch implementation
def torch_reshape(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(shape)

# JAX implementation
def jax_reshape(x: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return jax.numpy.reshape(x, shape)

# NumPy implementation
def numpy_reshape(x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.reshape(x, shape)

# Register the reshape primitive
ml.TorchBackend.register_primitive("reshape", torch_reshape)
ml.JaxBackend.register_primitive("reshape", jax_reshape)
ml.NumpyBackend.register_primitive("reshape", numpy_reshape)
```

### 2. Custom Gradients

For primitives with specialized gradient computations, you can define custom gradient functions:

```python
# PyTorch custom gradient example (using autograd)
class CustomGradSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        sigmoid_x = torch.sigmoid(x)
        result = x * sigmoid_x
        ctx.save_for_backward(x, sigmoid_x)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, sigmoid_x = ctx.saved_tensors
        dx = sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
        return grad_output * dx

def torch_custom_grad_swish(x: torch.Tensor) -> torch.Tensor:
    return CustomGradSwish.apply(x)

# Register with custom gradient
ml.TorchBackend.register_primitive("swish", torch_custom_grad_swish)
```

For JAX:

```python
# JAX custom gradient
@jax.custom_gradient
def jax_custom_grad_swish(x):
    sigmoid_x = jax.nn.sigmoid(x)
    y = x * sigmoid_x
    def grad_fn(grad_y):
        return grad_y * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
    return y, grad_fn

# Register with custom gradient
ml.JaxBackend.register_primitive("swish", jax_custom_grad_swish)
```

### 3. Stateful Primitives

Some primitives need to maintain state. Here's how to implement a stateful primitive:

```python
class BatchNorm(PrimitiveModel):
    """Batch normalization primitive with running statistics"""
    class_name = "BatchNorm"
    
    def __init__(
        self, 
        num_features: int, 
        momentum: float = 0.1, 
        eps: float = 1e-5, 
        *, 
        name: str | None = None
    ):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        super().__init__(
            "batch_norm", 
            name=name, 
            input=ml.IOKey("input"),
            running_mean=ml.IOKey("running_mean", shape=(num_features,), differentiable=False),
            running_var=ml.IOKey("running_var", shape=(num_features,), differentiable=False),
            weight=ml.IOKey("weight", shape=(num_features,), differentiable=True),
            bias=ml.IOKey("bias", shape=(num_features,), differentiable=True),
            momentum=ml.IOKey("momentum", value=momentum),
            eps=ml.IOKey("eps", value=eps),
            training=ml.IOKey("training", value=True),
            output=ml.IOKey("output")
        )
```

Backend implementation for PyTorch:

```python
def torch_batch_norm(
    x: torch.Tensor, 
    running_mean: torch.Tensor, 
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    momentum: float,
    eps: float,
    training: bool
) -> torch.Tensor:
    return torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, 
        training=training, momentum=momentum, eps=eps
    )

# Register the batch norm primitive
ml.TorchBackend.register_primitive("batch_norm", torch_batch_norm)
```

## Integrating with Mithril's Constraint System

Mithril's constraint system ensures that shapes and types are properly propagated through the model. Custom primitives should integrate with this system:

```python
from mithril.framework.common import Updates, UpdateType

class CustomConv2D(PrimitiveModel):
    """Custom 2D convolution with shape inference"""
    class_name = "CustomConv2D"
    
    def __init__(
        self, 
        out_channels: int, 
        kernel_size: int | tuple[int, int], 
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        *, 
        name: str | None = None
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
            
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        super().__init__(
            "custom_conv2d", 
            name=name, 
            input=ml.IOKey("input"),
            weight=ml.IOKey(
                "weight", 
                shape=(out_channels, None, *kernel_size), 
                differentiable=True
            ),
            bias=ml.IOKey("bias", shape=(out_channels,), differentiable=True),
            stride=ml.IOKey("stride", value=stride),
            padding=ml.IOKey("padding", value=padding),
            output=ml.IOKey("output")
        )
    
    def set_constraints(self, updates: Updates) -> None:
        """Set shape constraints for the custom convolution"""
        input_shape = self.input.shape
        if input_shape is None or len(input_shape) != 4:
            return
            
        # Get input channels for weight shape
        in_channels = input_shape[1]
        # Set weight shape fully
        updates |= self.weight.set_shape((self.out_channels, in_channels, *self.kernel_size))
        
        # Calculate output spatial dimensions
        h_in, w_in = input_shape[2], input_shape[3]
        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Set output shape
        updates |= self.output.set_shape((input_shape[0], self.out_channels, h_out, w_out))
```

## Creating Composite Primitives

Sometimes, you may want to create a new primitive that is composed of multiple existing primitives. This can be done by implementing a `create_model` method that returns a model composed of other primitives:

```python
from mithril.framework.logical.operator import Operator

class GELU(Operator):
    """Gaussian Error Linear Unit activation function"""
    
    def __init__(self, approximate: bool = False, *, name: str | None = None):
        self.approximate = approximate
        super().__init__(
            "gelu", 
            name=name or self.__class__.__name__,
            input=ml.IOKey("input"),
            approximate=ml.IOKey("approximate", value=approximate),
            output=ml.IOKey("output")
        )
    
    def create_model(self):
        model = ml.models.Model()
        input = ml.IOKey("input")
        
        if self.approximate:
            # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            constant = 0.7978845608 # sqrt(2/π)
            model |= ml.models.Power(exponent=3)(input=input, output="x_cubed")
            model |= ml.models.Multiply()(left="x_cubed", right=0.044715, output="scaled_cube")
            model |= ml.models.Add()(left=input, right="scaled_cube", output="inner")
            model |= ml.models.Multiply()(left="inner", right=constant, output="scaled_inner")
            model |= ml.models.Tanh()(input="scaled_inner", output="tanh_out")
            model |= ml.models.Add()(left="tanh_out", right=1.0, output="shifted")
            model |= ml.models.Multiply()(left=input, right="shifted", output="almost")
            model |= ml.models.Multiply()(left="almost", right=0.5, output=ml.IOKey("output"))
        else:
            # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            constant = 0.7071067812 # 1/sqrt(2)
            model |= ml.models.Multiply()(left=input, right=constant, output="scaled")
            model |= ml.models.Erf()(input="scaled", output="erf_out")
            model |= ml.models.Add()(left="erf_out", right=1.0, output="shifted")
            model |= ml.models.Multiply()(left=input, right="shifted", output="almost")
            model |= ml.models.Multiply()(left="almost", right=0.5, output=ml.IOKey("output"))
        
        return model
```

## Working with Multiple Backends

When creating custom primitives, it's important to ensure they work consistently across all backends you want to support. Here are some strategies for handling multiple backends:

```python
class CustomPrimitive(PrimitiveModel):
    """A primitive with implementations for multiple backends"""
    class_name = "CustomPrimitive"
    
    def __init__(self, param: float, *, name: str | None = None):
        self.param = param
        super().__init__(
            "custom_op", 
            name=name, 
            input=ml.IOKey("input"),
            param=ml.IOKey("param", value=param),
            output=ml.IOKey("output")
        )

# Register for different backends
def register_custom_primitive():
    backends = []
    
    # Check which backends are available
    if hasattr(ml, "TorchBackend"):
        backends.append((ml.TorchBackend, torch_custom_op))
    if hasattr(ml, "JaxBackend"):
        backends.append((ml.JaxBackend, jax_custom_op))
    if hasattr(ml, "NumpyBackend"):
        backends.append((ml.NumpyBackend, numpy_custom_op))
    if hasattr(ml, "MlxBackend"):
        backends.append((ml.MlxBackend, mlx_custom_op))
    
    # Register with all available backends
    for backend, implementation in backends:
        backend.register_primitive("custom_op", implementation)
```

## Testing Custom Primitives

It's important to test custom primitives to ensure they work correctly. Here's an example test suite:

```python
import pytest
import numpy as np

def test_custom_primitive():
    # Create input data
    input_data = np.random.randn(32, 100).astype(np.float32)
    
    # Test with PyTorch backend
    if hasattr(ml, "TorchBackend"):
        backend = ml.TorchBackend()
        torch_input = backend.array(input_data)
        
        # Create and compile model
        model = ml.models.Model()
        model += CustomPrimitive(param=1.5)(input="input", output="output")
        
        compiled_model = ml.compile(
            model=model,
            backend=backend,
            shapes={"input": [32, 100]},
            data_keys={"input"}
        )
        
        # Run the model
        result = compiled_model.evaluate({}, {"input": torch_input})["output"]
        
        # Verify result
        reference_result = torch_custom_op(torch_input, 1.5)
        np.testing.assert_allclose(
            backend.to_numpy(result), 
            backend.to_numpy(reference_result), 
            rtol=1e-5
        )
    
    # Test with JAX backend (similar pattern)
    if hasattr(ml, "JaxBackend"):
        # ... similar tests for JAX ...
        pass
```

## Best Practices for Custom Primitives

When creating custom primitives in Mithril, follow these best practices:

1. **Consistent Behavior**: Ensure your primitive behaves consistently across all backends.

2. **Clear Documentation**: Document your primitive's purpose, parameters, and behavior.

3. **Proper Shape Inference**: Implement robust shape inference to ensure your primitive works with Mithril's shape propagation system.

4. **Type Safety**: Handle type conversions carefully to maintain numerical stability.

5. **Gradient Support**: For differentiable primitives, ensure that gradients are properly defined.

6. **Efficient Implementation**: Optimize your primitive's implementation for each backend to maximize performance.

7. **Comprehensive Testing**: Test your primitive with various inputs and across all supported backends.

8. **Fallback Implementations**: For backends where native implementation isn't possible, provide a composition of other primitives.

## Conclusion

Custom primitives are a powerful way to extend Mithril's capabilities and tailor it to your specific needs. By following the guidelines in this document, you can create robust, efficient, and backend-agnostic primitives that integrate seamlessly with the Mithril framework.

Whether you're implementing specialized numerical operations, domain-specific functions, or optimized versions of common operations, Mithril's primitive system provides the flexibility and control you need to enhance your models.