# Primitives API

The Primitives API provides low-level building blocks for creating models in Mithril. These primitives are the fundamental operations that are combined to create more complex models.

## Core Primitives

### Tensor Operations

```python
from mithril.models.primitives import *
```

#### Basic Operations

```python
# Element-wise operations
add(x, y)          # Addition: x + y
subtract(x, y)     # Subtraction: x - y
multiply(x, y)     # Multiplication: x * y
divide(x, y)       # Division: x / y
power(x, y)        # Power: x ** y

# Mathematical functions
abs(x)             # Absolute value
sqrt(x)            # Square root
exp(x)             # Exponential
log(x)             # Natural logarithm

# Trigonometric functions
sin(x)             # Sine
cos(x)             # Cosine
tan(x)             # Tangent

# Hyperbolic functions
sinh(x)            # Hyperbolic sine
cosh(x)            # Hyperbolic cosine
tanh(x)            # Hyperbolic tangent
```

#### Matrix Operations

```python
matmul(x, y)       # Matrix multiplication
transpose(x)       # Matrix transpose
det(x)             # Determinant
trace(x)           # Trace
inverse(x)         # Matrix inverse
```

#### Reduction Operations

```python
reduce_sum(x, axis=None)      # Sum along axis
reduce_mean(x, axis=None)     # Mean along axis
reduce_max(x, axis=None)      # Maximum along axis
reduce_min(x, axis=None)      # Minimum along axis
reduce_prod(x, axis=None)     # Product along axis
reduce_any(x, axis=None)      # Any along axis (logical OR)
reduce_all(x, axis=None)      # All along axis (logical AND)
```

#### Shape Operations

```python
reshape(x, shape)             # Reshape tensor to new shape
flatten(x)                    # Flatten tensor to 1D
expand_dims(x, axis)          # Add dimension at position axis
squeeze(x, axis=None)         # Remove dimensions of size 1
concatenate(tensors, axis)    # Concatenate tensors along axis
stack(tensors, axis)          # Stack tensors along new axis
split(x, num_or_sections, axis)  # Split tensor into sub-tensors
slice(x, begin, size)         # Extract slice from tensor
```

#### Neural Network Operations

```python
# Activations
relu(x)                       # Rectified Linear Unit: max(0, x)
sigmoid(x)                    # Sigmoid: 1 / (1 + exp(-x))
softmax(x, axis=-1)           # Softmax normalization
leaky_relu(x, alpha=0.2)      # Leaky ReLU
elu(x, alpha=1.0)             # Exponential Linear Unit

# Convolutions and pooling
conv2d(x, filters, kernel_size, strides=(1, 1), padding="SAME")
max_pool2d(x, pool_size, strides=None, padding="SAME")
avg_pool2d(x, pool_size, strides=None, padding="SAME")

# Normalization
batch_norm(x, mean, variance, scale, offset, epsilon=1e-5)
layer_norm(x, scale, offset, epsilon=1e-5)
instance_norm(x, scale, offset, epsilon=1e-5)

# Dropout
dropout(x, rate, training=True, seed=None)
```

#### Loss Functions

```python
mean_squared_error(y_true, y_pred)       # MSE loss
categorical_crossentropy(y_true, y_pred)  # Cross-entropy loss
binary_crossentropy(y_true, y_pred)       # Binary cross-entropy
hinge_loss(y_true, y_pred)                # Hinge loss
huber_loss(y_true, y_pred, delta=1.0)     # Huber loss
```

## Usage in Models

Primitives are used within a model definition:

```python
from mithril.models import LogicalModel
import mithril as mi

model = LogicalModel()
with model:
    x = mi.Input(shape=(None, 784), name="input")
    w1 = mi.Parameter(shape=(784, 128), name="w1")
    b1 = mi.Parameter(shape=(128,), name="b1")
    
    # Use primitives to define computation
    h1 = mi.relu(mi.matmul(x, w1) + b1)
    
    w2 = mi.Parameter(shape=(128, 10), name="w2")
    b2 = mi.Parameter(shape=(10,), name="b2")
    logits = mi.matmul(h1, w2) + b2
    
    probs = mi.softmax(logits)
    mi.Output(probs, name="output")
```

## Custom Primitives

You can define custom primitives by extending the `Operator` class:

```python
from mithril.framework.logical import Operator, register_op

# Define a custom primitive
@register_op
class MyCustomOp(Operator):
    def __init__(self, x, factor, name=None):
        super().__init__(inputs=[x], name=name)
        self.factor = factor
    
    def infer_shape(self, input_shape):
        return input_shape
    
    def infer_type(self, input_type):
        return input_type

# Create a convenience function
def my_custom_op(x, factor):
    return MyCustomOp(x, factor)

# Use in a model
with model:
    y = my_custom_op(x, factor=2.0)
```

## Backend Implementations

Each primitive operation has corresponding implementations in the supported backends. For example, the `relu` primitive maps to:

- JAX: `jax.nn.relu`
- PyTorch: `torch.nn.functional.relu`
- NumPy: Custom implementation using `np.maximum`
- MLX: `mlx.nn.relu`

This mapping is handled automatically by the Mithril compilation process.

## Performance Considerations

When using primitives, consider these performance tips:

1. **Fusion opportunities**: Operations like `relu(matmul(x, w) + b)` can be fused by some backends
2. **Memory layout**: Some operations perform better with specific memory layouts
3. **Precision**: Lower precision operations (float16) can be significantly faster on GPUs

## Examples

For more examples of using primitives, see the [examples directory](https://github.com/example/mithril/tree/main/examples) in the Mithril repository.