# Type Inference

Type inference in Mithril automates the process of determining tensor data types throughout your model. This guide explains how type inference works and how to use it effectively.

## Overview

Mithril's type inference system automatically propagates and determines data types for all tensors in your model. Key benefits include:

- Ensuring type compatibility between operations
- Reducing manual type specification
- Enabling backend-specific optimizations
- Facilitating precision control (e.g., mixed precision training)

## Basic Type Inference

By default, Mithril infers types automatically when you provide input types:

```python
import mithril as mi
from mithril.models import LogicalModel

# Create a model
model = LogicalModel()
with model:
    x = mi.Input(shape=(None, 10), dtype="float32", name="input")
    w = mi.Parameter(shape=(10, 5), name="weights")  # Type will be inferred
    b = mi.Parameter(shape=(5,), name="bias")        # Type will be inferred
    y = mi.matmul(x, w) + b
    mi.Output(y, name="output")

# Type inference happens automatically
model.set_types({"input": "float32"})

# Now all tensor types are known
print(model.get_type("output"))  # float32
print(model.get_type("weights"))  # float32
```

## Default Types

Mithril assigns default types when not explicitly specified:

- Floating-point operations default to `float32`
- Integer operations default to `int32`
- Boolean operations default to `bool`

You can override the default floating-point type:

```python
from mithril.types import set_default_float_type

# Set default float type to float64
set_default_float_type("float64")

# Create model with float64 as default
model = LogicalModel()
with model:
    x = mi.Input(shape=(10, 10), name="input")  # Will be float64
    y = mi.sin(x)                              # Will be float64
    mi.Output(y, name="output")
```

## Type Conversion

You can explicitly convert between types:

```python
model = LogicalModel()
with model:
    x = mi.Input(shape=(10,), dtype="float32", name="input")
    y = mi.cast(x, dtype="int32")
    z = mi.cast(y, dtype="float64")
    mi.Output(z, name="output")

print(model.get_type("output"))  # float64
```

## Mixed Precision

Mithril supports mixed precision computations for improved performance:

```python
model = LogicalModel()
with model:
    x = mi.Input(shape=(None, 10), dtype="float16", name="input")
    w = mi.Parameter(shape=(10, 5), dtype="float16", name="weights")
    
    # Compute in float16
    y = mi.matmul(x, w)
    
    # Convert to float32 for better accuracy in reduction
    y_f32 = mi.cast(y, dtype="float32")
    output = mi.reduce_mean(y_f32, axis=1)
    
    mi.Output(output, name="output")
```

## Backend-Specific Types

Different backends support different data types:

| Type | JAX | PyTorch | NumPy | MLX | C |
|------|-----|---------|-------|-----|---|
| float16 | ✓ | ✓ | ✓ | ✓ | ✓ |
| float32 | ✓ | ✓ | ✓ | ✓ | ✓ |
| float64 | ✓ | ✓ | ✓ | ✓ | ✓ |
| bfloat16 | ✓ | ✓ | ✗ | ✓ | ✗ |
| int8 | ✓ | ✓ | ✓ | ✓ | ✓ |
| int16 | ✓ | ✓ | ✓ | ✓ | ✓ |
| int32 | ✓ | ✓ | ✓ | ✓ | ✓ |
| int64 | ✓ | ✓ | ✓ | ✓ | ✓ |
| bool | ✓ | ✓ | ✓ | ✓ | ✓ |

When compiling for a specific backend, Mithril checks type compatibility:

```python
from mithril.backends.with_autograd.jax_backend import JaxBackend

model = LogicalModel()
with model:
    x = mi.Input(shape=(10,), dtype="bfloat16", name="input")
    y = mi.sin(x)
    mi.Output(y, name="output")

# Works with JAX backend
jax_backend = JaxBackend()
physical_model_jax = model.compile(jax_backend)

# Would fail with NumPy backend since it doesn't support bfloat16
```

## Type Inference Rules

Mithril follows these type inference rules:

1. **Explicit types** take precedence over inferred types
2. **Widening conversions** are applied automatically (e.g., float32 + float64 → float64)
3. **Type compatibility** is checked for all operations
4. **Backend limitations** are respected during compilation

## Custom Type Rules

For custom operators, you can define custom type inference rules:

```python
from mithril.framework.logical import Operator

class MyCustomOp(Operator):
    def __init__(self, input, name=None):
        super().__init__(inputs=[input], name=name)
        
    def infer_type(self, input_type):
        # Custom type logic
        if input_type == "float32":
            return "float64"  # Promote precision
        return input_type
```

## Debugging Type Issues

When you encounter type issues, you can debug them using:

```python
# Print types of all tensors
model.print_types()

# Check specific tensor type
print(model.get_type("layer1/output"))

# Visualize the model with types
model.visualize(show_types=True)
```

## Best Practices

1. **Be Explicit When Needed**: Specify types explicitly for key tensors
2. **Use Mixed Precision Carefully**: Understand the trade-offs between performance and accuracy
3. **Check Backend Compatibility**: Ensure your model uses types supported by your target backend
4. **Prefer Wider Types for Accumulation**: Use wider types (e.g., float32) for reductions and accumulations

## Advanced Topics

### Quantization

Mithril supports quantized types for efficient inference:

```python
# Define a model with quantized weights
model = LogicalModel()
with model:
    x = mi.Input(shape=(None, 10), dtype="float32", name="input")
    w = mi.Parameter(shape=(10, 5), dtype="int8", name="weights")
    scale = mi.Parameter(shape=(1,), dtype="float32", name="scale")
    
    # Dequantize weights
    w_float = mi.cast(w, dtype="float32") * scale
    
    # Compute with dequantized weights
    y = mi.matmul(x, w_float)
    mi.Output(y, name="output")
```

### Type Constraints

You can enforce type constraints in your model:

```python
from mithril.framework import constraints

model = LogicalModel()
with model:
    x = mi.Input(shape=(10,), name="input")
    # Add constraint that input must be floating point
    constraints.add_type_constraint(x, lambda t: t.startswith("float"))
```

### Performance Considerations

Different data types have different performance characteristics:

- **float16/bfloat16**: Fastest on modern GPUs, may have accuracy issues
- **float32**: Good balance of speed and accuracy
- **float64**: Highest precision, but slowest performance
- **int8/int16**: Fast for inference when using quantization

Choose the appropriate type based on your accuracy and performance requirements.