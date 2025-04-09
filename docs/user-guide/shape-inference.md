# Shape Inference

Shape inference in Mithril automates the process of determining tensor shapes throughout your model. This guide explains how shape inference works and how to use it effectively.

## Overview

Shape inference analyzes your model's structure to automatically determine the shapes of intermediate tensors based on input shapes. This has several benefits:

- Ensures shape compatibility between operations
- Reduces manual shape specification
- Helps detect shape-related errors early
- Enables static memory planning

## Basic Shape Inference

By default, Mithril infers shapes automatically when you provide input shapes:

```python
import mithril as mi
from mithril.models import LogicalModel

# Create a model
model = LogicalModel()
with model:
    x = mi.Input(shape=(None, 10), name="input")
    w = mi.Parameter(shape=(10, 5), name="weights")
    b = mi.Parameter(shape=(5,), name="bias")
    y = mi.matmul(x, w) + b
    mi.Output(y, name="output")

# Shape inference happens automatically
model.set_shapes({"input": (32, 10)})

# Now all tensor shapes are known
print(model.get_shape("output"))  # (32, 5)
```

## Dynamic Shapes

Mithril supports dynamic shapes using `None` as a dimension placeholder:

```python
# Define a model with dynamic batch size
model = LogicalModel()
with model:
    x = mi.Input(shape=(None, 784), name="input")
    w = mi.Parameter(shape=(784, 10), name="weights")
    y = mi.matmul(x, w)
    mi.Output(y, name="output")

# Can run with any batch size
model.set_shapes({"input": (1, 784)})
print(model.get_shape("output"))  # (1, 10)

model.set_shapes({"input": (32, 784)})
print(model.get_shape("output"))  # (32, 10)
```

## Shape Functions

Each operator in Mithril has an associated shape function that determines its output shape based on input shapes. For example:

- `matmul`: Determines output shape based on matrix multiplication rules
- `conv2d`: Calculates output shape based on input shape, kernel size, padding, and stride
- `reshape`: Takes output shape directly from parameters

## Custom Shape Inference

For custom operators, you can define custom shape functions:

```python
from mithril.framework.logical import Operator

class MyCustomOp(Operator):
    def __init__(self, input, factor, name=None):
        super().__init__(inputs=[input], name=name)
        self.factor = factor
        
    def infer_shape(self, input_shape):
        # Custom shape logic
        return (input_shape[0], input_shape[1] * self.factor)
```

## Shape Verification

Mithril verifies shape compatibility during both model definition and execution:

```python
# This will raise a shape compatibility error
model = LogicalModel()
with model:
    x = mi.Input(shape=(10, 5), name="input")
    w = mi.Parameter(shape=(7, 3), name="weights")  # Incompatible dimensions
    y = mi.matmul(x, w)  # Will raise error: 5 != 7
    mi.Output(y, name="output")
```

## Debugging Shape Issues

When you encounter shape issues, you can debug them using:

```python
# Print shapes of all tensors
model.print_shapes()

# Check specific tensor shape
print(model.get_shape("layer1/output"))

# Visualize the model with shapes
model.visualize(show_shapes=True)
```

## Shape Inference and Compilation

Shape inference is a critical step during model compilation. The compiler uses inferred shapes to:

1. Allocate memory for tensors
2. Select optimized kernels based on shapes
3. Perform optimizations like operator fusion

## Best Practices

1. **Specify Input Shapes Early**: Set input shapes early to enable effective shape inference
2. **Use Dynamic Dimensions**: Use `None` for dimensions that can vary (like batch size)
3. **Check Shapes**: Verify shape compatibility before running expensive operations
4. **Debug with Visualization**: Use model visualization to debug shape issues

## Advanced Topics

### Broadcasting

Mithril supports NumPy-style broadcasting rules when combining tensors of different shapes:

```python
# Broadcasting example
x = mi.Input(shape=(10, 1), name="x")
y = mi.Input(shape=(1, 5), name="y")
z = x + y  # Result shape: (10, 5)
```

### Reshaping and Transposition

Reshaping and transposition operations modify tensor shapes in predictable ways:

```python
x = mi.Input(shape=(10, 5), name="x")
y = mi.reshape(x, shape=(5, 10))  # Explicitly specify new shape
z = mi.transpose(x)  # Result shape: (5, 10)
```

### Shape Constraints

You can enforce shape constraints in your model:

```python
from mithril.framework import constraints

model = LogicalModel()
with model:
    x = mi.Input(shape=(None, 10), name="input")
    # Add constraint that batch dimension must be even
    constraints.add_dimension_constraint(x, 0, lambda d: d % 2 == 0)
```