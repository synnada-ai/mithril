# Common Utilities API

The Common Utilities API provides helper functions and utilities that are used throughout the Mithril framework. These utilities handle tasks like type checking, shape manipulation, error handling, and more.

## Type Utilities

```python
from mithril.utils.type_utils import *
```

### Type Checking

```python
is_integer_type(dtype)      # Check if dtype is an integer type
is_floating_type(dtype)     # Check if dtype is a floating-point type
is_numeric_type(dtype)      # Check if dtype is a numeric type
is_boolean_type(dtype)      # Check if dtype is a boolean type

is_scalar(value)            # Check if value is a scalar
is_tensor(value)            # Check if value is a tensor

get_canonical_type(dtype)   # Get canonical type name (e.g., 'float32' for 'f32')
convert_to_dtype(value, dtype)  # Convert value to specified dtype
```

### Type Promotion

```python
promote_types(type1, type2)  # Determine result type from two input types
promote_numeric_types(types)  # Promote list of numeric types
get_common_dtype(values)      # Get common dtype for a list of values
```

## Shape Utilities

```python
from mithril.utils.utils import *
```

### Shape Manipulation

```python
shape_to_tuple(shape)         # Convert shape to canonical tuple form
broadcast_shapes(shape1, shape2)  # Compute broadcast shape
get_shape_size(shape)         # Compute total number of elements in shape
validate_axis(axis, ndim)      # Validate axis is within range for ndim
normalize_axis(axis, ndim)     # Normalize negative axis

# Check shapes are compatible for operation
are_shapes_compatible(shape1, shape2)

# Check if shape1 can be broadcast to shape2
can_broadcast_to(shape1, shape2)
```

### Shape Inference

```python
# Infer output shape for common operations
infer_elementwise_shape(shapes)    # Shape for elementwise operations
infer_reduce_shape(input_shape, axis, keepdims)  # Shape for reduction
infer_matmul_shape(a_shape, b_shape)  # Shape for matrix multiplication
infer_conv2d_shape(input_shape, filter_shape, strides, padding)  # Shape for 2D convolution
```

## Dictionary Conversions

```python
from mithril.utils.dict_conversions import *
```

### Model Serialization

```python
to_dict(model)                # Convert model to dictionary representation
from_dict(model_dict)         # Create model from dictionary representation

# Update model parameters from dictionary
update_from_dict(model, param_dict)

# Extract specific parts of model dictionary
extract_parameters(model_dict)
extract_structure(model_dict)
```

## Functional Utilities

```python
from mithril.utils.func_utils import *
```

### Function Manipulation

```python
curry(func, *args, **kwargs)  # Partially apply function
compose(f, g)                 # Function composition: f(g(x))

# Wrap function to track call count
track_calls(func)

# Cache function results
cached(func)

# Create vectorized function
vectorize(func, excluded=None)
```

## Evaluation Context

```python
from mithril.utils.evaluation_context import EvaluationContext
```

The `EvaluationContext` manages state during model evaluation:

```python
# Create an evaluation context
context = EvaluationContext()

# Store values
context.set_value("x", tensor_value)
context.set_shape("x", (10, 5))
context.set_type("x", "float32")

# Retrieve values
value = context.get_value("x")
shape = context.get_shape("x")
dtype = context.get_type("x")

# Check if keys exist
has_value = context.has_value("x")
has_shape = context.has_shape("x")
has_type = context.has_type("x")

# Clear context
context.clear()
```

## Error Handling

```python
from mithril.common import MithrilError, ShapeError, TypeError, ValidationError
```

Mithril provides custom exceptions for specific error cases:

```python
# Base error class
try:
    # Some operation
    pass
except MithrilError as e:
    print(f"Mithril error: {e}")

# Shape-related error
try:
    # Shape validation
    if not are_shapes_compatible(shape1, shape2):
        raise ShapeError(f"Incompatible shapes: {shape1} and {shape2}")
    pass
except ShapeError as e:
    print(f"Shape error: {e}")

# Type-related error
try:
    # Type validation
    if not is_numeric_type(dtype):
        raise TypeError(f"Expected numeric type, got {dtype}")
    pass
except TypeError as e:
    print(f"Type error: {e}")

# Validation error
try:
    # Parameter validation
    if value < 0:
        raise ValidationError(f"Value must be non-negative, got {value}")
    pass
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Logging

```python
from mithril.utils import get_logger
```

Mithril provides a standardized logging interface:

```python
# Get a logger
logger = get_logger("mithril.module_name")

# Log messages at different levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error")

# Log with additional context
logger.info("Processing model", extra={"model_name": "resnet18", "backend": "jax"})
```

## Configuration

```python
from mithril.utils import get_config, set_config
```

Mithril provides a configuration system:

```python
# Get configuration values
debug_mode = get_config("debug", default=False)
log_level = get_config("log_level", default="INFO")

# Set configuration values
set_config("default_float_type", "float32")
set_config("default_backend", "jax")
```

## Path Handling

```python
from mithril.utils import get_cache_dir, get_data_dir
```

Utilities for managing file paths:

```python
# Get standard directories
cache_dir = get_cache_dir()  # Returns ~/.cache/mithril
data_dir = get_data_dir()    # Returns ~/.mithril/data

# Create a file path in standard directory
model_path = os.path.join(get_cache_dir(), "models", "model.json")
```

## Performance Utilities

```python
from mithril.utils import timer, memory_usage
```

Utilities for measuring performance:

```python
# Measure execution time
with timer() as t:
    result = compute_intensive_function()
print(f"Execution time: {t.elapsed:.4f} seconds")

# Measure function execution time
@timer()
def slow_function():
    # Code here
    pass

slow_function()  # Will print execution time

# Measure memory usage
with memory_usage() as mem:
    result = memory_intensive_function()
print(f"Peak memory usage: {mem.peak} MB")
```

## Random Utilities

```python
from mithril.utils import set_random_seed, get_random_seed
```

Utilities for managing random number generation:

```python
# Set random seed for reproducibility
set_random_seed(42)

# Get current random seed
current_seed = get_random_seed()
```

## Examples

For more examples of using the Common Utilities API, see the [examples directory](https://github.com/example/mithril/tree/main/examples) in the Mithril repository.