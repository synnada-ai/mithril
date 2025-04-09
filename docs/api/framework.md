# Framework API

The Framework API provides the core abstractions and utilities for building and manipulating models in Mithril. This document covers the key components of the framework and how to use them.

## Logical Framework

The logical framework provides abstractions for defining models without tying them to a specific backend.

### Operators

```python
from mithril.framework.logical import Operator, register_op
```

Operators are the basic building blocks of logical models:

```python
@register_op
class MyCustomOp(Operator):
    def __init__(self, input1, input2, factor=1.0, name=None):
        super().__init__(inputs=[input1, input2], name=name)
        self.factor = factor
    
    def infer_shape(self, input1_shape, input2_shape):
        # Define how output shape is determined from input shapes
        return input1_shape
    
    def infer_type(self, input1_type, input2_type):
        # Define how output type is determined from input types
        return input1_type
```

### Primitives

Primitives are pre-defined operators that map to backend-specific implementations:

```python
from mithril.framework.logical.primitive import Primitive

# Define a primitive operator
@register_op
class Add(Primitive):
    def __init__(self, a, b, name=None):
        super().__init__(inputs=[a, b], name=name, op_name="add")
    
    def infer_shape(self, a_shape, b_shape):
        # Broadcasting rules
        return broadcast_shapes(a_shape, b_shape)
    
    def infer_type(self, a_type, b_type):
        # Type promotion rules
        return promote_types(a_type, b_type)
```

### Logical Models

Logical models compose operators into computation graphs:

```python
from mithril.framework.logical.model import LogicalModel

# Create a logical model
model = LogicalModel(name="my_model")

# Define the model graph
with model:
    x = Input(shape=(None, 10), name="input")
    w = Parameter(shape=(10, 5), name="weights")
    b = Parameter(shape=(5,), name="bias")
    
    y = matmul(x, w) + b
    Output(y, name="output")

# Analyze the model
print(model.summary())
```

## Physical Framework

The physical framework handles the compiled representation of models for specific backends.

### Data Store

```python
from mithril.framework.physical.data_store import DataStore
```

The `DataStore` manages tensor data and metadata:

```python
# Create a data store
data_store = DataStore()

# Store tensors
data_store.set_value("weights", backend.tensor(np.random.randn(10, 5)))
data_store.set_value("bias", backend.tensor(np.zeros(5)))

# Retrieve tensors
weights = data_store.get_value("weights")
bias = data_store.get_value("bias")

# Set and get metadata
data_store.set_metadata("weights", {"trainable": True, "initializer": "normal"})
meta = data_store.get_metadata("weights")
```

### Flat Graph

```python
from mithril.framework.physical.flat_graph import FlatGraph
```

The `FlatGraph` represents the flattened computation graph:

```python
# Create a flat graph
flat_graph = FlatGraph()

# Add nodes to the graph
flat_graph.add_node("matmul", inputs=["input", "weights"], outputs=["matmul_output"])
flat_graph.add_node("add", inputs=["matmul_output", "bias"], outputs=["output"])

# Mark inputs and outputs
flat_graph.mark_as_input("input")
flat_graph.mark_as_output("output")

# Analyze the graph
print(flat_graph.get_sorted_nodes())
print(flat_graph.get_input_keys())
print(flat_graph.get_output_keys())
```

### Physical Model

```python
from mithril.framework.physical.model import PhysicalModel
```

The `PhysicalModel` combines a flat graph and data store with a backend implementation:

```python
# Create a physical model
physical_model = PhysicalModel(
    flat_graph=flat_graph,
    data_store=data_store,
    backend=backend
)

# Run inference
input_data = {"input": backend.tensor(np.random.randn(32, 10))}
output = physical_model(input_data)

# Get output values
result = output["output"]

# Get parameters
params = physical_model.get_parameters()
```

## Code Generation

The code generation framework translates logical models to target-specific code:

```python
from mithril.framework.codegen import CodeGenerator
```

Different code generators target specific backends:

```python
# Python code generation
from mithril.framework.codegen.python_gen import PythonGenerator

python_gen = PythonGenerator()
code = python_gen.generate(logical_model)

# C code generation
from mithril.framework.codegen.c_gen import CGenerator

c_gen = CGenerator()
code = c_gen.generate(logical_model)

# NumPy code generation
from mithril.framework.codegen.numpy_gen import NumPyGenerator

numpy_gen = NumPyGenerator()
code = numpy_gen.generate(logical_model)
```

## Constraints

The constraints system enforces rules on models:

```python
from mithril.framework.constraints import Constraint, ConstraintViolation
```

Define and check constraints:

```python
# Define a custom constraint
class ShapeConstraint(Constraint):
    def __init__(self, op, axis, condition):
        self.op = op
        self.axis = axis
        self.condition = condition
    
    def check(self, context):
        shape = context.get_shape(self.op)
        if shape and self.axis < len(shape):
            if not self.condition(shape[self.axis]):
                return ConstraintViolation(
                    f"Shape constraint violated: {shape[self.axis]} does not satisfy condition"
                )
        return None

# Apply a constraint to a model
constraint = ShapeConstraint(model.get_node("input"), axis=1, condition=lambda x: x % 2 == 0)
model.add_constraint(constraint)

# Check constraints
violations = model.check_constraints()
for violation in violations:
    print(violation.message)
```

## Utility Functions

The framework provides various utility functions:

```python
from mithril.framework.utils import (
    get_unique_name,
    is_tensor_type,
    shape_to_tuple,
    broadcast_shapes,
    promote_types
)

# Generate a unique name
name = get_unique_name("layer")

# Check if a type is a tensor type
is_tensor = is_tensor_type("float32")

# Convert shape to tuple
shape_tuple = shape_to_tuple((None, 10))

# Broadcast shapes
result_shape = broadcast_shapes((3, 1), (1, 4))  # (3, 4)

# Promote types
result_type = promote_types("float32", "int64")  # "float64"
```

## Common Patterns

### Model Composition

Compose multiple models:

```python
from mithril.models import LogicalModel

# Create sub-models
model1 = LogicalModel(name="encoder")
model2 = LogicalModel(name="decoder")

# Compose models
combined_model = LogicalModel(name="autoencoder")
with combined_model:
    x = Input(shape=(None, 784), name="input")
    
    # Use model1 as a sub-model
    encoded = model1(x)
    
    # Use model2 as a sub-model
    decoded = model2(encoded)
    
    Output(decoded, name="output")
```

### Dynamic Execution

Create models with dynamic behavior:

```python
from mithril.framework.logical.operators import Conditional

model = LogicalModel()
with model:
    x = Input(shape=(None,), name="input")
    condition = reduce_any(greater(x, 0))
    
    # Define conditional execution
    y = Conditional(
        condition=condition,
        true_fn=lambda: x * 2,
        false_fn=lambda: x * -1
    )
    
    Output(y, name="output")
```

## Examples

See the [examples directory](https://github.com/example/mithril/tree/main/examples) for complete examples of using the Framework API.