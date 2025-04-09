# Model Serialization

Mithril provides robust mechanisms for saving and loading models. This document explains how to serialize models, save parameters, and handle different serialization formats.

## Overview

Model serialization in Mithril allows you to:

- Save and load model definitions
- Store and restore model parameters
- Share models across different environments
- Deploy models in production settings

Mithril supports both logical and physical model serialization, with different options for each.

## Serializing Logical Models

Logical models can be serialized to a JSON-compatible dictionary format, which captures the model structure without tying it to any specific backend.

### Saving Logical Models

```python
import mithril as mi
from mithril.models import LogicalModel
import json

# Create a simple model
model = LogicalModel("my_model")
with model:
    x = mi.Input(shape=(None, 10), name="input")
    w = mi.Parameter(shape=(10, 5), name="weights")
    b = mi.Parameter(shape=(5,), name="bias")
    y = mi.matmul(x, w) + b
    mi.Output(y, name="output")

# Convert model to dictionary
model_dict = model.to_dict()

# Save to a file
with open("model.json", "w") as f:
    json.dump(model_dict, f, indent=2)
```

The resulting dictionary contains the complete model structure, including operators, parameters, and connections.

### Loading Logical Models

```python
import json
from mithril.utils.dict_conversions import from_dict

# Load from a file
with open("model.json", "r") as f:
    model_dict = json.load(f)

# Recreate the model
loaded_model = from_dict(model_dict)
```

## Serializing Parameter Values

Model parameters can be serialized separately from the model structure, allowing for easy weight sharing and transfer learning.

### Saving Parameter Values

```python
import numpy as np

# First, create and compile a model
from mithril.backends.with_autograd.jax_backend import JaxBackend
backend = JaxBackend()
compiled_model = model.compile(backend)

# Initialize parameters with random values
compiled_model.set_values({
    "weights": backend.tensor(np.random.randn(10, 5)),
    "bias": backend.tensor(np.random.randn(5))
})

# Extract parameter values as NumPy arrays
params = {}
for name in compiled_model.get_parameter_names():
    params[name] = backend.to_numpy(compiled_model.get_value(name))

# Save parameters to a file
np.savez("model_params.npz", **params)
```

### Loading Parameter Values

```python
# Load parameters from file
loaded_params = np.load("model_params.npz")

# Set parameter values in a compiled model
for name, value in loaded_params.items():
    if name in compiled_model.get_parameter_names():
        compiled_model.set_value(name, backend.tensor(value))
```

## Serializing Physical Models

Physical models (compiled models) can also be serialized, but the approach varies by backend.

### Using Mithril's Built-in Serialization

```python
from mithril.framework.physical.model import PhysicalModel

# Save a compiled model
compiled_model.save("compiled_model.mithril")

# Load a compiled model
loaded_model = PhysicalModel.load("compiled_model.mithril", backend)
```

This approach saves both the model structure and parameter values.

### Backend-Specific Serialization

Some backends offer optimized serialization methods:

#### JAX Backend

```python
# For JAX, we can use pickle or specific JAX serialization
import pickle

# Save the model with pickle
with open("jax_model.pkl", "wb") as f:
    pickle.dump(compiled_model, f)

# Load the model
with open("jax_model.pkl", "rb") as f:
    loaded_jax_model = pickle.load(f)
```

#### PyTorch Backend

```python
# For PyTorch, we can extract and save state_dicts
from mithril.backends.with_autograd.torch_backend import TorchBackend
torch_backend = TorchBackend()

# Compile model with PyTorch backend
torch_model = model.compile(torch_backend)

# Extract state dict
state_dict = torch_model.get_state_dict()

# Save state dict
import torch
torch.save(state_dict, "torch_model.pt")

# Load state dict into a new model
new_torch_model = model.compile(torch_backend)
new_torch_model.load_state_dict(torch.load("torch_model.pt"))
```

## Model Versioning

Mithril includes model versioning to handle compatibility between different releases:

```python
# Check model version
version = model_dict.get("version", "unknown")
print(f"Model version: {version}")

# Save with explicit version
model_dict = model.to_dict(version="1.2.0")
```

Version information helps ensure that models can be correctly loaded even when there are changes to the serialization format.

## Parameter Initialization

When loading a model, you might want to initialize parameters in specific ways:

```python
# Define custom initializer functions
def glorot_initializer(shape, dtype=None):
    """Initialize weights with Glorot/Xavier initialization."""
    scale = np.sqrt(2.0 / (shape[0] + shape[1]))
    return np.random.randn(*shape) * scale

def zero_initializer(shape, dtype=None):
    """Initialize weights with zeros."""
    return np.zeros(shape)

# Initialize parameters
compiled_model.set_values({
    "weights": backend.tensor(glorot_initializer((10, 5))),
    "bias": backend.tensor(zero_initializer((5,)))
})
```

## Handling Model Compatibility

When loading models saved with previous versions of Mithril, compatibility utilities can help:

```python
from mithril.utils.dict_conversions import update_model_dict_version

# Update model dictionary to current version
updated_model_dict = update_model_dict_version(old_model_dict)

# Load the updated model
compatible_model = from_dict(updated_model_dict)
```

## Optimized Binary Format

For large models, Mithril provides an optimized binary format that is more compact and faster to load:

```python
from mithril.utils.serialization import save_binary, load_binary

# Save to binary format
save_binary(compiled_model, "model.mbin")

# Load from binary format
loaded_model = load_binary("model.mbin", backend)
```

## Partial Model Serialization

You can serialize parts of a model, which is useful for transfer learning:

```python
# Extract part of the model
feature_extractor = model.extract_submodel(["input", "hidden1", "features"])

# Serialize just the feature extractor
feature_extractor_dict = feature_extractor.to_dict()

# Save to file
with open("feature_extractor.json", "w") as f:
    json.dump(feature_extractor_dict, f, indent=2)
```

## Handling Custom Operators

If your model contains custom operators, they need to be registered before loading:

```python
from mithril.framework.logical import Operator, register_op

# Define and register custom operator
@register_op
class MyCustomOp(Operator):
    def __init__(self, x, factor=1.0, name=None):
        super().__init__(inputs=[x], name=name)
        self.factor = factor
    
    def infer_shape(self, input_shape):
        return input_shape
    
    def infer_type(self, input_type):
        return input_type

# Now loading a model with this custom operator will work
loaded_model = from_dict(model_dict)
```

## Cross-Platform Considerations

When sharing models across platforms, consider these best practices:

1. **Use logical models** for cross-platform compatibility
2. **Save parameters separately** in a standard format like NumPy's `.npz`
3. **Document required backends and dependencies**
4. **Include version information** for both Mithril and the model

## Example: Complete Serialization Workflow

```python
import mithril as mi
from mithril.models import LogicalModel
import numpy as np
import json
from mithril.backends.with_autograd.jax_backend import JaxBackend

# 1. Create a model
model = LogicalModel("simple_mlp")
with model:
    x = mi.Input(shape=(None, 10), name="input")
    h = mi.relu(mi.dense(x, 50, name="hidden"))
    y = mi.dense(h, 1, name="output_layer")
    mi.Output(y, name="output")

# 2. Save the logical model structure
model_dict = model.to_dict()
with open("model_structure.json", "w") as f:
    json.dump(model_dict, f, indent=2)

# 3. Compile the model
backend = JaxBackend()
compiled_model = model.compile(backend)

# 4. Initialize parameters
def init_parameters(compiled_model, backend):
    param_names = compiled_model.get_parameter_names()
    params = {}
    for name in param_names:
        shape = compiled_model.get_shape(name)
        scale = 0.1
        params[name] = backend.tensor(np.random.randn(*shape) * scale)
    compiled_model.set_values(params)

init_parameters(compiled_model, backend)

# 5. Save parameter values
def save_parameters(compiled_model, backend, filename):
    params = {}
    for name in compiled_model.get_parameter_names():
        params[name] = backend.to_numpy(compiled_model.get_value(name))
    np.savez(filename, **params)

save_parameters(compiled_model, backend, "model_params.npz")

# 6. Later, to rebuild the model
def rebuild_model(structure_file, params_file, backend):
    # Load structure
    with open(structure_file, "r") as f:
        model_dict = json.load(f)
    
    # Recreate logical model
    from mithril.utils.dict_conversions import from_dict
    model = from_dict(model_dict)
    
    # Compile with the specified backend
    compiled_model = model.compile(backend)
    
    # Load parameters
    params = np.load(params_file)
    for name, value in params.items():
        if name in compiled_model.get_parameter_names():
            compiled_model.set_value(name, backend.tensor(value))
    
    return compiled_model

# Rebuild model
rebuilt_model = rebuild_model("model_structure.json", "model_params.npz", backend)

# 7. Test the rebuilt model
test_input = np.random.randn(5, 10).astype(np.float32)
original_output = compiled_model({"input": backend.tensor(test_input)})["output"]
rebuilt_output = rebuilt_model({"input": backend.tensor(test_input)})["output"]

# Should be nearly identical
print("Original output shape:", backend.shape(original_output))
print("Rebuilt output shape:", backend.shape(rebuilt_output))
print("Max difference:", np.max(np.abs(
    backend.to_numpy(original_output) - backend.to_numpy(rebuilt_output)
)))
```

## Best Practices

1. **Version Your Models**: Always include version information with serialized models
2. **Test Reloading**: Verify that loaded models produce the same outputs as original models
3. **Separate Structure and Parameters**: For large models, keep structure and parameters separate
4. **Document Dependencies**: Note which backends and versions are compatible with your model
5. **Use Standard Formats**: Prefer widely-supported formats like JSON and NumPy for maximum compatibility

## Conclusion

Mithril's serialization features provide flexible options for saving and loading models across different environments. The separation of logical structure and parameter values enables efficient storage and transfer learning scenarios. By following the patterns described in this guide, you can effectively manage your models throughout their lifecycle.