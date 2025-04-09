# Terminal Management

Terminal management in Mithril refers to how input and output connections (terminals) are defined, connected, and managed within models. This guide explains the terminal management system and how to use it effectively.

## Overview

Terminals in Mithril are the entry and exit points of data in models. They define:

- How data flows into a model (`Input` terminals)
- How data flows out of a model (`Output` terminals)
- How models connect to each other

Proper terminal management ensures clean model composition and enables features like automatic shape and type inference.

## Input Terminals

Input terminals define the entry points for data into your model:

```python
import mithril as mi
from mithril.models import LogicalModel

model = LogicalModel("my_model")
with model:
    # Create an input terminal with a specific shape and type
    x = mi.Input(shape=(None, 10), dtype="float32", name="input")
    
    # Input tensor can be used in operations
    y = x * 2
    
    # Create an output terminal
    mi.Output(y, name="output")
```

Input terminals support:

- Dynamic shapes using `None` for dimensions that can vary
- Explicit type specification with the `dtype` parameter
- Custom naming for clear identification

## Output Terminals

Output terminals define the exit points for data from your model:

```python
model = LogicalModel("my_model")
with model:
    x = mi.Input(shape=(None, 10), name="features")
    
    # Process input
    hidden = mi.relu(mi.dense(x, 20))
    prediction = mi.dense(hidden, 1)
    
    # Create multiple output terminals
    mi.Output(prediction, name="prediction")
    mi.Output(hidden, name="hidden_features")  # Expose intermediate results
```

Output terminals enable:

- Exposing multiple results from a model
- Accessing intermediate computations for debugging or composition
- Clearly defining a model's interface

## Named Terminals

Naming terminals is important for:

1. **Clear model interfaces**: Well-named terminals make models more understandable
2. **Model composition**: Named terminals facilitate connecting models
3. **Debugging**: Named terminals make it easier to inspect model behavior

```python
# Good terminal naming
model = LogicalModel("face_recognition")
with model:
    image = mi.Input(shape=(None, 224, 224, 3), name="image")
    is_training = mi.Input(shape=(), dtype="bool", name="is_training")
    
    # ... model operations ...
    
    features = mi.dense(x, 128)
    logits = mi.dense(features, 10)
    
    mi.Output(features, name="face_embedding")
    mi.Output(logits, name="class_scores")
```

## Advanced Terminal Features

### Terminal Metadata

You can attach metadata to terminals using the `metadata` parameter:

```python
# Add metadata to terminals
x = mi.Input(
    shape=(None, 784),
    name="mnist_image",
    metadata={
        "description": "Flattened MNIST image",
        "preprocessing": "Normalized to [0, 1]",
        "batch_format": "NCHW"
    }
)
```

This metadata can be accessed later:

```python
metadata = model.get_metadata("mnist_image")
print(metadata["description"])  # "Flattened MNIST image"
```

### Terminal Groups

For models with many terminals, you can organize them into groups:

```python
# Group related terminals
model = LogicalModel("multi_task_model")
with model:
    # Input group
    image = mi.Input(shape=(None, 224, 224, 3), name="image")
    text = mi.Input(shape=(None, 100), dtype="int32", name="text")
    model.group_terminals(["image", "text"], group="inputs")
    
    # ... model operations ...
    
    # Output group
    class_pred = mi.dense(features, 10, name="classification")
    bbox = mi.dense(features, 4, name="bounding_box")
    
    mi.Output(class_pred, name="class_pred")
    mi.Output(bbox, name="bbox_pred")
    model.group_terminals(["class_pred", "bbox_pred"], group="predictions")
```

You can then access terminals by group:

```python
input_terminals = model.get_terminal_group("inputs")
prediction_terminals = model.get_terminal_group("predictions")
```

## Model Composition with Terminals

Terminals are key to model composition in Mithril:

```python
# Create an encoder model
encoder = LogicalModel("encoder")
with encoder:
    x = mi.Input(shape=(None, 784), name="input")
    h1 = mi.relu(mi.dense(x, 256))
    h2 = mi.relu(mi.dense(h1, 128))
    mi.Output(h2, name="embedding")

# Create a decoder model
decoder = LogicalModel("decoder")
with decoder:
    z = mi.Input(shape=(None, 128), name="embedding")
    h1 = mi.relu(mi.dense(z, 256))
    reconstruction = mi.sigmoid(mi.dense(h1, 784))
    mi.Output(reconstruction, name="output")

# Compose models using terminals
autoencoder = LogicalModel("autoencoder")
with autoencoder:
    x = mi.Input(shape=(None, 784), name="input")
    
    # Connect encoder input
    z = encoder(x)
    
    # Connect decoder input to encoder output
    reconstruction = decoder(z)
    
    # Define autoencoder output
    mi.Output(reconstruction, name="reconstruction")
    mi.Output(z, name="latent")
```

## Accessing Terminal Values

Once a model is compiled and running, you can access terminal values:

```python
# Compile the model
from mithril.backends.with_autograd.jax_backend import JaxBackend
backend = JaxBackend()
compiled_model = model.compile(backend)

# Run the model
inputs = {"input": backend.tensor(np.random.randn(32, 784))}
outputs = compiled_model(inputs)

# Access output terminal values
reconstruction = outputs["reconstruction"]
latent = outputs["latent"]
```

## Terminal Validation

Mithril performs validation on terminals to ensure model correctness:

- **Shape compatibility**: Ensures connected terminals have compatible shapes
- **Type compatibility**: Ensures connected terminals have compatible types
- **Name uniqueness**: Ensures terminal names are unique within a model

Errors are reported clearly to help you fix issues:

```python
try:
    # This will raise an error due to incompatible shapes
    model = LogicalModel()
    with model:
        x = mi.Input(shape=(None, 10), name="input")
        w = mi.Parameter(shape=(20, 30), name="weights")
        y = mi.matmul(x, w)  # Shape mismatch: 10 vs 20
        mi.Output(y, name="output")
except Exception as e:
    print(f"Error: {e}")
```

## Best Practices

1. **Use Clear Terminal Names**: Choose descriptive names for terminals
2. **Be Consistent**: Use consistent naming conventions across models
3. **Expose What's Needed**: Only create output terminals for values needed outside the model
4. **Document Terminals**: Use metadata to document the purpose and format of terminals
5. **Validate Early**: Set shapes and types early to catch errors during development

## Examples

### Simple Model with Named Terminals

```python
model = LogicalModel("linear_model")
with model:
    x = mi.Input(shape=(None, 10), name="features")
    w = mi.Parameter(shape=(10, 1), name="weights")
    b = mi.Parameter(shape=(1,), name="bias")
    
    y = mi.matmul(x, w) + b
    mi.Output(y, name="prediction")
```

### Complex Model with Multiple Terminals

```python
model = LogicalModel("multi_input_output")
with model:
    # Multiple inputs
    image = mi.Input(shape=(None, 224, 224, 3), name="image")
    metadata = mi.Input(shape=(None, 10), name="metadata")
    is_training = mi.Input(shape=(), dtype="bool", name="is_training")
    
    # Process image
    x = mi.conv2d(image, filters=32, kernel_size=3)
    x = mi.max_pool2d(x, pool_size=2)
    x = mi.flatten(x)
    
    # Combine with metadata
    combined = mi.concatenate([x, metadata], axis=1)
    
    # Generate multiple outputs
    features = mi.dense(combined, 128)
    class_pred = mi.dense(features, 10, activation="softmax")
    regression = mi.dense(features, 1)
    
    # Define outputs
    mi.Output(features, name="features")
    mi.Output(class_pred, name="class_probabilities")
    mi.Output(regression, name="regression_value")
```

## Conclusion

Effective terminal management is crucial for building maintainable and composable models in Mithril. By defining clear input and output terminals, you create models that are easier to understand, debug, and combine. The terminal system is designed to be flexible enough for simple models while scaling to complex multi-input, multi-output scenarios.