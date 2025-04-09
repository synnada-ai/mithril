# Quick Start: Write Once, Run Anywhere

This guide provides a hands-on introduction to Mithril, showing you how to define a model once and run it across multiple backends. You'll see firsthand how Mithril liberates your ML workflow from framework-specific constraints.

## The Mithril Workflow

The typical Mithril workflow consists of three simple steps:

1. **Define** a logical model architecture
2. **Compile** it for your target backend
3. **Execute** on any hardware

Let's see how this works in practice.

## Installation

First, install Mithril using pip:

```bash
pip install mithril
```

For development options, you can install from source:

```bash
git clone https://github.com/example/mithril.git
cd mithril
pip install -e .
```

## Your First Mithril Model

Let's create a simple two-layer neural network:

```python
import mithril as ml
from mithril.models import Model, Linear, Relu

# Define a logical model
model = Model()
model |= Linear(dimension=64)  # First layer connected to input
model += Relu()                # Activation function chained to previous
model += Linear(dimension=10)  # Output layer
```

That's it! This model definition is completely backend-agnostic. It describes the structure and connections of your model without any details about how it will be executed.

## Compile and Run with Different Backends

Now let's compile this same model for different backends and see how it runs:

### PyTorch Backend

```python
# Create a PyTorch backend
torch_backend = ml.TorchBackend(dtype=ml.float32)

# Compile for PyTorch
torch_model = ml.compile(
    model=model,
    backend=torch_backend,
    shapes={"input": [32, 20]},  # Batch size 32, input dimension 20
)

# Initialize parameters
params = torch_model.randomize_params()

# Create input data
inputs = {"input": torch_backend.randn(32, 20)}

# Run the model
outputs = torch_model.evaluate(params, inputs)
print(f"PyTorch output shape: {outputs['output'].shape}")  # [32, 10]
```

### JAX Backend

```python
# Create a JAX backend
jax_backend = ml.JaxBackend(dtype=ml.float32)

# Compile the SAME model for JAX
jax_model = ml.compile(
    model=model,
    backend=jax_backend,
    shapes={"input": [32, 20]},
    jit=True,  # Enable JIT compilation
)

# Convert parameters from PyTorch to JAX
jax_params = {
    k: jax_backend.array(torch_backend.to_numpy(v)) 
    for k, v in params.items()
}

# Create input data
jax_inputs = {"input": jax_backend.randn(32, 20)}

# Run the model
jax_outputs = jax_model.evaluate(jax_params, jax_inputs)
print(f"JAX output shape: {jax_outputs['output'].shape}")  # [32, 10]
```

### NumPy Backend

```python
# Create a NumPy backend (CPU only)
numpy_backend = ml.NumpyBackend(dtype=ml.float32)

# Compile for NumPy
numpy_model = ml.compile(
    model=model,
    backend=numpy_backend,
    shapes={"input": [32, 20]},
)

# Convert parameters to NumPy
numpy_params = {
    k: numpy_backend.array(torch_backend.to_numpy(v)) 
    for k, v in params.items()
}

# Create input data
numpy_inputs = {"input": numpy_backend.randn(32, 20)}

# Run the model
numpy_outputs = numpy_model.evaluate(numpy_params, numpy_inputs)
print(f"NumPy output shape: {numpy_outputs['output'].shape}")  # [32, 10]
```

## Computing Gradients

Mithril makes automatic differentiation easy with backends that support it:

```python
# Create input data
input_data = torch_backend.randn(32, 20)
inputs = {"input": input_data}

# Forward pass
outputs = torch_model.evaluate(params, inputs)
predictions = outputs["output"]

# Create target data
targets = torch_backend.randn(32, 10)  # Random targets for demonstration

# Compute loss (simple MSE loss)
diff = predictions - targets
loss = torch_backend.mean(diff * diff)

# Compute gradients of loss with respect to outputs
output_gradients = {"output": 2 * diff / diff.shape[0]}

# Compute gradients with respect to parameters
_, gradients = torch_model.evaluate(
    params, 
    inputs, 
    output_gradients=output_gradients
)

# Update parameters (simple SGD)
learning_rate = 0.01
for param_name, grad in gradients.items():
    params[param_name] = params[param_name] - learning_rate * grad
```

## Building a More Complex Model

Let's build something slightly more complex—a convolutional neural network for image classification:

```python
from mithril.models import Conv2d, MaxPool2d, Flatten, BatchNorm2d

# Create a CNN for image classification
cnn = Model()

# Convolutional block 1
cnn |= Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)(
    input="input", output="conv1"
)
cnn += BatchNorm2d(num_features=16)(input="conv1", output="bn1")
cnn += Relu()(input="bn1", output="relu1")
cnn += MaxPool2d(kernel_size=2)(input="relu1", output="pool1")

# Convolutional block 2
cnn += Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)(
    input="pool1", output="conv2"
)
cnn += BatchNorm2d(num_features=32)(input="conv2", output="bn2")
cnn += Relu()(input="bn2", output="relu2")
cnn += MaxPool2d(kernel_size=2)(input="relu2", output="pool2")

# Fully connected layers
cnn += Flatten()(input="pool2", output="flat")
cnn += Linear(dimension=128)(input="flat", output="fc1")
cnn += Relu()(input="fc1", output="relu3")
cnn += Linear(dimension=10)(input="relu3", output="output")

# Compile for PyTorch
torch_cnn = ml.compile(
    model=cnn,
    backend=torch_backend,
    shapes={"input": [32, 3, 32, 32]},  # CIFAR-10 sized images
)

# Initialize parameters
cnn_params = torch_cnn.randomize_params()

# Run the model
cnn_inputs = {"input": torch_backend.randn(32, 3, 32, 32)}
cnn_outputs = torch_cnn.evaluate(cnn_params, cnn_inputs)
print(f"CNN output shape: {cnn_outputs['output'].shape}")  # [32, 10]
```

## Training Loop

Here's a complete training loop example:

```python
# Create a simple dataset for demonstration
def create_dummy_data(backend, num_samples=1000, input_dim=20, output_dim=10):
    X = backend.randn(num_samples, input_dim)
    W = backend.randn(input_dim, output_dim)
    b = backend.randn(output_dim)
    Y = backend.matmul(X, W) + b + 0.1 * backend.randn(num_samples, output_dim)
    return X, Y

# Create data
X, Y = create_dummy_data(torch_backend)

# Training parameters
batch_size = 32
num_batches = len(X) // batch_size
epochs = 5
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    total_loss = 0
    
    for i in range(num_batches):
        # Get batch
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        
        # Forward pass
        inputs = {"input": X_batch}
        outputs = torch_model.evaluate(params, inputs)
        predictions = outputs["output"]
        
        # Compute loss
        diff = predictions - Y_batch
        loss = torch_backend.mean(diff * diff)
        total_loss += loss
        
        # Compute gradients
        output_gradients = {"output": 2 * diff / batch_size}
        _, gradients = torch_model.evaluate(params, inputs, output_gradients)
        
        # Update parameters (SGD)
        for param_name, grad in gradients.items():
            params[param_name] = params[param_name] - learning_rate * grad
    
    # Print progress
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
```

## Multi-Backend Workflow

The power of Mithril's approach becomes clear when you need to deploy the same model across different environments:

```python
# Research phase: Use PyTorch for rapid iteration
research_model = ml.compile(model, ml.TorchBackend())

# Scaling phase: Use JAX for distributed training on TPUs
scaling_model = ml.compile(
    model, 
    ml.JaxBackend(device="tpu", device_mesh=(8,))  # 8-way data parallelism
)

# Production phase: Use GGML for efficient deployment
deploy_model = ml.compile(
    model,
    ml.GGMLBackend(quantize="int8"),  # 8-bit quantization
    static_inference=True  # Optimized for inference
)
```

This eliminates the need for painful rewrites when moving between environments.

## Next Steps

Now that you've experienced the basics of Mithril, you can:

- Learn more about [Logical Models](../user-guide/logical-models.md)
- Explore different [Backends](../user-guide/backends.md)
- Dive into [Model Composition](../user-guide/model-composition.md)
- Check out complete [Examples](../examples/basic.md)

Congratulations! You've taken your first steps toward framework-agnostic machine learning development.