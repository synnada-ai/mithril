# Training Models

This guide covers how to train models using Mithril, including setting up training loops, using optimizers, and implementing common training patterns.

## Basic Training Loop

Mithril provides a flexible way to train models by computing gradients and updating parameters. Here's a basic training loop:

```python
import mithril as ml
from mithril.models import Linear
import numpy as np

# Create a model
model = Linear(dimension=10)

# Create a backend
backend = ml.JaxBackend(dtype=ml.float32)

# Compile the model
compiled_model = ml.compile(model, backend, shapes={"input": [32, 20]})

# Initialize parameters
params = compiled_model.randomize_params()

# Training data
X = backend.array(np.random.randn(32, 20))
y = backend.array(np.random.randn(32, 10))

# Training hyperparameters
learning_rate = 0.01
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    inputs = {"input": X}
    outputs, gradients = compiled_model.evaluate(
        params, 
        inputs,
        output_gradients={"output": y - outputs["output"]}  # Simple L2 loss gradient
    )
    
    # Update parameters (simple SGD)
    for param_name, grad in gradients.items():
        params[param_name] = params[param_name] + learning_rate * grad
    
    # Calculate loss
    loss = backend.mean(backend.square(y - outputs["output"]))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
```

## Using Optimizers

Mithril doesn't include built-in optimizers, but you can easily use optimizers from the backend frameworks:

### With JAX

```python
import optax

# Create optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

# In the training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs, gradients = compiled_model.evaluate(params, inputs, output_gradients=output_grads)
    
    # Update with optax
    updates, opt_state = optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)
```

### With PyTorch

```python
import torch

# Convert parameters to PyTorch tensors with requires_grad=True
torch_params = {}
for name, param in params.items():
    tensor = backend.to_torch(param)
    tensor.requires_grad = True
    torch_params[name] = tensor

# Create optimizer
optimizer = torch.optim.Adam([tensor for tensor in torch_params.values()], lr=0.01)

# In the training loop
for epoch in range(num_epochs):
    # Zero gradients
    optimizer.zero_grad()
    
    # Manual forward pass (alternative to using compiled_model.evaluate)
    torch_inputs = {"input": backend.to_torch(X)}
    torch_outputs = {} 
    
    # Call the forward function directly
    # This assumes you have the generated code available
    from generated_code import forward
    torch_outputs["output"] = forward(torch_params, torch_inputs)["output"]
    
    # Compute loss
    loss = torch.mean((torch_outputs["output"] - backend.to_torch(y))**2)
    
    # Backward pass
    loss.backward()
    
    # Step optimizer
    optimizer.step()
    
    # Update original params if needed
    for name, torch_param in torch_params.items():
        params[name] = backend.from_torch(torch_param.detach())
```

## Custom Training Loops

### Example: Classification with Cross-Entropy Loss

```python
import mithril as ml
from mithril.models import Model, Linear, Relu, Softmax
import numpy as np

# Create a classification model
model = Model()
model |= Linear(dimension=128)
model += Relu()
model += Linear(dimension=10)
model += Softmax()

# Compile model
backend = ml.JaxBackend(dtype=ml.float32)
compiled_model = ml.compile(model, backend)

# Generate random data (32 samples, 20 features, 10 classes)
X = backend.array(np.random.randn(32, 20))
y_one_hot = backend.array(np.eye(10)[np.random.choice(10, 32)])

# Initialize parameters
params = compiled_model.randomize_params()

# Training hyperparameters
learning_rate = 0.01
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    inputs = {"input": X}
    outputs = compiled_model.evaluate(params, inputs)
    predictions = outputs["output"]
    
    # Compute loss (cross-entropy)
    loss = -backend.sum(y_one_hot * backend.log(predictions + 1e-8)) / 32
    
    # Compute gradients of loss with respect to outputs
    output_gradients = {"output": (predictions - y_one_hot) / 32}
    
    # Backward pass
    _, gradients = compiled_model.evaluate(params, inputs, output_gradients=output_gradients)
    
    # Update parameters (SGD)
    for param_name, grad in gradients.items():
        params[param_name] = params[param_name] - learning_rate * grad
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
```

## Batch Training

```python
import mithril as ml
from mithril.models import Linear
import numpy as np

# Create model
model = Linear(dimension=10)
backend = ml.JaxBackend(dtype=ml.float32)
compiled_model = ml.compile(model, backend)

# Generate larger dataset
X_full = backend.array(np.random.randn(1000, 20))
y_full = backend.array(np.random.randn(1000, 10))

# Initialize parameters
params = compiled_model.randomize_params()

# Training hyperparameters
learning_rate = 0.01
num_epochs = 50
batch_size = 32

# Training loop with batching
for epoch in range(num_epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_full))
    
    total_loss = 0
    num_batches = len(X_full) // batch_size
    
    for i in range(num_batches):
        # Get batch
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        X_batch = X_full[batch_indices]
        y_batch = y_full[batch_indices]
        
        # Forward pass
        inputs = {"input": X_batch}
        outputs = compiled_model.evaluate(params, inputs)
        predictions = outputs["output"]
        
        # Compute loss
        loss = backend.mean(backend.square(predictions - y_batch))
        total_loss += loss
        
        # Compute output gradients
        output_gradients = {"output": (predictions - y_batch) / batch_size}
        
        # Backward pass
        _, gradients = compiled_model.evaluate(params, inputs, output_gradients=output_gradients)
        
        # Update parameters
        for param_name, grad in gradients.items():
            params[param_name] = params[param_name] - learning_rate * grad
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}, Avg Loss: {avg_loss}")
```

## Parallel Training

For training on multiple devices, you can use Mithril's parallelization features:

```python
import mithril as ml
from mithril.models import Linear

# Create model
model = Linear(dimension=128)

# Create backend with device mesh for data parallelism
backend = ml.JaxBackend(device_mesh=(2,))  # 2 devices

# Compile the model
compiled_model = ml.compile(model, backend)

# Initialize parameters (replicated across devices)
params = compiled_model.randomize_params()

# Create sharded training data
batch_size = 64
X = backend.randn(batch_size, 256, device_mesh=(2,), tensor_split=(0,))  # Sharded across batch dimension
y = backend.randn(batch_size, 128, device_mesh=(2,), tensor_split=(0,))

# Training loop
for epoch in range(num_epochs):
    # Forward pass with sharded inputs
    inputs = {"input": X}
    
    # Compute output gradients (also sharded)
    outputs = compiled_model.evaluate(params, inputs)
    output_gradients = {"output": (outputs["output"] - y) / batch_size}
    
    # Backward pass (gradients will be automatically reduced across devices)
    _, gradients = compiled_model.evaluate(params, inputs, output_gradients=output_gradients)
    
    # Update parameters
    for param_name, grad in gradients.items():
        params[param_name] = params[param_name] - learning_rate * grad
```

## Best Practices

1. **Start simple**: Begin with basic training loops and add complexity as needed
2. **Use backend-native optimizers**: Leverage optimizers from JAX, PyTorch, etc.
3. **Batch your data**: Use batching for memory efficiency and better generalization
4. **Monitor training**: Track loss, accuracy, and other metrics
5. **Checkpoint models**: Save trained parameters periodically
6. **Use validation data**: Evaluate on held-out data to detect overfitting
7. **Tune hyperparameters**: Experiment with learning rates, batch sizes, etc.
8. **Scale gradually**: Test on small data before scaling to large datasets and complex models