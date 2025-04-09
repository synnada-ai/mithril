# Optimizers

This guide explains how to use and implement optimization algorithms in Mithril for training models.

## Overview

Optimizers are algorithms used to adjust model parameters during training to minimize the loss function. Mithril provides a flexible approach to optimization that works with any backend, allowing you to:

- Implement standard optimization algorithms (SGD, Adam, etc.)
- Customize optimization behavior
- Use backend-specific optimizers
- Apply optimization techniques like learning rate scheduling

## Basic Optimization

### Manual Parameter Updates

The simplest form of optimization is manual parameter updates using gradient descent:

```python
import mithril as ml
from mithril.backends import JaxBackend

# Create and compile a model
model = ml.Model()
# ... define your model ...
backend = JaxBackend()
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Training data
inputs = {"input": backend.randn(32, 784)}  # Batch of 32 examples
targets = backend.randn(32, 10)             # Target outputs

# Learning rate
learning_rate = 0.01

# Forward pass
outputs = compiled_model.evaluate(params, inputs)

# Compute loss gradients
loss_gradients = {"output": 2 * (outputs["output"] - targets)}  # MSE loss gradient

# Backward pass to get parameter gradients
_, gradients = compiled_model.evaluate(
    params,
    inputs,
    output_gradients=loss_gradients
)

# Update parameters using gradient descent
for name, grad in gradients.items():
    params[name] = params[name] - learning_rate * grad
```

## Using Optimizer Classes

Mithril includes common optimization algorithms like SGD, Adam, and RMSProp through its optimizer utilities:

```python
from mithril.utils.optimizers import SGD, Adam, RMSProp

# Create an Adam optimizer
optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Initialize optimizer state
optimizer_state = optimizer.init(params)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = compiled_model.evaluate(params, inputs)
    
    # Compute loss gradients
    loss_gradients = {"output": 2 * (outputs["output"] - targets)}
    
    # Backward pass
    _, gradients = compiled_model.evaluate(
        params,
        inputs,
        output_gradients=loss_gradients
    )
    
    # Update parameters and optimizer state
    params, optimizer_state = optimizer.update(params, gradients, optimizer_state)
```

## Available Optimizers

### SGD (Stochastic Gradient Descent)

```python
from mithril.utils.optimizers import SGD

# Basic SGD
sgd = SGD(learning_rate=0.01)

# SGD with momentum
sgd_momentum = SGD(learning_rate=0.01, momentum=0.9)

# SGD with Nesterov momentum
sgd_nesterov = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
```

### Adam

```python
from mithril.utils.optimizers import Adam

# Adam optimizer
adam = Adam(
    learning_rate=0.001,
    beta1=0.9,       # Exponential decay rate for first moment
    beta2=0.999,     # Exponential decay rate for second moment
    epsilon=1e-8,    # Small constant for numerical stability
    weight_decay=0   # L2 regularization factor
)
```

### RMSProp

```python
from mithril.utils.optimizers import RMSProp

# RMSProp optimizer
rmsprop = RMSProp(
    learning_rate=0.001,
    decay=0.9,       # Decay rate for moving average of squared gradients
    epsilon=1e-8,    # Small constant for numerical stability
    momentum=0.0     # Optional momentum term
)
```

### AdaGrad

```python
from mithril.utils.optimizers import AdaGrad

# AdaGrad optimizer
adagrad = AdaGrad(
    learning_rate=0.01,
    epsilon=1e-8     # Small constant for numerical stability
)
```

## Learning Rate Scheduling

Mithril supports learning rate scheduling through its scheduler utilities:

```python
from mithril.utils.optimizers import Adam
from mithril.utils.lr_schedulers import StepLR, CosineAnnealingLR, ReduceLROnPlateau

# Create optimizer
optimizer = Adam(learning_rate=0.001)

# Step learning rate scheduler (reduces LR by a factor every n steps)
scheduler = StepLR(
    optimizer,
    step_size=30,     # Reduce LR every 30 epochs
    gamma=0.1         # Reduce LR by a factor of 0.1
)

# Cosine annealing scheduler (cosine decay with optional restarts)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,        # Cycle length
    eta_min=1e-6      # Minimum learning rate
)

# Reduce LR on plateau scheduler (reduces LR when metric stops improving)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',       # Monitor for minimizing (e.g., loss)
    factor=0.1,       # Reduce LR by a factor of 0.1
    patience=10,      # Wait 10 epochs for improvement before reducing
    threshold=0.001   # Minimum significant improvement
)

# Training loop with scheduler
for epoch in range(num_epochs):
    # Training steps
    # ...
    
    # Update learning rate
    scheduler.step()  # For StepLR or CosineAnnealingLR
    
    # Or, for ReduceLROnPlateau, provide validation loss
    # scheduler.step(val_loss)
    
    # Get current learning rate
    current_lr = optimizer.get_learning_rate()
    print(f"Epoch {epoch}, Learning rate: {current_lr}")
```

## Custom Optimizers

You can implement custom optimizers by extending the `Optimizer` base class:

```python
from mithril.utils.optimizers import Optimizer

class CustomOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, custom_param=0.5):
        super().__init__(learning_rate)
        self.custom_param = custom_param
    
    def init(self, params):
        """Initialize optimizer state."""
        # Create optimizer state for each parameter
        state = {}
        for name, param in params.items():
            state[name] = {
                "momentum": self.backend.zeros_like(param)
            }
        return state
    
    def update(self, params, gradients, state):
        """Update parameters using gradients."""
        new_params = {}
        new_state = {}
        
        for name, param in params.items():
            if name in gradients:
                grad = gradients[name]
                
                # Update momentum
                momentum = state[name]["momentum"]
                momentum = self.custom_param * momentum + grad
                
                # Update parameter
                new_params[name] = param - self.learning_rate * momentum
                
                # Store updated state
                new_state[name] = {"momentum": momentum}
            else:
                # Parameter not affected by gradients
                new_params[name] = param
                new_state[name] = state[name]
        
        return new_params, new_state
```

## Backend-Specific Optimizers

For maximum performance, you can leverage backend-specific optimizers:

### JAX Optimizers (Optax)

```python
import mithril as ml
from mithril.backends import JaxBackend
import optax

# Create a JAX backend
backend = JaxBackend()

# Create and compile a model
model = ml.Model()
# ... define your model ...
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Create an Optax optimizer
optax_optimizer = optax.adam(learning_rate=0.001)
opt_state = optax_optimizer.init(params)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = compiled_model.evaluate(params, inputs)
    
    # Compute loss gradients
    loss_gradients = {"output": 2 * (outputs["output"] - targets)}
    
    # Backward pass
    _, gradients = compiled_model.evaluate(
        params,
        inputs,
        output_gradients=loss_gradients
    )
    
    # Update parameters using Optax
    updates, opt_state = optax_optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)
```

### PyTorch Optimizers

```python
import mithril as ml
from mithril.backends import TorchBackend
import torch.optim as optim

# Create a PyTorch backend
backend = TorchBackend()

# Create and compile a model
model = ml.Model()
# ... define your model ...
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Create parameter groups for PyTorch optimizer
param_groups = [{"params": [param for param in params.values()]}]

# Create a PyTorch optimizer
torch_optimizer = optim.Adam(param_groups, lr=0.001)

# Training loop
for epoch in range(num_epochs):
    # Zero gradients
    torch_optimizer.zero_grad()
    
    # Forward pass
    outputs = compiled_model.evaluate(params, inputs)
    
    # Compute loss gradients
    loss_gradients = {"output": 2 * (outputs["output"] - targets)}
    
    # Backward pass
    _, gradients = compiled_model.evaluate(
        params,
        inputs,
        output_gradients=loss_gradients
    )
    
    # Set gradients in PyTorch parameters
    for name, grad in gradients.items():
        params[name].grad = backend.to_torch(grad)
    
    # Update parameters
    torch_optimizer.step()
    
    # Update Mithril parameters if needed
    # params = {name: param for name, param in params.items()}
```

## Advanced Optimization Techniques

### Weight Decay (L2 Regularization)

```python
from mithril.utils.optimizers import Adam

# Adam with weight decay
optimizer = Adam(learning_rate=0.001, weight_decay=1e-4)

# Or apply weight decay manually
for name, param in params.items():
    if "weight" in name:  # Apply only to weights, not biases
        gradients[name] = gradients[name] + weight_decay * param
```

### Gradient Clipping

```python
from mithril.utils.optimizers import clip_gradients

# Clip gradients by value
gradients = clip_gradients(gradients, clip_value=1.0)

# Clip gradients by norm
gradients = clip_gradients(gradients, clip_norm=5.0)
```

### Learning Rate Warmup

```python
from mithril.utils.optimizers import Adam
from mithril.utils.lr_schedulers import LinearWarmupScheduler

# Create optimizer
optimizer = Adam(learning_rate=0.001)

# Add warmup to the learning rate
scheduler = LinearWarmupScheduler(
    optimizer,
    warmup_steps=1000,           # Number of warmup steps
    start_lr=1e-6,               # Starting learning rate
    end_lr=0.001,                # Target learning rate after warmup
    after_scheduler=None         # Optional scheduler to use after warmup
)

# Training loop with warmup
for step in range(total_steps):
    # Training step
    # ...
    
    # Update learning rate
    scheduler.step()
```

### Mixed Precision Training

```python
import mithril as ml
from mithril.backends import TorchBackend
from mithril.utils.optimizers import Adam
from mithril.utils.mixed_precision import MixedPrecisionTrainer

# Create a backend with mixed precision support
backend = TorchBackend(dtype="float16")

# Create a model
model = ml.Model()
# ... define your model ...
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Create an optimizer
optimizer = Adam(learning_rate=0.001)
optimizer_state = optimizer.init(params)

# Create mixed precision trainer
mp_trainer = MixedPrecisionTrainer(
    model=compiled_model,
    optimizer=optimizer,
    params=params,
    optimizer_state=optimizer_state,
    loss_scale=128.0,            # Initial loss scale
    dynamic_loss_scaling=True    # Automatically adjust loss scale
)

# Training loop with mixed precision
for epoch in range(num_epochs):
    # Forward pass in mixed precision
    outputs = mp_trainer.forward(inputs)
    
    # Compute loss gradients
    loss_gradients = {"output": 2 * (outputs["output"] - targets)}
    
    # Backward pass and parameter update
    params, optimizer_state, success = mp_trainer.backward_and_update(
        loss_gradients
    )
    
    if not success:
        print("Skipping update due to overflow")
```

## Multi-Optimizer Training

For models with different parameter groups requiring different optimization strategies:

```python
from mithril.utils.optimizers import Adam, SGD

# Create optimizers for different parameter groups
embedding_optimizer = Adam(learning_rate=0.0001)  # Lower LR for embeddings
main_optimizer = SGD(learning_rate=0.01, momentum=0.9)  # Higher LR with momentum

# Initialize optimizer states
embedding_state = embedding_optimizer.init({
    name: param for name, param in params.items() if "embedding" in name
})
main_state = main_optimizer.init({
    name: param for name, param in params.items() if "embedding" not in name
})

# Training loop with multiple optimizers
for epoch in range(num_epochs):
    # Forward pass
    outputs = compiled_model.evaluate(params, inputs)
    
    # Compute loss gradients
    loss_gradients = {"output": 2 * (outputs["output"] - targets)}
    
    # Backward pass
    _, gradients = compiled_model.evaluate(
        params,
        inputs,
        output_gradients=loss_gradients
    )
    
    # Split gradients by parameter group
    embedding_grads = {
        name: grad for name, grad in gradients.items() if "embedding" in name
    }
    main_grads = {
        name: grad for name, grad in gradients.items() if "embedding" not in name
    }
    
    # Update embeddings
    embedding_params, embedding_state = embedding_optimizer.update(
        {name: params[name] for name in embedding_grads},
        embedding_grads,
        embedding_state
    )
    
    # Update other parameters
    main_params, main_state = main_optimizer.update(
        {name: params[name] for name in main_grads},
        main_grads,
        main_state
    )
    
    # Combine parameter updates
    params.update(embedding_params)
    params.update(main_params)
```

## Best Practices

1. **Choose the right optimizer**: 
   - SGD tends to generalize better for some tasks
   - Adam converges faster and often works better for larger models
   - RMSProp is good for RNNs and non-stationary objectives

2. **Tune hyperparameters**:
   - Learning rate is the most important hyperparameter to tune
   - Start with recommended defaults for other hyperparameters
   - Use learning rate scheduling for better convergence

3. **Monitor training**:
   - Track loss, accuracy, and gradient norms
   - Watch for exploding or vanishing gradients
   - Adjust optimization strategy if learning plateaus

4. **Use optimization techniques**:
   - Learning rate warmup for large batch sizes
   - Gradient clipping for RNNs and transformer models
   - Weight decay for regularization

5. **Consider hardware constraints**:
   - Use mixed precision to reduce memory usage and increase throughput
   - Choose backend-specific optimizers for maximum performance
   - Optimize memory usage with gradient accumulation for large models