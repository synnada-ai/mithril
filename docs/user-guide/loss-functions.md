# Loss Functions

This guide explains how to use and implement loss functions in Mithril for training models.

## Overview

Loss functions measure the difference between model predictions and target values, guiding the optimization process during training. In Mithril:

- Loss functions are computed outside the model
- Gradients of the loss are passed back for parameter updates
- Loss functions can be customized for any backend
- Common losses are provided through utility functions

## Basic Usage

### Computing Loss and Gradients

The basic pattern for using loss functions in Mithril involves:

1. Forward pass to get model outputs
2. Compute loss value for monitoring
3. Compute loss gradients for backpropagation
4. Backward pass with loss gradients to get parameter gradients

```python
import mithril as ml
from mithril.backends import JaxBackend
from mithril.utils.losses import mse_loss, mse_loss_gradient

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

# Forward pass
outputs = compiled_model.evaluate(params, inputs)
predictions = outputs["output"]  # Shape [32, 10]

# Compute loss value (for monitoring)
loss_value = mse_loss(predictions, targets)
print(f"Loss: {loss_value}")

# Compute loss gradients with respect to predictions
loss_gradients = mse_loss_gradient(predictions, targets)  # Shape [32, 10]

# Backward pass to get parameter gradients
_, gradients = compiled_model.evaluate(
    params,
    inputs,
    output_gradients={"output": loss_gradients}
)

# Update parameters (e.g., using gradient descent)
learning_rate = 0.01
for name, grad in gradients.items():
    params[name] = params[name] - learning_rate * grad
```

## Common Loss Functions

Mithril provides implementations of common loss functions in the `mithril.utils.losses` module.

### Mean Squared Error (MSE)

Used for regression tasks:

```python
from mithril.utils.losses import mse_loss, mse_loss_gradient

# Compute MSE loss
loss_value = mse_loss(predictions, targets)

# Compute gradients for backpropagation
loss_gradients = mse_loss_gradient(predictions, targets)
```

Implementation:

```python
def mse_loss(predictions, targets):
    """Mean squared error loss."""
    return backend.mean(backend.square(predictions - targets))

def mse_loss_gradient(predictions, targets):
    """Gradient of MSE loss with respect to predictions."""
    # Factor of 2/n from the mean of squared differences
    n = backend.size(predictions)
    return 2 * (predictions - targets) / n
```

### Cross Entropy Loss

Used for classification tasks:

```python
from mithril.utils.losses import cross_entropy_loss, cross_entropy_loss_gradient

# For multi-class classification with softmax outputs
loss_value = cross_entropy_loss(predictions, targets)
loss_gradients = cross_entropy_loss_gradient(predictions, targets)

# With logits (pre-softmax)
from mithril.utils.losses import cross_entropy_with_logits, cross_entropy_with_logits_gradient
loss_value = cross_entropy_with_logits(logits, targets)
loss_gradients = cross_entropy_with_logits_gradient(logits, targets)
```

### Binary Cross Entropy

Used for binary classification tasks:

```python
from mithril.utils.losses import binary_cross_entropy, binary_cross_entropy_gradient

# For binary classification with sigmoid outputs [0,1]
loss_value = binary_cross_entropy(predictions, targets)
loss_gradients = binary_cross_entropy_gradient(predictions, targets)

# With logits (pre-sigmoid)
from mithril.utils.losses import binary_cross_entropy_with_logits, binary_cross_entropy_with_logits_gradient
loss_value = binary_cross_entropy_with_logits(logits, targets)
loss_gradients = binary_cross_entropy_with_logits_gradient(logits, targets)
```

### Huber Loss

Used for regression tasks, less sensitive to outliers than MSE:

```python
from mithril.utils.losses import huber_loss, huber_loss_gradient

# Compute Huber loss (defaults to delta=1.0)
loss_value = huber_loss(predictions, targets, delta=1.0)
loss_gradients = huber_loss_gradient(predictions, targets, delta=1.0)
```

### Hinge Loss

Used for classification tasks, particularly with SVMs:

```python
from mithril.utils.losses import hinge_loss, hinge_loss_gradient

# Compute hinge loss
loss_value = hinge_loss(predictions, targets)
loss_gradients = hinge_loss_gradient(predictions, targets)
```

### Kullback-Leibler Divergence

Used to measure difference between probability distributions:

```python
from mithril.utils.losses import kl_divergence, kl_divergence_gradient

# Compute KL divergence loss
loss_value = kl_divergence(predictions, targets)
loss_gradients = kl_divergence_gradient(predictions, targets)
```

## Custom Loss Functions

You can implement custom loss functions tailored to your specific requirements:

```python
def custom_loss(predictions, targets, alpha=1.0):
    """Custom loss function combining MSE and L1 loss."""
    mse = backend.mean(backend.square(predictions - targets))
    l1 = backend.mean(backend.abs(predictions - targets))
    return mse + alpha * l1

def custom_loss_gradient(predictions, targets, alpha=1.0):
    """Gradient of custom loss with respect to predictions."""
    n = backend.size(predictions)
    # MSE gradient component
    mse_grad = 2 * (predictions - targets) / n
    # L1 gradient component
    l1_grad = backend.sign(predictions - targets) / n
    return mse_grad + alpha * l1_grad

# Usage
loss_value = custom_loss(predictions, targets, alpha=0.5)
loss_gradients = custom_loss_gradient(predictions, targets, alpha=0.5)
```

## Multi-Output Loss Functions

For models with multiple outputs, you can compute losses for each output and combine them:

```python
# Model with two outputs: classification and regression
outputs = compiled_model.evaluate(params, inputs)
class_predictions = outputs["class_output"]
reg_predictions = outputs["reg_output"]

# Compute separate losses
from mithril.utils.losses import cross_entropy_loss, mse_loss
class_loss = cross_entropy_loss(class_predictions, class_targets)
reg_loss = mse_loss(reg_predictions, reg_targets)

# Combine losses with weighting
total_loss = class_loss + 0.5 * reg_loss

# Compute gradients for each output
class_gradients = cross_entropy_loss_gradient(class_predictions, class_targets)
reg_gradients = mse_loss_gradient(reg_predictions, reg_targets)

# Backward pass with gradients for both outputs
_, gradients = compiled_model.evaluate(
    params,
    inputs,
    output_gradients={
        "class_output": class_gradients,
        "reg_output": 0.5 * reg_gradients  # Apply the same weight as in the loss
    }
)
```

## Loss Functions with Regularization

### L1 Regularization

```python
def loss_with_l1_reg(predictions, targets, params, lambda_reg=0.01):
    """Loss function with L1 regularization."""
    # Base loss (e.g., MSE)
    base_loss = backend.mean(backend.square(predictions - targets))
    
    # L1 regularization term
    l1_reg = 0
    for param in params.values():
        l1_reg += backend.sum(backend.abs(param))
    
    return base_loss + lambda_reg * l1_reg

def loss_gradient_with_l1_reg(predictions, targets, lambda_reg=0.01):
    """Gradient of base loss (regularization applied separately)."""
    n = backend.size(predictions)
    return 2 * (predictions - targets) / n

# When updating parameters, add regularization gradient
for name, grad in gradients.items():
    # L1 regularization gradient
    l1_grad = lambda_reg * backend.sign(params[name])
    # Apply combined gradient
    params[name] = params[name] - learning_rate * (grad + l1_grad)
```

### L2 Regularization (Weight Decay)

```python
def loss_with_l2_reg(predictions, targets, params, lambda_reg=0.01):
    """Loss function with L2 regularization."""
    # Base loss (e.g., MSE)
    base_loss = backend.mean(backend.square(predictions - targets))
    
    # L2 regularization term
    l2_reg = 0
    for param in params.values():
        l2_reg += backend.sum(backend.square(param))
    
    return base_loss + 0.5 * lambda_reg * l2_reg

def loss_gradient_with_l2_reg(predictions, targets, lambda_reg=0.01):
    """Gradient of base loss (regularization applied separately)."""
    n = backend.size(predictions)
    return 2 * (predictions - targets) / n

# When updating parameters, add regularization gradient
for name, grad in gradients.items():
    # L2 regularization gradient
    l2_grad = lambda_reg * params[name]
    # Apply combined gradient
    params[name] = params[name] - learning_rate * (grad + l2_grad)
```

## Advanced Loss Functions

### Focal Loss

Used for imbalanced classification tasks:

```python
from mithril.utils.losses import focal_loss, focal_loss_gradient

# Compute focal loss
loss_value = focal_loss(predictions, targets, gamma=2.0, alpha=0.25)
loss_gradients = focal_loss_gradient(predictions, targets, gamma=2.0, alpha=0.25)
```

### Triplet Loss

Used for metric learning tasks:

```python
from mithril.utils.losses import triplet_loss, triplet_loss_gradient

# Compute triplet loss
anchors = outputs["anchor"]
positives = outputs["positive"]
negatives = outputs["negative"]

loss_value = triplet_loss(anchors, positives, negatives, margin=1.0)
gradients = triplet_loss_gradient(anchors, positives, negatives, margin=1.0)

# Backward pass with triplet gradients
_, param_gradients = compiled_model.evaluate(
    params,
    inputs,
    output_gradients={
        "anchor": gradients["anchor"],
        "positive": gradients["positive"],
        "negative": gradients["negative"]
    }
)
```

### Contrastive Loss

Used for similarity learning:

```python
from mithril.utils.losses import contrastive_loss, contrastive_loss_gradient

# Compute contrastive loss
embeddings1 = outputs["embedding1"]
embeddings2 = outputs["embedding2"]
# 1 if same class, 0 if different
labels = inputs["labels"]

loss_value = contrastive_loss(embeddings1, embeddings2, labels, margin=1.0)
gradients = contrastive_loss_gradient(embeddings1, embeddings2, labels, margin=1.0)

# Backward pass with contrastive gradients
_, param_gradients = compiled_model.evaluate(
    params,
    inputs,
    output_gradients={
        "embedding1": gradients["embedding1"],
        "embedding2": gradients["embedding2"]
    }
)
```

## Backend-Specific Loss Functions

For optimal performance, you can use backend-native loss functions:

### JAX Backend

```python
import mithril as ml
from mithril.backends import JaxBackend
import jax
import jax.numpy as jnp

# Create a JAX backend
backend = JaxBackend()

# Define a JAX-native loss function
def jax_cross_entropy(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))

# Define the gradient function using JAX's grad
jax_cross_entropy_grad = jax.grad(
    lambda logits, targets: jax_cross_entropy(logits, targets),
    argnums=0
)

# Use in training loop
outputs = compiled_model.evaluate(params, inputs)
loss_value = jax_cross_entropy(outputs["output"], targets)
loss_gradients = jax_cross_entropy_grad(outputs["output"], targets)

_, gradients = compiled_model.evaluate(
    params,
    inputs,
    output_gradients={"output": backend.from_jax(loss_gradients)}
)
```

### PyTorch Backend

```python
import mithril as ml
from mithril.backends import TorchBackend
import torch.nn.functional as F

# Create a PyTorch backend
backend = TorchBackend()

# Use PyTorch's built-in loss functions
def torch_cross_entropy(predictions, targets):
    return F.cross_entropy(backend.to_torch(predictions), backend.to_torch(targets))

# For gradients, use PyTorch's autograd
def torch_cross_entropy_gradient(predictions, targets):
    predictions_torch = backend.to_torch(predictions)
    predictions_torch.requires_grad_(True)
    targets_torch = backend.to_torch(targets)
    
    loss = F.cross_entropy(predictions_torch, targets_torch)
    loss.backward()
    
    return backend.from_torch(predictions_torch.grad)

# Use in training loop
outputs = compiled_model.evaluate(params, inputs)
loss_value = torch_cross_entropy(outputs["output"], targets)
loss_gradients = torch_cross_entropy_gradient(outputs["output"], targets)

_, gradients = compiled_model.evaluate(
    params,
    inputs,
    output_gradients={"output": loss_gradients}
)
```

## Loss Function Composition

You can combine multiple loss terms for multi-task learning or specific objectives:

```python
def combined_loss(predictions, targets, aux_preds=None, aux_targets=None, 
                  lambda_aux=0.3, lambda_reg=0.01, params=None):
    """Combined loss with main task, auxiliary task, and regularization."""
    # Main task loss (e.g., classification)
    main_loss = cross_entropy_loss(predictions, targets)
    
    # Auxiliary task loss (e.g., regression)
    aux_loss = 0
    if aux_preds is not None and aux_targets is not None:
        aux_loss = lambda_aux * mse_loss(aux_preds, aux_targets)
    
    # Regularization loss
    reg_loss = 0
    if params is not None:
        for param in params.values():
            reg_loss += backend.sum(backend.square(param))
        reg_loss = 0.5 * lambda_reg * reg_loss
    
    # Combined loss
    return main_loss + aux_loss + reg_loss

# Compute gradients for main and auxiliary tasks
def combined_loss_gradients(predictions, targets, aux_preds=None, aux_targets=None, 
                            lambda_aux=0.3):
    """Compute gradients for main and auxiliary tasks."""
    main_gradients = cross_entropy_loss_gradient(predictions, targets)
    
    aux_gradients = None
    if aux_preds is not None and aux_targets is not None:
        aux_gradients = lambda_aux * mse_loss_gradient(aux_preds, aux_targets)
    
    return {"main": main_gradients, "aux": aux_gradients}

# Use in training loop
outputs = compiled_model.evaluate(params, inputs)
main_preds = outputs["main_output"]
aux_preds = outputs["aux_output"]

# Compute combined loss value
loss_value = combined_loss(
    main_preds, main_targets,
    aux_preds, aux_targets,
    lambda_aux=0.3, lambda_reg=0.01, params=params
)

# Compute gradients for each output
gradients_dict = combined_loss_gradients(
    main_preds, main_targets,
    aux_preds, aux_targets,
    lambda_aux=0.3
)

# Backward pass with task-specific gradients
_, gradients = compiled_model.evaluate(
    params,
    inputs,
    output_gradients={
        "main_output": gradients_dict["main"],
        "aux_output": gradients_dict["aux"]
    }
)

# Apply regularization gradient separately
for name, grad in gradients.items():
    # Apply task-specific gradient and regularization
    reg_grad = 0.01 * params[name]  # L2 regularization
    params[name] = params[name] - learning_rate * (grad + reg_grad)
```

## Numerical Stability

Many loss functions involve operations like logarithms or divisions that can cause numerical instability. Use these techniques for better stability:

### Stable Softmax Cross-Entropy

```python
def stable_cross_entropy(logits, targets):
    """Numerically stable cross-entropy loss."""
    # Shift logits to avoid overflow in exp
    max_logits = backend.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits
    
    # Compute log softmax
    exp_logits = backend.exp(shifted_logits)
    sum_exp_logits = backend.sum(exp_logits, axis=-1, keepdims=True)
    log_probs = shifted_logits - backend.log(sum_exp_logits)
    
    # Compute cross-entropy
    return -backend.mean(backend.sum(targets * log_probs, axis=-1))

def stable_cross_entropy_gradient(logits, targets):
    """Numerically stable gradient of cross-entropy loss."""
    # Compute softmax probabilities
    max_logits = backend.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits
    exp_logits = backend.exp(shifted_logits)
    sum_exp_logits = backend.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / sum_exp_logits
    
    # Gradient is (probabilities - targets) / batch_size
    batch_size = backend.shape(logits)[0]
    return (probs - targets) / batch_size
```

### Epsilon in Divisions

```python
def binary_cross_entropy_stable(predictions, targets, epsilon=1e-7):
    """Numerically stable binary cross-entropy loss."""
    # Clip predictions to avoid log(0) and log(1)
    clipped_preds = backend.clip(predictions, epsilon, 1.0 - epsilon)
    
    # Compute BCE
    return -backend.mean(
        targets * backend.log(clipped_preds) + 
        (1 - targets) * backend.log(1 - clipped_preds)
    )
```

## Best Practices

1. **Match loss to task**:
   - Classification: Cross-entropy
   - Regression: MSE or Huber
   - Ranking: Triplet or contrastive
   - Imbalanced data: Focal loss or weighted loss

2. **Monitor loss behavior**:
   - Track both training and validation loss
   - Watch for instability (NaN, exploding values)
   - Check for plateaus that might indicate learning issues

3. **Scale losses appropriately**:
   - When combining losses, use weighting factors to balance their contributions
   - Adjust regularization strength based on model size and dataset

4. **Handle class imbalance**:
   - Use class weighting in cross-entropy
   - Consider specialized losses like focal loss
   - Adjust sampling or augmentation strategy

5. **Use backend-specific optimizations**:
   - Leverage native implementations when available
   - Use mixed precision where appropriate
   - Consider performance implications for complex loss functions