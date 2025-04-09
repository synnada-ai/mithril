# Linear Regression Tutorial

This tutorial demonstrates how to implement and train a simple linear regression model using Mithril. Linear regression is a foundational machine learning algorithm that serves as an excellent starting point for understanding Mithril's model building, training, and evaluation workflow.

## Overview

In this tutorial, we will:

1. Create a linear regression model using Mithril's logical model API
2. Generate synthetic data for training and testing
3. Compile the model for different backends (JAX, PyTorch, NumPy)
4. Train the model using gradient descent
5. Evaluate the model's performance
6. Visualize the results

## Prerequisites

- Basic understanding of Python and linear algebra
- Familiarity with machine learning concepts
- Mithril installed (`pip install mithril`)
- NumPy and Matplotlib for data generation and visualization

## Creating a Linear Regression Model

Let's start by defining a simple linear regression model using Mithril:

```python
import mithril as mi
import numpy as np
import matplotlib.pyplot as plt
from mithril.models import LogicalModel

# Create a linear regression model
def create_linear_regression(input_dim=1, output_dim=1):
    model = LogicalModel("linear_regression")
    
    with model:
        # Input features
        x = mi.Input(shape=(None, input_dim), name="x")
        
        # Model parameters: weights and bias
        weights = mi.Parameter(shape=(input_dim, output_dim), name="weights")
        bias = mi.Parameter(shape=(output_dim,), name="bias")
        
        # Linear prediction: y = X * W + b
        y_pred = mi.matmul(x, weights) + bias
        
        # Output
        mi.Output(y_pred, name="y_pred")
    
    return model

# Create a linear regression model with single input and output
linear_model = create_linear_regression(input_dim=1, output_dim=1)
```

## Generating Synthetic Data

Let's generate some synthetic data for our linear regression task:

```python
# Generate synthetic data
def generate_data(n_samples=100, input_dim=1, noise_scale=0.1):
    # True parameters
    true_weights = np.random.randn(input_dim, 1) * 2  # Random weights
    true_bias = np.random.randn(1) * 0.5  # Random bias
    
    # Generate random input features
    X = np.random.rand(n_samples, input_dim) * 4 - 2  # Values between -2 and 2
    
    # Generate targets with some noise
    y_true = np.matmul(X, true_weights) + true_bias
    y = y_true + np.random.randn(n_samples, 1) * noise_scale
    
    return X, y, true_weights, true_bias

# Generate training and test data
X_train, y_train, true_weights, true_bias = generate_data(n_samples=100)
X_test, y_test, _, _ = generate_data(n_samples=20)

print(f"True weights: {true_weights.flatten()}")
print(f"True bias: {true_bias.flatten()}")
```

## Compiling the Model for Different Backends

One of Mithril's key features is its ability to compile models for different backends. Let's compile our model for JAX, PyTorch, and NumPy backends:

```python
# Compile the model with different backends
from mithril.backends.with_autograd.jax_backend import JaxBackend
from mithril.backends.with_autograd.torch_backend import TorchBackend
from mithril.backends.with_manualgrad.numpy_backend import NumpyBackend

# JAX backend
jax_backend = JaxBackend()
jax_model = linear_model.compile(jax_backend)

# Initialize parameters randomly
jax_model.set_values({
    "weights": jax_backend.tensor(np.random.randn(1, 1) * 0.1),
    "bias": jax_backend.tensor(np.random.randn(1) * 0.1)
})

# PyTorch backend
torch_backend = TorchBackend()
torch_model = linear_model.compile(torch_backend)

# Initialize with the same parameters as JAX model for comparison
torch_model.set_values({
    "weights": torch_backend.tensor(jax_backend.to_numpy(jax_model.get_value("weights"))),
    "bias": torch_backend.tensor(jax_backend.to_numpy(jax_model.get_value("bias")))
})

# NumPy backend
numpy_backend = NumpyBackend()
numpy_model = linear_model.compile(numpy_backend)

# Initialize with the same parameters as JAX model for comparison
numpy_model.set_values({
    "weights": numpy_backend.tensor(jax_backend.to_numpy(jax_model.get_value("weights"))),
    "bias": numpy_backend.tensor(jax_backend.to_numpy(jax_model.get_value("bias")))
})
```

## Implementing Training Loop

Now, let's implement a training loop for our linear regression model:

```python
# Define mean squared error loss function
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Training function that works with any Mithril backend
def train_model(model, X, y, learning_rate=0.01, epochs=200, batch_size=32):
    n_samples = X.shape[0]
    losses = []
    
    # Get the backend from the model
    backend = model.backend
    
    # Define training step
    def training_step(model, X_batch, y_batch):
        # Forward pass
        outputs = model({"x": X_batch})
        y_pred = outputs["y_pred"]
        
        # Compute loss
        loss = backend.reduce_mean(backend.power(y_pred - y_batch, 2))
        
        # Get parameters
        weights = model.get_value("weights")
        bias = model.get_value("bias")
        
        # Compute gradients
        if isinstance(backend, JaxBackend) or isinstance(backend, TorchBackend):
            # Use automatic differentiation for JAX and PyTorch
            grad_fn = backend.value_and_grad(lambda params: backend.reduce_mean(
                backend.power(
                    backend.matmul(X_batch, params["weights"]) + params["bias"] - y_batch, 
                    2
                )
            ))
            
            loss_val, grads = grad_fn({"weights": weights, "bias": bias})
            
            # Update parameters
            new_weights = weights - learning_rate * grads["weights"]
            new_bias = bias - learning_rate * grads["bias"]
        else:
            # Manual gradient computation for NumPy backend
            # Gradient of MSE with respect to weights: 2/n * X^T * (X*W + b - y)
            # Gradient of MSE with respect to bias: 2/n * (X*W + b - y)
            batch_size = backend.shape(X_batch)[0]
            error = backend.matmul(X_batch, weights) + bias - y_batch
            
            dw = (2.0 / batch_size) * backend.matmul(backend.transpose(X_batch), error)
            db = (2.0 / batch_size) * backend.reduce_sum(error, axis=0)
            
            # Update parameters
            new_weights = weights - learning_rate * dw
            new_bias = bias - learning_rate * db
        
        # Set new parameter values
        model.set_values({"weights": new_weights, "bias": new_bias})
        
        return backend.to_numpy(loss)
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle the data
        idx = np.random.permutation(n_samples)
        X_shuffled = X[idx]
        y_shuffled = y[idx]
        
        # Mini-batch training
        epoch_losses = []
        for i in range(0, n_samples, batch_size):
            X_batch = backend.tensor(X_shuffled[i:i+batch_size])
            y_batch = backend.tensor(y_shuffled[i:i+batch_size])
            
            loss = training_step(model, X_batch, y_batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Get final parameter values
    weights = backend.to_numpy(model.get_value("weights"))
    bias = backend.to_numpy(model.get_value("bias"))
    
    return weights, bias, losses
```

## Training the Model

Let's train our models with different backends and compare the results:

```python
# Train models with different backends
print("Training with JAX backend:")
jax_weights, jax_bias, jax_losses = train_model(jax_model, X_train, y_train, learning_rate=0.05, epochs=200)

print("\nTraining with PyTorch backend:")
torch_weights, torch_bias, torch_losses = train_model(torch_model, X_train, y_train, learning_rate=0.05, epochs=200)

print("\nTraining with NumPy backend:")
numpy_weights, numpy_bias, numpy_losses = train_model(numpy_model, X_train, y_train, learning_rate=0.05, epochs=200)

# Display final parameters
print("\nFinal Parameters:")
print(f"True weights: {true_weights.flatten()}, True bias: {true_bias.flatten()}")
print(f"JAX weights: {jax_weights.flatten()}, JAX bias: {jax_bias.flatten()}")
print(f"PyTorch weights: {torch_weights.flatten()}, PyTorch bias: {torch_bias.flatten()}")
print(f"NumPy weights: {numpy_weights.flatten()}, NumPy bias: {numpy_bias.flatten()}")
```

## Evaluating the Model

Now let's evaluate our trained models on the test data:

```python
# Evaluate models
def evaluate_model(model, X, y):
    backend = model.backend
    X_tensor = backend.tensor(X)
    y_tensor = backend.tensor(y)
    
    # Forward pass
    outputs = model({"x": X_tensor})
    y_pred = outputs["y_pred"]
    
    # Compute loss
    loss = backend.reduce_mean(backend.power(y_pred - y_tensor, 2))
    
    return backend.to_numpy(loss), backend.to_numpy(y_pred)

# Evaluate on test data
jax_test_loss, jax_preds = evaluate_model(jax_model, X_test, y_test)
torch_test_loss, torch_preds = evaluate_model(torch_model, X_test, y_test)
numpy_test_loss, numpy_preds = evaluate_model(numpy_model, X_test, y_test)

print("\nTest MSE:")
print(f"JAX: {jax_test_loss:.6f}")
print(f"PyTorch: {torch_test_loss:.6f}")
print(f"NumPy: {numpy_test_loss:.6f}")
```

## Visualizing Results

Finally, let's visualize our results:

```python
# Plot training loss curves
plt.figure(figsize=(12, 4))

# Plot loss curves
plt.subplot(1, 2, 1)
plt.plot(jax_losses, label='JAX')
plt.plot(torch_losses, label='PyTorch')
plt.plot(numpy_losses, label='NumPy')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.legend()

# Plot regression results
plt.subplot(1, 2, 2)

# Sort data for line plotting
sort_idx = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sort_idx]
y_test_sorted = y_test[sort_idx]
jax_preds_sorted = jax_preds[sort_idx]
torch_preds_sorted = torch_preds[sort_idx]
numpy_preds_sorted = numpy_preds[sort_idx]

# Plot data and predictions
plt.scatter(X_test[:, 0], y_test, color='black', alpha=0.5, label='Data')
plt.plot(X_test_sorted[:, 0], jax_preds_sorted, color='blue', linewidth=2, label='JAX')
plt.plot(X_test_sorted[:, 0], torch_preds_sorted, color='orange', linewidth=2, label='PyTorch')
plt.plot(X_test_sorted[:, 0], numpy_preds_sorted, color='green', linewidth=2, label='NumPy')

# True line
X_line = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100).reshape(-1, 1)
y_line = np.matmul(X_line, true_weights) + true_bias
plt.plot(X_line, y_line, 'r--', linewidth=2, label='True')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()

plt.tight_layout()
plt.show()
```

## Complete Example

Here's the complete example of linear regression with Mithril:

```python
import mithril as mi
import numpy as np
import matplotlib.pyplot as plt
from mithril.models import LogicalModel
from mithril.backends.with_autograd.jax_backend import JaxBackend
from mithril.backends.with_autograd.torch_backend import TorchBackend
from mithril.backends.with_manualgrad.numpy_backend import NumpyBackend

# Create a linear regression model
def create_linear_regression(input_dim=1, output_dim=1):
    model = LogicalModel("linear_regression")
    
    with model:
        # Input features
        x = mi.Input(shape=(None, input_dim), name="x")
        
        # Model parameters: weights and bias
        weights = mi.Parameter(shape=(input_dim, output_dim), name="weights")
        bias = mi.Parameter(shape=(output_dim,), name="bias")
        
        # Linear prediction: y = X * W + b
        y_pred = mi.matmul(x, weights) + bias
        
        # Output
        mi.Output(y_pred, name="y_pred")
    
    return model

# Generate synthetic data
def generate_data(n_samples=100, input_dim=1, noise_scale=0.1):
    # True parameters
    true_weights = np.random.randn(input_dim, 1) * 2  # Random weights
    true_bias = np.random.randn(1) * 0.5  # Random bias
    
    # Generate random input features
    X = np.random.rand(n_samples, input_dim) * 4 - 2  # Values between -2 and 2
    
    # Generate targets with some noise
    y_true = np.matmul(X, true_weights) + true_bias
    y = y_true + np.random.randn(n_samples, 1) * noise_scale
    
    return X, y, true_weights, true_bias

# Training function that works with any Mithril backend
def train_model(model, X, y, learning_rate=0.01, epochs=200, batch_size=32):
    n_samples = X.shape[0]
    losses = []
    
    # Get the backend from the model
    backend = model.backend
    
    # Define training step
    def training_step(model, X_batch, y_batch):
        # Forward pass
        outputs = model({"x": X_batch})
        y_pred = outputs["y_pred"]
        
        # Compute loss
        loss = backend.reduce_mean(backend.power(y_pred - y_batch, 2))
        
        # Get parameters
        weights = model.get_value("weights")
        bias = model.get_value("bias")
        
        # Compute gradients
        if isinstance(backend, JaxBackend) or isinstance(backend, TorchBackend):
            # Use automatic differentiation for JAX and PyTorch
            grad_fn = backend.value_and_grad(lambda params: backend.reduce_mean(
                backend.power(
                    backend.matmul(X_batch, params["weights"]) + params["bias"] - y_batch, 
                    2
                )
            ))
            
            loss_val, grads = grad_fn({"weights": weights, "bias": bias})
            
            # Update parameters
            new_weights = weights - learning_rate * grads["weights"]
            new_bias = bias - learning_rate * grads["bias"]
        else:
            # Manual gradient computation for NumPy backend
            batch_size = backend.shape(X_batch)[0]
            error = backend.matmul(X_batch, weights) + bias - y_batch
            
            dw = (2.0 / batch_size) * backend.matmul(backend.transpose(X_batch), error)
            db = (2.0 / batch_size) * backend.reduce_sum(error, axis=0)
            
            # Update parameters
            new_weights = weights - learning_rate * dw
            new_bias = bias - learning_rate * db
        
        # Set new parameter values
        model.set_values({"weights": new_weights, "bias": new_bias})
        
        return backend.to_numpy(loss)
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle the data
        idx = np.random.permutation(n_samples)
        X_shuffled = X[idx]
        y_shuffled = y[idx]
        
        # Mini-batch training
        epoch_losses = []
        for i in range(0, n_samples, batch_size):
            X_batch = backend.tensor(X_shuffled[i:i+batch_size])
            y_batch = backend.tensor(y_shuffled[i:i+batch_size])
            
            loss = training_step(model, X_batch, y_batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Get final parameter values
    weights = backend.to_numpy(model.get_value("weights"))
    bias = backend.to_numpy(model.get_value("bias"))
    
    return weights, bias, losses

# Evaluate model
def evaluate_model(model, X, y):
    backend = model.backend
    X_tensor = backend.tensor(X)
    y_tensor = backend.tensor(y)
    
    # Forward pass
    outputs = model({"x": X_tensor})
    y_pred = outputs["y_pred"]
    
    # Compute loss
    loss = backend.reduce_mean(backend.power(y_pred - y_tensor, 2))
    
    return backend.to_numpy(loss), backend.to_numpy(y_pred)

# Main function
def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create model
    linear_model = create_linear_regression(input_dim=1, output_dim=1)
    
    # Generate data
    X_train, y_train, true_weights, true_bias = generate_data(n_samples=100, noise_scale=0.2)
    X_test, y_test, _, _ = generate_data(n_samples=20, noise_scale=0.2)
    
    print(f"True weights: {true_weights.flatten()}")
    print(f"True bias: {true_bias.flatten()}")
    
    # Compile with different backends
    jax_backend = JaxBackend()
    jax_model = linear_model.compile(jax_backend)
    jax_model.set_values({
        "weights": jax_backend.tensor(np.random.randn(1, 1) * 0.1),
        "bias": jax_backend.tensor(np.random.randn(1) * 0.1)
    })
    
    torch_backend = TorchBackend()
    torch_model = linear_model.compile(torch_backend)
    torch_model.set_values({
        "weights": torch_backend.tensor(jax_backend.to_numpy(jax_model.get_value("weights"))),
        "bias": torch_backend.tensor(jax_backend.to_numpy(jax_model.get_value("bias")))
    })
    
    numpy_backend = NumpyBackend()
    numpy_model = linear_model.compile(numpy_backend)
    numpy_model.set_values({
        "weights": numpy_backend.tensor(jax_backend.to_numpy(jax_model.get_value("weights"))),
        "bias": numpy_backend.tensor(jax_backend.to_numpy(jax_model.get_value("bias")))
    })
    
    # Train models
    print("\nTraining with JAX backend:")
    jax_weights, jax_bias, jax_losses = train_model(jax_model, X_train, y_train, learning_rate=0.05, epochs=200)
    
    print("\nTraining with PyTorch backend:")
    torch_weights, torch_bias, torch_losses = train_model(torch_model, X_train, y_train, learning_rate=0.05, epochs=200)
    
    print("\nTraining with NumPy backend:")
    numpy_weights, numpy_bias, numpy_losses = train_model(numpy_model, X_train, y_train, learning_rate=0.05, epochs=200)
    
    # Display final parameters
    print("\nFinal Parameters:")
    print(f"True weights: {true_weights.flatten()}, True bias: {true_bias.flatten()}")
    print(f"JAX weights: {jax_weights.flatten()}, JAX bias: {jax_bias.flatten()}")
    print(f"PyTorch weights: {torch_weights.flatten()}, PyTorch bias: {torch_bias.flatten()}")
    print(f"NumPy weights: {numpy_weights.flatten()}, NumPy bias: {numpy_bias.flatten()}")
    
    # Evaluate on test data
    jax_test_loss, jax_preds = evaluate_model(jax_model, X_test, y_test)
    torch_test_loss, torch_preds = evaluate_model(torch_model, X_test, y_test)
    numpy_test_loss, numpy_preds = evaluate_model(numpy_model, X_test, y_test)
    
    print("\nTest MSE:")
    print(f"JAX: {jax_test_loss:.6f}")
    print(f"PyTorch: {torch_test_loss:.6f}")
    print(f"NumPy: {numpy_test_loss:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(jax_losses, label='JAX')
    plt.plot(torch_losses, label='PyTorch')
    plt.plot(numpy_losses, label='NumPy')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot regression results
    plt.subplot(1, 2, 2)
    
    # Sort data for line plotting
    sort_idx = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sort_idx]
    y_test_sorted = y_test[sort_idx]
    jax_preds_sorted = jax_preds[sort_idx]
    torch_preds_sorted = torch_preds[sort_idx]
    numpy_preds_sorted = numpy_preds[sort_idx]
    
    # Plot data and predictions
    plt.scatter(X_test[:, 0], y_test, color='black', alpha=0.5, label='Data')
    plt.plot(X_test_sorted[:, 0], jax_preds_sorted, color='blue', linewidth=2, label='JAX')
    plt.plot(X_test_sorted[:, 0], torch_preds_sorted, color='orange', linewidth=2, label='PyTorch')
    plt.plot(X_test_sorted[:, 0], numpy_preds_sorted, color='green', linewidth=2, label='NumPy')
    
    # True line
    X_line = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100).reshape(-1, 1)
    y_line = np.matmul(X_line, true_weights) + true_bias
    plt.plot(X_line, y_line, 'r--', linewidth=2, label='True')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

## Conclusion

In this tutorial, you've learned how to:

1. Create a linear regression model using Mithril's logical model API
2. Compile the same model for different backends (JAX, PyTorch, NumPy)
3. Implement training and evaluation loops that work with any backend
4. Compare performance across different backends

This example demonstrates the flexibility of Mithril's approach to building machine learning models. The same model definition can be compiled and executed on different backends without changing the model code, allowing you to leverage the strengths of each backend as needed.

## Next Steps

- Try modifying the model to handle multiple input features
- Experiment with different optimization algorithms
- Apply regularization techniques to prevent overfitting
- Explore Mithril's more advanced features, such as custom operators or model composition