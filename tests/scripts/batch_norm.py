import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
# Define the input tensor
a = torch.tensor([[1.0], [2]])

# Define the running mean and running variance
running_mean = torch.tensor([0.0])
running_var = torch.tensor([1.0])

# Apply batch normalization
output = F.batch_norm(a, running_mean, running_var)

print(output)

class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.momentum = momentum
        self.eps = eps
        self.gamma = jnp.ones(num_features)  # Scale parameter
        self.beta = jnp.zeros(num_features)  # Shift parameter
        self.running_mean = jnp.zeros(num_features)  # Running mean
        self.running_var = jnp.ones(num_features)   # Running variance

    def __call__(self, x, training=True):
        if training:
            # Compute batch statistics
            batch_mean = jnp.mean(x, axis=0)
            # Compute variance with Bessel's correction (n - 1)
            batch_var = jnp.sum((x - batch_mean) ** 2, axis=0) / (x.shape[0] - 1)
            
            # Update running statistics (aligned with PyTorch's formula)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize using batch statistics
            x_norm = (x - batch_mean) / jnp.sqrt(jnp.var(x, axis=0) + self.eps)
        else:
            # Normalize using running statistics
            x_norm = (x - self.running_mean) / jnp.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta

# PyTorch BatchNorm for comparison
class PyTorchBatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.batchnorm = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps, affine=False)
    
    def __call__(self, x, training=True):
        if training:
            self.batchnorm.train()
        else:
            self.batchnorm.eval()
        return self.batchnorm(x)

# Generate random data for testing
def generate_data(batch_size, num_features, num_batches):
    rng = np.random.RandomState(42)
    data = [rng.randn(batch_size, num_features).astype(np.float32) for _ in range(num_batches)]
    return data

# Loop-based comparison test
def test_batchnorm_vs_pytorch():
    # Hyperparameters
    num_features = 2
    batch_size = 4
    num_batches = 10
    momentum = 0.1
    eps = 1e-5

    # Initialize BatchNorm layers
    batchnorm_jax = BatchNorm(num_features, momentum=momentum, eps=eps)
    batchnorm_torch = PyTorchBatchNorm(num_features, momentum=momentum, eps=eps)

    # Generate random data
    data = generate_data(batch_size, num_features, num_batches)

    # Training loop
    for i, x_np in enumerate(data):
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        # JAX BatchNorm
        output_jax = batchnorm_jax(x_jax, training=True)
        
        # PyTorch BatchNorm
        output_torch = batchnorm_torch(x_torch, training=True)

        # Compare outputs
        assert jnp.allclose(output_jax, output_torch.detach().numpy(), atol=1e-5), f"Output mismatch at batch {i}"

        # Compare running statistics
        assert jnp.allclose(batchnorm_jax.running_mean, batchnorm_torch.batchnorm.running_mean.numpy(), atol=1e-5), f"Running mean mismatch at batch {i}"
        assert jnp.allclose(batchnorm_jax.running_var, batchnorm_torch.batchnorm.running_var.numpy(), atol=1e-5), f"Running var mismatch at batch {i}"

    print("All tests passed! Running statistics and outputs match between JAX and PyTorch.")

# Run the test
test_batchnorm_vs_pytorch()