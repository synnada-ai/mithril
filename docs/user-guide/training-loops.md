# Training Loops

This guide explains how to implement effective training loops for Mithril models.

## Overview

Training loops in Mithril involve:

1. Preparing data and models
2. Iterating through data in batches
3. Performing forward and backward passes
4. Updating model parameters
5. Monitoring training progress
6. Validating periodically
7. Saving checkpoints

Unlike some frameworks, Mithril doesn't provide built-in training loops, giving you complete control over the training process.

## Basic Training Loop

Here's a simple training loop for a Mithril model:

```python
import mithril as ml
from mithril.backends import JaxBackend
from mithril.utils.losses import mse_loss, mse_loss_gradient
from mithril.utils.optimizers import Adam

# Create and compile a model
model = ml.Model()
# ... define your model ...
backend = JaxBackend()
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Create optimizer
optimizer = Adam(learning_rate=0.001)
optimizer_state = optimizer.init(params)

# Training data
# ... load your data ...

# Training loop
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    # Training metrics
    epoch_loss = 0.0
    num_batches = 0
    
    # Iterate through training data
    for batch_inputs, batch_targets in data_loader(batch_size):
        # Forward pass
        outputs = compiled_model.evaluate(params, batch_inputs)
        predictions = outputs["output"]
        
        # Compute loss for monitoring
        loss_value = mse_loss(predictions, batch_targets)
        epoch_loss += loss_value
        num_batches += 1
        
        # Compute loss gradients
        loss_gradients = mse_loss_gradient(predictions, batch_targets)
        
        # Backward pass
        _, gradients = compiled_model.evaluate(
            params,
            batch_inputs,
            output_gradients={"output": loss_gradients}
        )
        
        # Update parameters
        params, optimizer_state = optimizer.update(
            params, 
            gradients, 
            optimizer_state
        )
    
    # Print epoch summary
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Optional: validation
    if epoch % 2 == 0:
        validate(compiled_model, params, validation_data)
    
    # Optional: save checkpoint
    if epoch % 5 == 0:
        save_checkpoint(params, optimizer_state, epoch, avg_loss)
```

## Components of a Training Loop

### Data Loading

Efficient data loading is critical for training:

```python
def data_loader(batch_size, shuffle=True):
    """A simple data loader that yields batches of data."""
    # Assuming data and targets are already loaded
    num_samples = len(data)
    indices = list(range(num_samples))
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Convert data to backend tensors
        inputs = {"input": backend.array(data[batch_indices])}
        targets = backend.array(targets[batch_indices])
        
        yield inputs, targets
```

For larger datasets, consider using a more sophisticated data loading system:

```python
# Using PyTorch's DataLoader with Mithril
import torch
from torch.utils.data import DataLoader, TensorDataset

# Create a PyTorch dataset
tensor_data = torch.tensor(data, dtype=torch.float32)
tensor_targets = torch.tensor(targets, dtype=torch.float32)
dataset = TensorDataset(tensor_data, tensor_targets)

# Create a DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Useful for GPU training
)

# Use in training loop
for batch_data, batch_targets in loader:
    # Convert to backend tensors
    inputs = {"input": backend.from_torch(batch_data)}
    targets = backend.from_torch(batch_targets)
    
    # Forward pass, loss computation, etc.
    # ...
```

### Validation

Regular validation helps monitor for overfitting:

```python
def validate(model, params, validation_data, batch_size=64):
    """Evaluate the model on validation data."""
    total_loss = 0.0
    num_batches = 0
    
    # Disable gradient tracking for validation (if applicable)
    with backend.no_grad():
        for val_inputs, val_targets in validation_data(batch_size, shuffle=False):
            # Forward pass
            outputs = model.evaluate(params, val_inputs)
            predictions = outputs["output"]
            
            # Compute validation loss
            val_loss = mse_loss(predictions, val_targets)
            total_loss += val_loss
            num_batches += 1
    
    avg_val_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    return avg_val_loss
```

### Checkpointing

Save model checkpoints to resume training or deploy models:

```python
def save_checkpoint(params, optimizer_state, epoch, loss, filepath=None):
    """Save a training checkpoint."""
    if filepath is None:
        filepath = f"checkpoint_epoch_{epoch}.pkl"
    
    # Convert backend-specific tensors to NumPy arrays
    np_params = {name: backend.to_numpy(param) for name, param in params.items()}
    np_optimizer_state = {
        name: {k: backend.to_numpy(v) for k, v in state.items()} 
        for name, state in optimizer_state.items()
    }
    
    checkpoint = {
        "epoch": epoch,
        "params": np_params,
        "optimizer_state": np_optimizer_state,
        "loss": loss
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, backend):
    """Load a training checkpoint."""
    with open(filepath, "rb") as f:
        checkpoint = pickle.load(f)
    
    # Convert NumPy arrays back to backend tensors
    params = {name: backend.array(param) for name, param in checkpoint["params"].items()}
    optimizer_state = {
        name: {k: backend.array(v) for k, v in state.items()} 
        for name, state in checkpoint["optimizer_state"].items()
    }
    
    return params, optimizer_state, checkpoint["epoch"], checkpoint["loss"]
```

## Advanced Training Techniques

### Learning Rate Scheduling

Adjust learning rates during training:

```python
from mithril.utils.optimizers import Adam
from mithril.utils.lr_schedulers import StepLR

# Create optimizer
optimizer = Adam(learning_rate=0.001)
optimizer_state = optimizer.init(params)

# Create scheduler
scheduler = StepLR(
    optimizer,
    step_size=30,     # Reduce LR every 30 epochs
    gamma=0.1         # Reduce LR by a factor of 0.1
)

# Training loop
for epoch in range(num_epochs):
    # Training steps
    # ...
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.get_learning_rate()
    print(f"Learning rate: {current_lr}")
```

### Gradient Accumulation

For training with larger effective batch sizes:

```python
accumulation_steps = 4  # Accumulate gradients over 4 batches
effective_batch_size = batch_size * accumulation_steps

# Initialize accumulated gradients
accumulated_gradients = {name: backend.zeros_like(param) for name, param in params.items()}

# Training loop with gradient accumulation
for epoch in range(num_epochs):
    # Reset accumulated gradients at the start of each epoch
    for name in accumulated_gradients:
        accumulated_gradients[name] = backend.zeros_like(params[name])
    
    # Iterate through training data
    for batch_idx, (batch_inputs, batch_targets) in enumerate(data_loader(batch_size)):
        # Forward pass
        outputs = compiled_model.evaluate(params, batch_inputs)
        predictions = outputs["output"]
        
        # Compute loss
        loss_value = mse_loss(predictions, batch_targets)
        
        # Compute loss gradients
        loss_gradients = mse_loss_gradient(predictions, batch_targets)
        
        # Backward pass
        _, gradients = compiled_model.evaluate(
            params,
            batch_inputs,
            output_gradients={"output": loss_gradients}
        )
        
        # Accumulate gradients
        for name, grad in gradients.items():
            accumulated_gradients[name] = accumulated_gradients[name] + grad
        
        # Update parameters after accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Scale gradients by the number of accumulation steps
            for name in accumulated_gradients:
                accumulated_gradients[name] = accumulated_gradients[name] / accumulation_steps
            
            # Update parameters
            params, optimizer_state = optimizer.update(
                params, 
                accumulated_gradients, 
                optimizer_state
            )
            
            # Reset accumulated gradients
            for name in accumulated_gradients:
                accumulated_gradients[name] = backend.zeros_like(params[name])
```

### Mixed Precision Training

For faster training with lower precision:

```python
from mithril.utils.mixed_precision import MixedPrecisionTrainer

# Create a backend with mixed precision support
backend = TorchBackend(default_dtype="float16")

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
    loss_scale=128.0,  # Initial loss scale
    dynamic_loss_scaling=True  # Automatically adjust loss scale
)

# Training loop with mixed precision
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in data_loader(batch_size):
        # Forward pass in mixed precision
        outputs = mp_trainer.forward(batch_inputs)
        predictions = outputs["output"]
        
        # Compute loss
        loss_value = mse_loss(predictions, batch_targets)
        
        # Compute loss gradients
        loss_gradients = mse_loss_gradient(predictions, batch_targets)
        
        # Backward pass and parameter update
        params, optimizer_state, success = mp_trainer.backward_and_update(
            {"output": loss_gradients}
        )
        
        if not success:
            print("Skipping update due to overflow")
```

### Distributed Training

For training across multiple devices:

```python
# Create a backend with a device mesh
backend = JaxBackend(device_mesh=(4,))  # 4 GPUs with data parallelism

# Create and compile a model
model = ml.Model()
# ... define your model ...
compiled_model = ml.compile(model, backend)

# Initialize parameters (replicated across devices)
params = compiled_model.get_parameters()

# Create optimizer
optimizer = Adam(learning_rate=0.001)
optimizer_state = optimizer.init(params)

# Training loop
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in data_loader(batch_size * 4):  # Larger batch
        # Create sharded inputs
        sharded_inputs = {
            "input": backend.array(
                batch_inputs["input"],
                device_mesh=(4,),    # 4 devices
                tensor_split=(0,)    # Split along batch dimension
            )
        }
        
        # Create sharded targets
        sharded_targets = backend.array(
            batch_targets,
            device_mesh=(4,),    # 4 devices
            tensor_split=(0,)    # Split along batch dimension
        )
        
        # Forward pass (runs in parallel across devices)
        outputs = compiled_model.evaluate(params, sharded_inputs)
        predictions = outputs["output"]
        
        # Compute loss gradients
        loss_gradients = mse_loss_gradient(predictions, sharded_targets)
        
        # Backward pass (gradients automatically averaged across devices)
        _, gradients = compiled_model.evaluate(
            params,
            sharded_inputs,
            output_gradients={"output": loss_gradients}
        )
        
        # Update parameters (happens on all devices)
        params, optimizer_state = optimizer.update(
            params, 
            gradients, 
            optimizer_state
        )
```

## Monitoring and Metrics

### Progress Tracking

Monitor training progress with metrics:

```python
# Initialize metrics
train_losses = []
val_losses = []
accuracies = []

# Training loop with metrics
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    for batch_inputs, batch_targets in data_loader(batch_size):
        # Forward pass
        outputs = compiled_model.evaluate(params, batch_inputs)
        predictions = outputs["output"]
        
        # Compute loss
        loss_value = cross_entropy_loss(predictions, batch_targets)
        epoch_loss += loss_value
        
        # Compute accuracy
        pred_classes = backend.argmax(predictions, axis=1)
        target_classes = backend.argmax(batch_targets, axis=1)
        correct_preds += backend.sum(backend.equal(pred_classes, target_classes))
        total_samples += batch_targets.shape[0]
        
        # Training steps
        # ...
    
    # Calculate epoch metrics
    avg_loss = epoch_loss / len(data_loader)
    accuracy = correct_preds / total_samples
    
    # Store metrics
    train_losses.append(avg_loss)
    accuracies.append(accuracy)
    
    # Validation
    val_loss = validate(compiled_model, params, validation_data)
    val_losses.append(val_loss)
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Accuracy: {accuracy:.4f}")
```

### Visualizing Metrics

```python
import matplotlib.pyplot as plt

# Plot training metrics
def plot_metrics(train_losses, val_losses, accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'g-', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

# Call the function after training
plot_metrics(train_losses, val_losses, accuracies)
```

### TensorBoard Integration

Using TensorBoard for monitoring:

```python
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter
writer = SummaryWriter(log_dir='./runs/experiment_1')

# Training loop with TensorBoard logging
for epoch in range(num_epochs):
    # Training steps
    # ...
    
    # Calculate epoch metrics
    # ...
    
    # Log to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('Accuracy', accuracy, epoch)
    
    # Optionally, log parameter histograms
    for name, param in params.items():
        writer.add_histogram(f'Parameters/{name}', backend.to_numpy(param), epoch)
    
    # Optionally, log gradient histograms
    for name, grad in gradients.items():
        writer.add_histogram(f'Gradients/{name}', backend.to_numpy(grad), epoch)

# Close the writer when done
writer.close()
```

## Specialized Training Loops

### Training GANs

For Generative Adversarial Networks:

```python
# Create generator and discriminator models
generator = ml.Model()
# ... define generator ...
discriminator = ml.Model()
# ... define discriminator ...

# Compile models
compiled_generator = ml.compile(generator, backend)
compiled_discriminator = ml.compile(discriminator, backend)

# Initialize parameters
gen_params = compiled_generator.get_parameters()
disc_params = compiled_discriminator.get_parameters()

# Create optimizers
gen_optimizer = Adam(learning_rate=0.0002, beta1=0.5)
disc_optimizer = Adam(learning_rate=0.0002, beta1=0.5)
gen_opt_state = gen_optimizer.init(gen_params)
disc_opt_state = disc_optimizer.init(disc_params)

# Training loop
for epoch in range(num_epochs):
    for real_images in data_loader(batch_size):
        # Generate random noise
        batch_size = real_images.shape[0]
        noise = backend.randn(batch_size, latent_dim)
        
        # -----------------
        # Train Discriminator
        # -----------------
        
        # Generate fake images
        fake_inputs = {"noise": noise}
        fake_outputs = compiled_generator.evaluate(gen_params, fake_inputs)
        fake_images = fake_outputs["output"]
        
        # Real images - forward pass
        real_inputs = {"input": real_images}
        real_outputs = compiled_discriminator.evaluate(disc_params, real_inputs)
        real_preds = real_outputs["output"]
        
        # Fake images - forward pass
        fake_disc_inputs = {"input": fake_images}
        fake_outputs = compiled_discriminator.evaluate(disc_params, fake_disc_inputs)
        fake_preds = fake_outputs["output"]
        
        # Create labels
        real_labels = backend.ones_like(real_preds)
        fake_labels = backend.zeros_like(fake_preds)
        
        # Compute discriminator loss
        real_loss = binary_cross_entropy(real_preds, real_labels)
        fake_loss = binary_cross_entropy(fake_preds, fake_labels)
        disc_loss = real_loss + fake_loss
        
        # Compute gradients for discriminator
        real_grad = binary_cross_entropy_gradient(real_preds, real_labels)
        _, real_disc_grads = compiled_discriminator.evaluate(
            disc_params, real_inputs, output_gradients={"output": real_grad}
        )
        
        fake_grad = binary_cross_entropy_gradient(fake_preds, fake_labels)
        _, fake_disc_grads = compiled_discriminator.evaluate(
            disc_params, fake_disc_inputs, output_gradients={"output": fake_grad}
        )
        
        # Combine gradients
        disc_grads = {
            name: real_disc_grads[name] + fake_disc_grads[name]
            for name in real_disc_grads
        }
        
        # Update discriminator parameters
        disc_params, disc_opt_state = disc_optimizer.update(
            disc_params, disc_grads, disc_opt_state
        )
        
        # -----------------
        # Train Generator
        # -----------------
        
        # Generate fake images
        fake_inputs = {"noise": noise}
        fake_outputs = compiled_generator.evaluate(gen_params, fake_inputs)
        fake_images = fake_outputs["output"]
        
        # Discriminator prediction on fake images
        fake_disc_inputs = {"input": fake_images}
        fake_outputs = compiled_discriminator.evaluate(disc_params, fake_disc_inputs)
        fake_preds = fake_outputs["output"]
        
        # We want the generator to fool the discriminator
        fake_labels = backend.ones_like(fake_preds)  # Label fake as real
        
        # Compute generator loss
        gen_loss = binary_cross_entropy(fake_preds, fake_labels)
        
        # Compute gradients through both models
        fake_grad = binary_cross_entropy_gradient(fake_preds, fake_labels)
        _, disc_grads = compiled_discriminator.evaluate(
            disc_params, fake_disc_inputs, output_gradients={"output": fake_grad}
        )
        
        # Backpropagate from discriminator to generator using fake_images
        # First, get gradients with respect to fake_images
        fake_images_grad = disc_grads["input_grad"]
        
        # Then, backpropagate through generator
        _, gen_grads = compiled_generator.evaluate(
            gen_params, fake_inputs, output_gradients={"output": fake_images_grad}
        )
        
        # Update generator parameters
        gen_params, gen_opt_state = gen_optimizer.update(
            gen_params, gen_grads, gen_opt_state
        )
        
        # Print metrics
        print(f"Disc Loss: {disc_loss:.4f}, Gen Loss: {gen_loss:.4f}")
```

### Self-Supervised Learning

For self-supervised learning approaches:

```python
# Create a model with two output heads
model = ml.Model()
# ... define base encoder ...
model += ml.Linear(dimension=128)(input="encoder_output", output="embedding")
model += ml.Linear(dimension=num_classes)(input="embedding", output="classifier_output")
model += ml.Linear(dimension=input_dim)(input="embedding", output="reconstruction_output")

# Compile model
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Create optimizer
optimizer = Adam(learning_rate=0.001)
optimizer_state = optimizer.init(params)

# Training loop
for epoch in range(num_epochs):
    for batch_inputs in data_loader(batch_size):
        inputs = batch_inputs["input"]
        
        # Create corrupted inputs for self-supervised task
        corrupted_inputs = corrupt_inputs(inputs)  # e.g., adding noise, masking
        
        # Forward pass
        model_inputs = {"input": corrupted_inputs}
        outputs = compiled_model.evaluate(params, model_inputs)
        reconstruction = outputs["reconstruction_output"]
        classification = outputs["classifier_output"]
        
        # Compute reconstruction loss
        recon_loss = mse_loss(reconstruction, inputs)
        recon_grad = mse_loss_gradient(reconstruction, inputs)
        
        # If you have some labeled data
        if "labels" in batch_inputs:
            labels = batch_inputs["labels"]
            class_loss = cross_entropy_loss(classification, labels)
            class_grad = cross_entropy_loss_gradient(classification, labels)
            
            total_loss = recon_loss + class_loss
            
            # Backward pass with multiple output gradients
            _, gradients = compiled_model.evaluate(
                params,
                model_inputs,
                output_gradients={
                    "reconstruction_output": recon_grad,
                    "classifier_output": class_grad
                }
            )
        else:
            # Only use reconstruction loss if no labels
            total_loss = recon_loss
            
            # Backward pass with just reconstruction gradient
            _, gradients = compiled_model.evaluate(
                params,
                model_inputs,
                output_gradients={
                    "reconstruction_output": recon_grad,
                    "classifier_output": backend.zeros_like(classification)
                }
            )
        
        # Update parameters
        params, optimizer_state = optimizer.update(
            params, gradients, optimizer_state
        )
```

## Interoperability with Other Frameworks

### Using PyTorch's Training Utilities

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create a model
model = ml.Model()
# ... define your model ...
backend = ml.TorchBackend()
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Convert parameters to PyTorch parameters with gradients
torch_params = {}
for name, param in params.items():
    torch_param = backend.to_torch(param)
    torch_param.requires_grad_(True)
    torch_params[name] = torch_param

# Create a PyTorch optimizer
optimizer = optim.Adam(torch_params.values(), lr=0.001)

# Create a PyTorch dataset and loader
# ... create your dataset ...
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in loader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        ml_inputs = {"input": backend.from_torch(batch_inputs)}
        outputs = compiled_model.evaluate(
            {name: backend.from_torch(param) for name, param in torch_params.items()},
            ml_inputs
        )
        predictions = outputs["output"]
        torch_predictions = backend.to_torch(predictions)
        
        # Compute loss
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(torch_predictions, batch_targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Update Mithril parameters
        for name, param in torch_params.items():
            params[name] = backend.from_torch(param.detach())
```

### Using JAX's Training Utilities

```python
import jax
import jax.numpy as jnp
import optax

# Create a model
model = ml.Model()
# ... define your model ...
backend = ml.JaxBackend()
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Convert to JAX arrays if needed
jax_params = {name: backend.to_jax(param) for name, param in params.items()}

# Create an Optax optimizer
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(jax_params)

# Define training step function
@jax.jit
def train_step(params, opt_state, inputs, targets):
    def loss_fn(params):
        # Forward pass
        outputs = compiled_model.evaluate(params, inputs)
        predictions = outputs["output"]
        
        # Compute MSE loss
        loss = jnp.mean(jnp.square(predictions - targets))
        return loss
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_inputs, batch_targets in data_loader(batch_size):
        jax_inputs = {"input": backend.to_jax(batch_inputs["input"])}
        jax_targets = backend.to_jax(batch_targets)
        
        # Train step
        jax_params, opt_state, loss = train_step(jax_params, opt_state, jax_inputs, jax_targets)
        epoch_loss += loss
        num_batches += 1
    
    # Update Mithril parameters
    params = {name: backend.from_jax(param) for name, param in jax_params.items()}
    
    # Print progress
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

## Best Practices

1. **Modularize your training code**:
   - Separate data loading, model definition, training, and evaluation
   - Create reusable functions for common operations

2. **Monitor and validate regularly**:
   - Track multiple metrics, not just the loss
   - Validate frequently to detect overfitting
   - Use visualization to understand training dynamics

3. **Save checkpoints strategically**:
   - Save checkpoints regularly but not too frequently
   - Keep the best model based on validation metrics
   - Include all necessary state for resuming training

4. **Handle errors gracefully**:
   - Catch and log exceptions during training
   - Save checkpoints before potentially problematic operations
   - Implement resumable training

5. **Optimize performance**:
   - Use efficient data loading with prefetching
   - Profile your training loop to identify bottlenecks
   - Consider mixed precision for faster training

6. **Use appropriate hardware resources**:
   - Scale batch size based on available memory
   - Distribute training across multiple devices when available
   - Consider gradient accumulation for large models on limited hardware

7. **Implement early stopping**:
   - Monitor validation metrics to detect overfitting
   - Stop training when performance degrades
   - Revert to the best checkpoint

## Common Patterns

### Early Stopping

```python
def train_with_early_stopping(model, params, train_data, val_data, patience=10):
    """Train with early stopping based on validation loss."""
    optimizer = Adam(learning_rate=0.001)
    optimizer_state = optimizer.init(params)
    
    best_val_loss = float('inf')
    best_params = None
    patience_counter = 0
    
    for epoch in range(1000):  # Large max epochs
        # Train for one epoch
        params, optimizer_state, train_loss = train_epoch(
            model, params, optimizer, optimizer_state, train_data
        )
        
        # Evaluate on validation data
        val_loss = evaluate(model, params, val_data)
        
        # Print progress
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {name: backend.copy(param) for name, param in params.items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    return best_params, best_val_loss
```

### Cross-Validation

```python
def cross_validate(model, data, targets, n_folds=5):
    """Perform k-fold cross-validation."""
    # Split data into folds
    fold_size = len(data) // n_folds
    fold_indices = [list(range(i * fold_size, (i + 1) * fold_size)) for i in range(n_folds)]
    
    # Adjust the last fold if needed
    if len(data) % n_folds != 0:
        fold_indices[-1].extend(range(n_folds * fold_size, len(data)))
    
    val_scores = []
    
    for fold in range(n_folds):
        print(f"Fold {fold+1}/{n_folds}")
        
        # Create train/val split
        val_idx = fold_indices[fold]
        train_idx = [i for fold_i in range(n_folds) if fold_i != fold for i in fold_indices[fold_i]]
        
        # Create data loaders
        train_data = create_data_loader(data[train_idx], targets[train_idx])
        val_data = create_data_loader(data[val_idx], targets[val_idx], shuffle=False)
        
        # Re-initialize model
        compiled_model = ml.compile(model, backend)
        params = compiled_model.get_parameters()
        
        # Train on this fold
        best_params, val_score = train_with_early_stopping(
            compiled_model, params, train_data, val_data
        )
        
        val_scores.append(val_score)
    
    # Return average score
    avg_score = sum(val_scores) / n_folds
    print(f"Cross-validation complete. Average score: {avg_score:.4f}")
    
    return val_scores, avg_score
```

### Hyperparameter Tuning

```python
def train_with_hyperparams(model, data, targets, hyperparams):
    """Train model with specific hyperparameters."""
    # Extract hyperparameters
    learning_rate = hyperparams["learning_rate"]
    batch_size = hyperparams["batch_size"]
    weight_decay = hyperparams.get("weight_decay", 0.0)
    
    # Create data loaders
    train_data, val_data = split_train_val(data, targets)
    
    # Compile model
    compiled_model = ml.compile(model, backend)
    params = compiled_model.get_parameters()
    
    # Create optimizer with these hyperparameters
    optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    optimizer_state = optimizer.init(params)
    
    # Train model
    best_params, val_score = train_with_early_stopping(
        compiled_model, params, train_data, val_data, batch_size=batch_size
    )
    
    return best_params, val_score

def grid_search(model, data, targets, param_grid):
    """Perform grid search over hyperparameters."""
    import itertools
    
    # Create all combinations of hyperparameters
    keys = param_grid.keys()
    values = param_grid.values()
    hyperparam_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    best_score = float('inf')
    best_hyperparams = None
    best_params = None
    
    for i, hyperparams in enumerate(hyperparam_combinations):
        print(f"Testing hyperparameters {i+1}/{len(hyperparam_combinations)}: {hyperparams}")
        
        # Train with these hyperparameters
        params, val_score = train_with_hyperparams(model, data, targets, hyperparams)
        
        # Check if this is the best so far
        if val_score < best_score:
            best_score = val_score
            best_hyperparams = hyperparams
            best_params = params
    
    print(f"Best hyperparameters: {best_hyperparams}, Score: {best_score:.4f}")
    return best_params, best_hyperparams, best_score
```

## Conclusion

Training loops in Mithril give you complete control over the training process while providing tools for optimization, validation, and monitoring. By following the patterns and best practices outlined in this guide, you can implement efficient and effective training loops for a wide range of machine learning tasks.

Remember that the best training loop depends on your specific task, model architecture, and available hardware resources. Don't hesitate to customize these examples to fit your unique requirements.