# Image Classification with Mithril

This tutorial demonstrates how to build, train, and evaluate a convolutional neural network (CNN) for image classification using Mithril. We'll use the CIFAR-10 dataset as an example.

## Overview

In this tutorial, you'll learn:

1. How to create a CNN model in Mithril
2. How to load and preprocess image data
3. How to train the model using JAX
4. How to evaluate the model's performance

## Prerequisites

Before starting, make sure you have the required libraries:

```bash
pip install mithril jax optax numpy matplotlib torch torchvision
```

## Dataset Setup

We'll use PyTorch's data utilities to load the CIFAR-10 dataset:

```python
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# Class names
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Function to convert PyTorch tensor to NumPy array
def tensor_to_numpy(tensor):
    return tensor.numpy()
```

## Building the CNN Model

Now, let's create a CNN model using Mithril:

```python
import mithril as ml
from mithril.models import Model, Conv2d, MaxPool2d, Relu, Linear, Flatten, BatchNorm2d, Dropout

def create_cifar10_cnn():
    model = Model()
    
    # Convolutional Block 1
    model |= Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)(
        input="input", output="conv1"
    )
    model += BatchNorm2d(num_features=32)(input="conv1", output="bn1")
    model += Relu()(input="bn1", output="relu1")
    model += MaxPool2d(kernel_size=2, stride=2)(input="relu1", output="pool1")
    
    # Convolutional Block 2
    model += Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)(
        input="pool1", output="conv2"
    )
    model += BatchNorm2d(num_features=64)(input="conv2", output="bn2")
    model += Relu()(input="bn2", output="relu2")
    model += MaxPool2d(kernel_size=2, stride=2)(input="relu2", output="pool2")
    
    # Convolutional Block 3
    model += Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)(
        input="pool2", output="conv3"
    )
    model += BatchNorm2d(num_features=128)(input="conv3", output="bn3")
    model += Relu()(input="bn3", output="relu3")
    model += MaxPool2d(kernel_size=2, stride=2)(input="relu3", output="pool3")
    
    # Flatten and fully connected layers
    model += Flatten()(input="pool3", output="flat")
    model += Linear(dimension=512)(input="flat", output="fc1")
    model += Relu()(input="fc1", output="relu4")
    model += Dropout(p=0.5)(input="relu4", output="drop1")
    model += Linear(dimension=10)(input="drop1", output="output")
    
    return model
```

## Compiling the Model

Let's compile the model with the JAX backend:

```python
import jax
import optax

# Create model and backend
model = create_cifar10_cnn()
backend = ml.JaxBackend(dtype=ml.float32)

# Compile the model
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 3, 32, 32]},  # [batch_size, channels, height, width]
    jit=True
)

# Initialize parameters
params = compiled_model.randomize_params()

# Initialize optimizer
learning_rate = 0.001
optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)

# Print model summary
print(model.summary())
```

## Training Functions

Let's define functions for training and evaluation:

```python
import jax.numpy as jnp

# Cross-entropy loss
def cross_entropy_loss(logits, targets):
    # Convert to one-hot
    one_hot = jnp.zeros((targets.shape[0], 10))
    one_hot = one_hot.at[jnp.arange(targets.shape[0]), targets].set(1)
    
    # Compute softmax and cross-entropy
    exp_logits = jnp.exp(logits - jnp.max(logits, axis=1, keepdims=True))
    probs = exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)
    loss = -jnp.mean(jnp.sum(one_hot * jnp.log(probs + 1e-8), axis=1))
    
    return loss

# Compute loss and its gradient
def compute_loss_and_grad(params, inputs, targets):
    outputs = compiled_model.evaluate(params, inputs)
    logits = outputs["output"]
    loss = cross_entropy_loss(logits, targets)
    
    # Compute gradient of loss with respect to outputs
    batch_size = logits.shape[0]
    exp_logits = jnp.exp(logits - jnp.max(logits, axis=1, keepdims=True))
    probs = exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)
    
    # Gradient of cross-entropy loss
    grad = probs.copy()
    grad = grad.at[jnp.arange(batch_size), targets].add(-1)
    grad = grad / batch_size
    
    # Compute gradients through the model
    _, gradients = compiled_model.evaluate(params, inputs, output_gradients={"output": grad})
    
    return loss, gradients

# Training step
def train_step(params, opt_state, inputs, targets):
    loss, gradients = compute_loss_and_grad(params, inputs, targets)
    updates, new_opt_state = optimizer.update(gradients, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# Evaluate accuracy
def evaluate(params, dataloader):
    correct = 0
    total = 0
    
    for data in dataloader:
        images, labels = data
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        inputs = {"input": backend.array(images_np)}
        outputs = compiled_model.evaluate(params, inputs)
        predictions = jnp.argmax(outputs["output"], axis=1)
        
        total += labels.size(0)
        correct += jnp.sum(predictions == labels_np)
    
    return correct / total
```

## Training Loop

Now, let's train the model:

```python
# Training configuration
num_epochs = 10
log_interval = 100

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for i, data in enumerate(trainloader):
        # Get data batch
        images, labels = data
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        # Prepare inputs
        inputs = {"input": backend.array(images_np)}
        
        # Train step
        params, opt_state, loss = train_step(params, opt_state, inputs, labels_np)
        
        # Log statistics
        running_loss += loss
        if i % log_interval == log_interval - 1:
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / log_interval:.4f}')
            running_loss = 0.0
    
    # Evaluate on test set
    accuracy = evaluate(params, testloader)
    print(f'Epoch: {epoch + 1}, Test Accuracy: {accuracy:.4f}')

print('Training complete')
```

## Visualizing Results

Let's visualize some predictions from the model:

```python
# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # Convert from [C, H, W] to [H, W, C]

# Function to display predictions
def show_predictions(images, labels, predictions):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(10, len(images))):
        ax = axes[i]
        imshow(images[i])
        ax.set_title(f'True: {classes[labels[i]]}\nPred: {classes[predictions[i]]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Get a batch of test images
dataiter = iter(testloader)
images, labels = next(dataiter)

# Convert to numpy
images_np = images.numpy()
labels_np = labels.numpy()

# Get predictions
inputs = {"input": backend.array(images_np)}
outputs = compiled_model.evaluate(params, inputs)
predictions = jnp.argmax(outputs["output"], axis=1)

# Show images with predictions
show_predictions(images_np, labels_np, predictions)
```

## Saving the Model

You can save the trained model's parameters and architecture:

```python
import pickle
import json

# Save parameters
with open('cifar10_cnn_params.pkl', 'wb') as f:
    pickle.dump({k: backend.to_numpy(v) for k, v in params.items()}, f)

# Save model architecture
with open('cifar10_cnn_model.json', 'w') as f:
    json.dump(model.to_dict(), f)

print("Model saved successfully")
```

## Loading the Model

You can load the saved model and parameters:

```python
# Load model architecture
with open('cifar10_cnn_model.json', 'r') as f:
    model_dict = json.load(f)
    loaded_model = Model.from_dict(model_dict)

# Create a new backend and compile model
backend = ml.JaxBackend(dtype=ml.float32)
compiled_model = ml.compile(
    model=loaded_model,
    backend=backend,
    shapes={"input": [32, 3, 32, 32]}
)

# Load parameters
with open('cifar10_cnn_params.pkl', 'rb') as f:
    loaded_params_np = pickle.load(f)
    loaded_params = {k: backend.array(v) for k, v in loaded_params_np.items()}

# Test the loaded model
accuracy = evaluate(loaded_params, testloader)
print(f'Loaded model test accuracy: {accuracy:.4f}')
```

## Advanced Topics

### 1. Data Augmentation

You can improve your model by using data augmentation:

```python
# Enhanced transformations with data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create new training dataset with augmentation
trainset_aug = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
trainloader_aug = torch.utils.data.DataLoader(trainset_aug, batch_size=32,
                                             shuffle=True, num_workers=2)
```

### 2. Learning Rate Scheduling

Implement learning rate scheduling for better convergence:

```python
# Learning rate schedule
schedule = optax.exponential_decay(
    init_value=0.001,
    transition_steps=1000,
    decay_rate=0.9
)

# Create optimizer with schedule
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping
    optax.adam(learning_rate=schedule)
)
opt_state = optimizer.init(params)
```

### 3. Model Validation During Training

Add a validation set to monitor overfitting:

```python
# Split training set into training and validation
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset_split, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

trainloader_split = torch.utils.data.DataLoader(trainset_split, batch_size=32,
                                              shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=32,
                                      shuffle=False, num_workers=2)

# Modify training loop to include validation
for epoch in range(num_epochs):
    # Training
    # ... [training code as before]
    
    # Validation
    val_accuracy = evaluate(params, valloader)
    print(f'Epoch: {epoch + 1}, Validation Accuracy: {val_accuracy:.4f}')
    
    # Early stopping logic
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = params.copy()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break
```

## Conclusion

In this tutorial, you learned how to:

1. Create a CNN architecture using Mithril's composable models
2. Compile the model with the JAX backend
3. Train the model using custom training loops
4. Evaluate the model on a test dataset
5. Save and load model parameters
6. Implement advanced techniques like data augmentation and learning rate scheduling

This example demonstrates how Mithril provides a flexible framework for creating and training neural networks with minimal boilerplate code, while allowing you to maintain full control over the training process.

---

**Next Steps:**

- Try modifying the model architecture to improve performance
- Experiment with different optimizers and learning rates
- Apply transfer learning by loading pre-trained weights
- Adapt the model for your own image classification tasks