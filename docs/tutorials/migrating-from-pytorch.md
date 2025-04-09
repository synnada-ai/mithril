# Migrating from PyTorch to Mithril

This guide helps experienced PyTorch users transition to Mithril, showing how familiar PyTorch patterns map to Mithril's logical-physical separation approach.

## Mental Model Shift

The biggest shift when moving from PyTorch to Mithril is separating **what** your model computes from **how** it's executed:

| PyTorch Pattern | Mithril Approach |
|-----------------|------------------|
| `nn.Module` defines both architecture and execution | `Model` defines architecture, `compile()` handles execution |
| Device placement in model definition | Device specified at compilation time |
| Framework-specific tensor operations | Backend-agnostic logical operations |
| Framework dictates parallelism approach | Parallelism as a compilation option |

## From torch.nn.Module to Mithril Models

### PyTorch Style

```python
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 8 * 8, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
# Create and use model
model = ConvNet().to('cuda')
x = torch.randn(32, 3, 32, 32).to('cuda')
output = model(x)
```

### Mithril Style

```python
import mithril as ml
from mithril.models import Model, Conv2d, BatchNorm2d, Relu, MaxPool2d, Flatten, Linear

# Define logical model
def create_conv_net():
    model = Model()
    model |= Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    model += BatchNorm2d(num_features=16)
    model += Relu()
    model += MaxPool2d(kernel_size=2)
    model += Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    model += BatchNorm2d(num_features=32)
    model += Relu()
    model += MaxPool2d(kernel_size=2)
    model += Flatten()
    model += Linear(dimension=10)(input_size=32*8*8)
    return model

# Create logical model
model = create_conv_net()

# Compile for PyTorch backend (or any other backend)
backend = ml.TorchBackend(device="cuda")
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 3, 32, 32]}
)

# Initialize parameters
params = compiled_model.randomize_params()

# Create input and run model
inputs = {"input": backend.randn(32, 3, 32, 32)}
outputs = compiled_model.evaluate(params, inputs)
output = outputs["output"]
```

## Layer Correspondence

| PyTorch Layer | Mithril Equivalent |
|---------------|-------------------|
| `nn.Linear` | `Linear` |
| `nn.Conv2d` | `Conv2d` |
| `nn.BatchNorm2d` | `BatchNorm2d` |
| `nn.ReLU` | `Relu` |
| `nn.MaxPool2d` | `MaxPool2d` |
| `nn.Flatten` | `Flatten` |
| `nn.LSTM` | `LSTM` |
| `nn.Dropout` | `Dropout` |
| `nn.Sequential` | Use `|=` and `+=` operators |

## Sequential vs. Operator Composition

### PyTorch Sequential

```python
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 6 * 6, 10)
)
```

### Mithril Composition

```python
model = Model()
model |= Conv2d(in_channels=3, out_channels=16, kernel_size=3)
model += Relu()
model += MaxPool2d(kernel_size=2)
model += Conv2d(in_channels=16, out_channels=32, kernel_size=3)
model += Relu()
model += MaxPool2d(kernel_size=2)
model += Flatten()
model += Linear(dimension=10)(input_size=32*6*6)
```

## Skip Connections and Complex Architectures

### PyTorch ResNet Block

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1)
            
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out += identity
        out = self.relu(out)
        return out
```

### Mithril ResNet Block

```python
from mithril.models import Add

def create_residual_block(in_channels, out_channels):
    block = Model()
    
    # Main path
    block |= Conv2d(in_channels=in_channels, out_channels=out_channels, 
                   kernel_size=3, padding=1)(input="input", output="conv1")
    block += BatchNorm2d(num_features=out_channels)(input="conv1", output="bn1")
    block += Relu()(input="bn1", output="relu1")
    block += Conv2d(in_channels=out_channels, out_channels=out_channels, 
                   kernel_size=3, padding=1)(input="relu1", output="conv2")
    block += BatchNorm2d(num_features=out_channels)(input="conv2", output="bn2")
    
    # Skip connection
    if in_channels != out_channels:
        block += Conv2d(in_channels=in_channels, out_channels=out_channels, 
                       kernel_size=1)(input="input", output="identity")
    else:
        # Direct connection
        identity_input = "input"
        
    # Add the residual connection
    block += Add()(left="bn2", right=identity_input if in_channels == out_channels else "identity", output="add")
    block += Relu()(input="add", output="output")
    
    return block
```

## Custom Modules and Reuse

### PyTorch Custom Module

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.double_conv(x)

# Using the custom module
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        # More layers...
```

### Mithril Custom Model

```python
def create_double_conv(in_channels, out_channels):
    double_conv = Model()
    double_conv |= Conv2d(in_channels=in_channels, out_channels=out_channels, 
                         kernel_size=3, padding=1)
    double_conv += BatchNorm2d(num_features=out_channels)
    double_conv += Relu()
    double_conv += Conv2d(in_channels=out_channels, out_channels=out_channels, 
                         kernel_size=3, padding=1)
    double_conv += BatchNorm2d(num_features=out_channels)
    double_conv += Relu()
    return double_conv

# Using the custom model
def create_unet():
    unet = Model()
    unet |= create_double_conv(3, 64)(output="down1")
    unet += create_double_conv(64, 128)(input="down1", output="down2")
    # More layers...
    return unet
```

## Parameters and Training

### PyTorch Training

```python
# Define model and optimizer
model = ConvNet().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Mithril Training

```python
# Define model and compile
model = create_conv_net()
backend = ml.TorchBackend(device="cuda")
compiled_model = ml.compile(model, backend, shapes={"input": [32, 3, 32, 32]})

# Initialize parameters
params = compiled_model.randomize_params()

# Define optimizer
learning_rate = 0.001
optimizer = ml.Adam(learning_rate=learning_rate)
optimizer_state = optimizer.init(params)

# Training loop
for epoch in range(10):
    for inputs_batch, targets_batch in train_loader:
        # Convert to backend format
        inputs = {"input": backend.array(inputs_batch)}
        targets = backend.array(targets_batch)
        
        # Forward pass
        outputs = compiled_model.evaluate(params, inputs)
        logits = outputs["output"]
        
        # Compute loss
        loss = backend.cross_entropy(logits, targets)
        
        # Compute gradients
        loss_grad = backend.cross_entropy_grad(logits, targets)
        output_gradients = {"output": loss_grad}
        _, gradients = compiled_model.evaluate(params, inputs, output_gradients)
        
        # Update parameters
        params, optimizer_state = optimizer.update(params, gradients, optimizer_state)
```

## Saving and Loading Models

### PyTorch Save/Load

```python
# Save model
torch.save(model.state_dict(), "model.pt")

# Load model
model = ConvNet()
model.load_state_dict(torch.load("model.pt"))
model.to('cuda')
```

### Mithril Save/Load

```python
import pickle
import json

# Save the logical model architecture
with open("model_architecture.json", "w") as f:
    json.dump(model.to_dict(), f)
    
# Save the parameters
with open("model_params.pkl", "wb") as f:
    # Convert to numpy for framework-agnostic storage
    numpy_params = {k: backend.to_numpy(v) for k, v in params.items()}
    pickle.dump(numpy_params, f)

# Load the model
with open("model_architecture.json", "r") as f:
    model_dict = json.load(f)
    loaded_model = Model.from_dict(model_dict)
    
# Compile model
compiled_model = ml.compile(loaded_model, backend)

# Load parameters
with open("model_params.pkl", "rb") as f:
    numpy_params = pickle.load(f)
    params = {k: backend.array(v) for k, v in numpy_params.items()}
```

## Handling Device Placement

### PyTorch Device Placement

```python
# Device placement in model/tensor definition
model = ConvNet().to('cuda:0')
x = torch.randn(32, 3, 32, 32, device='cuda:0')
```

### Mithril Device Placement

```python
# Device specified at backend creation
backend = ml.TorchBackend(device="cuda:0")
compiled_model = ml.compile(model, backend)

# Inputs created with that backend
inputs = {"input": backend.randn(32, 3, 32, 32)}
```

## Leveraging PyTorch Within Mithril

If you prefer PyTorch's execution environment, you can still benefit from Mithril's composability while compiling to PyTorch:

```python
# Define model with Mithril's composable API
model = Model()
model |= Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
model += Relu()
# ...rest of model definition

# Compile to PyTorch
torch_backend = ml.TorchBackend(device="cuda")
torch_model = ml.compile(model, torch_backend)

# Use familiar PyTorch training utilities
optimizer = torch.optim.Adam(torch_model.torch_parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop with PyTorch's optimizer
for epoch in range(10):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        
        # Wrap inputs for Mithril
        mithril_inputs = {"input": inputs}
        
        # Forward pass
        outputs = torch_model(mithril_inputs)
        loss = criterion(outputs["output"], targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Transitioning Gradually

You don't need to convert your entire codebase at once. Here's a strategy for gradual adoption:

1. Start by converting individual components to Mithril models
2. Compile these components to PyTorch for use in your existing PyTorch code
3. Gradually expand the scope of Mithril models in your codebase
4. Eventually, transition to Mithril's full logical-physical separation

## Key Advantages Over Pure PyTorch

1. **Backend Flexibility**: The same model definition works across JAX, PyTorch, NumPy, and more
2. **Cleaner Composition**: Explicit naming of terminals makes complex architectures clearer
3. **Framework Independence**: Seamless migration between frameworks as needs change
4. **Optimization Separation**: Performance tuning without changing model architecture
5. **Parallelism Configuration**: Change parallelism strategy without rewriting models

## Conclusion

Transitioning from PyTorch to Mithril requires a mental shift toward the separation of model architecture from execution details. While there's a learning curve, the benefits of framework agnosticism, cleaner composition, and flexible deployment make it worthwhile for many use cases.

Remember that you can:
- Use PyTorch as your backend while still getting Mithril's composition benefits
- Migrate gradually by converting components one at a time
- Leverage your PyTorch knowledge within Mithril's ecosystem

For more examples and patterns, see the [Examples](../examples/basic.md) section.