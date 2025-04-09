# Basic Examples

This guide provides step-by-step examples to help you get started with Mithril. We'll cover fundamental concepts and operations with increasing complexity.

## Example 1: Creating a Simple Model

Let's start by creating a basic linear model that performs `y = wx + b`.

```python
import mithril as ml
import numpy as np

# Step 1: Define a logical model
class LinearModel(ml.Model):
    def __init__(self):
        super().__init__()
        # Initialize weights and bias
        self.w = self.add_parameter("weight", shape=(1, 1))
        self.b = self.add_parameter("bias", shape=(1,))
    
    def forward(self, x):
        # Linear operation: y = wx + b
        return ml.matmul(x, self.w) + self.b

# Step 2: Create an instance of the model
model = LinearModel()

# Step 3: Compile the model with a backend
# Here we use NumPy backend for simplicity
physical_model = model.compile(ml.backends.NumpyBackend())

# Step 4: Initialize parameters
physical_model.set_parameter_values({
    "weight": np.array([[2.0]]),
    "bias": np.array([1.0])
})

# Step 5: Run inference
x = np.array([[3.0]])
result = physical_model(x)
print(f"Input: {x}, Output: {result}")  # Should output something like: Input: [[3.0]], Output: [[7.0]]
```

## Example 2: Training a Model

Now let's train our linear model to fit some data.

```python
import mithril as ml
import numpy as np

# Step 1: Define our model (same as before)
class LinearModel(ml.Model):
    def __init__(self):
        super().__init__()
        self.w = self.add_parameter("weight", shape=(1, 1))
        self.b = self.add_parameter("bias", shape=(1,))
    
    def forward(self, x):
        return ml.matmul(x, self.w) + self.b

# Step 2: Create synthetic data
np.random.seed(42)
x_train = np.random.randn(100, 1)
# True relationship: y = 3x + 2 + noise
y_train = 3 * x_train + 2 + np.random.randn(100, 1) * 0.1

# Step 3: Create model and loss function
model = LinearModel()

# Define MSE loss
def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    return ml.mean(ml.square(diff))

# Wrap model and loss in a training context
train_ctx = ml.train_model(model, loss_fn=mse_loss)

# Step 4: Compile with autograd-enabled backend (PyTorch for this example)
physical_model = train_ctx.compile(ml.backends.TorchBackend())

# Step 5: Initialize parameters randomly
physical_model.set_parameter_values({
    "weight": np.random.randn(1, 1),
    "bias": np.array([0.0])
})

# Step 6: Train the model
learning_rate = 0.01
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    y_pred = physical_model(x_train, y_train)
    
    # Compute gradients and update parameters
    grads = physical_model.gradient()
    
    # Manual gradient descent update
    current_params = physical_model.get_parameter_values()
    updated_params = {
        "weight": current_params["weight"] - learning_rate * grads["weight"],
        "bias": current_params["bias"] - learning_rate * grads["bias"]
    }
    physical_model.set_parameter_values(updated_params)
    
    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {y_pred}")

# Print final parameters
final_params = physical_model.get_parameter_values()
print(f"Learned weight: {final_params['weight']}, Learned bias: {final_params['bias']}")
print(f"True weight: 3.0, True bias: 2.0")
```

## Example 3: Working with Multiple Backends

Mithril's key advantage is its ability to use different backends. Let's see how to use the same model with multiple backends.

```python
import mithril as ml
import numpy as np

# Step 1: Define a model
class MLP(ml.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # First layer parameters
        self.w1 = self.add_parameter("weight1", shape=(input_size, hidden_size))
        self.b1 = self.add_parameter("bias1", shape=(hidden_size,))
        
        # Second layer parameters
        self.w2 = self.add_parameter("weight2", shape=(hidden_size, output_size))
        self.b2 = self.add_parameter("bias2", shape=(output_size,))
    
    def forward(self, x):
        # First layer with ReLU activation
        h = ml.matmul(x, self.w1) + self.b1
        h = ml.relu(h)
        
        # Output layer
        y = ml.matmul(h, self.w2) + self.b2
        return y

# Step 2: Create model instance
model = MLP(input_size=2, hidden_size=10, output_size=1)

# Step 3: Create sample data
x = np.random.randn(5, 2)  # 5 samples, 2 features each

# Step 4: Initialize parameters
init_params = {
    "weight1": np.random.randn(2, 10) * 0.1,
    "bias1": np.zeros(10),
    "weight2": np.random.randn(10, 1) * 0.1,
    "bias2": np.zeros(1)
}

# Step 5: Compile and run with different backends
backends = {
    "NumPy": ml.backends.NumpyBackend(),
    "PyTorch": ml.backends.TorchBackend(),
    # Uncomment to try other backends
    # "JAX": ml.backends.JaxBackend(),
    # "MLX": ml.backends.MLXBackend()
}

for name, backend in backends.items():
    print(f"Running with {name} backend:")
    physical_model = model.compile(backend)
    physical_model.set_parameter_values(init_params)
    
    result = physical_model(x)
    print(f"Output shape: {result.shape}")
    print(f"Output sample: {result[0]}")
    print()
```

## Example 4: Creating a Custom Primitive Operation

Mithril allows you to define custom primitive operations. Here's how to create a custom activation function.

```python
import mithril as ml
import numpy as np

# Step 1: Define the custom operation
@ml.primitive
def swish(x, beta=1.0):
    """Swish activation function: x * sigmoid(beta * x)"""
    return x * ml.sigmoid(beta * x)

# Step 2: Define a model using this custom primitive
class SwishMLP(ml.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.w1 = self.add_parameter("weight1", shape=(input_size, hidden_size))
        self.b1 = self.add_parameter("bias1", shape=(hidden_size,))
        self.w2 = self.add_parameter("weight2", shape=(hidden_size, output_size))
        self.b2 = self.add_parameter("bias2", shape=(output_size,))
    
    def forward(self, x):
        # First layer with Swish activation
        h = ml.matmul(x, self.w1) + self.b1
        h = swish(h, beta=1.0)  # Using our custom activation
        
        # Output layer
        y = ml.matmul(h, self.w2) + self.b2
        return y

# Step 3: Create and compile the model
model = SwishMLP(input_size=2, hidden_size=10, output_size=1)
physical_model = model.compile(ml.backends.NumpyBackend())

# Step 4: Initialize parameters
physical_model.set_parameter_values({
    "weight1": np.random.randn(2, 10) * 0.1,
    "bias1": np.zeros(10),
    "weight2": np.random.randn(10, 1) * 0.1,
    "bias2": np.zeros(1)
})

# Step 5: Run inference
x = np.random.randn(5, 2)
result = physical_model(x)
print(f"Output with Swish activation: {result}")
```

## Example 5: Composing Models

Mithril makes it easy to compose models. Let's create a sequence of models and connect them.

```python
import mithril as ml
import numpy as np

# Step 1: Define component models
class FeatureExtractor(ml.Model):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.w = self.add_parameter("extractor_weight", shape=(input_size, hidden_size))
        self.b = self.add_parameter("extractor_bias", shape=(hidden_size,))
    
    def forward(self, x):
        h = ml.matmul(x, self.w) + self.b
        return ml.relu(h)

class Classifier(ml.Model):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.w = self.add_parameter("classifier_weight", shape=(hidden_size, output_size))
        self.b = self.add_parameter("classifier_bias", shape=(output_size,))
    
    def forward(self, h):
        return ml.matmul(h, self.w) + self.b

# Step 2: Compose the models
class CompositeModel(ml.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Create component models
        self.feature_extractor = FeatureExtractor(input_size, hidden_size)
        self.classifier = Classifier(hidden_size, output_size)
        
        # Compose the models using the pipe operator
        self |= self.feature_extractor
        self |= self.classifier
    
    def forward(self, x):
        # Extract features
        h = self.feature_extractor(x)
        # Classify
        return self.classifier(h)

# Step 3: Create and compile the model
model = CompositeModel(input_size=2, hidden_size=10, output_size=3)
physical_model = model.compile(ml.backends.NumpyBackend())

# Step 4: Initialize parameters
np.random.seed(42)
physical_model.set_parameter_values({
    "extractor_weight": np.random.randn(2, 10) * 0.1,
    "extractor_bias": np.zeros(10),
    "classifier_weight": np.random.randn(10, 3) * 0.1,
    "classifier_bias": np.zeros(3)
})

# Step 5: Run inference
x = np.random.randn(5, 2)
result = physical_model(x)
print(f"Composite model output shape: {result.shape}")
print(f"Composite model output: {result}")
```

These examples demonstrate the fundamental capabilities of Mithril. Once you're comfortable with these concepts, you can explore more advanced features and real-world models in the other documentation sections.