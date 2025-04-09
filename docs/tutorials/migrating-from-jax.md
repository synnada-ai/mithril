# Migrating from JAX to Mithril

This guide helps JAX users transition to Mithril, showing how familiar JAX patterns map to Mithril's logical-physical separation approach while maintaining the performance benefits of JAX's transformations and compilation.

## Mental Model Shift

The primary shift when moving from JAX to Mithril is separating model definition from execution details:

| JAX Pattern | Mithril Approach |
|-------------|------------------|
| Functions for model definition with manual parameter passing | `Model` objects with explicit connections |
| Manual function transformations (`jit`, `vmap`, `pmap`) | Transformations handled during compilation |
| Explicit PRNG key management | PRNG management handled by backend |
| Manual parameter initialization | Parameter initialization through compiled model |
| Custom gradient functions | Backend abstraction for gradients |

## From JAX Functions to Mithril Models

### JAX Style

```python
import jax
import jax.numpy as jnp
from jax import random

def init_layer(key, in_dim, out_dim):
    k1, k2 = random.split(key)
    w_key, b_key = random.split(k1)
    w = random.normal(w_key, (in_dim, out_dim)) * 0.01
    b = random.normal(b_key, (out_dim,)) * 0.01
    return w, b

def forward_layer(params, x):
    w, b = params
    return jnp.dot(x, w) + b

def init_mlp(key, layer_sizes):
    keys = random.split(key, len(layer_sizes) - 1)
    params = []
    for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        params.append(init_layer(keys[i], m, n))
    return params

def forward_mlp(params, x):
    activations = x
    for i, layer_params in enumerate(params[:-1]):
        activations = forward_layer(layer_params, activations)
        activations = jax.nn.relu(activations)
    return forward_layer(params[-1], activations)

# Initialize and use MLP
key = random.PRNGKey(0)
layer_sizes = [784, 512, 256, 10]
params = init_mlp(key, layer_sizes)
x = random.normal(random.PRNGKey(1), (32, 784))
output = forward_mlp(params, x)

# JIT compilation
fast_forward_mlp = jax.jit(forward_mlp)
output = fast_forward_mlp(params, x)
```

### Mithril Style

```python
import mithril as ml
from mithril.models import Model, Linear, Relu

# Define logical model
def create_mlp(layer_sizes):
    model = Model()
    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if i == 0:
            model |= Linear(dimension=out_size)(input_size=in_size)
        else:
            model += Linear(dimension=out_size)(input_size=in_size)
        
        if i < len(layer_sizes) - 2:
            model += Relu()
    
    return model

# Create logical model
layer_sizes = [784, 512, 256, 10]
model = create_mlp(layer_sizes)

# Compile for JAX backend
backend = ml.JaxBackend(dtype=ml.float32)
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 784]},
    jit=True  # Use JAX JIT compilation
)

# Initialize parameters
params = compiled_model.randomize_params()

# Create input and run model
inputs = {"input": backend.randn(32, 784)}
outputs = compiled_model.evaluate(params, inputs)
output = outputs["output"]
```

## JAX Transformations and Mithril Compilation

### JAX Transformations

```python
# JIT compilation
jitted_forward = jax.jit(forward_mlp)

# Vectorization
batched_forward = jax.vmap(forward_mlp, in_axes=(None, 0))

# Parallelization
parallel_forward = jax.pmap(forward_mlp, axis_name='batch')

# Automatic differentiation
grad_fn = jax.grad(loss_fn)

# Combined transformations
training_step = jax.jit(jax.grad(loss_fn))
```

### Mithril Compilation Options

```python
# JIT compilation
jitted_model = ml.compile(
    model=model,
    backend=ml.JaxBackend(),
    jit=True
)

# Data parallelism (similar to vmap + pmap)
data_parallel = ml.compile(
    model=model,
    backend=ml.JaxBackend(device_mesh=(8,)),  # 8-way data parallelism
    parallelism="data"
)

# Model parallelism
model_parallel = ml.compile(
    model=model,
    backend=ml.JaxBackend(device_mesh=(1, 4)),  # 4-way model parallelism
    parallelism="model"
)

# Mixed parallelism
mixed_parallel = ml.compile(
    model=model,
    backend=ml.JaxBackend(device_mesh=(2, 2)),  # 2x2 mixed parallelism
    parallelism="mixed"
)

# Automatic differentiation (built into evaluate)
outputs, gradients = compiled_model.evaluate(params, inputs, output_gradients=grads)
```

## Convolutional Networks

### JAX CNN 

```python
def init_conv_layer(key, in_channels, out_channels, kernel_size=3):
    k1, k2 = random.split(key)
    w_shape = (kernel_size, kernel_size, in_channels, out_channels)
    w = random.normal(k1, w_shape) * 0.01
    b = random.normal(k2, (out_channels,)) * 0.01
    return w, b

def conv_forward(params, x):
    w, b = params
    y = jax.lax.conv(x, w, (1, 1), 'SAME')
    return y + b

def init_cnn(key):
    k1, k2, k3, k4, k5 = random.split(key, 5)
    
    # Conv layers
    conv1 = init_conv_layer(k1, 3, 16)
    conv2 = init_conv_layer(k2, 16, 32)
    
    # Fully connected layer
    w_fc = random.normal(k3, (32 * 8 * 8, 10)) * 0.01
    b_fc = random.normal(k4, (10,)) * 0.01
    
    return [conv1, conv2, (w_fc, b_fc)]

def cnn_forward(params, x):
    # First conv block
    y = conv_forward(params[0], x)
    y = jax.nn.relu(y)
    y = jax.lax.max_pool(y, (2, 2), (2, 2), 'VALID')
    
    # Second conv block
    y = conv_forward(params[1], y)
    y = jax.nn.relu(y)
    y = jax.lax.max_pool(y, (2, 2), (2, 2), 'VALID')
    
    # Flatten
    y = y.reshape((y.shape[0], -1))
    
    # Fully connected
    w_fc, b_fc = params[2]
    y = jnp.dot(y, w_fc) + b_fc
    
    return y

# Use the CNN
key = random.PRNGKey(0)
params = init_cnn(key)
x = random.normal(random.PRNGKey(1), (32, 32, 32, 3))  # NHWC format
output = cnn_forward(params, x)
```

### Mithril CNN

```python
from mithril.models import Conv2d, MaxPool2d, Flatten

def create_cnn():
    model = Model()
    
    # First conv block
    model |= Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same')
    model += Relu()
    model += MaxPool2d(kernel_size=2, stride=2)
    
    # Second conv block
    model += Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
    model += Relu()
    model += MaxPool2d(kernel_size=2, stride=2)
    
    # Flatten and FC
    model += Flatten()
    model += Linear(dimension=10)(input_size=32*8*8)
    
    return model

# Create and compile model
model = create_cnn()
backend = ml.JaxBackend()
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 3, 32, 32]},  # NCHW format (Mithril's default)
    jit=True
)

# Initialize and run
params = compiled_model.randomize_params()
inputs = {"input": backend.randn(32, 3, 32, 32)}
outputs = compiled_model.evaluate(params, inputs)
```

## Training and Optimization

### JAX Training

```python
def loss_fn(params, x, y):
    logits = forward_mlp(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=1))

@jax.jit
def update(params, x, y, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# Setup optimizer
import optax
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Training loop
for epoch in range(10):
    for x_batch, y_batch in data_loader:
        params, opt_state, loss = update(params, x_batch, y_batch, opt_state)
        print(f"Loss: {loss}")
```

### Mithril Training

```python
# Define loss function
def compute_loss_and_grad(compiled_model, params, inputs, targets):
    # Forward pass
    outputs = compiled_model.evaluate(params, inputs)
    logits = outputs["output"]
    
    # Compute loss
    loss = backend.cross_entropy(logits, targets)
    
    # Compute gradients for output
    grad = backend.cross_entropy_grad(logits, targets)
    output_gradients = {"output": grad}
    
    # Compute gradients through the model
    _, gradients = compiled_model.evaluate(params, inputs, output_gradients)
    
    return loss, gradients

# Initialize optimizer
import optax
learning_rate = 0.001
optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)

# Training loop
for epoch in range(10):
    for x_batch, y_batch in data_loader:
        # Convert to backend format
        inputs = {"input": backend.array(x_batch)}
        targets = backend.array(y_batch)
        
        # Compute loss and gradients
        loss, gradients = compute_loss_and_grad(
            compiled_model, params, inputs, targets
        )
        
        # Update parameters
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        
        print(f"Loss: {loss}")
```

## Handling PRNG Keys

### JAX PRNG Management

```python
def init_model(key):
    k1, k2, k3 = random.split(key, 3)
    # Use different keys for different parts
    params1 = init_part1(k1)
    params2 = init_part2(k2)
    params3 = init_part3(k3)
    return [params1, params2, params3]
    
# Generate initial key
key = random.PRNGKey(42)
model_key, dropout_key = random.split(key)
params = init_model(model_key)

# For stochastic operations like dropout
def forward_with_dropout(params, x, key):
    # ... use key for dropout mask
    dropout_mask = random.bernoulli(key, 0.5, x.shape)
    return x * dropout_mask / 0.5
```

### Mithril PRNG Management

```python
# Backend handles PRNG key management
backend = ml.JaxBackend(seed=42)
compiled_model = ml.compile(model, backend)

# Initialize with deterministic randomness
params = compiled_model.randomize_params()

# Stochastic operations like dropout use the backend's PRNG
model = Model()
model |= Linear(dimension=64)
model += Relu()
model += Dropout(p=0.5)(training=True)  # Training flag determines behavior
model += Linear(dimension=10)
```

## Custom Operations and Extensions

### JAX Custom Operations

```python
@jax.custom_vjp
def custom_op(x):
    return jnp.sin(x) * x

# Forward pass
def custom_op_fwd(x):
    return custom_op(x), x

# Backward pass (custom gradient)
def custom_op_bwd(res, grad):
    x = res
    return (grad * (jnp.sin(x) + x * jnp.cos(x)),)

custom_op.defvjp(custom_op_fwd, custom_op_bwd)
```

### Mithril Custom Primitives

```python
from mithril.models import Operator

# Define custom operator
class SinTimeX(Operator):
    def __init__(self):
        super().__init__()
    
    # Define the computation
    def compute(self, inputs):
        x = inputs["input"]
        return {"output": backend.sin(x) * x}
        
# Register with gradient
@ml.register_primitive("sin_times_x")
def sin_times_x(backend, x):
    return backend.sin(x) * x

@ml.register_primitive("sin_times_x_grad")
def sin_times_x_grad(backend, x, grad):
    return grad * (backend.sin(x) + x * backend.cos(x))

# Use in model
model = Model()
model |= Linear(dimension=64)
model += Relu()
model += SinTimeX()
model += Linear(dimension=10)
```

## Advanced JAX Features in Mithril

### XLA Compilation Control

```python
# JAX with XLA flags
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', True)

# Mithril with XLA options
backend = ml.JaxBackend(
    precision="float64",  # Use double precision
    xla_flags=["--xla_gpu_autotune_level=4"]  # XLA tuning
)

compiled_model = ml.compile(
    model=model,
    backend=backend,
    jit=True,
    jit_options={
        "static_argnums": (0,),
        "donate_argnums": (0,),
        "xla_computation": True  # Get access to XLA computation
    }
)
```

### TPU Usage

```python
# JAX on TPU
import jax
devices = jax.devices('tpu')
sharded_params = jax.device_put_replicated(params, devices)
results = jax.pmap(forward_fn)(sharded_params, sharded_inputs)

# Mithril on TPU
backend = ml.JaxBackend(
    device="tpu",
    device_mesh=(8,)  # 8 TPU cores
)

compiled_model = ml.compile(
    model=model,
    backend=backend,
    tpu_options={
        "replicate_inputs": True,
        "allow_spmd": True
    }
)
```

## JAX-Specific Optimization in Mithril

```python
# Use JAX's powerful optimizations through Mithril
backend = ml.JaxBackend(
    precision="bfloat16",  # Use TPU-optimized precision
    device="tpu"
)

compiled_model = ml.compile(
    model=model,
    backend=backend,
    jit=True,
    jit_options={
        "donate_argnums": (0,),  # Buffer reuse
    },
    static_shapes=True,  # Optimize for fixed shapes
    optimizations={
        "fusion": True,  # Operation fusion
        "buffer_reuse": True,  # Buffer reuse
        "layout_optimization": True  # Tensor layout optimization
    }
)
```

## Converting Haiku/Flax Models

### Haiku/Flax Model

```python
import haiku as hk

def haiku_fn(x):
    mlp = hk.nets.MLP([512, 256, 10])
    return mlp(x)

# Transform into pure functions
model = hk.transform(haiku_fn)

# Initialize
key = jax.random.PRNGKey(42)
dummy_input = jnp.zeros((1, 784))
params = model.init(key, dummy_input)

# Forward pass
outputs = model.apply(params, key, dummy_input)
```

### Mithril Equivalent

```python
# Create equivalent model in Mithril
def create_mlp():
    model = Model()
    model |= Linear(dimension=512)
    model += Relu()
    model += Linear(dimension=256)
    model += Relu()
    model += Linear(dimension=10)
    return model

# Import parameters from Haiku/Flax
def import_haiku_params(haiku_params, compiled_model):
    # Map Haiku parameter names to Mithril names
    # This requires knowledge of both parameter structures
    mithril_params = {}
    
    # Example mapping (actual mapping depends on model structure)
    for layer_idx, layer in enumerate(compiled_model.get_layer_names()):
        if f"mlp/linear_{layer_idx}" in haiku_params:
            haiku_layer = haiku_params[f"mlp/linear_{layer_idx}"]
            mithril_params[f"linear_{layer_idx}.weight"] = backend.array(haiku_layer["w"])
            mithril_params[f"linear_{layer_idx}.bias"] = backend.array(haiku_layer["b"])
    
    return mithril_params
```

## Key Advantages Over Pure JAX

1. **Modular Composition**: Clearer model structure with explicit connections
2. **Framework Flexibility**: The same model works across JAX, PyTorch, and more
3. **Simplified Parallelism**: Change parallelism strategy without rewriting models
4. **Automatic Parameter Management**: No manual parameter handling
5. **Separation of Concerns**: Architecture design separate from execution details

## Conclusion

Transitioning from JAX to Mithril lets you maintain JAX's performance advantages while gaining Mithril's logical-physical separation and cross-framework compatibility. The mental shift involves moving from functional programming with explicit parameters to object composition with managed parameters.

Key differences to remember:
- JAX uses functions and explicit parameter passing; Mithril uses models and connections
- JAX requires explicit transformation application; Mithril applies transformations during compilation
- JAX needs explicit PRNG management; Mithril handles randomness internally
- JAX parameters require manual structure; Mithril manages parameter organization

You can still use JAX's powerful features through Mithril's JAX backend while gaining the benefits of framework agnosticism and cleaner composition.