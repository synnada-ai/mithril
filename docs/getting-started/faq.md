# Frequently Asked Questions

This page answers common questions about Mithril, its usage, and troubleshooting tips.

## General Questions

### What is Mithril?

Mithril is a flexible machine learning library designed to simplify the composition and compilation of gradient-based models. It focuses on three core principles: versatile composability, framework-agnostic code generation, and easy parallelization.

### How does Mithril differ from other ML frameworks?

Unlike frameworks like PyTorch or TensorFlow that provide both model definition and execution, Mithril separates model architecture definition from execution details. This allows you to define a model once and compile it for different backends (JAX, PyTorch, NumPy, etc.) without changing the model definition.

### Is Mithril production-ready?

Mithril is still under active development. While it's stable for research and development use, some features are experimental and may change. Check the version notes for stability information.

### Who should use Mithril?

Mithril is ideal for:

- Researchers who want to experiment with model architectures without being tied to a specific framework
- ML engineers who need to deploy models across different platforms
- Developers who want more control over model composition and compilation
- Teams working with multiple ML frameworks who want a unified interface

## Installation and Setup

### What are the system requirements for Mithril?

Mithril requires Python 3.8 or later. Hardware requirements depend on the backends you plan to use (e.g., JAX with GPUs will require CUDA).

### How do I install Mithril with all backends?

```bash
pip install mithril[all] --upgrade
```

### Can I use Mithril with custom hardware?

Yes, Mithril supports custom hardware through its backend system. You can create custom backends for specific hardware or use existing backends optimized for different platforms (JAX for TPUs, PyTorch for GPUs, etc.).

### Is Mithril compatible with Python 3.7?

No, Mithril requires Python 3.8 or later due to its use of modern Python features like typed dictionaries.

## Model Creation

### How do I create a custom model?

You can create custom models by subclassing the `Model` class:

```python
from mithril.models import Model, Linear, Relu

class MyCustomModel(Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self |= Linear(dimension=hidden_dim)(input="input", output="hidden")
        self += Relu()(input="hidden", output="hidden_act")
        self += Linear(dimension=output_dim)(input="hidden_act", output="output")
```

### Can I use models from other frameworks with Mithril?

Mithril is designed to define and compile its own models, but you can:

1. Create Mithril models that mirror architectures from other frameworks
2. Use Mithril-compiled models within the execution context of other frameworks
3. In some cases, convert models from other frameworks to Mithril (experimental feature)

### How do I handle models with multiple inputs or outputs?

Simply define multiple input or output terminals in your model:

```python
model = Model()
model |= Linear(dimension=64)(input="image_input", output="image_features")
model += Linear(dimension=32)(input="text_input", output="text_features")
model += Concat()(inputs=["image_features", "text_features"], output="combined")
model += Linear(dimension=10)(input="combined", output="classification")
model += Linear(dimension=1)(input="combined", output="regression")
```

### How many layers/models can I compose together?

There's no hard limit on the number of layers or models you can compose. The practical limit depends on memory constraints and compilation time.

## Compilation

### Why does compilation take so long?

Compilation involves several steps including shape inference, optimization, and code generation. For complex models, this can take time. Enable caching to speed up subsequent compilations:

```python
compiled_model = ml.compile(model, backend, cache=True)
```

### How can I debug compilation errors?

Use the verbose flag to get detailed information about the compilation process:

```python
compiled_model = ml.compile(model, backend, verbose=True)
```

### Can I see the generated code?

Yes, you can output the generated code to a file:

```python
compiled_model = ml.compile(model, backend, file_path="generated_model.py")
```

### How do I handle dynamic shapes during compilation?

Use `None` for dimensions that should be dynamic:

```python
compiled_model = ml.compile(model, backend, shapes={"input": [None, 784]})
```

## Backends

### Which backend should I choose?

- **JAX**: For high-performance computing, especially on TPUs or when using JIT compilation
- **PyTorch**: For dynamic computation graphs and easy debugging
- **NumPy**: For simple CPU computations without automatic differentiation
- **MLX**: For Apple Silicon (M1/M2/M3) optimization

### Can I create my own backend?

Yes, you can create custom backends by implementing the `Backend` interface. See the documentation on [Custom Backends](../user-guide/custom-backends.md) for details.

### How do I switch between backends?

Simply compile the same model with different backends:

```python
jax_model = ml.compile(model, ml.JaxBackend())
torch_model = ml.compile(model, ml.TorchBackend())
```

### Can I use multiple backends in the same model?

Yes, you can specify different backends for different parts of your model using the `backend_map` parameter:

```python
compiled_model = ml.compile(
    model=model,
    backend=jax_backend,  # Default backend
    backend_map={"torch": torch_backend}  # Backend mapping
)
```

## Training

### How do I train a model in Mithril?

Mithril provides a flexible interface for training models. A basic training loop might look like:

```python
# Compute outputs and gradients
outputs, gradients = compiled_model.evaluate(
    params, 
    inputs, 
    output_gradients=compute_loss_gradient(y_pred, y_true)
)

# Update parameters using an optimizer
params = update_params(params, gradients, optimizer)
```

### Does Mithril provide built-in optimizers?

Mithril doesn't include built-in optimizers, but you can easily use optimizers from the backend frameworks (e.g., optax for JAX, torch.optim for PyTorch).

### How do I implement custom loss functions?

You compute gradients of your loss function and pass them as `output_gradients`:

```python
# Custom MSE loss gradient
def mse_loss_gradient(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_pred.shape[0]

# Compute outputs and gradients
outputs, gradients = compiled_model.evaluate(
    params, 
    inputs, 
    output_gradients={"output": mse_loss_gradient(outputs["output"], y_true)}
)
```

### Can I freeze certain parameters during training?

Yes, you can selectively update parameters by filtering the gradients:

```python
# Only update non-frozen parameters
for name, grad in gradients.items():
    if name not in frozen_params:
        params[name] = params[name] - learning_rate * grad
```

## Parallelization

### How do I train a model on multiple GPUs?

Use the device mesh feature with a PyTorch or JAX backend:

```python
# Create a backend with a 2-GPU device mesh
backend = ml.JaxBackend(device_mesh=(2,))

# Compile model
compiled_model = ml.compile(model, backend)

# Create sharded inputs
inputs = {"input": backend.ones(batch_size, features, device_mesh=(2,), tensor_split=(0,))}
```

### What's the difference between model parallelism and data parallelism?

- **Data Parallelism**: The same model is replicated across devices, each processing a different batch of data
- **Model Parallelism**: Different parts of the model are placed on different devices, processing the same data

### How do I choose between data and model parallelism?

- Use data parallelism when:
  - Your model fits on a single device
  - You want to process larger batch sizes
  - Computation is the bottleneck

- Use model parallelism when:
  - Your model is too large to fit on a single device
  - Memory is the bottleneck
  - You have layers with different computational requirements

### Can I combine data and model parallelism?

Yes, you can use a multi-dimensional device mesh to combine both approaches:

```python
# Create a 2D device mesh (2x2 = 4 devices)
backend = ml.JaxBackend(device_mesh=(2, 2))

# Use first dimension for model parallelism, second for data parallelism
weight = backend.ones(8192, 8192, device_mesh=(2, 1), tensor_split=(0, None))
inputs = {"input": backend.ones(batch_size, 8192, device_mesh=(1, 2), tensor_split=(None, 0))}
```

## Troubleshooting

### Common Errors and Solutions

#### "Incompatible shapes during compilation"

- **Cause**: Shape mismatch between connected model components
- **Solution**: Check your model's architecture and ensure tensor shapes are compatible between connections

#### "Terminal not found"

- **Cause**: Referencing a terminal that doesn't exist
- **Solution**: Verify all terminal names in your model connections

#### "Backend does not support operation X"

- **Cause**: Using an operation not supported by the chosen backend
- **Solution**: Use a different backend or implement the operation for your backend

#### "Out of memory during compilation"

- **Cause**: Model is too large for the available memory
- **Solution**: 
  - Use memory planning: `ml.compile(model, backend, memory_planning="greedy")`
  - Use lower precision: `ml.compile(model, ml.JaxBackend(dtype=ml.float16))`
  - Use gradient checkpointing: `ml.compile(model, backend, gradient_checkpointing=True)`

#### "Slow compilation times"

- **Cause**: Complex model or inefficient compilation
- **Solution**:
  - Enable caching: `ml.compile(model, backend, cache=True)`
  - Simplify model where possible
  - Use static shapes for better optimization

### How do I report bugs or request features?

You can report bugs or request features on the [Mithril GitHub repository](https://github.com/example/mithril).

## Miscellaneous

### Is Mithril open source?

Yes, Mithril is licensed under the Apache License 2.0.

### How can I contribute to Mithril?

Check out the [Contributing Guide](../contributing/development.md) for information on how to contribute to Mithril.

### Where can I get help with Mithril?

- Documentation: [https://example.com/mithril/docs](https://example.com/mithril/docs)
- GitHub: [https://github.com/example/mithril](https://github.com/example/mithril)
- Community Forum: [https://discuss.mithril.ai](https://discuss.mithril.ai)

### What's the origin of the name "Mithril"?

Mithril refers to a fictional metal in J.R.R. Tolkien's Middle-earth legendarium. It's described as being stronger than steel but much lighter, symbolizing the library's aim to be both powerful and flexible while maintaining a light API.