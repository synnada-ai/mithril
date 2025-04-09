# Multi-Backend Deployment with Mithril

This guide explores how Mithril enables the same model definition to run efficiently across multiple computational backends, demonstrating the framework's versatility for different deployment scenarios.

## Understanding Mithril's Backend Architecture

Mithril's core design separates model definitions from execution environments through:

1. **Logical Models**: Backend-agnostic representations of computational graphs
2. **Physical Models**: Compiled versions optimized for specific backends
3. **Backends**: Execution engines that implement the actual computation

This separation allows developers to define models once and run them anywhere, from development to production.

## Supported Backends

Mithril currently supports multiple backends:

- **PyTorch**: Full-featured backend with CPU and GPU support
- **JAX**: Optimized for XLA, TPUs, and high-performance computing
- **NumPy**: CPU-only backend for simplicity and compatibility
- **MLX**: Specialized for Apple Silicon devices
- **Raw C/GGML**: Low-level backends for edge deployment

## Running Models Across Different Backends

### Same Model, Multiple Backends

Here's how to run the same model on different backends:

```python
import mithril as ml
from mithril.models import resnet18

# Create a logical model
model = resnet18(num_classes=1000)

# Compile and run with PyTorch backend
torch_backend = ml.TorchBackend(device="cuda")
torch_model = ml.compile(
    model,
    backend=torch_backend,
    shapes={"input": [1, 3, 224, 224]},
    data_keys={"input"}
)

# Compile and run with JAX backend
jax_backend = ml.JaxBackend()
jax_model = ml.compile(
    model,
    backend=jax_backend,
    shapes={"input": [1, 3, 224, 224]},
    data_keys={"input"}
)

# Run with MLX backend on Apple Silicon
mlx_backend = ml.MlxBackend()
mlx_model = ml.compile(
    model,
    backend=mlx_backend,
    shapes={"input": [1, 3, 224, 224]},
    data_keys={"input"}
)
```

The model definition remains identical—only the backend changes.

### Benchmarking Across Backends

To compare performance across backends, use this simple benchmarking approach:

```python
import time
import numpy as np

def benchmark_model(model_name, input_shape, backends, num_runs=100):
    # Create the model
    if model_name == "resnet18":
        from mithril.models import resnet18
        model = resnet18(num_classes=1000)
    elif model_name == "transformer":
        from examples.gpt.model import create_gpt
        model = create_gpt(bias=True, block_size=128, dims=768, 
                          num_heads=12, num_layers=12, vocab_size=50304)
    
    # Prepare input data (once, as numpy)
    numpy_input = np.random.randn(*input_shape).astype(np.float32)
    
    results = {}
    for backend_name, backend_class in backends.items():
        backend = backend_class()
        
        # Compile model
        compiled_model = ml.compile(
            model,
            backend=backend,
            shapes={"input": input_shape},
            data_keys={"input"},
            jit=True
        )
        
        # Convert input to backend format
        backend_input = backend.array(numpy_input)
        
        # Warmup
        for _ in range(10):
            compiled_model.evaluate({}, {"input": backend_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            compiled_model.evaluate({}, {"input": backend_input})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        results[backend_name] = avg_time
    
    return results

# Run benchmark
backends = {
    "torch": ml.TorchBackend,
    "jax": ml.JaxBackend, 
    "numpy": ml.NumpyBackend
}
results = benchmark_model("resnet18", [1, 3, 224, 224], backends)
print(results)
```

This approach helps identify the optimal backend for specific models and deployment scenarios.

## Cross-Backend Transfer Learning

Mithril enables seamless weight transfer between backends:

```python
# Train on PyTorch backend
torch_backend = ml.TorchBackend(device="cuda")
torch_model = ml.compile(model, backend=torch_backend, data_keys={"input", "target"})
torch_params = train_model(torch_model, torch_backend, training_data)

# Convert parameters to numpy
numpy_params = {k: v.cpu().numpy() for k, v in torch_params.items()}

# Deploy on JAX backend
jax_backend = ml.JaxBackend()
jax_model = ml.compile(model, backend=jax_backend, data_keys={"input"})
jax_params = {k: jax_backend.array(v) for k, v in numpy_params.items()}

# Run inference with trained weights
outputs = jax_model.evaluate(jax_params, {"input": jax_input})
```

This pattern allows training on high-performance backends like PyTorch or JAX, then deploying to specialized backends like MLX or C/GGML.

## Backend-Specific Optimizations

While keeping the same logical model, you can apply backend-specific optimizations during compilation:

```python
# JAX backend with XLA compilation
jax_model = ml.compile(
    model,
    backend=ml.JaxBackend(),
    data_keys={"input"},
    jit=True,  # Enable JAX JIT compilation
    auto_scalar_promotion=True,  # Automatic handling of scalar types
)

# PyTorch backend with TorchScript
torch_model = ml.compile(
    model,
    backend=ml.TorchBackend(),
    data_keys={"input"},
    jit=True,  # Enable TorchScript
)

# MLX backend with graph capture
mlx_model = ml.compile(
    model,
    backend=ml.MlxBackend(),
    data_keys={"input"},
    jit=True,  # Enable MLX graph compilation
)
```

Each backend leverages its native compilation strategy while preserving the model's logical structure.

## Multi-Device Deployment

Mithril supports multi-device deployment strategies:

```python
# Data-parallel training across multiple GPUs
multi_gpu_backend = ml.TorchBackend(parallel_strategy="data_parallel", 
                                   devices=["cuda:0", "cuda:1"])
model = ml.compile(
    resnet18(num_classes=1000),
    backend=multi_gpu_backend,
    data_keys={"input", "target"}
)

# Model-parallel deployment across devices
model_parallel_backend = ml.TorchBackend(parallel_strategy="model_parallel",
                                        devices=["cuda:0", "cuda:1"])
```

This flexibility enables scaling models beyond single-device constraints.

## Real-World Multi-Backend Examples

### CLIP Model across Backends

CLIP (Contrastive Language-Image Pre-training) works across backends:

```python
from examples.clip.model import clip

# Define model once
clip_model = clip(
    embed_dim=512,
    image_resolution=224,
    vision_layers=(3, 4, 6, 3),  # ResNet-50
    vision_width=64,
    vision_patch_size=0,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
)

# Shapes for compilation
shapes = {"image": [1, 3, 224, 224], "text": [1, 77]}

# PyTorch compilation
torch_backend = ml.TorchBackend(device="cuda" if torch.cuda.is_available() else "cpu")
torch_clip = ml.compile(clip_model, torch_backend, shapes=shapes, 
                        data_keys={"image", "text"})

# JAX compilation (on TPU or GPU)
jax_backend = ml.JaxBackend()
jax_clip = ml.compile(clip_model, jax_backend, shapes=shapes, 
                      data_keys={"image", "text"})

# Apple Silicon acceleration with MLX
mlx_backend = ml.MlxBackend()  
mlx_clip = ml.compile(clip_model, mlx_backend, shapes=shapes, 
                     data_keys={"image", "text"})
```

### NLP Models with Different Backends

T5 and GPT models also benefit from multi-backend support:

```python
from examples.t5 import t5_encode, t5_decode
from examples.gpt.model import create_gpt

# T5 Encoder-Decoder with different backends
encoder_lm = t5_encode(config)
decoder_lm = t5_decode(config)

# JAX backend for encoder, PyTorch for decoder
encoder = ml.compile(encoder_lm, ml.JaxBackend(), data_keys={"input"})
decoder = ml.compile(decoder_lm, ml.TorchBackend(), data_keys={"input", "memory"})

# GPT model running on Apple Silicon
gpt = create_gpt(bias=True, block_size=100, dims=768, num_heads=12, 
                num_layers=12, vocab_size=50304)
mlx_gpt = ml.compile(gpt, ml.MlxBackend(), data_keys={"input"})
```

## Best Practices for Multi-Backend Deployment

1. **Keep Models Backend-Agnostic**: Avoid backend-specific operations in logical models
2. **Test All Target Backends**: Verify model behavior across all deployment backends
3. **Benchmark Performance**: Identify the optimal backend for specific workloads
4. **Consider Precision Requirements**: Some backends may have different precision behaviors
5. **Leverage JIT Compilation**: Enable just-in-time compilation where possible
6. **Monitor Memory Usage**: Different backends have different memory patterns
7. **Understand Device Constraints**: Be aware of platform-specific limitations

## Conclusion

Mithril's multi-backend architecture provides exceptional flexibility for deploying models across diverse computing environments. This approach allows researchers and engineers to:

1. Develop models with high-productivity frameworks
2. Train on high-performance hardware
3. Deploy to specialized environments
4. Adapt to hardware changes without rewriting models

By separating model definitions from execution environments, Mithril enables true "write once, run anywhere" machine learning development.