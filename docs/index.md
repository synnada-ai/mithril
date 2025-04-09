# Mithril: Transcend Frameworks, Embrace Composability

<div class="center-content" markdown>
![Mithril Logo](assets/logo.png){ width="300" }
</div>

## The Compiler Revolution for Machine Learning

Mithril is an Apache-licensed machine learning framework that brings the compiler revolution to ML development. By separating **what** your model does from **how** it executes, Mithril liberates developers from framework-specific constraints and enables true "write once, run anywhere" machine learning.

> "The future of machine learning development isn't about choosing the right framework—it's about transcending frameworks altogether."

## Key Principles

### Logical-Physical Separation

Mithril's core innovation is the clean separation between:

- **Logical Models**: Describe your model's architecture and connections
- **Physical Models**: Optimized implementations for specific hardware and frameworks

This separation—inspired by database query engines—eliminates the painful rewrites and optimizations that plague traditional ML workflows.

### Every Model is a Composable Block

In Mithril, everything from a simple `Linear` layer to a complete `ResNet` is a composable building block. This enables a truly hierarchical design approach:

```python
# Build a residual block
res_block = Model()
res_block |= Conv2d(64, 64)(output="conv1")
res_block += BatchNorm2d(64)(input="conv1", output="bn1")
res_block += ReLU()(input="bn1", output="relu1")
res_block += Conv2d(64, 64)(input="relu1", output="conv2")
res_block += BatchNorm2d(64)(input="conv2", output="conv2_bn")
res_block += Add()(left="input", right="conv2_bn", output="output")

# Use it in a larger model
model = Model()
model |= Conv2d(3, 64)
model += MaxPool2d(kernel_size=3, stride=2)
model += res_block
model += res_block
model += Flatten()
model += Linear(64, 10)
```

### Compile Once, Run Anywhere

Mithril compiles logical models into optimized implementations for any supported backend:

```python
# Same model, different backends
torch_model = ml.compile(model, ml.TorchBackend())  # PyTorch for development
jax_model = ml.compile(model, ml.JaxBackend())      # JAX for TPU training
ggml_model = ml.compile(model, ml.GGMLBackend())    # GGML for deployment
```

This eliminates the costly rewrites required when moving between research, scaling, and production environments.

## Break Free from the Framework Cycle

The traditional ML development cycle forces painful choices and trade-offs:

1. **Framework Lock-in**: "We must use TensorFlow because our serving infrastructure requires it"
2. **Knowledge Silos**: Research teams build expertise in frameworks different from production
3. **Late-Stage Surprises**: Deployment constraints force costly rewrites of working models
4. **Optimization Barriers**: Framework limitations prevent hardware-specific optimizations

Mithril breaks this cycle by making frameworks an implementation detail, not a fundamental architectural decision.

## Features That Matter

### Effortless Parallelism

Instead of baking parallelization into your model code, Mithril treats it as a compile-time concern:

```python
# Data parallelism across 8 GPUs
backend = ml.JaxBackend(device_mesh=(8,))
data_parallel = ml.compile(model, backend)  # That's it!

# Model parallelism for large models
backend = ml.JaxBackend(device_mesh=(4, 2))
model_parallel = ml.compile(model, backend)
```

### Unified Optimization

Optimizations happen at the compiler level, benefiting all models without requiring changes to their definitions:

```python
# Apply memory optimization, JIT compilation, and float16 precision
optimized = ml.compile(
    model=model,
    backend=ml.JaxBackend(precision="float16"),
    jit=True,
    memory_optimization="greedy"
)
```

### Framework Agnostic Training

Train models using a consistent API regardless of backend:

```python
# Define once, train anywhere
trainer = ml.Trainer(
    model=compiled_model,
    optimizer=ml.Adam(learning_rate=1e-3),
    loss=ml.CrossEntropyLoss()
)

# Train for 10 epochs
trainer.train(train_loader, epochs=10)
```

## Streamlined ML Development Pipeline

### Research & Prototyping

Compile with PyTorch for rapid iteration and easy debugging:

```python
# Fast development cycles with PyTorch
dev_model = ml.compile(model, ml.TorchBackend())
```

### Scaling & Training

Recompile with JAX for high-performance distributed training:

```python
# Scale to TPU pods with JAX
tpu_model = ml.compile(model, ml.JaxBackend(device="tpu"))
```

### Production & Deployment

Generate optimized code for deployment environments:

```python
# Lightweight deployment with GGML
prod_model = ml.compile(
    model, 
    ml.GGMLBackend(quantize="int8"),  # 8-bit quantization
    static_inference=True              # Optimized static shapes
)
```

## Getting Started

Mithril makes it easy to build powerful, portable ML models:

- [Installation](getting-started/installation.md)
- [Quick Start Guide](getting-started/quick-start.md)
- [Key Concepts](getting-started/key-concepts.md)
- [Examples](examples/basic.md)

---

**Mithril: Forge Your ML Future**