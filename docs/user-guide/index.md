# The Mithril User Guide

Welcome to the Mithril User Guide—your comprehensive resource for mastering the framework that's transforming how we develop machine learning models.

## A New Approach to ML Development

Mithril introduces a revolutionary paradigm shift in machine learning development:

1. **Define** logical models focused on architecture and connections
2. **Compile** to optimized physical implementations for any target
3. **Execute** efficiently on diverse hardware without rewriting code

This separation between logical model design and physical execution mirrors how query optimizers revolutionized database systems—allowing you to focus on what your model should compute, not how it's executed.

## The Mithril Development Lifecycle

![Development Lifecycle](/assets/mithril_lifecycle.png)

### Research & Design
Define your models with a focus on architecture and mathematical correctness, free from framework constraints.

### Development & Debugging
Compile to interactive backends like PyTorch for rapid iteration and easy debugging.

### Scaling & Training
Recompile to high-performance backends like JAX for distributed training and TPU acceleration.

### Deployment & Inference
Generate optimized code for production targets like GGML for efficient inference.

## Core Concepts

### Logical Models
Logical models are the architecture blueprints—pure descriptions of what operations should be performed and how they connect, without implementation details:

```python
# A logical model is framework-agnostic
encoder = Model()
encoder |= Conv2d(3, 64, kernel_size=3)
encoder += BatchNorm2d(64)
encoder += Relu()
```

Learn more in the [Logical Models](logical-models.md) section.

### Physical Models
Physical models are the compiled, executable implementations of logical models, optimized for specific backends:

```python
# Compile for different targets without changing the model
torch_model = ml.compile(model, ml.TorchBackend())
jax_model = ml.compile(model, ml.JaxBackend())
ggml_model = ml.compile(model, ml.GGMLBackend())
```

Explore further in the [Physical Models](physical-models.md) section.

### The Compilation Bridge
Compilation is the magic that transforms logical models into optimized physical implementations:

```python
# Compile with specific optimizations
optimized_model = ml.compile(
    model=model,
    backend=ml.JaxBackend(),
    jit=True,                # Enable JIT compilation
    static_shapes=True,      # Optimize for fixed shapes
    gradient_checkpointing=True  # Memory-efficient backprop
)
```

Discover more in the [Compilation](compilation.md) section.

## Using This Guide

This guide is structured to accompany you through your entire journey with Mithril:

### For New Users
Start with [Logical Models](logical-models.md) to understand the fundamental building blocks, then explore [Model Composition](model-composition.md) to learn how to connect them.

### For ML Engineers
Focus on [Backends](backends.md) and [Compilation](compilation.md) to understand how to optimize models for your specific use cases.

### For ML Researchers
Dive into [Custom Models](custom-models.md) and [Custom Primitives](custom-primitives.md) to extend Mithril for novel architectures.

### For Production Engineers
Explore [Performance Tuning](performance-tuning.md) and [Memory Optimization](memory-optimization.md) to optimize for deployment.

## Breaking Free from Framework Constraints

With Mithril, you're no longer forced to make painful choices based on framework limitations:

- **No more framework lock-in**: Define once, run anywhere
- **No more knowledge silos**: One model architecture for research and production
- **No more late-stage rewrites**: Compilation adapts to deployment targets
- **No more optimization barriers**: Backend-specific optimizations without changing models

## Overview of User Guide Sections

### Logical Models
Learn how to define model architectures, manage terminals, and create reusable components.

### Physical Models
Understand how models are flattened, optimized, and prepared for execution.

### Compilation
Explore the transformation from logical to physical models and available optimization options.

### Backends
Discover the different execution environments and their specific capabilities.

### Parallelization
Learn how to scale models across multiple devices without changing their architecture.

### Training
Master the unified training API that works consistently across backends.

### Advanced Topics
Dive deep into advanced features like conditional execution, custom primitives, and memory optimization.

---

Ready to transform your ML development workflow? Let's begin the journey.