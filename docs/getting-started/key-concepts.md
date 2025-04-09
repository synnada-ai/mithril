# Key Concepts: Breaking Free from Framework Constraints

Mithril introduces a paradigm shift in machine learning development. This page explains the core concepts that underpin Mithril's approach to liberating ML development from framework-specific constraints.

## The Great Divide: Logical vs. Physical Models

The fundamental insight of Mithril is the clean separation between model definition and execution:

### Logical Models

Logical models represent the abstract architecture of your machine learning model—what it should compute, not how. They define:

- Structure and components
- Connections and data flow
- Input/output relationships

They are framework-agnostic, focusing purely on the mathematical operations and their composition.

```python
# A logical model definition is framework-agnostic
model = Model()
model |= Linear(dimension=64)
model += Relu()
model += Linear(dimension=10)
```

### Physical Models

Physical models are the concrete, executable implementations of logical models, optimized for specific backends like JAX, PyTorch, or GGML. They handle:

- Specific tensor implementations
- Hardware acceleration
- Memory management
- Parallelization details
- Performance optimization

```python
# Same logical model compiled for different physical targets
torch_model = ml.compile(model, ml.TorchBackend())
jax_model = ml.compile(model, ml.JaxBackend())
numpy_model = ml.compile(model, ml.NumpyBackend())
```

This separation mirrors how database systems separate logical query plans from physical execution plans—a proven approach that enables optimization without changing the high-level specifications.

## Composition: The Building Blocks of Model Architecture

Mithril treats every model as a first-class composable building block, regardless of complexity. This enables true modular architecture design:

### Composition Operators

```python
# |= (pipe-assign): Connect to the input
model |= Linear(dimension=64)

# += (add-assign): Chain to the previous output
model += Relu()

# Explicit connections via function call syntax
model += Conv2d(in_channels=16, out_channels=32)(
    input="layer1",  # Take input from a specific terminal
    output="layer2"  # Send output to a named terminal
)
```

### Hierarchical Composition

Models can contain other models, creating a true component hierarchy:

```python
# Create a residual block
res_block = Model()
res_block |= Conv2d(64, 64)(output="conv1")
res_block += BatchNorm2d(64)(input="conv1", output="bn1")
res_block += ReLU()(input="bn1", output="relu1")
res_block += Conv2d(64, 64)(input="relu1", output="conv2")
res_block += Add()(left="input", right="conv2", output="output")

# Use it multiple times in a larger model
model = Model()
model |= Conv2d(3, 64)
model += res_block       # First residual block
model += res_block       # Second residual block (same definition)
model += GlobalAvgPool2d()
model += Linear(64, 10)
```

This hierarchical composition enables:
- Reusable component libraries
- Team collaboration around well-defined interfaces
- Model architecture as a true design discipline

## Terminals: Named Connection Points

Terminals are named connection points that allow for explicit data flow in your model:

```python
# A model with explicit input/output terminals
model = Model()
model |= Conv2d(3, 64)(input="image", output="features")
model += MaxPool2d()(input="features", output="pooled")
model += Flatten()(input="pooled", output="flat_features")
model += Linear(64 * 7 * 7, 10)(input="flat_features", output="logits")
```

This explicit naming enables:
- Complex, non-sequential architectures
- Multiple inputs and outputs
- Skip connections, residual paths, and other advanced patterns

Every model has default input and output terminals, but you can create and reference any custom terminals when needed.

## Backends: The Engine Room

Backends are the execution environments that implement the operations defined in your logical models:

```python
# Different backends for different needs
jax_backend = ml.JaxBackend(dtype=ml.float32, device="gpu")
torch_backend = ml.TorchBackend(dtype=ml.float32, device="cuda:0")
mlx_backend = ml.MLXBackend(dtype=ml.float32)  # Apple Silicon
numpy_backend = ml.NumpyBackend(dtype=ml.float64)  # CPU for debugging
ggml_backend = ml.GGMLBackend(quantize="int8")  # Inference deployment
```

Backends handle:
- Tensor creation and operations
- Device placement (CPU/GPU/TPU)
- Mathematical computations
- Automatic differentiation (for backends that support it)
- Memory management

This abstraction allows you to write your model once and run it anywhere, from research environments to production deployment.

## Compilation: The Transformation Process

Compilation is the process that bridges logical models and physical execution:

```python
# Basic compilation
compiled_model = ml.compile(model, backend)

# Compilation with additional information
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 3, 224, 224]},  # Batch of 32 images
    jit=True,  # Enable JIT compilation
    static_shapes=True  # Optimize for fixed shapes
)
```

During compilation, Mithril:

1. **Flattens** the model hierarchy
2. **Analyzes** data dependencies
3. **Infers** shapes and types
4. **Optimizes** the computation graph
5. **Generates** efficient code for the target backend

This compilation step is what enables the logical-physical separation at the heart of Mithril.

## Parallelization: A Configuration Detail, Not a Design Constraint

In Mithril, parallelism is a compile-time configuration detail, not something that must be embedded in your model design:

```python
# Data parallelism for training
data_parallel = ml.JaxBackend(device_mesh=(8,))  # 8-way data parallelism
parallel_model = ml.compile(model, data_parallel)

# Model parallelism for large models
model_parallel = ml.JaxBackend(device_mesh=(2, 4))  # 2D parallelism
sharded_model = ml.compile(large_model, model_parallel)
```

This approach allows:
- Separation of model architecture from execution strategy
- Easy experimentation with different parallelism approaches
- Adaptation to available hardware without model changes
- Combining different forms of parallelism

## Training: Unified API Across Backends

Mithril provides a consistent training API regardless of backend:

```python
# Create a trainer
trainer = ml.Trainer(
    model=compiled_model,
    optimizer=ml.Adam(learning_rate=0.001),
    loss=ml.CrossEntropyLoss()
)

# Train for 10 epochs
trainer.train(train_loader, epochs=10)

# Evaluate
metrics = trainer.evaluate(test_loader)
```

This consistent API eliminates the need to learn different training patterns for different frameworks.

## The Whole Picture: A New ML Development Workflow

With these concepts combined, Mithril enables a transformation in how ML models are developed:

1. **Design Phase**: Create logical models focusing on architecture, not implementation
2. **Development Phase**: Compile with interactive backends like PyTorch for rapid iteration
3. **Scaling Phase**: Recompile with high-performance backends like JAX for distributed training
4. **Deployment Phase**: Generate optimized code for production environments

This workflow eliminates the framework silos, knowledge gaps, and costly rewrites that plague traditional ML development.

## Beyond the Basics

These core concepts are just the beginning. Mithril also provides:

- **Conditional Execution**: Dynamic behavior based on inputs or training mode
- **Custom Primitives**: Extend Mithril with your own operations
- **Code Generation**: Generate standalone code for deployment
- **Shape and Type Inference**: Automatic propagation of constraints
- **Memory Optimization**: Smart memory management techniques

Each of these features builds on the same fundamental insight: separating what your model computes from how it executes enables a more flexible, maintainable, and efficient ML development process.

## Next Steps

Now that you understand the core concepts of Mithril, you can:

- Learn more about [Logical Models](../user-guide/logical-models.md)
- Explore different [Backends](../user-guide/backends.md)
- Understand the [Compilation](../user-guide/compilation.md) process
- Check out the [Examples](../examples/basic.md) section for practical applications