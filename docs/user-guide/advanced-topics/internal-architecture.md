# Internal Architecture

This document provides a deep dive into Mithril's internal architecture, explaining how the various components work together to provide a flexible, backend-agnostic machine learning framework.

## Architectural Overview

Mithril's architecture is built around a clear separation between model definition and execution, with several layers of abstraction:

```
                ┌─────────────────────┐
                │ Logical Model Layer │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Compilation       │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Physical Model Layer│
                └─────────────────────┘
                           │
                           ▼
┌───────────────┐  ┌─────────────────┐  ┌───────────────┐
│ JAX Backend   │  │ PyTorch Backend │  │ Other Backends│
└───────────────┘  └─────────────────┘  └───────────────┘
```

The flow from logical model to execution follows these steps:
1. User defines a logical model
2. Logical model is compiled into a physical model
3. Physical model is executed on a specific backend

## Core Components

### 1. Logical Model Layer

The logical model layer is responsible for model definition and composition. It provides a high-level API for creating and connecting model components.

#### Key Classes

- **Model**: The base class for all models, with methods for composition
- **Operator**: The base class for primitive operations
- **Connection**: Represents data flow between operations
- **Terminal**: Named endpoints for model inputs and outputs

#### Example: Model Definition

```python
model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")
model += Relu()(input="hidden", output="relu")
model += Linear(dimension=10)(input="relu", output="output")
```

### 2. Compilation Layer

The compilation layer transforms logical models into executable physical models. This is where much of the framework's magic happens.

#### Compilation Process

1. **Flattening**: Transforms nested logical models into a flat graph structure
2. **Name Resolution**: Maps logical names to physical names
3. **Constraint Propagation**: Infers shapes, types, and other constraints
4. **Graph Optimization**: Prunes duplicate and unused operations
5. **Code Generation**: Creates backend-specific code

#### Constraint System

Mithril's constraint system enables powerful inference capabilities:

```python
# Constrains are propagated automatically
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 10]}  # Only need to specify input shape
)
```

Constraints propagate through the graph, inferring:
- Tensor shapes
- Data types
- Memory requirements
- Differentiability

### 3. Physical Model Layer

The physical model layer is the executable representation of the logical model, optimized for a specific backend.

#### Key Components

- **PhysicalModel**: The executable model with methods for evaluation
- **FlatGraph**: Internal representation of the computation graph
- **DataStore**: Manages tensor data during execution

#### Execution Flow

1. Inputs are provided to the model
2. Data flows through the graph according to connections
3. Operations are executed in topological order
4. Results are collected from output terminals
5. (If requested) Gradients are computed via backpropagation

### 4. Backend Layer

The backend layer provides the implementation of tensor operations for specific frameworks.

#### Backend Responsibilities

- Tensor creation and manipulation
- Mathematical operations
- Device placement (CPU/GPU/TPU)
- Memory management
- Automatic differentiation

#### Backend Interface

All backends implement a common interface, ensuring consistent behavior:

```python
class Backend:
    def zeros(self, *shape, dtype=None): ...
    def ones(self, *shape, dtype=None): ...
    def array(self, data, dtype=None): ...
    def add(self, a, b): ...
    def matmul(self, a, b): ...
    # Many more operations...
```

## Code Generation

Mithril's code generation system allows for translating models to different target frameworks and languages.

### Code Generation Process

1. Model is analyzed to determine required operations
2. Operations are mapped to backend-specific implementations
3. Code templates are filled with appropriate operations
4. Generated code is compiled and linked to the runtime

### Supported Targets

- **Python-based**: JAX, PyTorch, NumPy, MLX
- **C-based**: Raw C, GGML (for efficient inference)

## Memory Management

Memory management differs between backends and compilation modes:

### Backends with Automatic Differentiation

JAX, PyTorch, and MLX backends leverage their built-in memory management:
- Automatic tensor allocation and deallocation
- Hardware-specific optimizations
- Memory pooling and caching

### Backends with Manual Gradient Computation

NumPy, C, and GGML backends require manual management:
- Explicit tensor allocation
- Manual buffer management
- Custom gradient computation

## Parallelization Architecture

Mithril's parallelization system provides a unified API across backends:

### Device Mesh

The device mesh abstraction allows for expressing complex parallelism strategies:

```python
# 2D mesh for model and data parallelism
backend = ml.JaxBackend(device_mesh=(2, 4))
```

### Parallelism Modes

- **Data Parallelism**: Batch split across devices
- **Model Parallelism**: Parameters split across devices
- **Pipeline Parallelism**: Model layers split across devices
- **Mixed Parallelism**: Combinations of the above

## Advanced Compilation Features

### JIT Compilation

Just-in-time compilation optimizes execution by compiling model functions at runtime:

```python
compiled_model = ml.compile(
    model=model,
    backend=backend,
    jit=True  # Enable JIT compilation
)
```

### Static Inference

Static inference pre-computes parts of the model that don't depend on inputs:

```python
compiled_model = ml.compile(
    model=model,
    backend=backend,
    static_inference=True
)
```

### Memory Planning

Memory planning optimizes buffer allocation and reuse:

```python
compiled_model = ml.compile(
    model=model,
    backend=backend,
    memory_planning="greedy"
)
```

## Internal Data Structures

### Physical Model Representation

The physical model structure:

```
PhysicalModel
├── FlatGraph
│   ├── Node1 (Operation)
│   ├── Node2 (Operation)
│   ├── Edge1 (Data)
│   └── Edge2 (Data)
├── DataStore
│   ├── Parameter1
│   ├── Parameter2
│   └── Intermediate tensors
├── Inputs
│   ├── Input1
│   └── Input2
└── Outputs
    ├── Output1
    └── Output2
```

### Execution Context

The execution context maintains:
- Current values of intermediate tensors
- Parameter values
- Gradient accumulation
- Device placement information

## Common Design Patterns

### Model Composition

Mithril uses operator overloading for intuitive model composition:
- `|=` (pipe-assign): Connect to input
- `+=` (add-assign): Append to chain
- Function call syntax: Explicit connections

### Functional Design

Pure functional design principles:
- Immutable model definitions
- Clear separation of parameters and computation
- Explicit data flow

### Backend Abstraction

Backend abstraction via:
- Common interface for all backends
- Consistent tensor operations
- Backend-specific optimizations hidden from users

## Extension Points

Mithril provides several extension points:

### Custom Operators

Create custom operators by subclassing `Operator`:

```python
class CustomOp(Operator):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    # Define how operation works
    def compute(self, inputs):
        # Implementation
        return {"output": result}
```

### Custom Backends

Create custom backends by implementing the `Backend` interface:

```python
class CustomBackend(Backend):
    def __init__(self, dtype=None, device=None):
        super().__init__(dtype, device)
        # Initialize backend-specific state
    
    # Implement tensor operations
    def zeros(self, *shape, dtype=None):
        # Implementation
```

### Custom Primitives

Extend the primitive operation set:

```python
@register_primitive("custom_op")
def custom_op(backend, x, param1, param2):
    # Implementation for specific backend
    return result
```

## Conclusion

Mithril's layered architecture provides a powerful yet flexible framework for machine learning model development. By separating logical model definition from physical execution and providing a unified interface across multiple backends, Mithril enables users to focus on model design while leveraging the strengths of different execution environments.

For more details on specific components, see the following guides:
- [Logical Models](../logical-models.md)
- [Physical Models](../physical-models.md)
- [Compilation](../compilation.md)
- [Custom Primitives](../custom-primitives.md)