# Mithril: A Modular Machine Learning Library for Model Composability

Mithril is an [Apache-licensed](http://www.apache.org/licenses/LICENSE-2.0) flexible machine learning library designed to simplify the composition and compilation of gradient-based models. It focuses on three core principles: versatile composability, framework-agnostic code generation, and easy parallelization. Mithril supports model creation across multiple lower-level libraries like [JAX](https://jax.readthedocs.io/en/latest/index.html), [PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/) (and in the near future, bare [CUDA](https://developer.nvidia.com/cuda-zone)), offering symbolic shape inference and unified model/data parallelism to streamline the development of scalable and trainable models.

## 🚀 Key Features

### 🧱 Versatile Composability

Mithril treats every model (e.g., `Add`, `Linear`, `MLP`, `ResNet`, `LSTM`, `Transformer`, `GPT`) as a standalone building block, regardless of its complexity. Users can start with a blank canvas and add these blocks to create more sophisticated models, with the flexibility to use any model as a subcomponent at any depth. Akin to how traditional query engines work, Mithril divides the model development process into two distinct phases:

1. *Logical Phase*: Model architecture design
2. *Physical Phase*: Model compilation

Example of building a logical model:

```python
from mithril.models import Add, LeakyRelu, Linear, Model, Relu

# A simple two-layer network where connections are
# made implicitly through "standard" inputs/outputs.
# Note the use of the "+=" operator for adding new models.
model1 = Model()
model1 += Linear(dimension=32)
model1 += Relu()
model1 += Linear(dimension=16)(output="output")

# Let's make another network just like the one above.
model2 = Model()
model2 += Linear(dimension=32)
model2 += LeakyRelu()
model2 += Linear(dimension=16)(output="output")

# For more complex connections, provide explicit connection
# information as below. I/O terminals of models can be named
# arbitrarily.
model = Model()
model += model1(output="output1")
model += model2(output="output2")
model += Add()(left="output1", right="output2", output="output")
```

### 🖨️ Language and Framework Agnostic Compilation

Mithril separates the model design process from compilation, allowing users to select the target language (e.g., Python and C 🚧) and the lower-level framework (e.g., NumPy, PyTorch, JAX, [MLX](https://ml-explore.github.io/mlx/build/html/index.html), [tinygrad](https://tinygrad.org/) 🚧 and CUDA 🚧) after the design phase. This decoupling enables users to compile and experiment with the same model using different backends within a single script, providing flexibility without needing to reimplement models for each scenario.

During compilation, the logical model —whether simple or nested— is flattened to apply various graph optimizations (such as model pruning, static calculations, and shape/type inference) before code generation. After this process, users can evaluate the compiled model or compute gradients directly. Mithril also allows users to view and utilize the generated code for the specified target language and framework, providing transparency and flexibility.

🚧: Coming soon, not yet fully open-source.

Example of compiling a logical model:

```python
import mithril as ml
from mithril.models import Model, Linear

# Build a simple linear model
model = Linear(16)

# Create backends, specify the precision
backend_jax = ml.JaxBackend(precision=64)
backend_numpy = ml.NumpyBackend(precision=32)

# Compile the model with different backends, optionally specify
# the file to write the generated code into and whether to use jit
# compilation
jax_model = ml.compile(
    model=model,
    backend=backend_jax,
    jit=False,
    file_path="generated_code.py",
)
numpy_model = ml.compile(
    model=model,
    backend=backend_numpy,
    shapes={"input": [3, 3]},
)

# Compile different logical models with the same backend
other_model = Model()
other_model += Linear(dimension=32)(input = "input")
jax_model1 = ml.compile(
    model=other_model,
    backend=backend_jax,
    shapes={"input": [3, 3]},
)

# Evaluate the compiled JAX model
params = jax_model1.randomize_params()
inputs = {"input": backend_jax.ones(3, 3)}
output = jax_model1.evaluate(params, inputs)

# Compute gradients of the compiled numpy model
params = numpy_model.randomize_params()
inputs = {"input": backend_numpy.ones(3, 3)}
output_gradients = {"output": backend_numpy.ones(3, 16)}
gradients = numpy_model.evaluate_gradients(params, inputs, output_gradients)
```

### 🔀 Flexible Parallelizability and Training

All inputs of the model, whether trainable or not, can be parallelized in any dimension via the same API for supporting frameworks (currently, PyTorch and JAX). Users simply create a backend with a device mesh and generate sharded tensors compatible with the chosen framework.

Example of using parallelizable inputs:

```python
import mithril as ml
from mithril.models import Linear

# Build a simple linear model
model = Linear(256)
# Generate a PyTorch backend with a (2, 1) device mesh
backend = ml.TorchBackend(device_mesh=(2,1))
# Compile the model
pm = ml.compile(model, backend, jit=False)
# Generate sharded data and parameters
params = {"w": backend.ones([128, 256]), "b": backend.ones([256])}
input = {"input": backend.ones(256, 128, device_mesh=(2,1))}
# Run the compiled model
output = pm.evaluate(params, input)
```

In addition to input parallelization flexibility, any input of any model can be trainable if differentiable, leading to several advantages:

- **Elasticity**: The ability to train any input simplifies architecture search for model development.
- **Functionalization**: Absence of fixed model-based weights makes the models more versatile for various applications.

## ⬇️ Installation

Install Mithril from [PyPI](https://pypi.org/project/mithril/):

```
pip install mithril --upgrade
```

For more detailed examples of different models and APIs, please refer to the `examples` directory in the repository.

---

**⚒️ Mithril: Forge Your ML Future**