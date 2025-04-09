# Code Generation

Mithril provides powerful code generation capabilities that can transform logical models into optimized code for various target platforms. This document explains how code generation works in Mithril and how to use it effectively.

## Overview

Code generation in Mithril converts your high-level model definitions into efficient, platform-specific code. This enables several key benefits:

- **Performance optimization**: Generate highly optimized code for specific targets
- **Deployability**: Create self-contained code for deployment in constrained environments
- **Debuggability**: Inspect generated code to understand model behavior
- **Customizability**: Tweak generation to suit particular platform requirements

## Basic Usage

To generate code from a logical model, you use the appropriate code generator for your target platform:

```python
import mithril as mi
from mithril.models import LogicalModel
from mithril.framework.codegen import CodeGenerator

# Define a simple model
model = LogicalModel("simple_mlp")
with model:
    x = mi.Input(shape=(None, 10), name="input")
    w1 = mi.Parameter(shape=(10, 20), name="w1")
    b1 = mi.Parameter(shape=(20,), name="b1")
    h = mi.relu(mi.matmul(x, w1) + b1)
    
    w2 = mi.Parameter(shape=(20, 1), name="w2")
    b2 = mi.Parameter(shape=(1,), name="b2")
    y = mi.matmul(h, w2) + b2
    
    mi.Output(y, name="output")

# Create a code generator
from mithril.framework.codegen.python_gen import PythonGenerator
python_generator = PythonGenerator()

# Generate Python code
python_code = python_generator.generate(model)
print(python_code)
```

## Available Generators

Mithril provides code generators for several target languages and platforms:

### Python Generator

```python
from mithril.framework.codegen.python_gen import PythonGenerator

python_generator = PythonGenerator()
code = python_generator.generate(model)
```

### NumPy Generator

```python
from mithril.framework.codegen.numpy_gen import NumPyGenerator

numpy_generator = NumPyGenerator()
code = numpy_generator.generate(model)
```

### C Generator

```python
from mithril.framework.codegen.c_gen import CGenerator

c_generator = CGenerator()
code = c_generator.generate(model)
```

### Raw C Generator

```python
from mithril.framework.codegen.raw_c_gen import RawCGenerator

raw_c_generator = RawCGenerator()
code = raw_c_generator.generate(model)
```

### GGML Generator

```python
from mithril.framework.codegen.ggml_gen import GGMLGenerator

ggml_generator = GGMLGenerator()
code = ggml_generator.generate(model)
```

## Generator Configuration

Code generators can be configured with various options to control the output:

```python
c_generator = CGenerator(
    # Enable optimization passes
    optimize=True,
    
    # Include debug information
    debug=True,
    
    # Use single precision floats
    precision="float32",
    
    # Generate header file
    generate_header=True,
    
    # Enable SIMD vectorization
    use_simd=True
)
```

## Saving Generated Code

To save the generated code to a file:

```python
# Generate the code
code = c_generator.generate(model)

# Save to file
with open("model.c", "w") as f:
    f.write(code)

# For generators that produce multiple files
c_generator = CGenerator(generate_header=True)
code, header = c_generator.generate_with_header(model)

with open("model.c", "w") as f:
    f.write(code)
    
with open("model.h", "w") as f:
    f.write(header)
```

## Code Generation Pipeline

The code generation process involves several key steps:

1. **Model Analysis**: The model graph is analyzed to understand data flow and dependencies
2. **IR Generation**: The model is converted to an intermediate representation (IR)
3. **Optimization**: The IR is optimized (e.g., operator fusion, constant folding)
4. **Code Emission**: Target-specific code is generated from the optimized IR
5. **Post-processing**: The generated code is formatted and organized

## Advanced Usage

### Custom Operator Mappings

You can define custom mappings for operators to target-specific implementations:

```python
# Define custom operator implementations
custom_ops = {
    "relu": "custom_relu_implementation",
    "matmul": "optimized_matrix_multiply"
}

# Create generator with custom operator mappings
generator = CGenerator(custom_operator_mappings=custom_ops)
code = generator.generate(model)
```

### Model Partitioning

For complex models, you can partition the model and generate code for different parts:

```python
# Partition model by device
cpu_part = model.extract_subgraph(["input", "hidden1"])
gpu_part = model.extract_subgraph(["hidden1", "output"])

# Generate code for each part
cpu_code = c_generator.generate(cpu_part)
gpu_code = cuda_generator.generate(gpu_part)
```

### Custom Templates

Many code generators support custom templates:

```python
# Create generator with custom template
generator = CGenerator(template_path="my_custom_template.j2")
code = generator.generate(model)
```

## Compilation and Execution

After generating code, you typically need to compile and execute it:

### Python/NumPy

```python
# Generate Python code
code = python_generator.generate(model)

# Execute the generated code
namespace = {}
exec(code, namespace)

# Create the model function from the namespace
model_func = namespace["model_forward"]

# Run the model
output = model_func(input_data, weights)
```

### C/C++

```python
# Generate C code
code = c_generator.generate(model)

# Save to file
with open("model.c", "w") as f:
    f.write(code)

# Compile (using subprocess)
import subprocess
subprocess.run(["gcc", "-O3", "model.c", "-o", "model"])

# Alternatively, you can use the compilation utility
from mithril.framework.codegen.utils import compile_c_code
compile_c_code(code, "model")
```

## Examples

### NumPy Code Generation Example

```python
import mithril as mi
from mithril.models import LogicalModel
from mithril.framework.codegen.numpy_gen import NumPyGenerator

# Define a simple model
model = LogicalModel("simple_cnn")
with model:
    x = mi.Input(shape=(None, 3, 32, 32), name="input")
    w1 = mi.Parameter(shape=(16, 3, 3, 3), name="conv1_weights")
    b1 = mi.Parameter(shape=(16,), name="conv1_bias")
    
    # First conv layer
    conv1 = mi.conv2d(x, w1, strides=(1, 1), padding="SAME")
    conv1 = conv1 + b1.reshape((1, 16, 1, 1))
    act1 = mi.relu(conv1)
    pool1 = mi.max_pool2d(act1, pool_size=(2, 2), strides=(2, 2))
    
    # Flatten and dense layer
    flattened = mi.flatten(pool1)
    w2 = mi.Parameter(shape=(16*16*16, 10), name="dense_weights")
    b2 = mi.Parameter(shape=(10,), name="dense_bias")
    logits = mi.matmul(flattened, w2) + b2
    
    mi.Output(logits, name="output")

# Generate NumPy code
numpy_generator = NumPyGenerator(optimize=True)
code = numpy_generator.generate(model)

# Save to file
with open("cnn_model.py", "w") as f:
    f.write(code)

# Execute the generated code
namespace = {}
exec(code, namespace)
forward_func = namespace["forward"]

# Run with random input
import numpy as np
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
parameters = {
    "conv1_weights": np.random.randn(16, 3, 3, 3).astype(np.float32),
    "conv1_bias": np.random.randn(16).astype(np.float32),
    "dense_weights": np.random.randn(16*16*16, 10).astype(np.float32),
    "dense_bias": np.random.randn(10).astype(np.float32)
}

# Forward pass
output = forward_func(input_data, parameters)
print(output.shape)  # Should be (1, 10)
```

### C Code Generation Example

```python
import mithril as mi
from mithril.models import LogicalModel
from mithril.framework.codegen.c_gen import CGenerator

# Define a simple model
model = LogicalModel("simple_mlp")
with model:
    x = mi.Input(shape=(None, 784), name="input")
    w1 = mi.Parameter(shape=(784, 128), name="w1")
    b1 = mi.Parameter(shape=(128,), name="b1")
    h1 = mi.relu(mi.matmul(x, w1) + b1)
    
    w2 = mi.Parameter(shape=(128, 10), name="w2")
    b2 = mi.Parameter(shape=(10,), name="b2")
    logits = mi.matmul(h1, w2) + b2
    
    mi.Output(logits, name="output")

# Generate C code with header
c_generator = CGenerator(generate_header=True, use_simd=True)
code, header = c_generator.generate_with_header(model)

# Save to files
with open("model.c", "w") as f:
    f.write(code)
    
with open("model.h", "w") as f:
    f.write(header)
```

## Best Practices

1. **Set Shapes and Types**: Always set concrete shapes and types before generating code for best optimization
2. **Use Named Tensors**: Give clear names to tensors to make generated code more readable
3. **Specify Precision**: Choose appropriate precision (float32, float16, etc.) based on your needs
4. **Test Generated Code**: Always verify that generated code produces the same results as the original model
5. **Start Simple**: Begin with simpler generators (Python/NumPy) before moving to C/GGML

## Limitations

- Some complex operations may not be supported by all generators
- Dynamic shapes can be challenging for some target platforms
- Custom operators require manual implementation for each target platform

## Troubleshooting

- If code generation fails, check that all operations are supported by the target generator
- Ensure shapes and types are properly specified before generation
- For C/GGML generation, consider using simpler operations that map well to the target
- Use debug mode to get more verbose output about the generation process