# Debugging Models in Mithril

This tutorial provides a comprehensive guide to debugging Mithril models, identifying and resolving common issues, and developing robust models for production.

## Overview

In this tutorial, you'll learn:

1. How to inspect and debug Mithril models
2. Common issues and their solutions
3. Techniques for validating model behavior
4. Advanced debugging tools and methods
5. Best practices for debugging across different backends

## Prerequisites

Before starting, ensure you have the required libraries:

```bash
pip install mithril jax optax numpy torch matplotlib pytest
```

## Model Inspection and Visualization

### Printing Model Summaries

Mithril models provide summary methods that help understand their structure:

```python
import mithril as ml
from mithril.models import resnet18, Model, Linear, Relu

# Create a simple model
def create_simple_mlp():
    model = Model()
    model += Linear(128)(input="input")
    model += Relu()
    model += Linear(64)
    model += Relu()
    model += Linear(10)(output="output")
    return model

# Create and print model summary
model = create_simple_mlp()
print(model.summary())

# Create and print more complex model summary
resnet = resnet18(num_classes=10)
print(resnet.summary())
```

The summary method provides a hierarchical view of your model's structure, including:
- Layer names and types
- Parameter shapes
- Connection flow

### Visualizing Model Structure

For more complex models, you can generate a visual representation:

```python
def visualize_model_graph(model, output_file="model_graph.png"):
    """Generate a visual representation of the model graph"""
    import graphviz
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Model Graph')
    
    # Add nodes for model components
    for name, component in model.components.items():
        # Extract component type
        component_type = component.__class__.__name__
        
        # Create node
        dot.node(name, f"{name}\n({component_type})")
    
    # Add edges for connections
    for name, component in model.components.items():
        # Get input connections
        for input_name in component.input_connections:
            if input_name in model.components:
                dot.edge(input_name, name)
    
    # Render the graph
    dot.render(output_file, format='png', cleanup=True)
    print(f"Model graph saved to {output_file}")

# Example usage
model = create_simple_mlp()
visualize_model_graph(model)
```

## Inspecting Model Compilation

Mithril's compilation process creates a physical model optimized for the target backend. Inspecting this process helps debug issues:

```python
import mithril as ml
from mithril.models import Linear, Relu, Model

# Create a simple model
model = Model()
model += Linear(64)(input="input")
model += Relu()
model += Linear(10)(output="output")

# Compile with debug info
backend = ml.TorchBackend()
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 100]},
    data_keys={"input"},
    debug=True  # Enable debug mode
)

# Inspect shapes and types after compilation
print("Parameter shapes:")
for key, shape in compiled_model.shapes.items():
    print(f"  {key}: {shape}")

print("\nConnection shapes:")
for key, connection in compiled_model.model.conns.items():
    print(f"  {key}: {connection.shape}")
```

## Common Debugging Scenarios

### 1. Shape Mismatches

Shape errors are common when building models. Let's see how to identify and fix them:

```python
def debug_shape_mismatch():
    # Intentionally create a model with shape mismatch
    model = Model()
    model += Linear(64)(input="input")
    model += Relu()
    model += Linear(32)
    
    # This will cause a shape mismatch
    model |= Reshape(shape=(16, 4))(input=model.cout, output="reshaped")
    model += Linear(10)(output="output")
    
    # Try to compile and handle error
    try:
        backend = ml.TorchBackend()
        compiled_model = ml.compile(
            model=model,
            backend=backend,
            shapes={"input": [32, 100]},
            data_keys={"input"}
        )
    except Exception as e:
        print(f"Compilation error: {e}")
        
        # Fix the model
        fixed_model = Model()
        fixed_model += Linear(64)(input="input")
        fixed_model += Relu()
        fixed_model += Linear(32)
        
        # Use correct reshape dimensions
        fixed_model |= Reshape(shape=(-1, 16, 2))(input=fixed_model.cout, output="reshaped")
        # Flatten before the final layer
        fixed_model += Flatten(start_dim=1)
        fixed_model += Linear(10)(output="output")
        
        # Try compiling again
        try:
            compiled_model = ml.compile(
                model=fixed_model,
                backend=backend,
                shapes={"input": [32, 100]},
                data_keys={"input"}
            )
            print("Fixed model compiled successfully!")
        except Exception as e:
            print(f"Error in fixed model: {e}")
```

### 2. Type Errors

Type errors occur when operations expect different data types:

```python
def debug_type_errors():
    # Create a model with potential type issues
    model = Model()
    model += Linear(64)(input="input")
    model += Relu()
    
    # This might cause issues if input_b is an integer type
    model |= Add()(left=model.cout, right="input_b", output="added")
    model += Linear(10)(output="output")
    
    # Try compiling with mixed types
    try:
        backend = ml.TorchBackend()
        compiled_model = ml.compile(
            model=model,
            backend=backend,
            shapes={"input": [32, 100], "input_b": [32, 64]},
            data_keys={"input", "input_b"}
        )
        
        # Test with mixed type inputs
        float_input = backend.ones([32, 100], dtype=ml.float32)
        int_input = backend.ones([32, 64], dtype=ml.int32)
        
        outputs = compiled_model.evaluate(
            params={},
            data={"input": float_input, "input_b": int_input}
        )
    except Exception as e:
        print(f"Type error: {e}")
        
        # Fix with explicit type casting
        fixed_model = Model()
        fixed_model += Linear(64)(input="input")
        fixed_model += Relu()
        
        # Cast input_b to the right type
        fixed_model |= ml.models.Cast(dtype=ml.float32)(
            input="input_b", output="input_b_float"
        )
        fixed_model |= Add()(
            left=fixed_model.cout, right="input_b_float", output="added"
        )
        fixed_model += Linear(10)(output="output")
        
        # Try compiling again
        try:
            compiled_model = ml.compile(
                model=fixed_model,
                backend=backend,
                shapes={"input": [32, 100], "input_b": [32, 64]},
                data_keys={"input", "input_b"}
            )
            print("Fixed model compiled successfully!")
        except Exception as e:
            print(f"Error in fixed model: {e}")
```

### 3. Runtime NaN Values

NaN (Not a Number) values can propagate through computations and cause models to produce invalid results:

```python
def debug_nan_values():
    # Create a model that might produce NaNs
    model = Model()
    model += Linear(64)(input="input")
    model += Relu()
    
    # Division operation that might cause NaNs
    model |= ml.models.Divide()(
        left=model.cout, right="divisor", output="divided"
    )
    model += Linear(10)(output="output")
    
    # Compile model
    backend = ml.TorchBackend()
    compiled_model = ml.compile(
        model=model,
        backend=backend,
        shapes={"input": [32, 100], "divisor": [32, 64]},
        data_keys={"input", "divisor"}
    )
    
    # Create test data with potential division by zero
    params = compiled_model.randomize_params()
    test_input = backend.ones([32, 100], dtype=ml.float32)
    small_values = backend.ones([32, 64], dtype=ml.float32) * 1e-10
    
    # Check for NaNs in output
    outputs = compiled_model.evaluate(
        params=params, 
        data={"input": test_input, "divisor": small_values}
    )
    
    # Function to detect NaNs
    def contains_nans(tensor):
        if backend.backend_type == "torch":
            return bool(backend.isnan(tensor).any())
        elif backend.backend_type == "jax":
            return bool(backend.isnan(tensor).any())
        else:
            # Fallback for other backends
            return "NaN" in str(tensor)
    
    if contains_nans(outputs["output"]):
        print("NaN values detected in output!")
        
        # Fix with a safe division
        fixed_model = Model()
        fixed_model += Linear(64)(input="input")
        fixed_model += Relu()
        
        # Add small epsilon to avoid division by zero
        fixed_model |= ml.models.Add()(
            left="divisor", right=1e-8, output="safe_divisor"
        )
        fixed_model |= ml.models.Divide()(
            left=fixed_model.cout, right="safe_divisor", output="divided"
        )
        fixed_model += Linear(10)(output="output")
        
        # Compile and test fixed model
        fixed_compiled = ml.compile(
            model=fixed_model,
            backend=backend,
            shapes={"input": [32, 100], "divisor": [32, 64]},
            data_keys={"input", "divisor"}
        )
        
        fixed_outputs = fixed_compiled.evaluate(
            params=compiled_model.randomize_params(),
            data={"input": test_input, "divisor": small_values}
        )
        
        if not contains_nans(fixed_outputs["output"]):
            print("Fixed model produces valid outputs!")
        else:
            print("Fixed model still produces NaNs!")
```

## Debugging Model Behavior

### Testing Subcomponents

Breaking down complex models and testing individual components helps isolate issues:

```python
def test_component(component, input_shapes, expected_output_shape):
    """Test a model component in isolation"""
    # Create a simple backend for testing
    backend = ml.TorchBackend()
    
    # Compile the component
    compiled = ml.compile(
        model=component,
        backend=backend,
        shapes=input_shapes,
        data_keys=set(input_shapes.keys())
    )
    
    # Create random inputs
    inputs = {
        key: backend.randn(shape, dtype=ml.float32)
        for key, shape in input_shapes.items()
    }
    
    # Initialize random parameters
    params = compiled.randomize_params()
    
    # Run component
    outputs = compiled.evaluate(params, inputs)
    
    # Verify output shape
    output_key = list(outputs.keys())[0]  # Assume single output
    actual_shape = outputs[output_key].shape
    
    print(f"Component: {component.__class__.__name__}")
    print(f"Expected output shape: {expected_output_shape}")
    print(f"Actual output shape: {actual_shape}")
    
    if actual_shape == expected_output_shape:
        print("✅ Shape test passed!")
    else:
        print("❌ Shape test failed!")
    
    return outputs

# Example usage
attention_block = ml.models.ScaledDotProduct(is_causal=False)
input_shapes = {
    "query": [2, 8, 16, 64],
    "key": [2, 8, 16, 64],
    "value": [2, 8, 16, 64]
}
expected_shape = (2, 8, 16, 64)
test_component(attention_block, input_shapes, expected_shape)
```

### Gradient Checking

Verify that gradients are flowing correctly through the model:

```python
def check_gradients(model, input_shape, backend=None):
    """Check gradients using numerical approximation"""
    if backend is None:
        backend = ml.TorchBackend()
    
    # Compile model
    compiled_model = ml.compile(
        model=model,
        backend=backend,
        shapes={"input": input_shape},
        data_keys={"input"}
    )
    
    # Initialize parameters
    params = compiled_model.randomize_params()
    
    # Create random input
    test_input = backend.randn(input_shape, dtype=ml.float32)
    
    # Forward pass
    outputs = compiled_model.evaluate(params, {"input": test_input})
    output = outputs["output"]
    
    # Create dummy gradient for output (sum all elements)
    output_grad = backend.ones_like(output)
    
    # Get analytical gradients
    _, gradients = compiled_model.evaluate(
        params, {"input": test_input}, 
        output_gradients={"output": output_grad}
    )
    
    # Check if gradients exist and are not all zero for key parameters
    gradient_issues = []
    for key, grad in gradients.items():
        if "weight" in key or "bias" in key:
            # Convert to numpy for backend-agnostic checks
            grad_np = backend.to_numpy(grad)
            
            if np.all(grad_np == 0):
                gradient_issues.append(f"{key}: All zeros")
            elif np.any(np.isnan(grad_np)):
                gradient_issues.append(f"{key}: Contains NaNs")
            elif np.any(np.isinf(grad_np)):
                gradient_issues.append(f"{key}: Contains Infs")
    
    if gradient_issues:
        print("Gradient issues detected:")
        for issue in gradient_issues:
            print(f"  {issue}")
    else:
        print("Gradient check passed! No issues detected.")
    
    return gradients

# Example usage
model = Model()
model += Linear(64)(input="input")
model += Relu()
model += Linear(10)(output="output")

check_gradients(model, [32, 100])
```

## Debugging Across Different Backends

Mithril's multi-backend support can sometimes lead to backend-specific issues. Here's how to debug them:

```python
def debug_across_backends(model, input_shape):
    """Test a model on multiple backends to identify backend-specific issues"""
    # List of backends to test
    backends = [
        ml.TorchBackend(),
        ml.JaxBackend()
    ]
    
    # Optional: Add MLX backend on macOS with Apple Silicon
    import platform
    if platform.system() == "Darwin" and platform.processor() == "arm":
        backends.append(ml.MlxBackend())
    
    results = {}
    for backend in backends:
        backend_name = backend.backend_type
        print(f"Testing {backend_name} backend...")
        
        try:
            # Compile model
            compiled_model = ml.compile(
                model=model,
                backend=backend,
                shapes={"input": input_shape},
                data_keys={"input"}
            )
            
            # Initialize parameters
            params = compiled_model.randomize_params()
            
            # Create random input
            test_input = backend.randn(input_shape, dtype=ml.float32)
            
            # Run model
            outputs = compiled_model.evaluate(params, {"input": test_input})
            output = outputs["output"]
            
            # Check for issues
            if backend.backend_type == "torch":
                has_nan = backend.isnan(output).any()
                has_inf = backend.isinf(output).any()
            elif backend.backend_type == "jax":
                has_nan = bool(backend.isnan(output).any())
                has_inf = bool(backend.isinf(output).any())
            else:
                # Fallback for other backends
                output_np = backend.to_numpy(output)
                has_nan = np.isnan(output_np).any()
                has_inf = np.isinf(output_np).any()
            
            status = "✅ Success"
            if has_nan:
                status = "❌ Contains NaNs"
            elif has_inf:
                status = "❌ Contains Infs"
                
            results[backend_name] = {
                "status": status,
                "shape": output.shape
            }
            
        except Exception as e:
            results[backend_name] = {
                "status": f"❌ Error: {str(e)}",
                "shape": None
            }
    
    # Print summary
    print("\nResults across backends:")
    for backend_name, result in results.items():
        print(f"{backend_name}: {result['status']}")
        if result['shape'] is not None:
            print(f"  Output shape: {result['shape']}")
    
    # Check for backend inconsistencies
    shapes = [r['shape'] for r in results.values() if r['shape'] is not None]
    if len(shapes) > 1 and not all(s == shapes[0] for s in shapes):
        print("\n⚠️ Warning: Output shapes differ across backends!")
    
    return results

# Example usage
model = Model()
model += Linear(64)(input="input")
model += Relu()
model += Linear(10)(output="output")

debug_across_backends(model, [32, 100])
```

## Setting Up Comprehensive Testing

To ensure model correctness, set up automated tests:

```python
import pytest

def create_test_suite(model_factory, input_shapes, expected_outputs=None):
    """Create a test suite for a model"""
    
    @pytest.mark.parametrize("backend_type", [ml.TorchBackend, ml.JaxBackend])
    def test_model_compilation(backend_type):
        """Test that the model compiles successfully"""
        model = model_factory()
        backend = backend_type()
        
        try:
            compiled_model = ml.compile(
                model=model,
                backend=backend,
                shapes=input_shapes,
                data_keys=set(input_shapes.keys())
            )
            assert compiled_model is not None
        except Exception as e:
            pytest.fail(f"Model compilation failed: {e}")
    
    @pytest.mark.parametrize("backend_type", [ml.TorchBackend, ml.JaxBackend])
    def test_model_forward(backend_type):
        """Test model forward pass"""
        model = model_factory()
        backend = backend_type()
        
        compiled_model = ml.compile(
            model=model,
            backend=backend,
            shapes=input_shapes,
            data_keys=set(input_shapes.keys())
        )
        
        # Initialize parameters
        params = compiled_model.randomize_params()
        
        # Create inputs
        inputs = {
            key: backend.ones(shape, dtype=ml.float32)
            for key, shape in input_shapes.items()
        }
        
        # Forward pass
        outputs = compiled_model.evaluate(params, inputs)
        
        # Check outputs
        assert "output" in outputs, "Model should have an output named 'output'"
        
        # Check output shape if expected_output_shape is provided
        if expected_outputs and "shape" in expected_outputs:
            assert outputs["output"].shape == expected_outputs["shape"], \
                f"Expected shape {expected_outputs['shape']}, got {outputs['output'].shape}"
    
    @pytest.mark.parametrize("backend_type", [ml.TorchBackend, ml.JaxBackend])
    def test_model_gradients(backend_type):
        """Test model gradients"""
        model = model_factory()
        backend = backend_type()
        
        compiled_model = ml.compile(
            model=model,
            backend=backend,
            shapes=input_shapes,
            data_keys=set(input_shapes.keys())
        )
        
        # Initialize parameters
        params = compiled_model.randomize_params()
        
        # Create inputs
        inputs = {
            key: backend.ones(shape, dtype=ml.float32)
            for key, shape in input_shapes.items()
        }
        
        # Forward pass with gradients
        outputs, gradients = compiled_model.evaluate(
            params, inputs, 
            output_gradients={"output": backend.ones_like(outputs["output"])}
        )
        
        # Check that gradients exist
        assert len(gradients) > 0, "No gradients computed"
        
        # Check gradients are not all zero
        for key, grad in gradients.items():
            if "weight" in key:  # Focus on main parameters
                grad_np = backend.to_numpy(grad)
                assert not np.all(grad_np == 0), f"Gradient {key} is all zeros"
                assert not np.any(np.isnan(grad_np)), f"Gradient {key} has NaNs"
                assert not np.any(np.isinf(grad_np)), f"Gradient {key} has Infs"
    
    # Return the test functions so they can be added to a module
    return [
        test_model_compilation,
        test_model_forward,
        test_model_gradients
    ]

# Example usage
def sample_model_factory():
    model = Model()
    model += Linear(64)(input="input")
    model += Relu()
    model += Linear(10)(output="output")
    return model

input_shapes = {"input": [32, 100]}
expected_outputs = {"shape": (32, 10)}

# Create test suite
tests = create_test_suite(sample_model_factory, input_shapes, expected_outputs)

# These tests could be added to a test module
# For demonstration purposes, we'll simply print their names
for test in tests:
    print(f"Generated test: {test.__name__}")
```

## Advanced Debugging Techniques

### 1. Adding Debug Probes

Insert probes into your model to inspect intermediate values:

```python
def add_debug_probes(model):
    """Add debug probes to a model to inspect intermediate values"""
    # Clone the model to avoid modifying the original
    from copy import deepcopy
    debug_model = deepcopy(model)
    
    # Get all existing connections
    connections = list(debug_model.conns.items())
    
    # Add debug outputs for key connections
    for idx, (name, conn) in enumerate(connections):
        # Skip input and output connections
        if name in debug_model.input_keys or name in debug_model.output_keys:
            continue
        
        # Add a Buffer that exposes this connection as a named output
        debug_model |= ml.models.Buffer()(
            input=name, output=f"debug_{idx}_{name}"
        )
    
    return debug_model

# Example usage
model = Model()
model += Linear(64, name="layer1")(input="input")
model += Relu(name="relu1")
model += Linear(32, name="layer2")
model += Relu(name="relu2")
model += Linear(10, name="output_layer")(output="output")

debug_model = add_debug_probes(model)

# Compile and run the debug model
backend = ml.TorchBackend()
compiled_debug = ml.compile(
    model=debug_model,
    backend=backend,
    shapes={"input": [1, 100]},
    data_keys={"input"}
)

# Create a sample input
test_input = backend.ones([1, 100], dtype=ml.float32)

# Run the model and collect all intermediate outputs
results = compiled_debug.evaluate(
    compiled_debug.randomize_params(),
    {"input": test_input}
)

# Print statistics about intermediate layers
print("Layer statistics:")
for key, value in sorted(results.items()):
    if key.startswith("debug_"):
        # Extract tensor statistics
        if hasattr(value, "shape"):
            value_np = backend.to_numpy(value)
            print(f"{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Min: {np.min(value_np):.4f}, Max: {np.max(value_np):.4f}")
            print(f"  Mean: {np.mean(value_np):.4f}, Std: {np.std(value_np):.4f}")
            
            # Check for extreme values
            if np.max(np.abs(value_np)) > 100:
                print("  ⚠️ Warning: Contains extreme values!")
            if np.std(value_np) < 1e-6:
                print("  ⚠️ Warning: Very low standard deviation!")
```

### 2. Visualizing Activations

Visualize activation patterns to detect model behavior issues:

```python
import matplotlib.pyplot as plt

def visualize_activations(model, test_input, backend=None):
    """Visualize activation patterns across model layers"""
    if backend is None:
        backend = ml.TorchBackend()
    
    # Add debug probes
    debug_model = add_debug_probes(model)
    
    # Compile debug model
    compiled_debug = ml.compile(
        model=debug_model,
        backend=backend,
        shapes={"input": test_input.shape},
        data_keys={"input"}
    )
    
    # Generate random parameters
    params = compiled_debug.randomize_params()
    
    # Run the model and collect all intermediate outputs
    results = compiled_debug.evaluate(params, {"input": test_input})
    
    # Collect activation data
    activation_data = {}
    for key, value in results.items():
        if key.startswith("debug_") and "relu" in key:
            # Extract activation data
            value_np = backend.to_numpy(value)
            
            # For visualization, reshape to 2D if needed
            if len(value_np.shape) > 2:
                value_np = value_np.reshape(value_np.shape[0], -1)
            
            # Store for visualization
            activation_data[key] = value_np
    
    # Visualize activations
    num_layers = len(activation_data)
    if num_layers == 0:
        print("No ReLU activations found in the model.")
        return
    
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 4, 3))
    if num_layers == 1:
        axes = [axes]
    
    for (key, value), ax in zip(activation_data.items(), axes):
        im = ax.imshow(value, aspect='auto', cmap='viridis')
        ax.set_title(key)
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Batch item")
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    # Also plot activation statistics
    plt.figure(figsize=(10, 4))
    
    # Plot mean activation per layer
    means = [np.mean(v) for v in activation_data.values()]
    stds = [np.std(v) for v in activation_data.values()]
    sparsity = [np.mean(v == 0) for v in activation_data.values()]
    
    layer_names = [key.split('_')[-1] for key in activation_data.keys()]
    
    plt.subplot(1, 3, 1)
    plt.bar(layer_names, means)
    plt.title("Mean Activation")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.bar(layer_names, stds)
    plt.title("Activation Std Dev")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    plt.bar(layer_names, sparsity)
    plt.title("Activation Sparsity")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
```

### 3. Analyzing Weight Distributions

Inspect weight distributions to detect potential issues:

```python
def analyze_weights(model, params, backend=None):
    """Analyze model weight distributions"""
    if backend is None:
        backend = ml.TorchBackend()
    
    # Collect weight data
    weight_data = {}
    for key, value in params.items():
        if "weight" in key or "bias" in key:
            # Convert to numpy for analysis
            weight_data[key] = backend.to_numpy(value)
    
    # Plot weight distributions
    num_weights = len(weight_data)
    if num_weights == 0:
        print("No weights found in the model.")
        return
    
    rows = (num_weights + 2) // 3  # Calculate number of rows needed
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 3))
    axes = axes.flatten()
    
    for i, (key, value) in enumerate(weight_data.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Flatten weights for histogram
            flat_weights = value.flatten()
            
            # Plot histogram
            ax.hist(flat_weights, bins=50, alpha=0.7)
            ax.set_title(key)
            
            # Add statistics
            mean = np.mean(flat_weights)
            std = np.std(flat_weights)
            min_val = np.min(flat_weights)
            max_val = np.max(flat_weights)
            
            stats_text = f"Mean: {mean:.4f}\nStd: {std:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print potential issues
    print("Potential weight issues:")
    for key, value in weight_data.items():
        flat_weights = value.flatten()
        mean = np.mean(flat_weights)
        std = np.std(flat_weights)
        
        # Check for issues
        if std < 1e-6:
            print(f"⚠️ {key}: Very low standard deviation, weights might not be properly initialized")
        
        if np.max(np.abs(flat_weights)) > 10:
            print(f"⚠️ {key}: Contains extreme values, might cause exploding activations")
        
        if "weight" in key and np.all(flat_weights >= 0):
            print(f"⚠️ {key}: All weights are positive, unusual for neural networks")
```

## Best Practices for Debugging Mithril Models

### 1. Start Small

Build up complex models gradually, testing each component:

```python
def incremental_model_building():
    """Demonstrate incremental model building with testing"""
    # Start with a simple component
    print("Step 1: Building and testing input embedding")
    embedding = Model()
    embedding += Linear(64, name="embedding")(input="input")
    embedding += Relu()
    
    # Test it
    backend = ml.TorchBackend()
    compiled_embedding = ml.compile(
        model=embedding,
        backend=backend,
        shapes={"input": [32, 100]},
        data_keys={"input"}
    )
    
    # Check embedding works
    test_input = backend.ones([32, 100], dtype=ml.float32)
    embedding_params = compiled_embedding.randomize_params()
    embedding_output = compiled_embedding.evaluate(
        embedding_params, {"input": test_input}
    )[compiled_embedding.cout.name]
    
    print(f"Embedding output shape: {embedding_output.shape}")
    
    # Add another layer
    print("\nStep 2: Adding hidden layer")
    hidden = Model()
    hidden += Linear(64, name="embedding")(input="input")
    hidden += Relu()
    hidden += Linear(32, name="hidden")
    hidden += Relu()
    
    # Test it
    compiled_hidden = ml.compile(
        model=hidden,
        backend=backend,
        shapes={"input": [32, 100]},
        data_keys={"input"}
    )
    
    hidden_params = compiled_hidden.randomize_params()
    hidden_output = compiled_hidden.evaluate(
        hidden_params, {"input": test_input}
    )[compiled_hidden.cout.name]
    
    print(f"Hidden layer output shape: {hidden_output.shape}")
    
    # Complete model
    print("\nStep 3: Completing the model")
    full_model = Model()
    full_model += Linear(64, name="embedding")(input="input")
    full_model += Relu()
    full_model += Linear(32, name="hidden")
    full_model += Relu()
    full_model += Linear(10, name="output")(output="output")
    
    # Test it
    compiled_full = ml.compile(
        model=full_model,
        backend=backend,
        shapes={"input": [32, 100]},
        data_keys={"input"}
    )
    
    full_params = compiled_full.randomize_params()
    full_output = compiled_full.evaluate(
        full_params, {"input": test_input}
    )["output"]
    
    print(f"Final model output shape: {full_output.shape}")
```

### 2. Use Consistent Testing Patterns

Develop reusable testing patterns:

```python
class ModelTester:
    """Reusable model testing class"""
    
    def __init__(self, model, input_shapes, backend=None):
        self.model = model
        self.input_shapes = input_shapes
        self.backend = backend or ml.TorchBackend()
        
        # Compile model
        self.compiled_model = ml.compile(
            model=self.model,
            backend=self.backend,
            shapes=self.input_shapes,
            data_keys=set(self.input_shapes.keys())
        )
        
        # Generate parameters
        self.params = self.compiled_model.randomize_params()
        
        # Create test inputs
        self.test_inputs = {
            key: self.backend.ones(shape, dtype=ml.float32)
            for key, shape in self.input_shapes.items()
        }
    
    def test_forward(self):
        """Test forward pass"""
        try:
            outputs = self.compiled_model.evaluate(
                self.params, self.test_inputs
            )
            print("Forward pass successful!")
            
            # Print output shapes
            for key, value in outputs.items():
                print(f"  {key}: shape {value.shape}")
            
            return outputs
        except Exception as e:
            print(f"Forward pass failed: {e}")
            return None
    
    def test_gradients(self):
        """Test gradient computation"""
        try:
            # Forward pass
            outputs = self.compiled_model.evaluate(
                self.params, self.test_inputs
            )
            
            # Create dummy gradients
            output_gradients = {
                key: self.backend.ones_like(value)
                for key, value in outputs.items()
            }
            
            # Compute gradients
            _, gradients = self.compiled_model.evaluate(
                self.params, self.test_inputs, output_gradients=output_gradients
            )
            
            print("Gradient computation successful!")
            print(f"  Number of gradient parameters: {len(gradients)}")
            
            # Check for issues
            issues = 0
            for key, grad in gradients.items():
                grad_np = self.backend.to_numpy(grad)
                
                if np.all(grad_np == 0):
                    print(f"  ⚠️ {key}: All gradients are zero")
                    issues += 1
                
                if np.any(np.isnan(grad_np)):
                    print(f"  ⚠️ {key}: Contains NaN gradients")
                    issues += 1
                
                if np.any(np.isinf(grad_np)):
                    print(f"  ⚠️ {key}: Contains Inf gradients")
                    issues += 1
            
            if issues == 0:
                print("  No gradient issues detected!")
            
            return gradients
        except Exception as e:
            print(f"Gradient computation failed: {e}")
            return None
    
    def test_shapes_consistent(self):
        """Test that shapes are consistent across the model"""
        try:
            # Get all connections
            connections = self.compiled_model.model.conns
            
            # Check shape consistency
            issues = 0
            for name, conn in connections.items():
                if conn.shape is None:
                    print(f"  ⚠️ {name}: Shape is None")
                    issues += 1
                elif any(d is None for d in conn.shape):
                    print(f"  ⚠️ {name}: Contains None dimensions: {conn.shape}")
                    issues += 1
            
            if issues == 0:
                print("All shapes are consistent!")
            
            return issues == 0
        except Exception as e:
            print(f"Shape consistency check failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("Running all tests:")
        print("1. Forward pass test:")
        self.test_forward()
        print("\n2. Gradient test:")
        self.test_gradients()
        print("\n3. Shape consistency test:")
        self.test_shapes_consistent()

# Example usage
model = Model()
model += Linear(64)(input="input")
model += Relu()
model += Linear(10)(output="output")

tester = ModelTester(model, {"input": [32, 100]})
tester.run_all_tests()
```

### 3. Document Common Issues

Create a reference guide for common issues:

```python
def common_issues_reference():
    """Display reference guide for common Mithril issues"""
    issues = [
        {
            "issue": "Shape mismatch in Linear layer",
            "symptoms": "Error: Input shape ... doesn't match weight shape ...",
            "cause": "Input features don't match the Linear layer's in_features",
            "solution": "Adjust the Linear layer dimensions or reshape the input"
        },
        {
            "issue": "NaN values in output",
            "symptoms": "Model produces NaN values or loss becomes NaN",
            "cause": "Division by zero, log of negative number, or explosion of gradients",
            "solution": "Add epsilon to denominators, use stable implementations, check for extreme values"
        },
        {
            "issue": "Backend-specific failures",
            "symptoms": "Model works on one backend but fails on another",
            "cause": "Backend-specific operations or precision issues",
            "solution": "Use backend-agnostic operations, test on all target backends early"
        },
        {
            "issue": "Memory issues with large models",
            "symptoms": "Out of memory errors during compilation or forward pass",
            "cause": "Model too large for available GPU/CPU memory",
            "solution": "Use smaller batch sizes, optimize model structure, use model parallelism"
        },
        {
            "issue": "Model doesn't learn (loss doesn't decrease)",
            "symptoms": "Training loss stays constant or fluctuates without decreasing",
            "cause": "Incorrect loss function, learning rate issues, gradient flow problems",
            "solution": "Check loss function, adjust learning rate, inspect gradients, investigate activation patterns"
        }
    ]
    
    print("Common Mithril Debugging Issues Reference Guide")
    print("================================================")
    
    for i, issue in enumerate(issues):
        print(f"\n{i+1}. {issue['issue']}")
        print(f"   Symptoms: {issue['symptoms']}")
        print(f"   Cause: {issue['cause']}")
        print(f"   Solution: {issue['solution']}")
    
    return issues

# Display the reference guide
common_issues_reference()
```

## Conclusion

In this tutorial, you've learned:

1. How to inspect and debug Mithril models using various techniques
2. Common issues in model development and how to fix them
3. Methods for visualizing model behavior and detecting problems
4. Best practices for building reliable models
5. Systematic approaches to debugging across different backends

Effective debugging is essential for successful model development. By understanding Mithril's model structure and using the techniques in this tutorial, you can quickly identify and resolve issues, leading to more robust and reliable models.

## Next Steps

- Create custom debugging tools tailored to your specific models
- Implement comprehensive test suites for your projects
- Set up continuous integration to catch issues early
- Explore advanced profiling tools for performance optimization
- Contribute improved error messages and debugging tools to Mithril