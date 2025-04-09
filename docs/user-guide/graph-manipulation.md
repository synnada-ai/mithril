# Graph Manipulation in Mithril

Mithril provides powerful tools for manipulating computational graphs, allowing you to optimize performance, customize model structure, and implement complex transformations. This guide explores how to work with Mithril's graph representations and perform common graph manipulation operations.

## Understanding Mithril's Graph Representation

Mithril represents models as computational graphs with two distinct abstractions:

1. **Logical Models**: High-level, user-friendly representations that define the model structure
2. **Physical Models**: Compiled, backend-optimized graphs used for execution

During compilation, Mithril transforms logical models into physical graphs, applying optimizations and transformations along the way. Understanding this process is key to effective graph manipulation.

### Graph Components

The core components of Mithril's graph representation include:

- **Nodes**: Operators that perform computations (e.g., Linear, Convolution2D)
- **Connections**: Edges that connect operators and carry tensors between them
- **Keys**: Unique identifiers for connections and model interfaces

This structure is represented by the `FlatGraph` class, which stores the complete computational graph and enables various manipulations.

## Inspecting a Model's Graph Structure

To understand a model's graph structure, you can use various inspection tools:

```python
import mithril as ml
from mithril.models import Model, Linear, Relu

# Create a simple model
model = Model()
model += Linear(64)(input="input")
model += Relu()
model += Linear(10)(output="output")

# Compile the model to create the physical graph
backend = ml.TorchBackend()
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 100]},
    data_keys={"input"}
)

# Print model summary
print(model.summary())

# Access the physical model's flat graph
flat_graph = compiled_model.model.graph

# Inspect connections
print("Connections:")
for key, conn in flat_graph.connections.items():
    print(f"  {key}: sources={conn.source_keys}, targets={conn.target_keys}")

# Inspect topological order
print("Topological order:")
for key in flat_graph.topological_order:
    print(f"  {key}")
```

## Common Graph Manipulation Operations

### 1. Adding New Connections

You can modify a model's graph by adding new connections:

```python
from mithril.models import Add, Multiply

def add_skip_connection(model: Model, from_layer: str, to_layer: str):
    """Add a skip connection between two layers in a model"""
    # Create a placeholder for the skip connection
    skip_name = f"skip_{from_layer}_to_{to_layer}"
    
    # Add a buffer to capture the output of the source layer
    model |= ml.models.Buffer()(
        input=from_layer, 
        output=skip_name
    )
    
    # Add the skip connection to the target layer
    model |= Add()(
        left=to_layer, 
        right=skip_name, 
        output=f"{to_layer}_with_skip"
    )
    
    return model

# Example usage:
model = Model()
model += Linear(64, name="layer1")(input="input")
model += Relu(name="relu1")
model += Linear(64, name="layer2")
model += Relu(name="relu2")
model += Linear(10, name="output_layer")(output="output")

# Add skip connection from layer1 to layer2
model = add_skip_connection(model, "layer1", "layer2")
```

### 2. Inserting Operators

You can insert new operators into the graph at specific positions:

```python
def insert_batch_norm(model: Model, target_layer: str):
    """Insert batch normalization after a specific layer"""
    # Get the source connection
    conn = model.conns.get(target_layer)
    if conn is None:
        raise ValueError(f"Layer {target_layer} not found in model")
    
    # Create a batch normalization operator
    bn_name = f"{target_layer}_bn"
    bn = ml.models.BatchNorm2D(
        num_features=conn.shape[1],  # Assume NCHW format
        name=bn_name
    )
    
    # Find all target connections of the source layer
    targets = []
    for name, connection in model.conns.items():
        if target_layer in connection.input_connections:
            targets.append(name)
    
    # Insert the batch normalization layer
    model |= bn(input=target_layer, output=f"{target_layer}_bn_out")
    
    # Reconnect all targets to use the batch normalization output
    for target in targets:
        # Update the input connection
        input_conn = model.components[target].input_connections
        input_conn[target_layer] = f"{target_layer}_bn_out"
    
    return model
```

### 3. Pruning Connections

Removing unused connections can optimize model performance:

```python
def prune_unused_connections(compiled_model):
    """Prune unused connections from a compiled model"""
    # Get the flat graph
    flat_graph = compiled_model.model.graph
    
    # Find the model's output keys
    output_keys = set(flat_graph.output_keys)
    
    # Find connections that are necessary to compute the outputs
    necessary_keys = set()
    queue = list(output_keys)
    while queue:
        key = queue.pop(0)
        if key in necessary_keys:
            continue
        
        necessary_keys.add(key)
        source_keys = flat_graph.get_source_keys(key)
        queue.extend(source_keys)
    
    # Identify connections that can be pruned
    all_keys = set(flat_graph.connections.keys())
    prunable_keys = all_keys - necessary_keys - set(flat_graph.input_keys)
    
    # Print pruning information
    print(f"Found {len(prunable_keys)} connections that can be pruned:")
    for key in prunable_keys:
        print(f"  {key}")
    
    # Note: In a real implementation, you would modify the flat_graph to remove these connections
    return prunable_keys
```

### 4. Merging Operations

Combining multiple operations can improve performance:

```python
def merge_consecutive_linear_layers(model: Model):
    """Merge consecutive linear layers without activations"""
    # Find linear layers in the model
    linear_layers = []
    for name, component in model.components.items():
        if isinstance(component, ml.models.Linear):
            linear_layers.append((name, component))
    
    # Find consecutive linear layers
    merged_layers = []
    for i, (name1, layer1) in enumerate(linear_layers):
        for name2, layer2 in linear_layers:
            # Check if layer2 directly uses the output of layer1
            if layer2.input_connections.get("input") == name1:
                # These layers can be merged
                merged_layers.append((name1, name2))
                print(f"Found mergeable layers: {name1} -> {name2}")
                
                # In a real implementation, you would create a new Linear layer
                # with the combined weights and replace the two layers
    
    return merged_layers
```

## Advanced Graph Transformations

### 1. Model Fusion

Fusing models combines multiple smaller models into a single optimized model:

```python
def fuse_models(models: list[Model]) -> Model:
    """Fuse multiple models into a single model"""
    fused_model = Model()
    
    # Keep track of key mappings
    key_mappings = {}
    
    # Add each model's components
    for idx, model in enumerate(models):
        model_prefix = f"model_{idx}_"
        
        # Add all components with prefixed names
        for name, component in model.components.items():
            prefixed_name = f"{model_prefix}{name}"
            
            # Create a copy of the component with the new name
            component_copy = component.clone(name=prefixed_name)
            
            # Update input and output connections
            input_connections = {}
            for in_key, conn in component.input_connections.items():
                if conn in key_mappings:
                    # Use existing mapping
                    input_connections[in_key] = key_mappings[conn]
                else:
                    # Create new prefixed connection
                    prefixed_conn = f"{model_prefix}{conn}"
                    input_connections[in_key] = prefixed_conn
                    key_mappings[conn] = prefixed_conn
            
            # Add the component to the fused model
            fused_model.add_component(component_copy, input_connections)
            
            # Update mappings for this component's output
            if component.output_key:
                key_mappings[component.output_key] = f"{model_prefix}{component.output_key}"
    
    # Set input and output keys for the fused model
    fused_model.input_keys = {key_mappings[key] for key in models[0].input_keys}
    fused_model.output_keys = {key_mappings[key] for key in models[-1].output_keys}
    
    return fused_model
```

### 2. Graph Rewiring

Rewiring allows you to change the flow of data through a model:

```python
def rewire_model(model: Model, rewiring_map: dict[str, str]) -> Model:
    """Rewire a model based on a mapping of source to target connections"""
    # Create a deep copy of the model to avoid modifying the original
    rewired_model = model.clone()
    
    # Update connections based on the rewiring map
    for source_key, target_key in rewiring_map.items():
        # Find all components that use source_key as input
        for name, component in rewired_model.components.items():
            for in_key, conn in list(component.input_connections.items()):
                if conn == source_key:
                    # Rewire the connection to the target key
                    component.input_connections[in_key] = target_key
                    print(f"Rewired {name}.{in_key} from {source_key} to {target_key}")
    
    return rewired_model
```

### 3. Model Quantization

Graph transformation can enable model quantization for reduced memory footprint and faster inference:

```python
def quantize_model(model: Model, bit_width: int = 8) -> Model:
    """Quantize a model to lower precision"""
    # Create a quantized model
    quantized_model = Model()
    
    # Add quantization parameters
    quantized_model.scales = {}
    quantized_model.zero_points = {}
    
    # Process each component in the original model
    for name, component in model.components.items():
        # Handle linear layers specially
        if isinstance(component, ml.models.Linear):
            # Create quantizable linear layer
            quantized_component = ml.models.QuantizedLinear(
                in_features=component.in_features,
                out_features=component.out_features,
                bit_width=bit_width,
                name=name
            )
        else:
            # For other layers, just copy the component
            quantized_component = component.clone()
        
        # Add the component with the same connections
        quantized_model.add_component(
            quantized_component, 
            dict(component.input_connections)
        )
    
    # Copy input and output keys
    quantized_model.input_keys = set(model.input_keys)
    quantized_model.output_keys = set(model.output_keys)
    
    return quantized_model
```

## Working with Physical Graphs

While logical models provide a high-level interface, sometimes you need to work directly with physical graphs for advanced optimizations:

```python
def optimize_physical_graph(compiled_model):
    """Apply optimizations directly to the physical graph"""
    # Get the flat graph
    flat_graph = compiled_model.model.graph
    
    # 1. Prune duplicate connections
    flat_graph.prune_duplicate_connections(
        flat_graph.all_data,
        compiled_model.model.constant_keys
    )
    
    # 2. Infer static keys
    updates = flat_graph.infer_static_keys()
    
    # 3. Update constraints based on the inferred static keys
    flat_graph.constraint_solver(updates)
    
    # 4. Perform backend-specific graph updates
    flat_graph.graph_update()
    
    return compiled_model
```

## Custom Graph Constraints

You can define custom constraints for graph optimization:

```python
from mithril.framework.common import Updates, UpdateType

def add_custom_constraint(model: Model, source_key: str, target_key: str):
    """Add a custom constraint between two connections"""
    # Compile the model to access the physical graph
    backend = ml.TorchBackend()
    compiled_model = ml.compile(
        model=model,
        backend=backend,
        shapes={"input": [32, 100]},
        data_keys={"input"}
    )
    
    # Access the flat graph
    flat_graph = compiled_model.model.graph
    
    # Get the data objects for the source and target
    source_data = flat_graph.all_data[source_key]
    target_data = flat_graph.all_data[target_key]
    
    # Create an update that matches the shapes
    updates = Updates()
    updates |= source_data.shape.match(target_data.shape)
    
    # Apply the updates
    flat_graph.constraint_solver(updates)
    
    return compiled_model
```

## Best Practices for Graph Manipulation

When working with Mithril's graph representation, follow these best practices:

1. **Understand the Graph Structure**: Before manipulating a graph, inspect its structure to understand connections and dependencies.

2. **Prefer High-Level APIs**: When possible, use Mithril's high-level APIs for model composition rather than directly manipulating the graph.

3. **Validate after Modifications**: After modifying a graph, validate that it remains coherent by checking for hanging connections or cycles.

4. **Consider Performance Implications**: Graph manipulations can have significant performance implications. Test your modifications to ensure they improve rather than degrade performance.

5. **Use Named Components**: Giving meaningful names to components makes graph manipulation easier by providing clear references.

6. **Test Across Backends**: Graph optimizations might have different effects across backends. Test your modifications on all target backends.

## Debugging Graph Issues

If you encounter issues with graph manipulation, these techniques can help:

```python
def debug_graph(compiled_model):
    """Debug issues in a model's graph"""
    # Get the flat graph
    flat_graph = compiled_model.model.graph
    
    # Check for hanging connections
    hanging_keys = flat_graph.hanging_keys
    if hanging_keys:
        print(f"Warning: Found {len(hanging_keys)} hanging connections:")
        for key in hanging_keys:
            print(f"  {key}")
    
    # Check for shape inconsistencies
    for key, conn in flat_graph.connections.items():
        if key in flat_graph.all_data:
            data = flat_graph.all_data[key]
            if data.is_tensor and data.shape.is_valued:
                print(f"  {key}: shape={data.shape.value}")
            else:
                print(f"  {key}: shape not fully determined")
    
    # Check for type inconsistencies
    for key in flat_graph.all_keys:
        if key in flat_graph.all_data:
            data = flat_graph.all_data[key]
            if data.value is not None:
                print(f"  {key}: type={type(data.value)}")
    
    return compiled_model
```

## Conclusion

Graph manipulation in Mithril provides powerful tools for optimizing models, implementing custom transformations, and achieving fine-grained control over model architecture. By understanding the structure of Mithril's graphs and the available manipulation techniques, you can create highly optimized and efficient models tailored to your specific needs.

Whether you're working with high-level logical models or low-level physical graphs, Mithril's flexible architecture enables a wide range of graph manipulations that can significantly improve model performance and capabilities.