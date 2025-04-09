# Computer Vision Models with Mithril

This document explores how to implement various computer vision models in Mithril, from simple CNN architectures to state-of-the-art models like ResNet and vision transformers.

## Basic CNN Architectures

### Simple CNN for Time Series Forecasting

Mithril makes it easy to build and train CNN models for various tasks. Here's an example of a 1D CNN for time series forecasting:

```python
# Create a simple CNN model
cnn_model = Model()
cnn_model |= Convolution1D(out_channels=16, kernel_size=5, stride=1, padding=2)
cnn_model += Relu()
cnn_model += MaxPool1D(kernel_size=2, stride=2)
cnn_model += Flatten(start_dim=1)
cnn_model += Linear(64)
cnn_model += Relu()
cnn_model += Linear(1)
```

This model uses 1D convolutions to process time series data, with pooling layers to reduce dimensionality and fully connected layers to make predictions.

### Multi-task CNN with SVM Classifier

Mithril supports building more complex models with multiple outputs and different loss functions:

```python
# Define backbone
backbone = Model()
backbone += Convolution2D(kernel_size=3, out_channels=8, stride=2, padding=1)
backbone += MaxPool2D(kernel_size=2, stride=2)
backbone += Convolution2D(kernel_size=3, out_channels=16, stride=2, padding=1)
backbone += MaxPool2D(kernel_size=2, stride=2)
backbone += Convolution2D(kernel_size=3, out_channels=32, stride=2, padding=1)
backbone += Flatten(start_dim=1)

# Define two MLP towers for two tasks
age_head = MLP(
    activations=[LeakyRelu(), LeakyRelu(), Buffer()], 
    dimensions=[256, 64, 1]
)

gender_head = MLP(
    activations=[LeakyRelu(), LeakyRelu(), Buffer()], 
    dimensions=[32, 16, 1]
)

# Define classifier with SVM
SVM_Model = LinearSVM()

logical_model = Model()
logical_model |= backbone
logical_model |= age_head(input=backbone.cout, output=ml.IOKey("age"))
logical_model |= gender_head(input=backbone.cout)
logical_model |= SVM_Model(
    input=gender_head.output,
    output=ml.IOKey("gender"),
    decision_output=ml.IOKey("pred_gender"),
)
```

This model demonstrates a multi-task architecture with a shared backbone and separate task-specific heads, including a Support Vector Machine classifier.

## ResNet Models

ResNet revolutionized deep learning with its residual connections that enable training very deep networks. Mithril provides a clean implementation:

### Basic ResNet Building Blocks

```python
def basic_block(
    out_channels: int, stride: int = 1, downsample: Model | None = None
) -> Model:
    block = Model()
    block += Convolution2D(
        kernel_size=3, out_channels=out_channels, padding=1, stride=stride
    )
    model_input = block.cin
    block += Relu()
    block += Convolution2D(kernel_size=3, out_channels=out_channels, padding=1)
    skip_in = block.cout

    if downsample is not None:
        block |= downsample(input=model_input)
        block |= Add()(left=downsample.cout, right=skip_in)
    else:
        block |= Add()(left=model_input, right=skip_in)

    block += Relu()
    return block
```

### Building Full ResNet Models

Mithril provides implementations for different ResNet variants:

```python
def resnet(n_classes: int, block: Callable, layers: list[int]) -> Model:
    resnet = Model()
    resnet += Convolution2D(kernel_size=7, out_channels=64, stride=2, padding=3)
    resnet += make_layer(64, block, n_blocks=layers[0], stride=1)
    resnet += make_layer(128, block, n_blocks=layers[1], stride=2)
    resnet += make_layer(256, block, n_blocks=layers[2], stride=2)
    resnet += make_layer(512, block, n_blocks=layers[3], stride=2)
    resnet += Flatten(start_dim=1)
    resnet |= Linear(dimension=n_classes)(
        input=resnet.cout, output=IOKey(name="output")
    )
    return resnet

# Predefined ResNet variants
def resnet18(n_classes: int):
    return resnet(n_classes=n_classes, block=basic_block, layers=[2, 2, 2, 2])

def resnet34(n_classes: int):
    return resnet(n_classes=n_classes, block=basic_block, layers=[2, 4, 6, 3])

def resnet50(n_classes: int):
    return resnet(n_classes=n_classes, block=bottleneck, layers=[2, 4, 6, 3])

def resnet101(n_classes: int):
    return resnet(n_classes=n_classes, block=bottleneck, layers=[3, 4, 23, 3])

def resnet152(n_classes: int):
    return resnet(n_classes=n_classes, block=bottleneck, layers=[3, 8, 36, 3])
```

## Vision Transformers (ViT)

Mithril also supports implementing Vision Transformers for image recognition, which use self-attention mechanisms instead of convolutions:

```python
def vision_transformer(
    input_resolution: int,
    patch_size: int,
    width: int,
    layers: int,
    heads: int,
    output_dim: int,
    use_proj: bool = False,
    name: str | None = None,
):
    block = Model(name=name)
    input = IOKey("input")

    # Image to patches via convolution
    block |= Convolution2D(
        kernel_size=patch_size,
        out_channels=width,
        stride=patch_size,
        use_bias=False,
        name="conv1",
    )(input=input, output="conv1")
    
    # Reshape and add positional and class embeddings
    # ... (additional processing code)
    
    # Apply transformer blocks
    transformer_visual = transformer(width, layers, heads, name="transformer")
    block |= transformer_visual(
        input=block.ln_1.transpose((1, 0, 2)),
        output="transformer",
    )
    
    # Output projection and normalization
    # ... (final processing code)
    
    return block
```

This implementation demonstrates how Mithril can express complex vision transformer architectures with clear, composable code.

## CLIP: Multimodal Learning

Mithril can implement advanced multimodal models like CLIP (Contrastive Language-Image Pre-training), which connects images and text:

```python
def clip(
    embed_dim: int,
    # vision
    image_resolution: int,
    vision_layers: tuple[int, int, int, int] | int,
    vision_width: int,
    vision_patch_size: int,
    # text
    context_length: int,
    vocab_size: int,
    transformer_width: int,
    transformer_heads: int,
    transformer_layers: int,
    name: str | None = None,
):
    block = Model(name=name)
    image = IOKey("image", type=ml.Tensor, shape=["N", 3, image_resolution, image_resolution])
    text = IOKey("text", type=ml.Tensor, shape=["M", context_length])

    # Vision encoder (either ResNet or ViT)
    if isinstance(vision_layers, tuple | list):
        # Use ResNet
        vision_heads = vision_width * 32 // 64
        visual = modified_resnet(...)
    else:
        # Use Vision Transformer
        vision_heads = vision_width // 64
        visual = vision_transformer(...)
    
    # Text encoder (Transformer)
    block |= Embedding(vocab_size, transformer_width, name="token_embedding")(...)
    transformer_main = transformer(...)
    
    # Compute similarity between image and text embeddings
    image_features = block.image_features / block.image_features_norm
    text_features = block.text_features / block.text_features_norm
    logits_per_image = logit_scale.exp() * (image_features @ text_features.transpose())
    logits_per_text = logits_per_image.transpose()
    
    return block
```

CLIP combines vision and text models to learn a joint embedding space, enabling powerful zero-shot image classification and text-image retrieval capabilities.

## Backend Compatibility

All these computer vision models can be compiled to run on different backends:

```python
# Choose a backend
backend = ml.TorchBackend()  # or ml.JaxBackend(), ml.MlxBackend()

# Compile the model
compiled_model = ml.compile(
    model=resnet18(10),
    backend=backend,
    shapes={"input": [32, 3, 224, 224]},
    data_keys={"input"},
)

# Run inference
outputs = compiled_model.evaluate(params, {"input": images})
```

This backend-agnostic approach allows the same model code to run efficiently on different hardware platforms and frameworks.