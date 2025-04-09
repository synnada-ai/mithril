# CLIP Example

This example demonstrates how to implement the CLIP (Contrastive Language-Image Pretraining) model using Mithril. CLIP is a neural network trained on a variety of image-text pairs, allowing it to learn visual concepts from natural language supervision.

## Overview

CLIP consists of two encoders:
1. A vision encoder (typically a transformer or CNN) that processes images
2. A text encoder (typically a transformer) that processes text

The model is trained to maximize the similarity between the embeddings of matching image-text pairs while minimizing the similarity between non-matching pairs.

## Implementation

The following example shows how to implement CLIP in Mithril:

```python
import mithril as mi
from mithril.models import LogicalModel
import numpy as np
from PIL import Image

# Vision Encoder
def create_vision_encoder(input_shape=(3, 224, 224), embed_dim=512):
    model = LogicalModel("vision_encoder")
    with model:
        # Input is an RGB image
        image = mi.Input(shape=input_shape, name="image")
        
        # Use a simple CNN backbone for demonstration
        # In practice, this would be a more complex model like ResNet or ViT
        x = mi.conv2d(image, filters=64, kernel_size=7, strides=2, padding="SAME")
        x = mi.relu(x)
        x = mi.max_pool2d(x, pool_size=3, strides=2, padding="SAME")
        
        # Additional convolutional blocks would be here
        x = mi.conv2d(x, filters=128, kernel_size=3, strides=1, padding="SAME")
        x = mi.relu(x)
        x = mi.max_pool2d(x, pool_size=2, strides=2, padding="SAME")
        
        x = mi.conv2d(x, filters=256, kernel_size=3, strides=1, padding="SAME")
        x = mi.relu(x)
        x = mi.max_pool2d(x, pool_size=2, strides=2, padding="SAME")
        
        # Global pooling and projection to embedding dimension
        x = mi.global_avg_pool2d(x)
        x = mi.flatten(x)
        image_embedding = mi.dense(x, embed_dim)
        
        # Normalize embedding to unit length
        image_embedding = mi.normalize(image_embedding, axis=-1)
        
        mi.Output(image_embedding, name="image_embedding")
    
    return model

# Text Encoder
def create_text_encoder(vocab_size=10000, max_seq_len=77, embed_dim=512):
    model = LogicalModel("text_encoder")
    with model:
        # Input is a sequence of token indices
        text = mi.Input(shape=(max_seq_len,), dtype="int32", name="text")
        
        # Token embedding
        token_embedding = mi.Parameter(shape=(vocab_size, embed_dim), name="token_embedding")
        x = mi.embedding(text, token_embedding)
        
        # Position embedding
        position_embedding = mi.Parameter(shape=(max_seq_len, embed_dim), name="position_embedding")
        positions = mi.range(0, max_seq_len, dtype="int32")
        pos_embed = mi.embedding(positions, position_embedding)
        
        # Add position embeddings
        x = x + pos_embed
        
        # Transformer layers would be here
        # For simplicity, we'll use a simple MLP instead
        x = mi.dense(x, embed_dim*2)
        x = mi.relu(x)
        x = mi.dense(x, embed_dim)
        
        # Average pooling over sequence length
        text_embedding = mi.reduce_mean(x, axis=1)
        
        # Normalize embedding to unit length
        text_embedding = mi.normalize(text_embedding, axis=-1)
        
        mi.Output(text_embedding, name="text_embedding")
    
    return model

# Combined CLIP model
def create_clip_model(vision_encoder, text_encoder):
    model = LogicalModel("clip")
    with model:
        # Image input
        image = mi.Input(shape=(3, 224, 224), name="image")
        
        # Text input
        text = mi.Input(shape=(77,), dtype="int32", name="text")
        
        # Encode image and text
        image_embedding = vision_encoder(image)
        text_embedding = text_encoder(text)
        
        # Compute similarity
        # Scale by temperature (learned or fixed)
        temperature = mi.Parameter(shape=(1,), name="temperature", initializer=0.07)
        similarity = mi.matmul(image_embedding, mi.transpose(text_embedding)) / temperature
        
        mi.Output(image_embedding, name="image_embedding")
        mi.Output(text_embedding, name="text_embedding")
        mi.Output(similarity, name="similarity")
    
    return model
```

## Using the Model

Here's how to use the CLIP model for inference:

```python
# Create encoders and combined model
vision_encoder = create_vision_encoder()
text_encoder = create_text_encoder()
clip_model = create_clip_model(vision_encoder, text_encoder)

# Compile the model with a JAX backend
from mithril.backends.with_autograd.jax_backend import JaxBackend
backend = JaxBackend()
compiled_model = clip_model.compile(backend)

# Load an image
from PIL import Image
image = Image.open("cat.jpg").resize((224, 224))
image_array = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0

# Tokenize text
# In practice, you'd use a proper tokenizer
def simple_tokenize(text, max_len=77, vocab_size=10000):
    # Dummy tokenization - just for demonstration
    tokens = np.zeros(max_len, dtype=np.int32)
    for i, char in enumerate(text[:max_len]):
        tokens[i] = ord(char) % vocab_size
    return tokens

texts = [
    "a photo of a cat",
    "a photo of a dog",
    "a rendering of a 3D model",
    "a painting of a landscape"
]
tokenized_texts = np.stack([simple_tokenize(text) for text in texts])

# Run inference
# Batch the image to match text batch size
batched_image = np.repeat(image_array[np.newaxis], len(texts), axis=0)
inputs = {"image": batched_image, "text": tokenized_texts}
outputs = compiled_model(inputs)

# Get the similarity scores
similarity_scores = outputs["similarity"]
print("Similarity scores:")
for i, text in enumerate(texts):
    print(f"{text}: {similarity_scores[0, i]}")

# Get the most similar text
best_match_idx = np.argmax(similarity_scores[0])
print(f"\nBest match: {texts[best_match_idx]}")
```

## Training the Model

Here's a simplified example of how to train the CLIP model:

```python
from mithril.models.train_model import train

# Create loss function for contrastive learning
def contrastive_loss(outputs):
    similarity = outputs["similarity"]
    batch_size = similarity.shape[0]
    
    # Labels are the diagonal elements (matching pairs)
    labels = np.eye(batch_size)
    
    # Cross-entropy loss (both directions)
    loss_i2t = mi.categorical_crossentropy(labels, mi.softmax(similarity, axis=1))
    loss_t2i = mi.categorical_crossentropy(labels, mi.softmax(similarity, axis=0))
    
    # Average the losses
    loss = (loss_i2t + loss_t2i) / 2.0
    return loss

# Create optimizer
from mithril.models.optimizers import Adam
optimizer = Adam(learning_rate=5e-5)

# Train the model
train(
    model=compiled_model,
    train_data=train_dataloader,  # Your data loader
    optimizer=optimizer,
    loss_fn=contrastive_loss,
    epochs=50,
    validate_fn=validation_function,  # Your validation function
    val_data=val_dataloader         # Your validation data
)
```

## Complete Example

A complete example with training and evaluation is available in the Mithril repository:

```bash
git clone https://github.com/example/mithril.git
cd mithril/examples/clip
python model.py
```

## Multi-Backend Support

One of the key benefits of Mithril is the ability to run the same model on different backends. Here's how to use CLIP with different backends:

```python
# PyTorch backend
from mithril.backends.with_autograd.torch_backend import TorchBackend
torch_backend = TorchBackend(device="cuda" if torch.cuda.is_available() else "cpu")
clip_torch = clip_model.compile(torch_backend)

# MLX backend (Apple Silicon)
from mithril.backends.with_autograd.mlx_backend import MlxBackend
mlx_backend = MlxBackend()
clip_mlx = clip_model.compile(mlx_backend)

# Compare outputs from different backends
outputs_jax = compiled_model(inputs)
outputs_torch = clip_torch(inputs)
outputs_mlx = clip_mlx(inputs)

print("Similarity scores from JAX backend:", outputs_jax["similarity"][0])
print("Similarity scores from PyTorch backend:", outputs_torch["similarity"][0])
print("Similarity scores from MLX backend:", outputs_mlx["similarity"][0])
```

## Performance Optimization

To optimize CLIP performance, you can use Mithril's advanced compilation features:

```python
# Enable JIT compilation for better performance
jax_backend = JaxBackend(jit_compile=True)

# Compile with optimizations
optimized_clip = clip_model.compile(
    backend=jax_backend,
    optimize=True,
    inline_constants=True,
    fuse_operations=True
)
```

## Conclusion

This example demonstrates how to implement the CLIP model using Mithril. The same model definition can be compiled to run efficiently on different backends, demonstrating Mithril's flexibility and backend-agnostic approach.