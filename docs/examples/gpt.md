# GPT Example

This example demonstrates how to implement a GPT (Generative Pre-trained Transformer) model using Mithril. GPT models are autoregressive language models that use the transformer architecture to generate text.

## Overview

GPT-style models consist of a stack of transformer decoder blocks, which include:

1. Self-attention mechanisms
2. Feed-forward neural networks
3. Layer normalization
4. Residual connections

This example implements a simplified version of the GPT architecture while showcasing Mithril's capabilities for building complex models.

## Implementation

Below is the implementation of a basic GPT model in Mithril:

```python
import mithril as mi
from mithril.models import LogicalModel
import numpy as np

def create_gpt_model(
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768,
    context_length=1024,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
):
    """Create a GPT model with the specified parameters."""
    model = LogicalModel("gpt")
    
    with model:
        # Input is a sequence of token indices
        tokens = mi.Input(shape=(None, context_length), dtype="int32", name="tokens")
        
        # Token embedding table
        wte = mi.Parameter(shape=(vocab_size, n_embd), name="wte")
        
        # Position embedding table
        wpe = mi.Parameter(shape=(context_length, n_embd), name="wpe")
        
        # Get token embeddings
        token_embeddings = mi.embedding(tokens, wte)
        
        # Create position indices and get position embeddings
        batch_size = mi.shape(tokens)[0]
        positions = mi.range(0, context_length, dtype="int32")
        positions = mi.reshape(positions, (1, context_length))
        positions = mi.broadcast_to(positions, (batch_size, context_length))
        position_embeddings = mi.embedding(positions, wpe)
        
        # Combined embeddings
        h = token_embeddings + position_embeddings
        
        # Embedding dropout
        h = mi.dropout(h, embd_pdrop, training=mi.Input(shape=(), dtype="bool", name="training"))
        
        # Transformer blocks
        for i in range(n_layer):
            h = transformer_block(
                h, 
                n_head=n_head, 
                n_embd=n_embd, 
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop, 
                layer_idx=i
            )
        
        # Final layer normalization
        ln_f_g = mi.Parameter(shape=(n_embd,), name="ln_f_g")
        ln_f_b = mi.Parameter(shape=(n_embd,), name="ln_f_b")
        h = mi.layer_norm(h, ln_f_g, ln_f_b)
        
        # Language modeling head
        h = mi.matmul(h, mi.transpose(wte))  # Tied weights with token embedding
        
        # Output logits
        mi.Output(h, name="logits")
    
    return model

def transformer_block(x, n_head, n_embd, attn_pdrop, resid_pdrop, layer_idx):
    """Implement a single transformer block."""
    # Layer normalization for attention
    ln_1_g = mi.Parameter(shape=(n_embd,), name=f"ln_1_g_{layer_idx}")
    ln_1_b = mi.Parameter(shape=(n_embd,), name=f"ln_1_b_{layer_idx}")
    ln_1 = mi.layer_norm(x, ln_1_g, ln_1_b)
    
    # Multi-head attention
    attn = attention(ln_1, n_head, n_embd, attn_pdrop, layer_idx)
    
    # Residual connection
    x = x + attn
    
    # Layer normalization for MLP
    ln_2_g = mi.Parameter(shape=(n_embd,), name=f"ln_2_g_{layer_idx}")
    ln_2_b = mi.Parameter(shape=(n_embd,), name=f"ln_2_b_{layer_idx}")
    ln_2 = mi.layer_norm(x, ln_2_g, ln_2_b)
    
    # MLP
    mlp_out = mlp(ln_2, n_embd, layer_idx)
    
    # Residual connection and dropout
    x = x + mi.dropout(mlp_out, resid_pdrop, training=mi.get_node("training"))
    
    return x

def attention(x, n_head, n_embd, attn_pdrop, layer_idx):
    """Implement multi-head attention."""
    # Reshape input to [batch, seq_len, n_head, n_embd/n_head]
    batch_size, seq_len, _ = mi.shape(x)
    head_dim = n_embd // n_head
    
    # Query, key, value projections
    c_attn_w = mi.Parameter(shape=(n_embd, 3 * n_embd), name=f"c_attn_w_{layer_idx}")
    c_attn_b = mi.Parameter(shape=(3 * n_embd,), name=f"c_attn_b_{layer_idx}")
    
    qkv = mi.matmul(x, c_attn_w) + c_attn_b
    
    # Split into query, key, value
    query, key, value = mi.split(qkv, 3, axis=-1)
    
    # Reshape for multi-head attention
    # [batch, seq_len, n_head, head_dim]
    query = mi.reshape(query, (batch_size, seq_len, n_head, head_dim))
    key = mi.reshape(key, (batch_size, seq_len, n_head, head_dim))
    value = mi.reshape(value, (batch_size, seq_len, n_head, head_dim))
    
    # Transpose to [batch, n_head, seq_len, head_dim]
    query = mi.transpose(query, (0, 2, 1, 3))
    key = mi.transpose(key, (0, 2, 1, 3))
    value = mi.transpose(value, (0, 2, 1, 3))
    
    # Compute attention scores
    # [batch, n_head, seq_len, seq_len]
    scale = mi.constant(1.0 / (head_dim ** 0.5), dtype="float32")
    scores = mi.matmul(query, mi.transpose(key, (0, 1, 3, 2))) * scale
    
    # Mask (lower triangular for causal/auto-regressive attention)
    mask = mi.tril(mi.ones((seq_len, seq_len), dtype="float32"))
    mask = mi.reshape(mask, (1, 1, seq_len, seq_len))
    mask = mi.broadcast_to(mask, (batch_size, n_head, seq_len, seq_len))
    
    # Apply mask (add large negative value to masked positions)
    masked_scores = scores + (1.0 - mask) * -1e10
    
    # Apply attention weights
    attn_weights = mi.softmax(masked_scores, axis=-1)
    attn_weights = mi.dropout(attn_weights, attn_pdrop, training=mi.get_node("training"))
    
    # Compute weighted sum
    # [batch, n_head, seq_len, head_dim]
    attn_output = mi.matmul(attn_weights, value)
    
    # Reshape back to [batch, seq_len, n_embd]
    attn_output = mi.transpose(attn_output, (0, 2, 1, 3))
    attn_output = mi.reshape(attn_output, (batch_size, seq_len, n_embd))
    
    # Output projection
    c_proj_w = mi.Parameter(shape=(n_embd, n_embd), name=f"c_proj_w_{layer_idx}")
    c_proj_b = mi.Parameter(shape=(n_embd,), name=f"c_proj_b_{layer_idx}")
    
    return mi.matmul(attn_output, c_proj_w) + c_proj_b

def mlp(x, n_embd, layer_idx):
    """Implement the MLP block of a transformer."""
    # Expand into larger representation
    c_fc_w = mi.Parameter(shape=(n_embd, 4 * n_embd), name=f"c_fc_w_{layer_idx}")
    c_fc_b = mi.Parameter(shape=(4 * n_embd,), name=f"c_fc_b_{layer_idx}")
    
    h = mi.matmul(x, c_fc_w) + c_fc_b
    h = mi.gelu(h)
    
    # Project back to residual size
    c_proj_w = mi.Parameter(shape=(4 * n_embd, n_embd), name=f"c_mlp_proj_w_{layer_idx}")
    c_proj_b = mi.Parameter(shape=(n_embd,), name=f"c_mlp_proj_b_{layer_idx}")
    
    return mi.matmul(h, c_proj_w) + c_proj_b
```

## Using the Model

Here's how to use the GPT model for text generation:

```python
# Create the model
gpt_model = create_gpt_model(
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768,
    context_length=1024
)

# Compile the model with a JAX backend
from mithril.backends.with_autograd.jax_backend import JaxBackend
backend = JaxBackend()
compiled_model = gpt_model.compile(backend)

# Create a simple tokenizer (for demonstration only)
def simple_tokenizer(text, vocab_size=50257):
    # In a real application, you'd use a proper tokenizer like GPT-2's tokenizer
    tokens = [ord(c) % vocab_size for c in text]
    return np.array(tokens, dtype=np.int32)

def generate_text(model, prompt, max_length=100, temperature=1.0):
    # Tokenize the prompt
    tokens = simple_tokenizer(prompt)
    
    # Pad or truncate to context_length - 1 (leave room for one prediction)
    if len(tokens) >= 1023:
        tokens = tokens[:1023]
    else:
        tokens = np.pad(tokens, (0, 1023 - len(tokens)), mode='constant')
    
    # Reshape for model input [batch_size=1, context_length=1024]
    token_input = np.expand_dims(tokens, axis=0)
    
    generated_text = prompt
    
    # Generate tokens one by one
    for _ in range(max_length):
        # Run the model
        outputs = model({"tokens": token_input, "training": False})
        logits = outputs["logits"]
        
        # Get the next token prediction (last position)
        next_token_logits = logits[0, len(tokens) - 1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Convert to probabilities
        probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
        
        # Sample from the distribution
        next_token = np.random.choice(len(probs), p=probs)
        
        # Convert to character and append to result
        next_char = chr(next_token % 128)  # Map to ASCII for demonstration
        generated_text += next_char
        
        # Update token input for next iteration
        tokens = np.append(tokens[1:], next_token)
        token_input = np.expand_dims(tokens, axis=0)
    
    return generated_text

# Example usage
prompt = "Once upon a time in a faraway land, there lived a"
generated_text = generate_text(compiled_model, prompt, max_length=100, temperature=0.7)
print(generated_text)
```

## Training the Model

Here's a simplified example of how to train the GPT model:

```python
from mithril.models.train_model import train

# Create loss function for language modeling
def lm_loss(outputs):
    logits = outputs["logits"]  # [batch, seq_len, vocab_size]
    
    # Get the input tokens
    tokens = outputs["tokens"]
    
    # Shift targets (predict next token)
    targets = tokens[:, 1:]  # Remove first token
    shifted_logits = logits[:, :-1, :]  # Remove last prediction
    
    # Compute cross-entropy loss
    loss = mi.categorical_crossentropy(
        targets, 
        shifted_logits,
        from_logits=True
    )
    
    # Average over non-padding tokens
    return mi.reduce_mean(loss)

# Create optimizer
from mithril.models.optimizers import Adam
optimizer = Adam(learning_rate=3e-5)

# Train the model
train(
    model=compiled_model,
    train_data=train_dataloader,  # Your data loader
    optimizer=optimizer,
    loss_fn=lm_loss,
    epochs=3,
    validate_fn=validation_function,  # Your validation function
    val_data=val_dataloader         # Your validation data
)
```

## Complete Example

A complete example with a more robust implementation is available in the Mithril repository:

```bash
git clone https://github.com/example/mithril.git
cd mithril/examples/gpt
python model.py
python run_sample.py
```

## Multi-Backend Support

One of the key benefits of Mithril is the ability to run the same model on different backends. Here's how to use GPT with different backends:

```python
# PyTorch backend
from mithril.backends.with_autograd.torch_backend import TorchBackend
torch_backend = TorchBackend(device="cuda" if torch.cuda.is_available() else "cpu")
gpt_torch = gpt_model.compile(torch_backend)

# MLX backend (Apple Silicon)
from mithril.backends.with_autograd.mlx_backend import MlxBackend
mlx_backend = MlxBackend()
gpt_mlx = gpt_model.compile(mlx_backend)

# Compare generation speed across backends
import time

prompt = "Once upon a time"

# JAX
start = time.time()
generated_jax = generate_text(compiled_model, prompt, max_length=100)
end = time.time()
print(f"JAX generation time: {end - start:.2f}s")

# PyTorch
start = time.time()
generated_torch = generate_text(gpt_torch, prompt, max_length=100)
end = time.time()
print(f"PyTorch generation time: {end - start:.2f}s")

# MLX
start = time.time()
generated_mlx = generate_text(gpt_mlx, prompt, max_length=100)
end = time.time()
print(f"MLX generation time: {end - start:.2f}s")
```

## Performance Optimization

To optimize GPT performance, you can use Mithril's advanced compilation features:

```python
# Enable JIT compilation for better performance
jax_backend = JaxBackend(jit_compile=True)

# Compile with optimizations
optimized_gpt = gpt_model.compile(
    backend=jax_backend,
    optimize=True,
    inline_constants=True,
    fuse_operations=True
)
```

## Deploying with C Code Generation

For deployment in constrained environments, you can use Mithril's code generation:

```python
from mithril.framework.codegen.c_gen import CGenerator

# Generate C code for the model
c_generator = CGenerator(optimize=True, use_simd=True)
code, header = c_generator.generate_with_header(gpt_model)

# Save to files
with open("gpt_model.c", "w") as f:
    f.write(code)
    
with open("gpt_model.h", "w") as f:
    f.write(header)

# Compile with gcc
import subprocess
subprocess.run(["gcc", "-O3", "-mavx2", "gpt_model.c", "-o", "gpt_model"])
```

## Conclusion

This example demonstrates how to implement and use a GPT model with Mithril. The same model definition can be compiled to run efficiently on different backends, highlighting Mithril's flexibility and backend-agnostic approach. The ability to generate optimized code for various targets makes Mithril suitable for both research and deployment scenarios.