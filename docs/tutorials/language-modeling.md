# Language Modeling with Mithril

This tutorial demonstrates how to build and train a transformer-based language model using Mithril. We'll implement a GPT-style autoregressive transformer and train it on a text dataset.

## Overview

In this tutorial, you'll learn:

1. How to create a transformer-based language model in Mithril
2. How to prepare and tokenize text data
3. How to train the model using your preferred backend
4. How to generate text with the trained model

## Prerequisites

Before starting, make sure you have the required libraries:

```bash
pip install mithril jax optax numpy tiktoken
```

## Model Architecture

Let's start by implementing the key components of our transformer-based language model:

```python
import mithril as ml
from mithril import IOKey
from mithril.models import (
    Add, 
    Arange, 
    Embedding, 
    Gelu, 
    LayerNorm, 
    Linear, 
    Model, 
    ScaledDotProduct,
    Dropout,
    Split,
    Transpose
)

# Create self-attention layer
def causal_attention(input_dim, num_heads, dropout_rate=0.1):
    if (input_dim % num_heads) != 0:
        raise ValueError("Requires input dims to be divisible by num_heads")

    model = Model(name="attn")
    
    # Combined projection for query, key, value
    model |= Linear(input_dim * 3, name="c_attn")(
        input="input", output="c_attn_out"
    )

    # Reshape for multi-head attention
    t_axes = (0, 2, 1, 3)
    shp_con = model.input.shape  # type: ignore
    reshape_con = (shp_con[0], shp_con[1], num_heads, -1)

    # Split QKV
    model |= Split(3, axis=-1)(
        model.c_attn_out, output="split_out"
    )  
    
    # Prepare query, key, value tensors
    tq = model.split_out[0].reshape(reshape_con).transpose(t_axes)  # type: ignore
    tk = model.split_out[1].reshape(reshape_con).transpose(t_axes)  # type: ignore
    tv = model.split_out[2].reshape(reshape_con).transpose(t_axes)  # type: ignore

    # Apply scaled dot-product attention with causal mask
    model |= ScaledDotProduct(is_causal=True)(
        query=tq, key=tk, value=tv, output="sdp_out"
    )
    
    # Reshape output
    t_sdp = model.sdp_out.transpose(t_axes).reshape(shp_con[:3])  # type: ignore
    
    # Final projection
    model |= Linear(input_dim, name="c_proj")(
        t_sdp, output="proj_out"
    )
    
    # Apply dropout
    model |= Dropout(p=dropout_rate)(
        input="proj_out", output=IOKey("output")
    )
    
    return model

# MLP block
def mlp(n_embd, dropout_rate=0.1):
    block = Model(name="mlp")
    block += Linear(n_embd * 4, name="c_fc")(input="input")
    block += Gelu()
    block += Linear(n_embd, name="c_proj")
    block += Dropout(p=dropout_rate)(output=IOKey("output"))
    return block

# Transformer block
def create_block(name, dims, num_heads, dropout_rate=0.1):
    block = Model(name=name)
    
    # First sub-block: normalization and self-attention
    block |= LayerNorm(eps=1e-5, name="ln_1")(
        input="input", output="ln_1_out"
    )
    block |= causal_attention(
        dims, num_heads, dropout_rate
    )(input="ln_1_out", output="attn_out")
    block |= Add()(left="input", right="attn_out", output="add_1_out")
    
    # Second sub-block: normalization and MLP
    block |= LayerNorm(eps=1e-5, name="ln_2")(
        input="add_1_out", output="ln_2_out"
    )
    block |= mlp(dims, dropout_rate)(
        input="ln_2_out", output="mlp_out"
    )
    block |= Add()(left="add_1_out", right="mlp_out", output=IOKey("output"))
    
    return block

# Complete GPT-style language model
def create_language_model(
    vocab_size=50257,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    max_seq_len=1024,
    dropout_rate=0.1
):
    model = Model(name="language_model")
    
    # Token and position embeddings
    model += Arange(start=0, stop=max_seq_len)(output="positions")
    model |= Embedding(
        num_embeddings=vocab_size, 
        dim=hidden_dim, 
        name="wte"
    )(input="input", output="token_embeddings")
    model |= Embedding(
        num_embeddings=max_seq_len, 
        dim=hidden_dim, 
        name="wpe"
    )(input="positions", output="position_embeddings")
    
    # Add token and position embeddings
    positions_slice = model.positions[:model.input.shape[1]]  # type: ignore
    position_embeddings_slice = model.wpe(positions_slice)  # type: ignore
    model |= Add()(
        left="token_embeddings", 
        right=position_embeddings_slice, 
        output="embeddings"
    )
    model |= Dropout(p=dropout_rate)(
        input="embeddings", output="embeddings_dropout"
    )
    
    # Add transformer blocks
    input_key = "embeddings_dropout"
    for i in range(num_layers):
        block = create_block(
            name=f"h_{i}",
            dims=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        model |= block(input=input_key, output=f"block_{i}_out")
        input_key = f"block_{i}_out"
    
    # Final layer norm and output
    model |= LayerNorm(eps=1e-5, name="ln_f")(
        input=input_key, output="final_norm"
    )
    model |= Linear(vocab_size, name="lm_head", use_bias=False)(
        input="final_norm", output=IOKey("output")
    )
    
    return model
```

## Data Preparation

Next, let's prepare a text dataset using a tokenizer. We'll use the `tiktoken` library, which is compatible with OpenAI's tokenizers:

```python
import tiktoken
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Tokenizer setup
def get_tokenizer(model_name="gpt2"):
    return tiktoken.get_encoding(model_name)

# Text dataset class
class TextDataset(Dataset):
    def __init__(self, text_file, tokenizer, seq_length=128):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize the entire text
        self.tokens = tokenizer.encode(text)
        self.seq_length = seq_length
        
        # Calculate number of samples
        self.num_samples = max(0, len(self.tokens) - seq_length)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get input and target sequences
        x = self.tokens[idx:idx + self.seq_length]
        y = self.tokens[idx + 1:idx + self.seq_length + 1]
        
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

# Create dataloaders
def create_dataloaders(text_file, tokenizer, seq_length=128, batch_size=32):
    dataset = TextDataset(text_file, tokenizer, seq_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    return dataloader
```

## Model Training

Now, let's set up the training procedure:

```python
import jax
import jax.numpy as jnp
import optax

# Create model and compile it
def setup_model(vocab_size, seq_length, batch_size, backend_type="jax"):
    # Create model
    model = create_language_model(
        vocab_size=vocab_size,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=seq_length,
        dropout_rate=0.1
    )
    
    # Create backend
    if backend_type == "jax":
        backend = ml.JaxBackend(dtype=ml.float32)
    elif backend_type == "torch":
        backend = ml.TorchBackend(dtype=ml.float32)
    elif backend_type == "mlx":
        backend = ml.MlxBackend(dtype=ml.float32)
    else:
        raise ValueError(f"Unsupported backend: {backend_type}")
    
    # Compile model
    compiled_model = ml.compile(
        model=model,
        backend=backend,
        shapes={"input": [batch_size, seq_length]},
        data_keys={"input"},
        jit=True
    )
    
    # Initialize parameters
    params = compiled_model.randomize_params()
    
    return compiled_model, params, backend

# Cross-entropy loss function
def compute_loss(logits, targets):
    # Get vocabulary size
    vocab_size = logits.shape[-1]
    
    # Convert targets to one-hot
    targets_one_hot = jnp.zeros((targets.shape[0], targets.shape[1], vocab_size))
    targets_flat = targets.reshape(-1)
    
    # Create indices for flattened one-hot matrix
    batch_size, seq_len = targets.shape
    batch_indices = jnp.arange(batch_size).repeat(seq_len)
    seq_indices = jnp.tile(jnp.arange(seq_len), batch_size)
    
    # Set one-hot values
    targets_one_hot = targets_one_hot.at[batch_indices, seq_indices, targets_flat].set(1)
    targets_one_hot = targets_one_hot.reshape(batch_size, seq_len, vocab_size)
    
    # Compute cross-entropy loss
    log_softmax = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(targets_one_hot * log_softmax) / (batch_size * seq_len)
    
    return loss

# Training step function
def train_step(params, opt_state, inputs, targets, compiled_model, optimizer):
    # Forward pass
    outputs = compiled_model.evaluate(params, {"input": inputs})
    logits = outputs["output"]
    
    # Compute loss
    loss = compute_loss(logits, targets)
    
    # Compute gradients
    grad_fn = jax.grad(lambda p: compute_loss(
        compiled_model.evaluate(p, {"input": inputs})["output"], 
        targets
    ))
    grads = grad_fn(params)
    
    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss

# Training loop
def train_model(
    compiled_model, 
    params, 
    dataloader, 
    backend, 
    num_epochs=10, 
    learning_rate=3e-4
):
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate)
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in dataloader:
            # Convert to backend arrays
            inputs = backend.array(x_batch)
            targets = backend.array(y_batch)
            
            # Training step
            params, opt_state, loss = train_step(
                params, opt_state, inputs, targets, compiled_model, optimizer
            )
            
            total_loss += loss
            num_batches += 1
            
            # Print progress
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss:.4f}")
        
        # Print epoch summary
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(params, f"model_checkpoint_epoch_{epoch+1}.pkl", backend)
    
    return params

# Save model checkpoint
def save_checkpoint(params, filename, backend):
    import pickle
    
    # Convert parameters to numpy
    numpy_params = {k: backend.to_numpy(v) for k, v in params.items()}
    
    # Save parameters
    with open(filename, 'wb') as f:
        pickle.dump(numpy_params, f)
    
    print(f"Checkpoint saved to {filename}")
```

## Text Generation

After training the model, we can use it to generate text:

```python
# Text generation function
def generate_text(
    compiled_model, 
    params, 
    tokenizer, 
    prompt, 
    max_length=100, 
    temperature=0.7, 
    top_k=40, 
    backend=None
):
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = backend.array([prompt_tokens])
    
    # Generate tokens sequentially
    for _ in range(max_length):
        # Get model output for current sequence
        outputs = compiled_model.evaluate(
            params, {"input": input_ids[:, -min(input_ids.shape[1], 1024):]}
        )
        logits = outputs["output"][:, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            # Get top k values and their indices
            v = backend.topk(logits, min(top_k, logits.shape[-1]))[0]
            # Create a mask for values below the threshold
            mask = backend.where(logits < v[:, [-1]], -float('inf'), 0)
            # Apply the mask to logits
            logits = logits + mask
        
        # Convert to probabilities
        probs = backend.softmax(logits, axis=-1)
        
        # Sample from the distribution
        next_token = backend.multinomial(probs, num_samples=1)
        
        # Append to input_ids
        input_ids = backend.concat([input_ids, next_token], axis=1)
        
        # Check if EOS token is generated
        if backend.to_numpy(next_token)[0, 0] == tokenizer.encode("<|endoftext|>")[0]:
            break
    
    # Decode generated tokens
    generated_text = tokenizer.decode(backend.to_numpy(input_ids)[0].tolist())
    return generated_text
```

## Complete Example

Let's put everything together and run the full training and generation pipeline:

```python
import os

# Main function
def main():
    # Configuration
    text_file = "dataset.txt"  # Your text file
    seq_length = 128
    batch_size = 32
    num_epochs = 3
    learning_rate = 3e-4
    backend_type = "jax"  # Options: "jax", "torch", "mlx"
    
    # Set up tokenizer
    tokenizer = get_tokenizer("gpt2")
    vocab_size = tokenizer.n_vocab
    
    # Create dataloaders
    dataloader = create_dataloaders(
        text_file, tokenizer, seq_length, batch_size
    )
    
    # Setup model
    compiled_model, params, backend = setup_model(
        vocab_size, seq_length, batch_size, backend_type
    )
    
    # Train model
    trained_params = train_model(
        compiled_model, params, dataloader, backend, num_epochs, learning_rate
    )
    
    # Generate text
    prompt = "Once upon a time"
    generated_text = generate_text(
        compiled_model, trained_params, tokenizer, prompt, 
        max_length=200, temperature=0.7, top_k=40, backend=backend
    )
    
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
```

## Loading a Pre-trained Model

To leverage existing pre-trained models like GPT-2, we can load pre-trained weights:

```python
from transformers import GPT2LMHeadModel

def load_pretrained_weights(model_name="gpt2"):
    # Load pre-trained model from Hugging Face
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Get state dictionary
    state_dict = hf_model.state_dict()
    
    # Create mapping from HF to Mithril parameter names
    hf_to_mithril = {
        "wte.weight": "wte_weight",
        "wpe.weight": "wpe_weight",
        "ln_f.weight": "ln_f_weight",
        "ln_f.bias": "ln_f_bias",
    }
    
    # Add mappings for each transformer block
    for i in range(12):  # Assuming 12 layers for gpt2
        # Layer norm 1
        hf_to_mithril[f"h.{i}.ln_1.weight"] = f"h_{i}_ln_1_weight"
        hf_to_mithril[f"h.{i}.ln_1.bias"] = f"h_{i}_ln_1_bias"
        
        # Attention
        hf_to_mithril[f"h.{i}.attn.c_attn.weight"] = f"h_{i}_attn_c_attn_weight"
        hf_to_mithril[f"h.{i}.attn.c_attn.bias"] = f"h_{i}_attn_c_attn_bias"
        hf_to_mithril[f"h.{i}.attn.c_proj.weight"] = f"h_{i}_attn_c_proj_weight"
        hf_to_mithril[f"h.{i}.attn.c_proj.bias"] = f"h_{i}_attn_c_proj_bias"
        
        # Layer norm 2
        hf_to_mithril[f"h.{i}.ln_2.weight"] = f"h_{i}_ln_2_weight"
        hf_to_mithril[f"h.{i}.ln_2.bias"] = f"h_{i}_ln_2_bias"
        
        # MLP
        hf_to_mithril[f"h.{i}.mlp.c_fc.weight"] = f"h_{i}_mlp_c_fc_weight"
        hf_to_mithril[f"h.{i}.mlp.c_fc.bias"] = f"h_{i}_mlp_c_fc_bias" 
        hf_to_mithril[f"h.{i}.mlp.c_proj.weight"] = f"h_{i}_mlp_c_proj_weight"
        hf_to_mithril[f"h.{i}.mlp.c_proj.bias"] = f"h_{i}_mlp_c_proj_bias"
    
    # Add LM head
    hf_to_mithril["lm_head.weight"] = "lm_head_weight"
    
    # Convert parameters
    mithril_params = {}
    for hf_name, mithril_name in hf_to_mithril.items():
        if hf_name in state_dict:
            # Handle special cases for weights that need transposition
            if "c_attn.weight" in hf_name or "c_proj.weight" in hf_name or "c_fc.weight" in hf_name:
                mithril_params[mithril_name] = state_dict[hf_name].numpy().T
            else:
                mithril_params[mithril_name] = state_dict[hf_name].numpy()
    
    return mithril_params

# Use pre-trained weights
def use_pretrained_model():
    # Load pre-trained weights
    pretrained_params = load_pretrained_weights("gpt2")
    
    # Create backend
    backend = ml.JaxBackend(dtype=ml.float32)
    
    # Create model
    model = create_language_model(
        vocab_size=50257,  # GPT-2 vocabulary size
        hidden_dim=768,    # GPT-2 small hidden size
        num_layers=12,     # GPT-2 small layers
        num_heads=12,      # GPT-2 small heads
        max_seq_len=1024,  # GPT-2 context length
        dropout_rate=0.0   # Disable dropout for inference
    )
    
    # Compile model
    compiled_model = ml.compile(
        model=model,
        backend=backend,
        shapes={"input": [1, 128]},  # Batch size 1 for inference
        data_keys={"input"},
        jit=True
    )
    
    # Convert parameters to backend format
    backend_params = {k: backend.array(v) for k, v in pretrained_params.items()}
    
    # Create tokenizer
    tokenizer = get_tokenizer("gpt2")
    
    # Generate text with pre-trained model
    prompt = "Artificial intelligence has the potential to"
    generated_text = generate_text(
        compiled_model, backend_params, tokenizer, prompt, 
        max_length=100, temperature=0.7, top_k=40, backend=backend
    )
    
    print("\nGenerated Text from Pre-trained Model:")
    print(generated_text)
```

## Advanced Techniques

### 1. Fine-tuning on a Specific Dataset

To fine-tune a pre-trained model on your own dataset:

```python
def fine_tune_model(
    pretrained_model_name="gpt2",
    text_file="your_dataset.txt",
    num_epochs=3,
    learning_rate=3e-5  # Use lower learning rate for fine-tuning
):
    # Load pre-trained weights
    pretrained_params = load_pretrained_weights(pretrained_model_name)
    
    # Set up tokenizer
    tokenizer = get_tokenizer(pretrained_model_name)
    vocab_size = tokenizer.n_vocab
    
    # Create dataloaders
    dataloader = create_dataloaders(
        text_file, tokenizer, seq_length=128, batch_size=16
    )
    
    # Setup model
    model = create_language_model(
        vocab_size=vocab_size,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=1024,
        dropout_rate=0.1
    )
    
    # Create backend
    backend = ml.JaxBackend(dtype=ml.float32)
    
    # Compile model
    compiled_model = ml.compile(
        model=model,
        backend=backend,
        shapes={"input": [16, 128]},
        data_keys={"input"},
        jit=True
    )
    
    # Convert parameters to backend format
    backend_params = {k: backend.array(v) for k, v in pretrained_params.items()}
    
    # Fine-tune model
    fine_tuned_params = train_model(
        compiled_model, backend_params, dataloader, backend, 
        num_epochs=num_epochs, learning_rate=learning_rate
    )
    
    return compiled_model, fine_tuned_params, tokenizer, backend
```

### 2. Implementing Beam Search for Better Text Generation

For higher-quality generation, we can implement beam search:

```python
def beam_search_generation(
    compiled_model, 
    params, 
    tokenizer, 
    prompt, 
    beam_width=5, 
    max_length=100, 
    backend=None
):
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = backend.array([prompt_tokens])
    
    # Initialize beam sequences and scores
    sequences = [(input_ids, 0.0)]
    
    # Generate tokens with beam search
    for _ in range(max_length):
        all_candidates = []
        
        # Expand each current beam
        for seq, score in sequences:
            # Get model output for current sequence
            outputs = compiled_model.evaluate(
                params, {"input": seq[:, -min(seq.shape[1], 1024):]}
            )
            logits = outputs["output"][:, -1, :]
            
            # Convert to log probabilities
            log_probs = backend.log_softmax(logits, axis=-1)
            
            # Get top-k next tokens
            top_log_probs, top_indices = backend.topk(log_probs, beam_width)
            
            # Create new candidates
            for i in range(beam_width):
                next_token = top_indices[0, i:i+1]
                next_score = score + backend.to_numpy(top_log_probs[0, i]).item()
                
                # Create new sequence by appending the next token
                next_seq = backend.concat([seq, next_token.reshape(1, 1)], axis=1)
                all_candidates.append((next_seq, next_score))
        
        # Select top beam_width candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        sequences = all_candidates[:beam_width]
        
        # Check if all beams end with EOS token
        if all(backend.to_numpy(seq[0, -1]).item() == tokenizer.encode("<|endoftext|>")[0] 
               for seq, _ in sequences):
            break
    
    # Return best sequence
    best_seq, _ = sequences[0]
    return tokenizer.decode(backend.to_numpy(best_seq)[0].tolist())
```

### 3. Adding Distributed Training Support

For training on multiple devices:

```python
def distributed_train_step(params, opt_state, inputs, targets, compiled_model, optimizer):
    # Define per-device training step
    def per_device_train_step(params, inputs, targets):
        outputs = compiled_model.evaluate(params, {"input": inputs})
        logits = outputs["output"]
        loss = compute_loss(logits, targets)
        return loss, jax.grad(compute_loss)(logits, targets)
    
    # Parallelize across devices
    parallel_train_step = jax.pmap(per_device_train_step, axis_name='devices')
    
    # Split inputs and targets across devices
    num_devices = jax.device_count()
    batch_size = inputs.shape[0]
    per_device_batch_size = batch_size // num_devices
    
    inputs_split = inputs.reshape(num_devices, per_device_batch_size, -1)
    targets_split = targets.reshape(num_devices, per_device_batch_size, -1)
    
    # Run training step in parallel
    losses, grads = parallel_train_step(params, inputs_split, targets_split)
    
    # Combine gradients
    grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)
    
    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    # Calculate average loss across devices
    loss = jnp.mean(losses)
    
    return new_params, new_opt_state, loss
```

## Conclusion

In this tutorial, you've learned how to:

1. Create a transformer-based language model using Mithril
2. Prepare and tokenize text data
3. Implement a training pipeline with JAX backend
4. Generate text with the trained model
5. Use advanced techniques like loading pre-trained weights, beam search, and distributed training

Mithril's flexible architecture allows you to easily adapt this model to different backends and use cases, from research prototypes to production deployments.

## Next Steps

- Try adapting the model to other architectures like T5 or BERT
- Experiment with different training strategies like mixed-precision training
- Fine-tune the model on domain-specific data
- Implement more advanced techniques like contrastive learning or reinforcement learning from human feedback