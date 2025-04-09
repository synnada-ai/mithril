# Natural Language Processing with Mithril

This document explores how to implement various natural language processing (NLP) models in Mithril, from transformer-based language models to encoder-decoder architectures like T5.

## GPT-style Language Models

Mithril provides a clean implementation of GPT-style autoregressive language models, similar to the nanoGPT architecture by Andrej Karpathy.

### Building Blocks for Transformers

Let's examine how to implement the core components:

#### Self-Attention with Causal Masking

```python
def causal_attention(input_dim, num_heads, bias=True):
    if (input_dim % num_heads) != 0:
        raise ValueError("Requires input dims to be divisible by num_heads")

    model = Model(name="attn")
    model |= Linear(input_dim * 3, name="c_attn")("input", output="c_attn_out")

    t_axes = (0, 2, 1, 3)
    shp_con = model.input.shape
    reshape_con = (shp_con[0], shp_con[1], num_heads, -1)

    model |= Split(3, axis=-1)(model.c_attn_out, output="split_out")
    tq = model.split_out[0].reshape(reshape_con).transpose(t_axes)
    tk = model.split_out[1].reshape(reshape_con).transpose(t_axes)
    tv = model.split_out[2].reshape(reshape_con).transpose(t_axes)

    model |= ScaledDotProduct()(query=tq, key=tk, value=tv, output="sdp_out")
    t_sdp = model.sdp_out.transpose(t_axes).reshape(shp_con[:3])
    model |= Linear(input_dim, name="c_proj")(t_sdp)
    return model
```

This implementation efficiently handles multi-head attention with causal masking, preventing tokens from attending to future positions.

#### MLP Block

```python
def mlp(n_embd: int):
    block = Model(name="mlp")
    block += Linear(n_embd * 4, name="c_fc")(input="input")
    block += Gelu()
    block += Linear(n_embd, name="c_proj")(output=IOKey("output"))
    return block
```

The MLP expands the dimensionality by a factor of 4, applies GELU activation, then projects back to the original dimension.

#### Transformer Block

```python
def create_block(name, dims, num_heads, bias=True, eps=1e-5):
    block = Model(name=name)
    block += LayerNorm(use_bias=bias, eps=eps, name="ln_1")("input")
    block += causal_attention(dims, num_heads, bias)
    block |= Add()("input", block.cout, "add_out")
    block += LayerNorm(use_bias=bias, eps=eps, name="ln_2")
    block += mlp(dims)
    block |= Add()("add_out", right=block.cout)
    return block
```

Each transformer block includes pre-normalization, self-attention, residual connections, and an MLP, following the GPT architecture.

### Assembling the GPT Model

```python
def create_gpt(bias, block_size, dims, num_heads, num_layers, vocab_size):
    # Create Position Embedding model
    transformer = Model(name="transformer")
    transformer += Size(dim=1)("input")
    transformer += Arange(start=0, step=1)
    transformer += Embedding(name="wpe", num_embeddings=block_size, dim=dims)(
        output="pos_out"
    )
    transformer |= Embedding(name="wte", num_embeddings=vocab_size, dim=dims)(
        "input", output="token_out"
    )
    transformer |= Add()("pos_out", "token_out")

    blocks = Model(name="h")
    for idx in range(num_layers):
        blocks += create_block(f"{idx}", dims, num_heads)
    transformer += blocks
    transformer += LayerNorm(use_bias=bias, name="ln_f")

    # Create GPT
    gpt = Model()
    gpt += transformer(input="input")
    gpt += Linear(vocab_size, use_bias=False, name="lm_head")(output=IOKey("output"))
    gpt.set_differentiability({gpt.input: False})
    return gpt
```

This assembly creates a complete GPT model with token embeddings, positional embeddings, stacked transformer blocks, and an output projection layer.

### Text Generation with GPT

Mithril makes it easy to run inference and generate text:

```python
def generate(
    model: PhysicalModel[Any],
    block_size: int,
    weights: dict[str, ml.DataType],
    idx: ml.DataType,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    decode: Callable | None = None,
):
    for _ in range(max_new_tokens):
        # If the sequence context is growing too long we must crop it
        idx_cond = idx if idx.shape[1] <= block_size else idx[:, -block_size:]
        # Forward the model to get the logits
        outputs = model.evaluate(weights, data={"input": idx_cond})
        logits = outputs["output"]
        # Pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # Optionally crop the logits to only the top k options
        if top_k is not None:
            v = model.backend.topk(logits, min(top_k, logits.shape[-1]))
            logits = model.backend.where(
                logits < v[:, [-1]], -model.backend.inf, logits
            )
        # Apply softmax to convert logits to probabilities
        probs = model.backend.softmax(logits, dim=-1)
        # Sample from the distribution
        idx_next = model.backend.multinomial(probs, num_samples=1)
        # Append sampled index to the running sequence
        idx = model.backend.cat([idx, idx_next], axis=1)
        print(decode(idx_next[0].tolist()), end="", flush=False)
    return idx
```

This generation function supports temperature sampling and top-k filtering for controlled text generation.

## T5 Encoder-Decoder Architecture

Mithril also supports implementing more complex encoder-decoder models like T5, which excel at tasks like translation and summarization.

### T5 Multi-Head Attention

```python
def multihead_attention(
    config: dict[str, Any],
    use_mask: bool = False,
    *,
    name: str | None = None,
):
    d_kv = config["d_kv"]
    d_model = config["d_model"]
    num_heads = config["num_heads"]

    inner_dim = d_kv * num_heads
    block = Model(name=name)
    queries = IOKey("queries", shape=(None, None, d_model))
    keys = IOKey("keys", shape=(None, None, d_model))
    values = IOKey("values", shape=(None, None, d_model))

    block |= Linear(inner_dim, name="query_proj", use_bias=False)(
        queries, output="queries_proj"
    )
    block |= Linear(inner_dim, name="key_proj", use_bias=False)(
        keys, output="keys_proj"
    )
    block |= Linear(inner_dim, name="value_proj", use_bias=False)(
        values, output="values_proj"
    )

    # Reshape and transpose for multi-head attention
    # ...

    scores = queries @ keys

    if use_mask:
        scores = scores + IOKey("mask").cast(scores.dtype())

    block |= Softmax(axis=-1)(scores.cast(ml.float32), output="attention_weights")

    scores = block.attention_weights.cast(scores.dtype())
    values_hat = (scores @ values).transpose((0, 2, 1, 3)).reshape((B, L, -1))
    block |= Linear(d_model, name="out_proj", use_bias=False)(
        values_hat, output=IOKey("output")
    )
    
    # ...
    
    return block
```

T5's attention implementation supports both self-attention and cross-attention mechanisms, with relative position bias.

### Encoder and Decoder Blocks

```python
def transformer_encoder_layer(config: dict[str, Any], *, name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    mask = IOKey("mask")

    block |= rms_norm(config["d_model"], name="ln1")(input=input, output="input_norm")
    block |= multihead_attention(config=config, use_mask=True, name="attention")(
        queries="input_norm",
        keys="input_norm",
        values="input_norm",
        mask=mask,
        output="attn_out",
    )

    block |= Add()(left="input", right="attn_out", output="attn_out2")
    block |= rms_norm(config["d_model"], name="ln2")(input="attn_out2", output="norm2")
    block |= dense_activation(config=config, name="dense")(
        input="norm2", output="ff_out"
    )
    block |= Add()(left="attn_out2", right="ff_out", output=IOKey("output"))
    block.set_cout("output")
    return block
```

The T5 architecture uses pre-layer normalization and specialized components like RMS normalization instead of layer normalization.

### T5 Encode and Decode Functions

```python
def t5_encode(config: dict[str, Any], name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    block |= Embedding(
        name="wte", num_embeddings=config["vocab_size"], dim=config["d_model"]
    )(input, output="wte_out")
    block |= transformer_encoder(config, name="encoder")(
        input="wte_out", pos_bias="pos_bias", output=IOKey("output")
    )
    return block

def t5_decode(config: dict[str, Any], *, name: str | None = None):
    tie_word_embeddings = config.get("tie_word_embeddings", True)

    block = Model(name=name)
    input = IOKey("input")
    memory = IOKey("memory")
    wte = Embedding(
        name="wte", num_embeddings=config["vocab_size"], dim=config["d_model"]
    )
    block |= wte(input, output="wte_out")
    block |= transformer_decoder(config, name="decoder")(
        input="wte_out", memory=memory, output="decoder_out"
    )

    if not tie_word_embeddings:
        block |= output_head(config, name="lm_head")(
            input="decoder_out", output=IOKey("output")
        )
    else:
        decoder_out = block.decoder_out
        decoder_out *= config["d_model"] ** -0.5
        block |= MatrixMultiply()(
            decoder_out, wte.weight.transpose(), output=IOKey("output")
        )
    return block
```

These functions encapsulate the encoder and decoder parts of the T5 model, handling token embedding, transformation through the encoder/decoder blocks, and output projections.

### Text Generation with T5

```python
def generate(
    prompt: str,
    encoder: PhysicalModel,
    decoder: PhysicalModel,
    tokenizer: Tokenizer,
    weights: dict,
    backend: ml.Backend,
):
    prompt = tokenizer.encode(prompt)
    memory = encoder.evaluate(weights, {"input": prompt})["output"]

    y = backend.array([tokenizer.decoder_start_id])[None]
    while True:
        logits = decoder.evaluate(weights, {"input": y, "memory": memory})["output"]
        _y = logits[:, -1, :].argmax(axis=-1)[None]
        y = backend.concat([y, _y], axis=1)
        yield _y
```

This generator function shows how encoder-decoder models like T5 can be used for text generation, first encoding the input, then autogressively generating output tokens.

## Backend Flexibility

All these NLP models can run on various Mithril backends:

```python
# Create a model and run with PyTorch backend
gpt_model = create_gpt(bias=True, block_size=100, dims=768, num_heads=12, 
                      num_layers=12, vocab_size=50304)
compiled_model = ml.compile(
    gpt_model, ml.TorchBackend(), data_keys={"input"}, jit=False
)

# Same model can run on JAX backend
jax_compiled = ml.compile(
    gpt_model, ml.JaxBackend(), data_keys={"input"}, jit=True
)

# Or MLX for Apple silicon
mlx_compiled = ml.compile(
    gpt_model, ml.MlxBackend(), data_keys={"input"}, jit=False
)
```

This backend-agnostic approach allows the same model code to run efficiently on different platforms and frameworks, making Mithril ideal for developing and deploying NLP models across various environments.