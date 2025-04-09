# Flux: Image Generation with Text-to-Image Diffusion Model

This example demonstrates how to implement Flux, a text-to-image diffusion model using Mithril. Flux is designed to generate high-quality images from text prompts by combining different modalities (text and images) through a dual-stream architecture.

## Architecture Overview

Flux consists of several key components:

1. **Text Encoders**: Uses both CLIP and T5 to encode text prompts
2. **Diffusion Process**: A noise prediction network with timestep conditioning
3. **Dual-Stream Architecture**: Processes text and image tokens simultaneously
4. **Position Embeddings**: Uses rotary position embeddings (RoPE)
5. **Decoder**: Converts latent representations back to images

The model architecture includes specialized blocks:

- `double_stream_block`: Processes text and image tokens in parallel with cross-attention
- `single_stream_block`: Processes concatenated tokens with self-attention
- `modulation`: Provides conditioning through shift, scale, and gate mechanisms

## Key Implementation Features

### Adaptive Layer Normalization

Flux uses a form of adaptive layer normalization where conditioning vectors modulate the normalization process:

```python
txt_modulated = (1 + block.txt_mod_1[1]) * block.txt_norm + block.txt_mod_1[0]
```

### Rotary Position Embeddings

Position information is encoded using rotary position embeddings (RoPE):

```python
def rope(dim: int, theta: int, *, name: str | None = None) -> Model:
    block = Model(name=name)
    input = IOKey("input", type=Tensor)
    block |= Arange(start=0, stop=dim, step=2)(output="arange")
    
    omega = 1.0 / (theta ** (block.arange.cast(ml.float32) / dim))
    out = input[..., None] * omega
    
    # Transform into rotation matrices
    block |= Concat(axis=-1)(
        input=[
            out.cos()[..., None],
            -out.sin()[..., None],
            out.sin()[..., None],
            out.cos()[..., None],
        ],
        output="concat_out",
    )
    # ...
```

### Conditional Diffusion

The model implements a diffusion process with classifier-free guidance:

```python
if params.guidance_embed:
    guidance = IOKey("guidance", shape=[1])
    flux |= timestep_embedding(dim=256)(input=guidance, output="guidance_embed")
    flux |= mlp_embedder(params.hidden_size, name="guidance_in")(
        input="guidance_embed", output="guidance_vec"
    )
    flux |= Add()(left="vec", right="guidance_vec", output="guided_vec")
```

## Using the Flux Example

To run the Flux example:

```python
from examples.flux.main import run

# Generate an image with a text prompt
run(
    model_name="flux-dev",          # Model variant to use
    backend_name="torch",           # Backend (torch or jax)
    width=1024,                     # Output image width
    height=1024,                    # Output image height
    prompt="A mountain landscape with a lake at sunset", # Text description
    device="cuda",                  # Device to run on
    num_steps=28,                   # Diffusion steps (higher = better quality)
    guidance=3.5                    # Guidance scale (higher = more adherence to prompt)
)
```

## Backend Compatibility

Flux is designed to work with multiple backends in Mithril:

- **PyTorch**: Full support with GPU acceleration
- **JAX**: Optimized performance with XLA compilation
- **MLX**: Support for Apple Silicon devices

The model can switch between backends without changing the model definition, demonstrating Mithril's backend-agnostic design.

## Model Composition

The Flux implementation demonstrates how to compose complex models using Mithril's operator-based API:

```python
# Adding components to the model
flux |= timestep_embedding(dim=256)(input=timesteps, output="time_embed")
flux |= mlp_embedder(params.hidden_size, name="time_in")(
    input="time_embed", output="time_vec"
)
flux |= Linear(params.hidden_size, name="img_in")(input=img, output="img_vec")

# Using the += operator for sequential blocks
block |= Linear(hidden_dim, name="in_layer")(input="input")
block += SiLU()
block += Linear(hidden_dim, name="out_layer")(output=IOKey("output"))
```

## Customization Options

The Flux model can be customized through its parameters:

```python
@dataclass
class FluxParams:
    in_channels: int            # Input image channels
    vec_in_dim: int             # Conditioning vector dimension
    context_in_dim: int         # Context dimension
    hidden_size: int            # Model hidden dimension
    mlp_ratio: float            # MLP expansion ratio
    num_heads: int              # Number of attention heads
    depth: int                  # Depth of dual-stream blocks
    depth_single_blocks: int    # Depth of single-stream blocks
    axes_dim: list[int]         # Dimensions for position embeddings
    theta: int                  # RoPE theta parameter
    qkv_bias: bool              # Whether to use bias in QKV projection
    guidance_embed: bool        # Whether to use guidance embeddings
```

## Performance Optimizations

The example includes performance optimizations such as:

1. Batched processing of inputs
2. BFloat16 precision for faster computation
3. JIT compilation for optimized execution
4. Pre-compilation warmup for more consistent performance

These optimizations help Flux run efficiently on various hardware configurations.