# Whisper: Automatic Speech Recognition with Mithril

This example demonstrates how to implement OpenAI's Whisper, a powerful automatic speech recognition (ASR) model, using Mithril. Whisper is designed to transcribe speech across various languages by converting audio inputs into text transcriptions.

## Model Architecture

Whisper consists of two main components:

1. **Encoder**: Processes audio input (mel spectrograms) to extract features
2. **Decoder**: Autoregressively generates text transcriptions based on encoded audio features

The architecture follows a standard encoder-decoder structure with multi-head attention:

```
Audio Input → Mel Spectrogram → Encoder → Decoder → Text Output
```

### Encoder Architecture

The encoder processes mel spectrograms using:

1. Convolutional layers to extract features
2. Positional embeddings to provide sequence context
3. A stack of transformer blocks with self-attention

```python
def whisper_encoder(num_layers: int, input_dim: int, num_heads: int, ffn_dim: int):
    model = Model(name="encoder")
    model |= Convolution1D(
        out_channels=input_dim, kernel_size=3, padding=1, name="conv1"
    )(input="input", output="conv1_out")
    model |= Gelu()(input="conv1_out", output="gelu1_out")
    model |= Convolution1D(
        out_channels=input_dim, kernel_size=3, stride=2, padding=1, name="conv2"
    )(input="gelu1_out", output="conv2_out")
    model |= Gelu()(input="conv2_out", output="gelu2_out")
    processed_out = model.gelu2_out.transpose((0, 2, 1))
    model |= Arange()(stop=1500, output="embedding_in")
    model |= Embedding(name="embed_positions", num_embeddings=1500, dim=input_dim)(
        input="embedding_in", output="pos_out"
    )  # Sinusiodal positional embeddings
    model |= Add()(left="pos_out", right=processed_out, output="attention_input")
    model.set_cout("attention_input")
    encoder_layers = encoder_block(num_layers, input_dim, num_heads, ffn_dim)
    model += encoder_layers
    model += LayerNorm(name="layer_norm")(output="encoder_hidden_states")
    return model
```

### Decoder Architecture

The decoder generates text using:

1. Token embeddings for input text
2. Positional embeddings to provide sequence context
3. Self-attention layers (causal, masked)
4. Cross-attention layers to attend to encoder outputs
5. Feed-forward neural networks

```python
def whisper_decoder(num_layers: int, input_dim: int, num_heads: int, ffn_dim: int):
    model = Model(name="decoder")
    model |= Embedding(name="embed_tokens", num_embeddings=51865, dim=input_dim)(
        input="decoder_input_ids", output="embedded_tokens"
    )
    model |= Size(dim=1)("embedded_tokens")  # Decoder id token embeddings
    model += Arange()
    model += Embedding(name="embed_positions", num_embeddings=448, dim=input_dim)(
        output="embedded_positions"
    )  # Positional embedding for decoder ids
    model |= Add()("embedded_tokens", model.cout)
    decoder_layers = decoder_block(num_layers, input_dim, num_heads, ffn_dim)
    model += decoder_layers(encoder_hidden_states="encoder_hidden_states")
    model += LayerNorm(name="layer_norm")
    return model
```

## Audio Input Processing

Whisper uses mel spectrograms as input representations. The process includes:

1. Loading audio files and resampling to 16kHz
2. Converting audio to mel spectrograms
3. Normalizing the spectrograms

```python
def process_input(audio: np.ndarray) -> np.ndarray:
    sr = 16000  # Sampling rate
    target_samples = (3000 - 1) * 160 + 400  # Exactly 3000 samples is needed.
    n_mels = 80
    # Length adjustment for shorter sequence by padding
    y = librosa.util.fix_length(audio, size=target_samples)

    # Mel spectrogram generation
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels, power=2, center=False
    )

    # Converting to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalization steps
    mean = np.mean(mel_spec_db)
    std = np.std(mel_spec_db)
    standardized_mel = (mel_spec_db - mean) / std  # Zero mean, unit variance

    mel_min = np.min(standardized_mel)
    mel_max = np.max(standardized_mel)
    normalized_mel = (
        2 * (standardized_mel - mel_min) / (mel_max - mel_min) - 1
    )  # Normalize to [-1, 1]

    # Reshape for model input (a single sample is considered)
    input_features = normalized_mel.reshape((1, n_mels, 3000))
    return input_features
```

## Using Pre-trained Weights

This example demonstrates how to use pre-trained weights from Hugging Face's Transformers library:

```python
def get_weights(backend: Backend) -> dict[str, Any]:
    model_hf = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    sd_hf = model_hf.state_dict()
    params = {}
    for key in sd_hf:
        ml_key = key.replace(".", "_")
        if "conv" in ml_key and "bias" in ml_key:
            params[ml_key] = backend.array(np.array(sd_hf[key])).reshape(1, -1, 1)
        else:
            params[ml_key] = backend.array(np.array(sd_hf[key]))
    return params
```

## Inference Process

The transcription process works by:

1. Processing audio into mel spectrograms
2. Initializing with a start token
3. Autoregressively generating tokens one by one
4. Converting tokens back to text

```python
# Simple greedy decoding
while token != 50257:  # Start token for whisper
    outputs = compiled_model.evaluate(
        params,
        data={"input": input_features, "decoder_input_ids": decoder_input_ids},
    )
    logits = outputs["output"]
    logits = logits[:, -1, :]
    token = backend_obj.argmax(logits)
    decoded_text.append(int(token))
    decoder_input_ids = backend_obj.array([decoded_text])
```

## Running the Example

To run the Whisper example:

```bash
python examples/whisper/run_whisper.py --file_path path/to/audio.flac --backend torch
```

The example supports multiple backends:
- `torch`: PyTorch backend
- `jax`: JAX backend
- `numpy`: NumPy backend
- `mlx`: MLX backend (Apple Silicon only)

## Multi-Backend Testing

The Whisper example includes tests to verify consistent output across different backends:

```python
@pytest.mark.parametrize("backend", backend_strings)
def test_run_sample(self, backend: str, run_sample_fn: RunSampleType):
    with redirect_stdout(StringIO()) as prompt_output:
        run_sample_fn(
            file_path="examples/whisper/1040-133433-0001.flac", backend=backend
        )
    output = prompt_output.getvalue()
    assert result_prompt in output
```

This demonstrates Mithril's ability to provide consistent results across different computation backends, showcasing the framework's flexibility and backend-agnostic design.