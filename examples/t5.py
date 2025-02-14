# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import math
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

import mithril as ml
from mithril import IOKey
from mithril.models import (
    Add,
    Arange,
    Buffer,
    Embedding,
    Gelu,
    Linear,
    Log,
    Minimum,
    Model,
    Multiply,
    PhysicalModel,
    Relu,
    SiLU,
    Softmax,
    Transpose,
    Where,
    ZerosLike,
)


class Tokenizer:
    def __init__(self, config, model_name, backend):
        self._decoder_start_id = config["decoder_start_token_id"]
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            legacy=False,
            model_max_length=config.get("n_positions", 512),
        )
        self.backend = backend

    @property
    def eos_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def decoder_start_id(self) -> int:
        return self._decoder_start_id

    def encode(self, s: str):
        return self.backend.array(
            self._tokenizer(
                s,
                return_tensors="np",
                return_attention_mask=False,
            )["input_ids"]
        )

    def decode(self, t: list[int], with_sep: bool = True) -> str:
        tokens = self._tokenizer.convert_ids_to_tokens(t)
        return "".join(t.replace("▁", " " if with_sep else "") for t in tokens)


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

    queries: ml.Connection = block.queries_proj  # type: ignore
    keys: ml.Connection = block.keys_proj  # type: ignore
    values: ml.Connection = block.values_proj  # type: ignore

    B, L = queries.shape[0], queries.shape[1]
    S = keys.shape[1]
    queries = queries.reshape((B, L, num_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    keys = keys.reshape((B, S, num_heads, -1)).transpose((0, 2, 3, 1))  # type: ignore
    values = values.reshape((B, S, num_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore

    scores = queries @ keys

    if use_mask:
        scores = scores + IOKey("mask").cast(scores.dtype())

    block |= Softmax(axis=-1)(scores.cast(ml.float32), output="attention_weights")

    scores = block.attention_weights.cast(scores.dtype())  # type: ignore
    values_hat = (scores @ values).transpose((0, 2, 1, 3)).reshape((B, L, -1))
    block |= Linear(d_model, name="out_proj", use_bias=False)(
        values_hat, output=IOKey("output")
    )
    block |= Buffer()(keys, output=IOKey("keys_out"))
    block |= Buffer()(values, output=IOKey("values_out"))

    return block


def rms_norm(dim: int, *, name: str | None = None):
    # TODO: check original implementation they use astype and cast to float32
    block = Model(name=name)
    input = IOKey("input")
    weight = IOKey(
        "weight", shape=[dim], differantiable=True
    )  # TODO: weight must be initialized with ones.
    rrms = input / ((input**2).mean(axis=-1, keepdim=True) + 1e-5).sqrt()
    block += Multiply()(left=rrms, right=weight, output=IOKey("output"))

    return block


def dense_activation(config: dict[str, Any], *, name: str | None = None):
    mlp_dims = config["d_ff"] or config["d_model"] * 4

    is_gated = hasattr(config, "feed_forward_proj")
    activation_name = (
        "relu" if not is_gated else config["feed_forward_proj"].removeprefix("gated-")
    )

    if activation_name == "relu":
        activation: Model = Relu()
    elif activation_name == "gelu":
        activation = Gelu()
    elif activation_name == "silu":
        activation = SiLU()
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")

    block = Model(name=name)
    input = IOKey("input")

    if is_gated:
        block += Linear(mlp_dims, use_bias=False, name="wi_0")(input)
        block += activation(output="hidden_act")
        block += Linear(mlp_dims, use_bias=False, name="wi_1")(input, output="lin_out")
        block += Multiply()(left="hidden_act", right="lin_out", output="hidden_out")
    else:
        block += Linear(mlp_dims, name="wi", use_bias=False)(input)
        block += activation(output="hidden_out")

    block += Linear(config["d_model"], name="wo", use_bias=False)(
        output=IOKey("output")
    )
    return block


def relative_position_bucket(
    bidirectional=True, num_buckets=32, max_distance=128, *, name: str | None = None
):
    block = Model(name=name)
    relative_position = IOKey("relative_position")
    relative_buckets = IOKey(value=0)

    if bidirectional:
        num_buckets //= 2

        relative_buckets += (relative_position > 0).cast(ml.int16) * num_buckets
        relative_position = relative_position.abs()  # type: ignore
    else:
        block |= ZerosLike()(relative_position, output="zeros_like")
        block |= Minimum()(
            left=relative_position, right="zeros_like", output="minimum_out"
        )

        relative_position: ml.Connection = -block.minimum_out  # type: ignore
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    scale = (num_buckets - max_exact) / math.log(max_distance / max_exact)

    block |= Log()((relative_position.cast(ml.float32) / max_exact), output="log_out")

    block |= Minimum()(
        left=(max_exact + block.log_out * scale).cast(ml.int16),  # type: ignore
        right=ml.Tensor(num_buckets - 1),
        output="relative_position_if_large_2",
    )
    block |= Where()(
        cond=is_small,
        input1=relative_position,
        input2="relative_position_if_large_2",
        output="where_out",
    )

    block |= Add()(relative_buckets, "where_out", output=IOKey("output"))

    return block


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

    return block


def relative_position_bias(
    config: dict[str, Any], bidirectional: bool, *, name: str | None = None
):
    num_buckets = config["relative_attention_num_buckets"]
    max_distance = config.get("relative_attention_max_distance", 128)
    num_heads = config["num_heads"]

    block = Model(name=name)
    query_length = IOKey("query_length", type=int)
    key_length = IOKey("key_length", type=int)
    offset = IOKey("offset", type=int)

    block |= Arange(start=ml.TBD)(
        start=offset, stop=query_length, output="context_position"
    )
    block |= Arange()(stop=key_length, output="memory_position")

    context_position: ml.Connection = block.context_position[:, None]  # type: ignore
    memory_position: ml.Connection = block.memory_position[None, :]  # type: ignore

    relative_position = memory_position - context_position

    block |= relative_position_bucket(
        bidirectional=bidirectional, num_buckets=num_buckets, max_distance=max_distance
    )(relative_position=relative_position, output="relative_position_buckets")
    block |= Embedding(num_embeddings=num_buckets, dim=num_heads, name="embeddings")(
        input="relative_position_buckets", output="values"
    )
    block |= Transpose(axes=(2, 0, 1))(input="values", output=IOKey("output"))

    return block


def transformer_encoder(config: dict[str, Any], *, name: str | None = None):
    input = IOKey("input")
    block = Model(name=name)

    block |= relative_position_bias(
        config, bidirectional=True, name="relative_attention_bias"
    )(
        query_length=input.shape[1],
        key_length=input.shape[1],
        offset=0,
        output=IOKey("pos_bias"),
    )

    input_key = "input"
    for idx in range(config["num_layers"]):
        block |= transformer_encoder_layer(config, name=f"layers_{idx}")(
            input=input_key, mask="pos_bias", output=f"output_{idx}"
        )
        input_key = f"output_{idx}"

    block |= rms_norm(config["d_model"], name="ln")(
        input=input_key, output=IOKey("output")
    )
    return block


def transformer_decoder_layer(
    config: dict[str, Any], use_mask: bool = False, *, name: str | None = None
):
    block = Model(name=name)
    input = IOKey("input")
    mask = IOKey("mask")
    memory = IOKey("memory")

    block |= rms_norm(config["d_model"], name="ln1")(input=input, output="input_norm")
    block |= multihead_attention(
        config=config, use_mask=use_mask, name="self_attention"
    )(
        queries="input_norm",
        keys="input_norm",
        values="input_norm",
        mask=mask,
        output="self_attn_out",
    )
    block |= Add()(left="input", right="self_attn_out", output="self_attn_out2")
    block |= rms_norm(config["d_model"], name="ln2")(
        input="self_attn_out2", output="norm2"
    )
    block |= multihead_attention(config=config, use_mask=False, name="cross_attention")(
        queries="norm2", keys=memory, values=memory, output="cross_attn_out"
    )
    block |= Add()(
        left="self_attn_out2", right="cross_attn_out", output="cross_attn_out2"
    )
    block |= rms_norm(config["d_model"], name="ln3")(
        input="cross_attn_out2", output="norm3"
    )
    block |= dense_activation(config=config, name="dense")(
        input="norm3", output="ff_out"
    )
    block |= Add()(left="cross_attn_out2", right="ff_out", output=IOKey("output"))

    return block


def transformer_decoder(config: dict[str, Any], *, name: str | None = None):
    n_layers = config.get("num_decoder_layers", config["num_layers"])
    offset = 0

    block = Model(name=name)
    input = IOKey("input")
    memory = IOKey("memory")

    block += relative_position_bias(
        config, bidirectional=False, name="relative_attention_bias"
    )(
        query_length=input.shape[1],
        key_length=input.shape[1],
        offset=offset,
        output="pos_bias",
    )

    input_key = "input"
    for idx in range(n_layers):
        block |= transformer_decoder_layer(config, True, name=f"layers_{idx}")(
            input=input_key, mask="pos_bias", memory=memory, output=f"output_{idx}"
        )
        input_key = f"output_{idx}"

    block |= rms_norm(config["d_model"], name="ln")(
        input=input_key, output=IOKey("output")
    )

    return block


def output_head(config: dict[str, Any], *, name: str | None = None):
    block = Model(name=name)
    block += Linear(config["vocab_size"], use_bias=False, name="linear")(
        IOKey("input"), output=IOKey("output")
    )
    return block


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
        decoder_out = block.decoder_out  # type: ignore
        decoder_out *= config["d_model"] ** -0.5
        decoder_out = decoder_out @ wte.weight.transpose()
        block |= Buffer()(decoder_out, IOKey("output"))

    return block


def load_weights(name: str):
    x = snapshot_download(
        repo_id="t5-small", allow_patterns=["*.json", "*.safetensors", "*.model"]
    )
    path = Path(x)
    with open(path / "config.json") as f:
        config = json.load(f)

    weights = {}
    with safe_open(str(path / "model.safetensors"), framework="np", device="cpu") as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            weights[k] = f.get_tensor(k)

    weights = sanitize(weights)

    return weights, config


def sanitize(weights):
    shared_replacement_patterns = [
        (".block.", ".layers."),
        (".k.", ".key_proj."),
        (".o.", ".out_proj."),
        (".q.", ".query_proj."),
        (".v.", ".value_proj."),
        ("shared.", "wte."),
        ("lm_head.", "lm_head.linear."),
        (".layer.0.layer_norm.", ".ln1."),
        (".layer.1.layer_norm.", ".ln2."),
        (".layer.2.layer_norm.", ".ln3."),
        (".final_layer_norm.", ".ln."),
        (
            "layers.0.layer.0.SelfAttention.relative_attention_bias.",
            "relative_attention_bias.embeddings.",
        ),
    ]

    encoder_replacement_patterns = [
        (".layer.0.SelfAttention.", ".attention."),
        (".layer.1.DenseReluDense.", ".dense."),
    ]

    decoder_replacement_patterns = [
        (".layer.0.SelfAttention.", ".self_attention."),
        (".layer.1.EncDecAttention.", ".cross_attention."),
        (".layer.2.DenseReluDense.", ".dense."),
    ]

    ignored_keys = ["decoder.layers.0.cross_attention.relative_attention_bias.weight"]

    def replace_key(key: str) -> str:
        for old, new in shared_replacement_patterns:
            key = key.replace(old, new)
        if key.startswith("encoder."):
            for old, new in encoder_replacement_patterns:
                key = key.replace(old, new)
        elif key.startswith("decoder."):
            for old, new in decoder_replacement_patterns:
                key = key.replace(old, new)
        return key

    weights = {replace_key(k): v for k, v in weights.items()}
    for key in ignored_keys:
        if key in weights:
            del weights[key]
    return weights


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
        _y = logits[:, -1, :].argmax(axis=-1)[None]  # type: ignore
        y = backend.concat([y, _y], axis=1)
        yield _y


def run(prompt: str, backend: ml.Backend):
    weights, config = load_weights("t5-small")
    for key in list(weights.keys()):
        weights[key.replace(".", "_")] = backend.array(weights.pop(key))

    tokenizer = Tokenizer(config, "t5-small", backend)

    encoder_lm = t5_encode(config)
    decoder_lm = t5_decode(config)
    encoder_pm = ml.compile(
        encoder_lm,
        backend,
        data_keys={"input"},
        shapes={"input": [1, None]},
        jit=False,
        use_short_namings=False,
    )
    decoder_pm = ml.compile(
        decoder_lm,
        backend,
        data_keys={"input", "memory"},
        shapes={"input": [1, None], "memory": [1, None, 512]},
        jit=False,
        use_short_namings=False,
    )

    print("Prompt:", prompt)
    print("Generated text:", end="", flush=True)
    for token in generate(prompt, encoder_pm, decoder_pm, tokenizer, weights, backend):
        if token.item() == tokenizer.eos_id:
            print()
            break
        print(tokenizer.decode(token.item()), end="", flush=True)


# TODO: Cache is not supported

if __name__ == "__main__":
    prompt = "translate English to German: I am not in danger, I am the danger."
    run(prompt, ml.TorchBackend(dtype=ml.bfloat16))
