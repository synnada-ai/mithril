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

from typing import Any

import mithril as ml
from mithril import IOKey
from mithril.models import (
    Arange,
    ArgMax,
    Buffer,
    Concat,
    Convolution2D,
    Embedding,
    LayerNorm,
    Linear,
    Max,
    Model,
    Power,
    Randn,
    ScaledDotProduct,
    Sigmoid,
    Sqrt,
    Sum,
    Tensor,
    Where,
    ZerosLike,
)

backend_torch = ml.TorchBackend()


def quick_gelu(name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    block |= Sigmoid()((1.702 * input), output="sigmoid")
    block |= Buffer()(input * block.sigmoid, output=IOKey("output"))  # type: ignore
    return block


def attention(
    dims: int,
    num_heads: int,
    query_input_dims: int | None = None,
    key_input_dims: int | None = None,
    value_input_dims: int | None = None,
    value_dims: int | None = None,
    value_output_dims: int | None = None,
    bias: bool = False,
    use_mask: bool = False,
    name: str | None = None,
):
    block = Model(name=name)
    query_input_dims = query_input_dims or dims
    key_input_dims = key_input_dims or dims
    value_input_dims = value_input_dims or key_input_dims
    value_dims = value_dims or dims
    value_output_dims = value_output_dims or dims

    queries = IOKey("queries")
    keys = IOKey("keys")
    values = IOKey("values")

    block |= Linear(dims, name="q_proj", use_bias=bias)(queries, output="queries_proj")
    block |= Linear(dims, name="k_proj", use_bias=bias)(keys, output="keys_proj")
    block |= Linear(value_dims, name="v_proj", use_bias=bias)(
        values, output="values_proj"
    )

    queries: ml.Connection = block.queries_proj  # type: ignore
    keys: ml.Connection = block.keys_proj  # type: ignore
    values: ml.Connection = block.values_proj  # type: ignore

    B, L = queries.shape[0], queries.shape[1]
    S = keys.shape[1]
    queries = queries.reshape((B, L, num_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    keys = keys.reshape((B, S, num_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    values = values.reshape((B, S, num_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore

    if use_mask:
        block |= (mask_model := build_attention_mask())
        block |= ScaledDotProduct(is_causal=False, use_attn_mask=True)(
            query=queries,
            key=keys,
            value=values,
            attn_mask=mask_model.cout,
            output="scores",
        )
    else:
        block |= ScaledDotProduct(is_causal=False, use_attn_mask=False)(
            query=queries, key=keys, value=values, output="scores"
        )

    values_hat = block.scores.transpose((0, 2, 1, 3)).reshape((B, L, -1))  # type: ignore
    block |= Linear(value_output_dims, name="out_proj")(
        values_hat, output=IOKey("output")
    )
    # block |= Buffer()(input=block.out, output=IOKey("output"))  # type: ignore
    return block


def mlp(config: dict[str, Any], name: str | None = None):
    block = Model(name=name)
    block |= Linear(config["intermediate_size"], name="fc1")(
        input="input", output="fc1_output"
    )
    block |= quick_gelu(name="activation_fn")(
        input=block.fc1_output,  # type: ignore[attr-defined]
        output="gelu_output",
    )
    block |= Linear(config["hidden_size"], name="fc2")(
        input=block.gelu_output,  # type: ignore
        output=IOKey("output"),
    )
    return block


def encode_layer(
    config: dict[str, Any], use_mask: bool = False, name: str | None = None
):
    block = Model(name=name)
    input = IOKey("input")
    block |= LayerNorm(eps=config["layer_norm_eps"], name="layer_norm1")(
        input=input, output="ln_1_output"
    )
    block |= attention(
        config["hidden_size"],
        config["num_attention_heads"],
        bias=True,
        use_mask=use_mask,
        name="self_attn",
    )(
        queries=block.ln_1_output,  # type: ignore[attr-defined]
        keys=block.ln_1_output,  # type: ignore[attr-defined]
        values=block.ln_1_output,  # type: ignore[attr-defined]
        output="attn_output",
    )
    block |= LayerNorm(eps=config["layer_norm_eps"], name="layer_norm2")(
        input=input + block.attn_output,  # type: ignore[attr-defined]
        output="ln_2_output",
    )
    block |= mlp(config, name="mlp")(
        input=block.ln_2_output,  # type: ignore[attr-defined]
        output="mlp_output",
    )
    block |= Buffer()(
        input + block.attn_output + block.mlp_output,  # type: ignore[attr-defined]
        output=IOKey("output"),
    )
    return block


def encoder(
    config: dict[str, Any],
    use_mask: bool = False,
    name: str | None = None,
):
    block = Model(name=name)
    input_key = "input"
    for idx in range(config["num_hidden_layers"]):
        block |= encode_layer(config, use_mask=use_mask, name=f"layers_{idx}")(
            input=input_key, output=f"attn_output_{idx}"
        )
        input_key = f"attn_output_{idx}"
    block |= Buffer()(input=f"attn_output_{idx}", output=IOKey("output"))
    return block


def text_embeddings(
    config: dict[str, Any],
    name: str | None = None,
):
    block = Model(name=name)
    input = IOKey("input")
    embed_dim = config["hidden_size"]
    block |= Embedding(config["vocab_size"], embed_dim, name="token_embedding")(
        input=input, output="token_embedding_output"
    )
    block |= Embedding(
        config["max_position_embeddings"], embed_dim, name="position_embedding"
    )(
        input=input,
        weight="position_embedding_weight",
        output="position_embedding_output",
    )
    block |= Buffer()(
        input=block.token_embedding_output  # type: ignore
        + block.position_embedding_weight[: input.shape[1]],  # type: ignore
        output=IOKey("output"),
    )
    return block


def clip_text_model(config: dict[str, Any], name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    B, _ = input.shape[0], input.shape[1]

    block |= text_embeddings(config, name="embeddings")(
        input=input, output="embeddings_output"
    )

    block |= encoder(config, use_mask=True, name="encoder")(
        input=block.embeddings_output,  # type: ignore
        output="t_encoder_output",
    )

    block |= LayerNorm(name="final_layer_norm")(
        input=block.t_encoder_output,  # type: ignore
        output="last_hidden_state",
    )
    block |= Arange(dtype=ml.int64)(stop=B, output="arange_output")  # type: ignore
    block |= ArgMax(axis=-1)(input=input, output="argmax_output")

    # TODO: Add block.argmax_output
    block |= Buffer()(
        input=block.last_hidden_state[block.arange_output, block.argmax_output],  # type: ignore
        output=IOKey("output"),
    )
    return block


def vision_embeddings(config: dict[str, Any], name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    batch_size = input.shape[0]
    c_embed_dim = config["hidden_size"]
    c_image_size = config["image_size"]
    c_patch_size = config["patch_size"]
    num_positions = ((c_image_size // c_patch_size) ** 2) + 1
    block |= Convolution2D(
        kernel_size=c_patch_size,
        out_channels=c_embed_dim,
        stride=c_patch_size,
        use_bias=False,
        name="patch_embedding",
    )(input=input, output="patch_embeddings")
    patch_embeddings: ml.Connection = block.patch_embeddings.reshape(  # type: ignore
        (batch_size, c_embed_dim, -1)
    ).transpose((0, 2, 1))

    block |= Randn()(shape=(batch_size, 1, c_embed_dim), output="rand_1")  # type: ignore
    block |= ZerosLike()(input=block.rand_1, output="zeros_out")  # type: ignore
    class_embedding = IOKey("class_embedding", differentiable=True, shape=[c_embed_dim])

    block |= Concat(axis=1)(
        input=[class_embedding + block.zeros_out, patch_embeddings],  # type: ignore
        output="embeddings",
    )
    block |= Embedding(num_positions, c_embed_dim, name="position_embedding")(
        input=input,
        weight="position_embedding_weight",
        output="position_embedding_output",
    )
    block |= Buffer()(
        input=(block.embeddings + block.position_embedding_weight),  # type: ignore
        output=IOKey("output"),
    )
    return block


def clip_vision_model(config: dict[str, Any], name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    block |= vision_embeddings(config, name="embeddings")(
        input=input, output="v_embeddings_output"
    )
    block |= LayerNorm(name="pre_layrnorm")(
        input=block.v_embeddings_output,  # type: ignore
        output="pre_layrnorm_output",
    )
    block |= encoder(config, False, name="encoder")(
        input=block.pre_layrnorm_output,  # type: ignore
        output="v_encoder_output",
    )
    block |= LayerNorm(name="post_layernorm")(
        input=block.v_encoder_output,  # type: ignore
        output="post_layernorm_output",
    )
    block |= Buffer()(
        input=block.post_layernorm_output[:, 0, :],  # type: ignore
        output=IOKey("output"),
    )

    return block


# for torch.tensor.norm
def norm(
    p: str | int = 2,
    axis: int | None = None,
    keepdim: bool = False,
    name: str | None = None,
):
    block = Model(name=name)
    input = IOKey("input")
    if p == "inf":
        block += Max(axis=axis, keepdim=keepdim)(input=input.abs())
    elif p == 1:
        block += Sum(axis=axis, keepdim=keepdim)(input=input.abs())
    elif p == 2:
        block += Sum(axis=axis, keepdim=keepdim)(input=(input**2))
        block += Sqrt()
    else:
        assert isinstance(p, int)
        block += Sum(axis=axis, keepdim=keepdim)(input=(input.abs() ** p))
        block += Power(exponent=(1 / p))
    block += Buffer()(output=IOKey("output"))
    return block


def clip_model(config: dict[str, Any], name: str | None = None):
    block = Model(name=name)
    input_ids = IOKey("input_ids")
    pixel_values = IOKey("pixel_values")
    text_embed_dim = config["text_config"]["hidden_size"]
    projection_dim = config["projection_dim"]

    text_model = clip_text_model(config["text_config"], name="text_model")
    block |= text_model(input=input_ids, output="text_pooler_output")
    text_projection_weight = IOKey(
        "text_projection_weight",
        differentiable=True,
        shape=[text_embed_dim, projection_dim],
    )

    text_projection_output = (
        block.text_pooler_output @ text_projection_weight.transpose()  # type: ignore
    )

    block |= norm(p=2, axis=1, keepdim=True)(
        input=text_projection_output, output="norm_text_output"
    )
    text_embeds = text_projection_output / block.norm_text_output  # type: ignore

    vision_model = clip_vision_model(config["vision_config"], name="vision_model")
    block |= vision_model(input=pixel_values, output="visual_pooler_output")

    block |= Linear(projection_dim, use_bias=False, name="visual_projection")(
        input=block.visual_pooler_output,  # type: ignore
        output="visual_projection_output",
    )

    block |= norm(p=2, axis=1, keepdim=True)(
        input=block.visual_projection_output,  # type: ignore
        output="norm_visual_output",
    )
    image_embeds = block.visual_projection_output / block.norm_visual_output  # type: ignore

    block |= Buffer()(input=image_embeds, output=IOKey("image_embeds"))
    block |= Buffer()(input=text_embeds, output=IOKey("text_embeds"))
    return block


def build_attention_mask() -> Model:
    block = Model()
    block |= Arange(stop=77)(output="arange_out_1")
    block |= Arange(stop=77)(output="arange_out_2")
    upper_bool_triu = block.arange_out_1[..., None] >= block.arange_out_2[None, ...]  # type: ignore
    block |= Where()(
        cond=upper_bool_triu,
        input1=Tensor(0.0),
        input2=Tensor(float("-inf")),
        output=IOKey("output"),
    )
    return block
