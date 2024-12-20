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

"""
This example is equivalent of nanoGPT by karpathy.
Torch implementation of nanoGPT: https://github.com/karpathy/nanoGPT
"""

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
    Size,
    Split,
)


# Create self causal attention model.
def causal_attention(input_dim, num_heads, bias=True):
    if (input_dim % num_heads) != 0:
        raise ValueError("Requires input dims to be divisible by num_heads")

    model = Model(name="attn")
    model += Linear(input_dim * 3, name="c_attn")("input", output="c_attn_out")

    t_axes = (0, 2, 1, 3)
    shp_con = model.input.shape()  # type: ignore
    reshape_con = (shp_con[0], shp_con[1], num_heads, -1)

    model += Split(3, axis=-1)(model.c_attn_out, output="split_out")  # type: ignore
    tq = model.split_out[0].reshape(reshape_con).transpose(t_axes)  # type: ignore
    tk = model.split_out[1].reshape(reshape_con).transpose(t_axes)  # type: ignore
    tv = model.split_out[2].reshape(reshape_con).transpose(t_axes)  # type: ignore

    model += ScaledDotProduct()(query=tq, key=tk, value=tv, output="sdp_out")
    t_sdp = model.sdp_out.transpose(t_axes).reshape(shp_con[:3])  # type: ignore
    model += Linear(input_dim, name="c_proj")(t_sdp)

    return model


def mlp(n_embd: int):
    block = Model(name="mlp")
    block += Linear(n_embd * 4, name="c_fc")(input="input")
    block += Gelu()
    block += Linear(n_embd, name="c_proj")(output=IOKey("output"))

    return block


# Create Transformer Encoder Block.
def create_block(name, dims, num_heads, bias=True, eps=1e-5):
    block = Model(name=name)
    block += LayerNorm(use_bias=bias, eps=eps, name="ln_1")("input")
    block += causal_attention(dims, num_heads, bias)
    block += Add()("input", block.canonical_output, "add_out")
    block += LayerNorm(use_bias=bias, eps=eps, name="ln_2")
    block += mlp(dims)

    block += Add()("add_out", right=block.canonical_output)
    return block


def create_gpt(bias, block_size, dims, num_heads, num_layers, vocab_size):
    # Create Position Embedding model
    transformer = Model(name="transformer")
    transformer += Size(dim=1)("input")
    transformer += Arange(start=0, step=1)
    transformer += Embedding(name="wpe", num_embeddings=block_size, dim=dims)(
        output="pos_out"
    )
    transformer += Embedding(name="wte", num_embeddings=vocab_size, dim=dims)(
        "input", output="token_out"
    )
    transformer += Add()("pos_out", "token_out")

    blocks = Model(name="h")
    for idx in range(num_layers):
        blocks += create_block(f"{idx}", dims, num_heads)
    transformer += blocks
    transformer += LayerNorm(use_bias=bias, name="ln_f")

    # Create GPT
    gpt = Model()
    gpt += transformer(input="input")
    gpt += Linear(vocab_size, use_bias=False, name="lm_head")(output=IOKey("output"))
    gpt.input.set_differentiable(False)  # type: ignore
    return gpt
