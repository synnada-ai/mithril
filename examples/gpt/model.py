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
)


# Create self causal attention model.
def causal_attention(input_dim, num_heads, bias=True):
    if (input_dim % num_heads) != 0:
        raise ValueError("Requires input dims to be divisible by num_heads")

    model = Model()
    model += Linear(input_dim)("input", output="lq_out")
    model += Linear(input_dim)("input", output="lk_out")
    model += Linear(input_dim)("input", output="lv_out")

    t_axes = (0, 2, 1, 3)
    shp_con = model.input.shape()  # type: ignore
    reshape_con = (shp_con[0], shp_con[1], num_heads, -1)

    tq = model.lq_out.reshape(reshape_con).transpose(t_axes)  # type: ignore
    tk = model.lk_out.reshape(reshape_con).transpose(t_axes)  # type: ignore
    tv = model.lv_out.reshape(reshape_con).transpose(t_axes)  # type: ignore

    model += ScaledDotProduct()(query=tq, key=tk, value=tv, output="sdp_out")
    t_sdp = model.sdp_out.transpose(t_axes).reshape(shp_con[:3])  # type: ignore
    model += Linear(input_dim)(t_sdp)
    return model


# Create Transformer Encoder Block.
def create_block(dims, num_heads, mlp_dims, bias=True, eps=1e-5):
    block = Model()
    block += LayerNorm(use_bias=bias, eps=eps)("input")
    block += causal_attention(dims, num_heads, bias)
    block += Add()("input", block.canonical_output, "add_out")
    block += LayerNorm(use_bias=bias, eps=eps)
    block += Linear(mlp_dims)
    block += Gelu()
    block += Linear(dims)
    block += Add()("add_out", right=block.canonical_output)
    return block


def create_gpt(bias, block_size, dims, mlp_dims, num_heads, num_layers, vocab_size):
    # Create Position Embedding model
    position = Model()
    position += Size(dim=1)("input")
    position += Arange(start=0, step=1)
    position += Embedding(block_size, dims)(
        input=position.canonical_output, output="output"
    )
    # Create GPT
    gpt = Model()
    gpt += Embedding(vocab_size, dims)("input", output="token_out")
    gpt += position(input="input")
    gpt += Add()("token_out", position.canonical_output)

    for _ in range(num_layers):
        gpt += create_block(dims, num_heads, mlp_dims)

    gpt += LayerNorm(use_bias=bias)
    gpt += Linear(vocab_size, use_bias=False)
    return gpt
