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

    input = IOKey("input")
    lin_out = Linear(input_dim * 3, name="c_attn")(input=input)

    t_axes = (0, 2, 1, 3)
    shp_con = input.shape
    reshape_con = (shp_con[0], shp_con[1], num_heads, -1)

    split_out = Split(3, axis=-1)(input=lin_out)
    tq = split_out[0].reshape(reshape_con).transpose(t_axes)
    tk = split_out[1].reshape(reshape_con).transpose(t_axes)
    tv = split_out[2].reshape(reshape_con).transpose(t_axes)

    sdp_out = ScaledDotProduct()(query=tq, key=tk, value=tv)
    t_sdp = sdp_out.transpose(t_axes).reshape(shp_con[:3])
    output = Linear(input_dim, name="c_proj")(input=t_sdp)
    return Model.create(output, name="attn")



def mlp(n_embd: int):
    fc_out = Linear(n_embd * 4, name="c_fc")(input=IOKey("input"))
    gelu_out = Gelu()(input=fc_out)
    output = Linear(n_embd, name="c_proj")(input=gelu_out)
    return Model.create(output, name="mlp")


# Create Transformer Encoder Block.
def create_block(name, dims, num_heads, bias=True, eps=1e-5):
    input = IOKey("input")
    ln1_out = LayerNorm(use_bias=bias, eps=eps, name="ln_1")(input=input)
    attn_out = causal_attention(dims, num_heads, bias)(input=ln1_out)
    add1_out = Add()(left=input, right=attn_out)
    ln2_out = LayerNorm(use_bias=bias, eps=eps, name="ln_2")(input=add1_out)
    mlp_out = mlp(dims)(input=ln2_out)
    output = Add()(left=add1_out, right=mlp_out)
    return Model.create(output, name=name)


def create_gpt(bias, block_size, dims, num_heads, num_layers, vocab_size):
    # Create Position Embedding model
    input = IOKey("input")
    s_out = Size(dim=1)(input=input)
    arr_out = Arange(start=0, step=1)(stop=s_out)
    pos_out = Embedding(name="wpe", num_embeddings=block_size, dim=dims)(input=arr_out)
    token_out = Embedding(name="wte", num_embeddings=vocab_size, dim=dims)(input=input)

    block_out = IOKey("input")
    for idx in range(num_layers):
        block_out = create_block(f"{idx}", dims, num_heads)(input=block_out)
    blocks_out = Model.create(block_out, name="h")(input=pos_out + token_out)
    
    ln_out = LayerNorm(use_bias=bias, name="ln_f")(input=blocks_out)
    transformer = Model.create(ln_out, name="transformer")
    # Create GPT
    t_out = transformer(input=IOKey("input", differentiable=False))
    output = Linear(vocab_size, use_bias=False, name="lm_head")(input=t_out)
    return Model.create(output=output, name="gpt")