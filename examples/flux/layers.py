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

from copy import deepcopy

from mithril import IOKey
from mithril.models import (
    Buffer,
    Concat,
    Gelu,
    LayerNorm,
    Linear,
    Model,
    Multiply,
    Reshape,
    ScaledDotProduct,
    Sigmoid,
    SiLU,
    Split,
    Transpose,
)


def mlp_embedder(hidden_dim: int, name: str | None = None):
    block = Model(name=name)
    block += Linear(hidden_dim, name="in_layer")(input="input")
    block += SiLU()
    block += Linear(hidden_dim, name="out_layer")(output=IOKey("output"))

    return block


def rms_norm(dim: int, name: str | None = None):
    # TODO: check original implementation they use astype and cast to float32
    input = IOKey("input")
    scale = IOKey("scale", shape=[dim])  # TODO: scale must be initialized with ones.
    rrms = 1 / ((input**2).mean(axis=-1, keepdim=True) + 1e-6).sqrt()
    # NOTE: Temporarily, we have to use Buffer to attach the functional connections
    # to the model. This is a workaround for the current limitation of the API.
    block = Model(name=name)
    block += Buffer()(rrms, output=IOKey("rrms"))
    block += Buffer()(input * rrms * scale, output=IOKey("output"))

    return block


def apply_rope() -> Model:
    block = Model()
    # We define the input connections
    xq = IOKey("xq")
    xk = IOKey("xk")
    freqs_cis = IOKey("freqs_cis")

    xq_shape = xq.shape()
    xk_shape = xk.shape()
    B, L, H = xq_shape[0], xq_shape[1], xq_shape[2]
    block += Reshape()(xq, shape=(B, L, H, -1, 1, 2), output="xq_")
    B, L, H = xk_shape[0], xk_shape[1], xk_shape[2]
    # B,L,H = *xk.shape is not supported yet.
    block += Reshape()(xk, shape=(B, L, H, -1, 1, 2), output="xk_")
    # Do the math
    xq_out = (
        freqs_cis[..., 0] * block.xq_[..., 0] + freqs_cis[..., 1] * block.xq_[..., 1]  # type: ignore[attr-defined]
    )
    xk_out = (
        freqs_cis[..., 0] * block.xk_[..., 0] + freqs_cis[..., 1] * block.xk_[..., 1]  # type: ignore[attr-defined]
    )

    # We are explicitly defining the output connections with IOKey
    block += Reshape()(xq_out, shape=xq_shape, output=IOKey("xq_out"))
    block += Reshape()(xk_out, shape=xk_shape, output=IOKey("xk_out"))
    return block


def attention() -> Model:
    block = Model()
    block += apply_rope()(
        xq="q", xk="k", freqs_cis="pe", xq_out="q_out", xk_out="k_out"
    )
    block += ScaledDotProduct(is_causal=False)(
        query="q_out", key="k_out", value="v", output="context"
    )

    # We can get named connection as model.'connection_name'
    context_shape = block.context.shape()  # type: ignore[attr-defined]
    block += Transpose(axes=(0, 2, 1, 3))(block.context)  # type: ignore[attr-defined]
    # NOTE: Reshape input is automatically connected to Transpose output
    block += Reshape()(
        shape=(context_shape[0], context_shape[2], -1), output=IOKey("output")
    )

    return block


def qk_norm(dim: int, name: str | None = None):
    block = Model(name=name)
    query_norm = rms_norm(dim, name="query_norm")
    key_norm = rms_norm(dim, name="key_norm")

    block += query_norm(input="q_in", output=IOKey("q_out"))
    block += key_norm(input="k_in", output=IOKey("k_out"))
    return block


def modulation(dim: int, double: bool, name: str | None = None):
    multiplier = 6 if double else 3

    block = Model(name=name)
    block += SiLU()(input="input")
    block += Linear(dim * multiplier, name="lin")(output="lin_out")
    lin_out = block.lin_out[:, None, :]  # type: ignore[attr-defined]
    if double:
        modulation = lin_out.split(2, axis=-1)
        block += Buffer()(modulation[0].split(3, axis=-1), IOKey("mod_1"))
        block += Buffer()(modulation[1].split(3, axis=-1), IOKey("mod_2"))
    else:
        modulation = lin_out.split(3, axis=-1)
        block += Buffer()(modulation, IOKey("mod_1"))

    return block


def rearrange(num_heads: int):
    block = Model()
    input = IOKey("input")
    input_shaepe = input.shape()
    B, L = input_shaepe[0], input_shaepe[1]
    block += Reshape()(shape=(B, L, 3, num_heads, -1))
    block += Transpose(axes=(2, 0, 3, 1, 4))(output=IOKey("output"))
    return block


def double_stream_block(
    hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False
):
    img = IOKey("img")
    txt = IOKey("txt")
    vec = IOKey("vec")
    pe = IOKey("pe")

    mlp_hidden_dim = int(hidden_size * mlp_ratio)

    block = Model()
    block += modulation(hidden_size, double=True, name="img_mod")(
        input=vec, mod_1="img_mod_1", mod_2="img_mod_2"
    )
    block += LayerNorm(use_scale=False, use_bias=False, eps=1e-6, name="img_norm1")(
        input=img, output="img_norm"
    )

    img_modulated = (1 + block.img_mod_1[1]) * block.img_norm + block.img_mod_1[0]  # type: ignore[attr-defined]

    block += Linear(hidden_size * 3, use_bias=qkv_bias, name="img_attn_qkv")(
        img_modulated, output="img_qkv"
    )

    # Rearrange
    block += rearrange(num_heads=num_heads)(
        input=block.img_qkv,  # type: ignore[attr-defined]
        output="img_rearrange_out",
    )

    rearrange_out = block.txt_rearrange_out  # type: ignore[attr-defined]
    img_q, img_k, img_v = (rearrange_out[0], rearrange_out[1], rearrange_out[2])
    block += qk_norm(hidden_size // num_heads, name="img_attn_norm")(
        q_in=img_q, k_in=img_k, q_out="q_out", k_out="k_out"
    )
    img_q, img_k = block.q_out, block.k_out  # type: ignore[attr-defined]

    block += modulation(hidden_size, double=True, name="txt_mod")(
        input=vec, mod_1="txt_mod_1", mod_2="txt_mod_2"
    )
    block += LayerNorm(use_scale=False, use_bias=False, eps=1e-6, name="txt_norm1")(
        input=txt, output="txt_norm"
    )

    txt_modulated = (1 + block.txt_mod_1[1]) * block.txt_norm + block.txt_mod_1[0]  # type: ignore[attr-defined]

    block += Linear(hidden_size * 3, use_bias=qkv_bias, name="txt_attn_qkv")(
        txt_modulated, output=IOKey("txt_qkv")
    )

    # Rearrange
    block += rearrange(num_heads)(input=block.txt_qkv, output="txt_rearrange_out")  # type: ignore[attr-defined]

    rearrange_out = block.txt_rearrange_out  # type: ignore[attr-defined]
    txt_q, txt_k, txt_v = rearrange_out[0], rearrange_out[1], rearrange_out[2]
    block += qk_norm(hidden_size // num_heads, name="txt_attn_norm")(
        q_in=txt_q, k_in=txt_k, q_out="txt_q_out", k_out="txt_k_out"
    )
    txt_q, txt_k = block.txt_q_out, block.txt_k_out  # type: ignore[attr-defined]

    block += Concat(axis=2, n=2)(input1=txt_q, input2=img_q, output="q_concat")
    block += Concat(axis=2, n=2)(input1=txt_k, input2=img_k, output="k_concat")
    block += Concat(axis=2, n=2)(input1=txt_v, input2=img_v, output="v_concat")

    block += attention()(q="q_concat", k="k_concat", v="v_concat", pe=pe, output="attn")
    # TODO: use'[:, txt.shape()[1] :]' when fixed.
    img_attn = block.attn[:, 256:]  # type: ignore[attr-defined]

    block += Linear(hidden_size, name="img_attn_proj")(img_attn, output="img_proj")
    img = img + block.img_mod_1[2] * block.img_proj  # type: ignore[attr-defined]

    block += LayerNorm(use_scale=False, use_bias=False, name="img_norm2", eps=1e-6)(
        img, output="img_norm_2"
    )
    img_norm_2 = block.img_norm_2  # type: ignore[attr-defined]

    img_mlp = Model(name="img_mlp")
    img_mlp += Linear(mlp_hidden_dim, name="0")(input="input")
    img_mlp += Gelu(approximate=True)
    img_mlp += Linear(hidden_size, name="2")(output="output")

    txt_mlp = deepcopy(img_mlp)
    txt_mlp.name = "txt_mlp"

    block += img_mlp(
        input=(1 + block.img_mod_2[1]) * img_norm_2 + block.img_mod_2[0],  # type: ignore[attr-defined]
        output="img_mlp",
    )
    img = img + block.img_mod_2[2] * block.img_mlp  # type: ignore[attr-defined]

    # TODO: Use txt.shape()[1]]
    txt_attn = block.attn[:, :256]  # type: ignore[attr-defined]
    block += Linear(hidden_size, name="txt_attn_proj")(txt_attn, output="txt_proj")

    txt = txt + block.txt_mod_1[2] * block.txt_proj  # type: ignore[attr-defined]

    block += LayerNorm(use_scale=False, use_bias=False, name="txt_norm2", eps=1e-6)(
        txt, output="txt_norm_2"
    )
    txt_norm_2 = block.txt_norm_2  # type: ignore[attr-defined]

    block += txt_mlp(
        input=(1 + block.txt_mod_2[1]) * txt_norm_2 + block.txt_mod_2[0],  # type: ignore[attr-defined]
        output="txt_mlp",
    )
    txt = txt + block.txt_mod_2[2] * block.txt_mlp  # type: ignore[attr-defined]

    block += Buffer()(img, output=IOKey("img_out"))
    block += Buffer()(txt, output=IOKey("txt_out"))
    return block


def single_stream_block(hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    input = IOKey("input")
    vec = IOKey("vec")
    pe = IOKey("pe")

    head_dim = hidden_size // num_heads
    mlp_hidden_dim = int(hidden_size * mlp_ratio)

    block = Model()
    block += modulation(hidden_size, False, name="modulation")(input=vec, mod_1="mod")
    block += LayerNorm(use_scale=False, use_bias=False, name="pre_norm")(
        input=input, output="pre_norm"
    )

    x_mod = (1 + block.mod[1]) * block.pre_norm + block.mod[0]  # type: ignore[attr-defined]

    block += Linear(hidden_size * 3 + mlp_hidden_dim, name="linear1")(
        input=x_mod, output="lin1_out"
    )

    # Split
    qkv = block.lin1_out[..., : 3 * hidden_size]  # type: ignore[attr-defined]
    mlp = block.lin1_out[..., 3 * hidden_size :]  # type: ignore[attr-defined]

    # Rearrange
    block += rearrange(num_heads)(input=qkv, output="rearrange_out")

    q = block.rearrange_out[0]  # type: ignore[attr-defined]
    k = block.rearrange_out[1]  # type: ignore[attr-defined]
    v = block.rearrange_out[2]  # type: ignore[attr-defined]

    block += qk_norm(dim=head_dim, name="norm")(
        q_in=q, k_in=k, q_out="q_out", k_out="k_out"
    )
    block += attention()(q="q_out", k="k_out", v=v, pe=pe, output="attn")
    block += Gelu(approximate=True)(input=mlp, output="mlp_act")
    block += Concat(n=2, axis=2)(input1="attn", input2="mlp_act", output="concat_out")
    block += Linear(hidden_size, name="linear2")(input="concat_out", output="lin2_out")
    block += Buffer()(input + block.mod[2] * block.lin2_out, output=IOKey("output"))  # type: ignore[attr-defined]

    return block


def last_layer(hidden_size: int, patch_size: int, out_channels: int):
    adaLN_modulation = Model(name="adaLN_modulation")
    adaLN_modulation += Sigmoid()(input="input")
    adaLN_modulation += Multiply()(right="input")
    adaLN_modulation += Linear(hidden_size * 2, name="1")(output=IOKey("output"))

    block = Model()
    input = IOKey("input")
    vec = IOKey("vec")

    block += adaLN_modulation(input=vec, output="mod")
    block += Split(split_size=2, axis=1)(input="mod", output="mod_split")
    block += LayerNorm(use_scale=False, use_bias=False, name="norm_final")(
        input=input, output="pre_norm"
    )
    shift = block.mod_split[0]  # type: ignore[attr-defined]
    scale = block.mod_split[1]  # type: ignore[attr-defined]
    input = (1 + scale[:, None, :]) * block.pre_norm + shift[:, None, :]  # type: ignore[attr-defined]
    block += Linear(patch_size * patch_size * out_channels, name="linear")(
        input=input, output=IOKey("output")
    )

    return block
