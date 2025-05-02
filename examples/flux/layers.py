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

import math
from copy import deepcopy

import mithril as ml
from mithril import IOKey, Tensor
from mithril.models import (
    Arange,
    Buffer,
    Cast,
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
    ZerosLike,
)


def apply_rope(*, name: str | None = None) -> Model:
    block = Model(name=name)
    # We define the input connections
    xq = IOKey("xq", type=Tensor)
    xk = IOKey("xk", type=Tensor)
    freqs_cis = IOKey("freqs_cis", type=Tensor)

    xq_shape = xq.shape
    xk_shape = xk.shape
    B, L, H = xq_shape[0], xq_shape[1], xq_shape[2]
    xq_ = xq.reshape(shape=(B, L, H, -1, 1, 2))
    B, L, H = xk_shape[0], xk_shape[1], xk_shape[2]
    xk_ = xk.reshape(shape=(B, L, H, -1, 1, 2))
    # Do the math
    xq_out = (
        freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]  # type: ignore[attr-defined]
    )
    xk_out = (
        freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]  # type: ignore[attr-defined]
    )

    block |= Reshape().connect(xq_out, shape=xq_shape, output="xq_out_raw")
    block |= Reshape().connect(xk_out, shape=xk_shape, output="xk_out_raw")
    block |= Cast().connect(input="xq_out_raw", dtype=xq.dtype(), output="xq_out")
    block |= Cast().connect(input="xk_out_raw", dtype=xk.dtype(), output="xk_out")
    block.expose_keys("xq_out", "xk_out")
    return block


def attention(*, name: str | None = None) -> Model:
    block = Model(name=name)
    block |= apply_rope().connect(
        xq="q", xk="k", freqs_cis="pe", xq_out="q_out", xk_out="k_out"
    )
    block |= ScaledDotProduct(is_causal=False).connect(
        query="q_out", key="k_out", value="v", output="context"
    )

    # We can get named connection as model.'connection_name'
    context_shape = block.context.shape  # type: ignore[attr-defined]

    # NOTE: Reshape input is automatically connected to Transpose output
    block |= Reshape().connect(
        block.context.transpose(axes=(0, 2, 1, 3)),  # type: ignore[attr-defined]
        shape=(context_shape[0], context_shape[2], -1),
        output="output",
    )

    block.expose_keys("output")
    block.set_cout("output")
    return block


def timestep_embedding(
    dim: int,
    max_period: int = 10_000,
    time_factor: float = 1000.0,
    *,
    name: str | None = None,
):
    """
    Create sinusoidal timestep embeddings.
    """
    block = Model(name=name)

    input = IOKey("input")

    input = (input * time_factor)[:, None]  # type: ignore

    half = dim // 2

    block |= Arange(start=0.0, stop=half).connect(output="arange_out")
    freqs = (
        -math.log(max_period) * block.arange_out.cast(dtype=ml.float32) / half  # type: ignore[attr-defined]
    ).exp()

    args = input.cast(dtype=ml.float32) * freqs[None]  # type: ignore[attr-defined]

    block |= Concat(axis=-1).connect(input=[args.cos(), args.sin()], output="embedding")

    if dim % 2:
        block |= ZerosLike().connect(block.embedding[:, :1], output="zeros_like_out")  # type: ignore[attr-defined]
        block |= Concat(axis=-1).connect(
            input=[block.embedding, block.zeros_like_out],  # type: ignore
            output="embedding",
        )
        block |= Cast().connect(
            input="zeros_like_out", dtype=input.dtype(), output="output"
        )
    else:
        block |= Cast().connect(input="embedding", dtype=input.dtype(), output="output")

    block.expose_keys("output")
    return block


def mlp_embedder(hidden_dim: int, *, name: str | None = None):
    block = Model(name=name)
    block |= Linear(hidden_dim, name="in_layer").connect(input="input")
    block += SiLU()
    block += Linear(hidden_dim, name="out_layer").connect(output="output")

    block.expose_keys("output")
    return block


def rms_norm(dim: int, *, name: str | None = None):
    # TODO: check original implementation they use astype and cast to float32
    block = Model(name=name)
    input = IOKey("input")
    scale = IOKey(
        "scale", shape=[dim], differentiable=True
    )  # TODO: scale must be initialized with ones.
    block |= Cast(dtype=ml.float).connect(input=input, output="input_casted")
    rrms = 1 / ((block.input_casted**2).mean(axis=-1, keepdim=True) + 1e-6).sqrt()  # type: ignore[attr-defined]

    block |= Cast().connect(
        block.input_casted * rrms,  # type: ignore[attr-defined]
        dtype=input.dtype(),
        output="casted",  # type: ignore[attr-defined]
    )
    block |= Multiply().connect(left="casted", right=scale, output="output")

    block.expose_keys("output", "casted")
    return block


def qk_norm(dim: int, *, name: str | None = None):
    block = Model(name=name)
    query_norm = rms_norm(dim, name="query_norm")
    key_norm = rms_norm(dim, name="key_norm")

    block |= query_norm.connect(input="q_in", output="q_out")
    block |= key_norm.connect(input="k_in", output="k_out")

    block.expose_keys("q_out", "k_out")
    return block


def modulation(dim: int, double: bool, *, name: str | None = None):
    multiplier = 6 if double else 3

    block = Model(name=name)
    block |= SiLU().connect(input="input")
    block += Linear(dim * multiplier, name="lin").connect(output="lin_out")
    lin_out = block.lin_out[:, None, :]  # type: ignore[attr-defined]
    if double:
        modulation = IOKey("modulation")
        block |= Split(split_size=2, axis=-1).connect(input=lin_out, output=modulation)
        block |= Split(split_size=3, axis=-1).connect(
            input=modulation[0], output=IOKey("mod_1")
        )
        block |= Split(split_size=3, axis=-1).connect(
            input=modulation[1], output=IOKey("mod_2")
        )
    else:
        block |= Split(split_size=3, axis=-1).connect(lin_out, IOKey("mod_1"))

    block.expose_keys("mod_1")
    if double:
        block.expose_keys("mod_2")
    return block


def rearrange(num_heads: int, *, name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    input_shaepe = input.shape
    B, L = input_shaepe[0], input_shaepe[1]
    block |= Reshape().connect(input, shape=(B, L, 3, num_heads, -1))
    block += Transpose(axes=(2, 0, 3, 1, 4)).connect(output=IOKey("output"))
    block.expose_keys("output")
    return block


def double_stream_block(
    hidden_size: int,
    num_heads: int,
    mlp_ratio: float,
    qkv_bias: bool = False,
    *,
    name: str | None = None,
):
    img = IOKey("img", shape=[1, 4096, 3072])
    txt = IOKey("txt", shape=(1, 512, 3072))
    vec = IOKey("vec", shape=(1, 3072))
    pe = IOKey("pe", shape=(1, 1, 4608, 64, 2, 2))

    mlp_hidden_dim = int(hidden_size * mlp_ratio)

    block = Model(name=name)
    block |= modulation(hidden_size, double=True, name="img_mod").connect(
        input=vec, mod_1=IOKey("img_mod_1"), mod_2=IOKey("img_mod_2")
    )
    block |= LayerNorm(
        use_scale=False, use_bias=False, eps=1e-6, name="img_norm1"
    ).connect(input=img, output=IOKey("img_norm"))

    img_modulated = (1 + block.img_mod_1[1]) * block.img_norm + block.img_mod_1[0]  # type: ignore[attr-defined]

    block |= Linear(hidden_size * 3, use_bias=qkv_bias, name="img_attn_qkv").connect(
        img_modulated, output=IOKey("img_qkv")
    )

    # Rearrange
    block |= rearrange(num_heads=num_heads).connect(
        input=block.img_qkv,  # type: ignore[attr-defined]
        output="img_rearrange_out",
    )

    rearrange_out = block.img_rearrange_out  # type: ignore[attr-defined]
    img_q, img_k, img_v = (rearrange_out[0], rearrange_out[1], rearrange_out[2])
    block |= qk_norm(hidden_size // num_heads, name="img_attn_norm").connect(
        q_in=img_q, k_in=img_k, q_out="q_out", k_out="k_out"
    )
    img_q, img_k = block.q_out, block.k_out  # type: ignore[attr-defined]

    block |= modulation(hidden_size, double=True, name="txt_mod").connect(
        input=vec, mod_1="txt_mod_1", mod_2="txt_mod_2"
    )
    block |= LayerNorm(
        use_scale=False, use_bias=False, eps=1e-6, name="txt_norm1"
    ).connect(input=txt, output="txt_norm")

    txt_modulated = (1 + block.txt_mod_1[1]) * block.txt_norm + block.txt_mod_1[0]  # type: ignore[attr-defined]

    block |= Linear(hidden_size * 3, use_bias=qkv_bias, name="txt_attn_qkv").connect(
        txt_modulated, output="txt_qkv"
    )

    # Rearrange
    block |= rearrange(num_heads).connect(
        input=block.txt_qkv,  # type: ignore[attr-defined]
        output="txt_rearrange_out",
    )

    rearrange_out = block.txt_rearrange_out  # type: ignore[attr-defined]
    txt_q, txt_k, txt_v = rearrange_out[0], rearrange_out[1], rearrange_out[2]
    block |= qk_norm(hidden_size // num_heads, name="txt_attn_norm").connect(
        q_in=txt_q, k_in=txt_k, q_out="txt_q_out", k_out="txt_k_out"
    )
    txt_q, txt_k = block.txt_q_out, block.txt_k_out  # type: ignore[attr-defined]

    block |= Concat(axis=2).connect(input=[txt_q, img_q], output=IOKey("q_concat"))
    block |= Concat(axis=2).connect(input=[txt_k, img_k], output=IOKey("k_concat"))
    block |= Concat(axis=2).connect(input=[txt_v, img_v], output=IOKey("v_concat"))

    block |= attention().connect(
        q="q_concat", k="k_concat", v="v_concat", pe=pe, output="attn"
    )
    img_attn = block.attn[:, txt.shape[1] :]  # type: ignore

    block |= Linear(hidden_size, name="img_attn_proj").connect(
        img_attn, output="img_proj"
    )
    img = img + block.img_mod_1[2] * block.img_proj  # type: ignore[attr-defined]

    block |= LayerNorm(
        use_scale=False, use_bias=False, name="img_norm2", eps=1e-6
    ).connect(img, output="img_norm_2")
    img_norm_2 = block.img_norm_2  # type: ignore[attr-defined]

    img_mlp = Model(name="img_mlp")
    img_mlp |= Linear(mlp_hidden_dim, name="0").connect(input="input")
    img_mlp += Gelu(approximate=True)
    img_mlp += Linear(hidden_size, name="2").connect(output="output")

    txt_mlp = deepcopy(img_mlp)
    txt_mlp.name = "txt_mlp"

    block |= img_mlp.connect(
        input=(1 + block.img_mod_2[1]) * img_norm_2 + block.img_mod_2[0],  # type: ignore[attr-defined]
        output="img_mlp",
    )
    img = img + block.img_mod_2[2] * block.img_mlp  # type: ignore[attr-defined]

    # TODO: Use txt.shape[1]]
    txt_attn = block.attn[:, : txt.shape[1]]  # type: ignore
    block |= Linear(hidden_size, name="txt_attn_proj").connect(
        txt_attn, output="txt_proj"
    )

    txt = txt + block.txt_mod_1[2] * block.txt_proj  # type: ignore[attr-defined]

    block |= LayerNorm(
        use_scale=False, use_bias=False, name="txt_norm2", eps=1e-6
    ).connect(txt, output="txt_norm_2")
    txt_norm_2 = block.txt_norm_2  # type: ignore[attr-defined]

    block |= txt_mlp.connect(
        input=(1 + block.txt_mod_2[1]) * txt_norm_2 + block.txt_mod_2[0],  # type: ignore[attr-defined]
        output="txt_mlp",
    )
    txt = txt + block.txt_mod_2[2] * block.txt_mlp  # type: ignore[attr-defined]

    block |= Buffer().connect(img, output="img_out")
    block |= Buffer().connect(txt, output="txt_out")
    block.expose_keys("img_out", "txt_out")
    return block


def single_stream_block(
    hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, *, name: str | None = None
):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    input = IOKey("input")
    vec = IOKey("vec")
    pe = IOKey("pe")

    head_dim = hidden_size // num_heads
    mlp_hidden_dim = int(hidden_size * mlp_ratio)

    block = Model(name=name)
    block |= modulation(hidden_size, False, name="modulation").connect(
        input=vec, mod_1="mod"
    )
    block |= LayerNorm(use_scale=False, use_bias=False, name="pre_norm").connect(
        input=input, output="pre_norm"
    )

    x_mod = (1 + block.mod[1]) * block.pre_norm + block.mod[0]  # type: ignore[attr-defined]

    block |= Linear(hidden_size * 3 + mlp_hidden_dim, name="linear1").connect(
        input=x_mod, output="lin1_out"
    )

    # Split
    qkv = block.lin1_out[..., : 3 * hidden_size]  # type: ignore[attr-defined]
    mlp = block.lin1_out[..., 3 * hidden_size :]  # type: ignore[attr-defined]

    # Rearrange
    block |= rearrange(num_heads).connect(input=qkv, output="rearrange_out")

    q = block.rearrange_out[0]  # type: ignore[attr-defined]
    k = block.rearrange_out[1]  # type: ignore[attr-defined]
    v = block.rearrange_out[2]  # type: ignore[attr-defined]

    block |= qk_norm(dim=head_dim, name="norm").connect(
        q_in=q, k_in=k, q_out="q_out", k_out="k_out"
    )
    block |= attention().connect(q="q_out", k="k_out", v=v, pe=pe, output="attn")
    block |= Gelu(approximate=True).connect(input=mlp, output="mlp_act")
    block |= Concat(axis=2).connect(
        input=[block.attn, block.mlp_act],  # type: ignore[attr-defined]
        output="concat_out",
    )
    block |= Linear(hidden_size, name="linear2").connect(
        input="concat_out", output="lin2_out"
    )
    block |= Buffer().connect(
        input + block.mod[2] * block.lin2_out,  # type: ignore[attr-defined]
        output=IOKey("output"),
    )

    return block


def last_layer(
    hidden_size: int, patch_size: int, out_channels: int, *, name: str | None = None
):
    adaLN_modulation = Model(name="adaLN_modulation")
    adaLN_modulation |= Sigmoid().connect(input="input")
    adaLN_modulation += Multiply().connect(right="input")
    adaLN_modulation += Linear(hidden_size * 2, name="1").connect(
        output=IOKey("output")
    )

    block = Model(name=name)
    input = IOKey("input")
    vec = IOKey("vec")

    block |= adaLN_modulation.connect(input=vec, output="mod")
    block |= Split(split_size=2, axis=1).connect(input="mod", output="mod_split")
    block |= LayerNorm(use_scale=False, use_bias=False, name="norm_final").connect(
        input=input, output="pre_norm"
    )

    shift = block.mod_split[0]  # type: ignore[attr-defined]
    scale = block.mod_split[1]  # type: ignore[attr-defined]
    input = (1 + scale[:, None, :]) * block.pre_norm + shift[:, None, :]  # type: ignore[attr-defined]
    block |= Linear(patch_size * patch_size * out_channels, name="linear").connect(
        input=input, output=IOKey("output")
    )

    return block


def embed_nd(theta: int, axes_dim: list[int], *, name: str | None = None) -> Model:
    block = Model(name=name)
    input = IOKey("input")

    for i in range(len(axes_dim)):
        rope_B = rope(axes_dim[i], theta)
        block |= rope_B.connect(input=input[..., i], output=f"out{i}")

    block |= Concat(axis=-3).connect(
        input=[getattr(block, f"out{i}") for i in range(len(axes_dim))],
        output="concat_out",
    )

    block |= Buffer().connect(block.concat_out[:, None], output=IOKey("output"))  # type: ignore [attr-defined]
    block.set_cin("input")
    block.set_cout("output")

    return block


def rope(dim: int, theta: int, *, name: str | None = None) -> Model:
    assert dim % 2 == 0
    block = Model(name=name)
    input = IOKey("input", type=Tensor)
    block |= Arange(start=0, stop=dim, step=2).connect(output="arange")

    omega = 1.0 / (theta ** (block.arange.cast(ml.float32) / dim))  # type: ignore
    out = input[..., None] * omega

    out_shape = out.shape
    B, N, D = out_shape[0], out_shape[1], out_shape[2]

    block |= Concat(axis=-1).connect(
        input=[
            out.cos()[..., None],
            -out.sin()[..., None],
            out.sin()[..., None],
            out.cos()[..., None],
        ],
        output="concat_out",
    )
    rope_shape = (B, N, D, 2, 2)
    block |= Reshape().connect("concat_out", shape=rope_shape, output="reshape_out")
    block |= Cast(dtype=ml.float32).connect(input="reshape_out", output=IOKey("output"))
    block.set_cin("input")
    block.set_cout("output")
    return block
