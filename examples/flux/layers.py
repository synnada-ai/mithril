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

from mithril import IOKey
from mithril.models import (
    Arange,
    Buffer,
    Concat,
    Cosine,
    Model,
    Reshape,
    ScaledDotProduct,
    Sine,
    Transpose,
)


def rms_norm(dim: int) -> Model:
    # TODO: check original implementation they use astype and cast to float32
    input = IOKey("input")
    scale = IOKey("scale", shape=[dim])  # TODO: scale must be initialized with ones.
    rrms = 1 / ((input**2).mean(axis=-1, keepdim=True) + 1e-6).sqrt()
    # NOTE: Temporarily, we have to use Buffer to attach the functional connections
    # to the model. This is a workaround for the current limitation of the API.
    block = Model()
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
    apply_rope_block = apply_rope()
    block += apply_rope_block(
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


def embed_nd(theta: int, axes_dim: list[int]) -> Model:
    block = Model()
    input = IOKey("input")

    for i in range(len(axes_dim)):
        rope_B = rope(axes_dim[i], theta)
        block += rope_B(input=input[..., i], output=f"out{i}")

    block += Concat(n=len(axes_dim), axis=-3)(
        **{f"input{i+1}": f"out{i}" for i in range(len(axes_dim))}, output="concat_out"
    )

    block += Buffer()(block.concat_out[:, None], output=IOKey("output"))  # type: ignore [attr-defined]

    return block


def rope(dim: int, theta: int) -> Model:
    assert dim % 2 == 0
    block = Model()
    input = IOKey("input")
    block += Arange(start=0, stop=dim, step=2)(output="arange")

    omega = 1.0 / (theta ** (block.arange / dim))  # type: ignore
    out = input[..., None] * omega

    out_shape = out.shape()
    B, N, D = out_shape[0], out_shape[1], out_shape[2]

    block += Cosine()(out, output="cos")
    block += Sine()(out, output="sin")

    block += Concat(n=4, axis=-1)(
        input1=block.cos[..., None],  # type: ignore
        input2=-block.sin[..., None],  # type: ignore
        input3=block.sin[..., None],  # type: ignore
        input4=block.cos[..., None],  # type: ignore
    )
    rope_shape = (B, N, D, 2, 2)
    block += Reshape()(shape=rope_shape, output=IOKey("output"))
    block.set_canonical_input("input")
    return block
