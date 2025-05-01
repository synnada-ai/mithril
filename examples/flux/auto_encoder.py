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

from dataclasses import dataclass

from mithril import IOKey
from mithril.models import (
    Add,
    BroadcastTo,
    Buffer,
    Convolution2D,
    GroupNorm,
    Model,
    Multiply,
    Pad,
    Randn,
    Reshape,
    ScaledDotProduct,
    SiLU,
    Subtract,
    Transpose,
)


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def resnet_block(
    in_channels: int, out_channels: int | None = None, *, name: str | None = None
):
    out_channels = in_channels if out_channels is None else out_channels

    input = IOKey("input", shape=[None, in_channels, None, None])

    block = Model(name=name)
    block |= GroupNorm(num_groups=32, eps=1e-6, name="norm1").connect(input)
    block += SiLU()
    block += Convolution2D(3, out_channels, padding=1, name="conv1")

    block += GroupNorm(num_groups=32, eps=1e-6, name="norm2")
    block += SiLU()
    block += Convolution2D(3, out_channels, padding=1, name="conv2").connect(output="h")

    # TODO:  We need to solve the implementation below.
    # It is a conditional skip connection.
    if in_channels != out_channels:
        block |= Convolution2D(1, out_channels, name="nin_shortcut").connect(
            input=input, output="con_out"
        )
        block |= Add().connect(left="con_out", right="h", output=IOKey("output"))

    else:
        block |= Add().connect(input, "h", output=IOKey("output"))

    return block


def attn_block(n_channels: int, *, name: str | None = None):
    block = Model(name=name)
    block |= GroupNorm(num_groups=32, eps=1e-6, name="norm").connect(
        IOKey("input", shape=[None, 512, None, None]), "normalized"
    )
    block |= Convolution2D(1, n_channels, name="q").connect(
        "normalized", output="query"
    )
    block |= Convolution2D(1, n_channels, name="k").connect("normalized", output="key")
    block |= Convolution2D(1, n_channels, name="v").connect(
        "normalized", output="value"
    )

    query = block.query  # type: ignore[attr-defined]
    key = block.key  # type: ignore[attr-defined]
    value = block.value  # type: ignore[attr-defined]

    shape = query.shape  # type: ignore[attr-defined]

    query = query.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    key = key.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    value = value.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    block |= ScaledDotProduct(is_causal=False).connect(
        query, key, value, output="sdp_out"
    )
    block.set_cout("sdp_out")

    block += Reshape().connect(shape=(shape[0], shape[2], shape[3], shape[1]))
    block += Transpose(axes=(0, 3, 1, 2))
    block += Convolution2D(1, n_channels, name="proj_out")
    block += Add().connect(right="input", output=IOKey("output"))

    return block


def downsample(n_channels: int, *, name: str = "downsample"):
    block = Model(name=name)
    block |= Pad(pad_width=((0, 0), (0, 0), (0, 1), (0, 1))).connect("input")
    block += Convolution2D(3, n_channels, stride=2, name="conv").connect(
        output=IOKey("output")
    )

    return block


def upsample(n_channels: int, *, name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    input_shape = input.shape

    B, C, H, W = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    input = input[:, :, :, None, :, None]  # type: ignore

    block |= BroadcastTo().connect(input, shape=(B, C, H, 2, W, 2))
    block += Reshape().connect(shape=(B, C, H * 2, W * 2))
    block += Convolution2D(3, n_channels, padding=1, name="conv").connect(
        output=IOKey("output")
    )
    return block


def encoder(
    ch: int,
    ch_mult: list[int],
    num_res_blocks: int,
    z_channels: int,
    *,
    name: str | None = None,
    **kwargs,
):
    encoder = Model(name=name)
    encoder |= Convolution2D(3, ch, stride=1, padding=1, name="conv_in").connect(
        "input"
    )

    in_ch_mult = (1,) + tuple(ch_mult)

    # Downsample
    down = Model(name="down")
    for level_i in range(len(ch_mult)):
        block = Model(name="block")
        model = Model(name=str(level_i))

        block_in = ch * in_ch_mult[level_i]
        block_out = ch * ch_mult[level_i]
        for idx in range(num_res_blocks):
            block += resnet_block(block_in, block_out, name=str(idx))
            block_in = block_out

        model += block

        if level_i != len(ch_mult) - 1:
            model += downsample(block_in)

        down += model

    encoder += down

    # Middle
    encoder += resnet_block(block_in, block_in, name="mid_block_1")
    encoder += attn_block(block_in, name="mid_attn_1")
    encoder += resnet_block(block_in, block_in, name="mid_block_2")

    # end
    encoder += GroupNorm(32, eps=1e-6, name="norm_out")
    encoder += SiLU()
    encoder += Convolution2D(3, 2 * z_channels, padding=1, name="conv_out").connect(
        output=IOKey("output")
    )

    return encoder


def decoder(
    ch: int,
    out_ch: int,
    ch_mult: list[int],
    num_res_blocks: int,
    *,
    name: str | None = None,
    **kwargs,
):
    decoder = Model(name=name)
    block_in = ch * ch_mult[-1]
    decoder |= Convolution2D(3, block_in, padding=1, name="conv_in").connect("input")

    # Middle
    decoder += resnet_block(block_in, block_in, name="mid_block_1")
    decoder += attn_block(block_in, name="mid_attn_1")
    decoder += resnet_block(block_in, block_in, name="mid_block_2")

    # Upsample
    up = Model(name="up")
    for level_i in range(len(ch_mult)):
        name_idx_i = len(ch_mult) - level_i - 1
        block = Model(name="block")
        model = Model(name=str(name_idx_i))

        block_out = ch * ch_mult[len(ch_mult) - level_i - 1]
        for idx in range(num_res_blocks + 1):
            block += resnet_block(block_in, block_out, name=f"{idx}")
            block_in = block_out

        model += block

        if name_idx_i != 0:
            model += upsample(block_out, name="upsample")

        up += model

    decoder += up

    decoder += GroupNorm(num_groups=32, eps=1e-6, name="norm_out")
    decoder += SiLU()
    decoder += Convolution2D(3, out_ch, padding=1, name="conv_out").connect(
        output=IOKey("output")
    )

    return decoder


def diagonal_gaussian(sample: bool = True, chunk_dim: int = 1):
    block = Model()

    input = IOKey("input")
    input = input.split(2, axis=1)  # type: ignore

    if sample:
        std = (input[1] * 0.5).exp()
        mean = input[0]
        block |= Randn().connect(shape=mean.shape, output="randn")
        output = mean + std * block.randn  # type: ignore[attr-defined]
    else:
        output = input[0]

    block |= Buffer().connect(input=output, output=IOKey("output"))
    return block


def auto_encoder(
    ae_params: AutoEncoderParams,
):
    model = Model()
    model |= encoder(
        ae_params.ch,
        ae_params.ch_mult,
        ae_params.num_res_blocks,
        ae_params.z_channels,
        name="encoder",
    ).connect(input="input")
    model += diagonal_gaussian()

    model += Subtract().connect(right=ae_params.shift_factor)
    model += Multiply().connect(right=ae_params.scale_factor)

    model += decoder(
        ae_params.ch,
        ae_params.out_ch,
        ae_params.ch_mult,
        ae_params.num_res_blocks,
        name="decoder",
    ).connect(output="output")

    return model


def decode(ae_params: AutoEncoderParams):
    model = Model()
    input = IOKey("input")
    input = input / ae_params.scale_factor - ae_params.shift_factor  # type: ignore
    model |= decoder(
        ae_params.ch,
        ae_params.out_ch,
        ae_params.ch_mult,
        ae_params.num_res_blocks,
        name="decoder",
    ).connect(input=input, output="output")

    return model
