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
    Add,
    BroadcastTo,
    Convolution2D,
    GroupNorm,
    Model,
    Pad,
    Reshape,
    ScaledDotProduct,
    SiLU,
    Transpose,
)


def resnet_block(
    in_channels: int, out_channels: int | None = None, name: str | None = None
):  # TODO: RETURN TYPE
    out_channels = in_channels if out_channels is None else out_channels

    block = Model(name=name)
    block += GroupNorm(num_groups=32, eps=1e-6, name="norm1")("input")
    block += SiLU()()
    block += Convolution2D(3, out_channels, padding=1, name="conv1")

    block += GroupNorm(num_groups=32, eps=1e-6, name="norm2")
    block += SiLU()
    block += Convolution2D(3, out_channels, padding=1, name="conv2")(output="h")

    # TODO:  We need to solve the implementation below.
    # It is a conditional skip connection.
    if in_channels != out_channels:
        block += Convolution2D(1, out_channels, name="nin_shortcut")(input="input")
        block += Add()(right="h", output=IOKey("output"))

    else:
        block += Add()("input", "h", output=IOKey("output"))

    return block


def attn_block(n_channels: int, name: str | None = None):
    block = Model(name=name)
    block += GroupNorm(num_groups=32, eps=1e-6, name="norm")(
        IOKey("input", shape=[8, 512, 32, 32]), "normilized"
    )
    block += Convolution2D(1, n_channels, name="q")("normilized", output="query")
    block += Convolution2D(1, n_channels, name="k")("normilized", output="key")
    block += Convolution2D(1, n_channels, name="v")("normilized", output="value")

    query = block.query  # type: ignore[attr-defined]
    key = block.key  # type: ignore[attr-defined]
    value = block.value  # type: ignore[attr-defined]

    shape = query.shape()

    query = query.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    key = key.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    value = value.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    block += ScaledDotProduct(is_causal=False)(query, key, value)

    block += Reshape()(shape=(shape[0], shape[2], shape[3], shape[1]))
    block += Transpose(axes=(0, 3, 1, 2))
    block += Convolution2D(1, n_channels, name="proj_out")
    block += Add()(right="input", output=IOKey("output"))

    return block


def downsample(n_channels: int):
    block = Model(name="downsample")
    block += Pad(pad_width=((0, 0), (0, 0), (0, 1), (0, 1)))("input")
    block += Convolution2D(3, n_channels, stride=2, name="conv")(output=IOKey("output"))

    return block


def upsample(n_channels: int, name: str | None = None):
    block = Model(enforce_jit=False, name=name)  # TODO: Remove enfor jit false
    input = IOKey("input")
    input_shape = input.shape()

    B, C, H, W = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    input = input[:, :, :, None, :, None]

    block += BroadcastTo()(input, shape=(B, C, H, 2, W, 2))
    block += Reshape()(shape=(B, C, (H.tensor() * 2).item(), (W.tensor() * 2).item()))
    block += Convolution2D(3, n_channels, padding=1, name="conv")(
        output=IOKey("output")
    )
    return block


def encoder(
    ch: int,
    ch_mult: list[int],
    num_res_blocks: int,
    z_channels: int,
):
    encoder = Model()
    encoder += Convolution2D(3, ch, stride=1, padding=1, name="conv_in")("input")

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
    encoder += Convolution2D(3, 2 * z_channels, padding=1, name="conv_out")(
        output=IOKey("output")
    )

    return encoder


def decoder(ch: int, out_ch: int, ch_mult: list[int], num_res_blocks: int):
    decoder = Model(enforce_jit=False)
    block_in = ch * ch_mult[-1]
    decoder += Convolution2D(3, block_in, padding=1, name="conv_in")("input")

    # Middle
    decoder += resnet_block(block_in, block_in, "mid_block_1")
    decoder += attn_block(block_in, "mid_attn_1")
    decoder += resnet_block(block_in, block_in, name="mid_block_2")

    # Upsample
    up = Model(name="up", enforce_jit=False)
    for level_i in range(len(ch_mult)):
        name_idx_i = len(ch_mult) - level_i - 1
        block = Model(name="block", enforce_jit=False)
        model = Model(name=str(name_idx_i), enforce_jit=False)

        block_out = ch * ch_mult[len(ch_mult) - level_i - 1]
        for idx in range(num_res_blocks + 1):
            block += resnet_block(block_in, block_out, name=f"{idx}")
            block_in = block_out

        model += block

        if name_idx_i != 0:
            model += upsample(block_out, "upsample")

        up += model

    decoder += up

    decoder += GroupNorm(num_groups=32, eps=1e-6, name="norm_out")
    decoder += SiLU()
    decoder += Convolution2D(3, out_ch, padding=1, name="conv_out")(
        output=IOKey("output")
    )

    return decoder
