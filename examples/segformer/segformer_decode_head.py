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

from mithril.framework import IOKey, Tensor
from mithril.models import (
    Arange,
    BatchNorm2D,
    Concat,
    Convolution2D,
    Flatten,
    Linear,
    Model,
    Relu,
)
from mithril.types import Dtype


def segformer_bilinear_interpolate(*, name: str | None = None):
    input = IOKey("input", type=Tensor[float])
    size = IOKey("size", type=tuple[int, ...])

    input_dtype = input.dtype()
    shape = input.shape
    H_in, W_in = shape[2], shape[3]
    H_out, W_out = size[0], size[1]
    # Scale factors
    scale_y = H_in / H_out
    scale_x = W_in / W_out

    # Create normalized grid (i.e., in input pixel space)
    yy = ((Arange()(stop=H_out) + 0.5) * scale_y - 0.5).clamp(0, H_in - 1)
    xx = ((Arange()(stop=W_out) + 0.5) * scale_x - 0.5).clamp(0, W_in - 1)

    y0 = yy.floor().cast(Dtype.int64)
    x0 = xx.floor().cast(Dtype.int64)
    y1 = (y0 + 1).clamp(0, H_in - 1)
    x1 = (x0 + 1).clamp(0, W_in - 1)

    ly = (yy - y0.cast(input_dtype)).reshape((1, 1, H_out, 1))
    lx = (xx - x0.cast(input_dtype)).reshape((1, 1, 1, W_out))
    hy = 1.0 - ly
    hx = 1.0 - lx

    Ia = input[:, :, y0[:, None], x0]  # top-left
    Ib = input[:, :, y1[:, None], x0]  # bottom-left
    Ic = input[:, :, y0[:, None], x1]  # top-right
    Id = input[:, :, y1[:, None], x1]  # bottom-right

    wa = hy * hx
    wb = ly * hx
    wc = hy * lx
    wd = ly * lx
    out = Ia * wa + Ib * wb + Ic * wc + Id * wd

    return Model.create(name=name, output=out)


def segformer_mlp(decoder_hidden_size, *, name: str | None = None) -> Model:
    input = IOKey("input", type=Tensor[float])

    hidden_states = Flatten(start_dim=2)(input=input).transpose(axes=(0, 2, 1))
    hidden_states = Linear(decoder_hidden_size, name="proj")(input=hidden_states)
    return Model.create(name=name, output=hidden_states)


def segformer_decode_head(config, *, name: str | None = None) -> Model:
    hidden_states = IOKey(
        "input", value=tuple(Tensor() for _ in range(config.num_encoder_blocks))
    )

    hidden_shapes = [
        [1, 64, 128, 128],
        [1, 128, 64, 64],
        [1, 320, 32, 32],
        [1, 512, 16, 16],
    ]
    batch_size = hidden_states[0].shape[0]
    size = hidden_states[0].shape[2:]
    num_encoder_blocks = config.num_encoder_blocks
    all_hidden_states: list[IOKey] = []

    for i in range(num_encoder_blocks):
        encoder_hidden_state: IOKey = hidden_states[i]
        encoder_hidden_state.set_shapes(hidden_shapes[i])
        # Unify channel dimension.
        shape = encoder_hidden_state.shape
        height, width = shape[2], shape[3]
        encoder_hidden_state = segformer_mlp(
            config.decoder_hidden_size, name=f"linear_c_{i}"
        )(input=encoder_hidden_state)
        encoder_hidden_state = encoder_hidden_state.transpose(axes=(0, 2, 1))
        encoder_hidden_state = encoder_hidden_state.reshape(
            shape=(batch_size, -1, height, width)
        )
        # Upsample.
        encoder_hidden_state = segformer_bilinear_interpolate()(
            input=encoder_hidden_state, size=size
        )
        all_hidden_states.append(encoder_hidden_state)

    # Concatenate all hidden states.
    hidden_states = Concat(axis=1)(input=all_hidden_states[::-1])
    hidden_states = Convolution2D(
        out_channels=config.decoder_hidden_size,
        kernel_size=1,
        use_bias=False,
        name="linear_fuse",
    )(input=hidden_states)
    hidden_states = BatchNorm2D(config.decoder_hidden_size, name="batch_norm")(
        input=hidden_states
    )
    hidden_states = Relu()(input=hidden_states)
    hidden_states = Convolution2D(
        out_channels=config.num_labels,
        kernel_size=1,
        name="classifier",
    )(input=hidden_states)
    return Model.create(name=name, logits=hidden_states)
