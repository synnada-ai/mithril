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
from collections.abc import Callable

import torch

import mithril as ml
from mithril.models import (
    Arange,
    BroadcastTo,
    Concat,
    IOKey,
    Multiply,
    Ones,
    Randn,
    Reshape,
    Transpose,
)


def prepare_logical(
    block: ml.models.Model,
    t5: ml.models.Model,
    clip: ml.models.Model,
    num_samples: int,
    height: int,
    width: int,
):
    c = 16
    h = 2 * math.ceil(height / 16)
    w = 2 * math.ceil(width / 16)

    block |= Randn(shape=(num_samples, (h // 2) * (w // 2), c * 2 * 2))(
        output=IOKey("img")
    )

    block |= Ones(shape=(num_samples, h // 2, w // 2, 1))(output="ones")
    block |= Multiply()(left="ones", right=0, output="img_ids_preb")
    block |= Arange(stop=(w // 2))(output="arange_1")
    block |= BroadcastTo(shape=(num_samples, h // 2, w // 2))(
        block.arange_1[None, :, None],  # type: ignore
        output="arange_1_bcast",
    )
    block |= Arange(stop=(h // 2))(output="arange_2")
    block |= BroadcastTo(shape=(num_samples, h // 2, w // 2))(
        block.arange_2[None, None, :],  # type: ignore
        output="arange_2_bcast",
    )
    block |= Concat(axis=-1)(
        input=[
            block.img_ids_preb,  # type: ignore
            block.arange_1_bcast[..., None],  # type: ignore
            block.arange_2_bcast[..., None],  # type: ignore
        ],
        output="img_ids_cat",
    )

    block |= Reshape(shape=(num_samples, -1, 3))(
        block.img_ids_cat,  # type: ignore
        output=IOKey("img_ids"),
    )

    block |= t5(input=IOKey("t5_tokens"), output=IOKey("txt"))
    block |= Ones()(shape=(num_samples, block.txt.shape[1], 3), output="txt_ids_preb")  # type: ignore
    block |= Multiply()(left="txt_ids_preb", right=0, output=IOKey("txt_ids"))

    block |= clip(input=IOKey("clip_tokens"), output=IOKey("y"))


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
    *,
    backend: ml.Backend,
) -> list[float]:
    # extra step for zero
    timesteps = backend.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def unpack_logical(
    model: ml.models.Model, input: ml.models.Connection, height: int, width: int
) -> ml.models.Model:
    b = 1
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)
    ph = 2
    pw = 2

    model |= Reshape(shape=(b, h, w, -1, ph, pw))(input=input, output=IOKey("reshaped"))

    model |= Transpose(axes=(0, 3, 1, 4, 2, 5))(
        input="reshaped", output=IOKey("transposed")
    )

    model |= Reshape(shape=(b, -1, h * ph, w * pw))(
        input="transposed", output=IOKey("result")
    )

    return model.result  # type: ignore


def denoise(
    model: ml.models.PhysicalModel,
    params: dict,
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    # sampling parameters
    timesteps: list[float],
    backend: ml.Backend,
    guidance: float = 4.0,
):
    # this is ignored for schnell
    guidance_vec = backend.ones((img.shape[0],)) * guidance
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=False):
        t_vec = backend.ones((img.shape[0],)) * t_curr
        pred = model.evaluate(
            params,
            {
                "img": img,
                "img_ids": img_ids,
                "txt": txt,
                "txt_ids": txt_ids,
                "y": vec,
                "timesteps": t_vec,
                "guidance": guidance_vec,
            },
        )

        img = img + (t_prev - t_curr) * pred["output"]  # type: ignore[operator]

    return img
