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
from conditioner import HFEmbedder
from einops import rearrange, repeat

import mithril as ml


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    backend: ml.Backend,
):
    return backend.randn(
        num_samples,
        16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=ml.bfloat16,
    )


def prepare(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: torch.Tensor,
    prompt: str | list[str],
    backend: ml.Backend,
) -> dict[str, torch.Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    # Rearrange "b c (h ph) (w pw) -> b (h w) (c ph pw)" ph=2, pw=2
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = backend.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + backend.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + backend.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]

    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = backend.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }


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


def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


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
