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


import os
from dataclasses import dataclass

import numpy as np
from PIL import Image
from sampling import denoise, get_noise, get_schedule, prepare, rearrange, unpack
from t5 import download_t5_encoder_weights, load_t5_encoder, load_t5_tokenizer
from util import configs, load_clip, load_decoder, load_flow_model

import mithril as ml

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None = None


def run(
    model_name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    prompt: str = "A girl with green eyes on the a wooden bridge",
    device: str = "cuda",
    output_dir: str = "temp",
    num_steps: int | None = None,
    guidance: float = 3.5,
    seed: int = 42,
):
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(
            f"Got unknown model name: {model_name}, chose from {available}"
        )

    if num_steps is None:
        num_steps = 4 if model_name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    backend = ml.TorchBackend(device="cuda", dtype=ml.bfloat16)
    backend.seed = seed

    t5 = load_t5_encoder(backend)
    t5_tokenizer = load_t5_tokenizer(backend, pad=False)
    t5_np_weights = download_t5_encoder_weights(backend)
    t5_weights = {key: backend.array(value) for key, value in t5_np_weights.items()}

    clip = load_clip(device=device).to("cuda")

    flow_model, flow_params = load_flow_model(model_name, backend=backend)
    decoder, decoder_params = load_decoder(model_name, backend=backend)

    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    noise = get_noise(1, opts.height, opts.width, backend)
    inp = prepare(
        t5, t5_weights, t5_tokenizer, clip, noise, prompt=opts.prompt, backend=backend
    )

    timesteps = get_schedule(
        opts.num_steps,
        inp["img"].shape[1],
        shift=(model_name != "flux-schnell"),
        backend=backend,
    )

    for key, value in inp.items():
        inp[key] = backend.array(np.array(value.to("cpu").float()))

    x = denoise(flow_model, flow_params, **inp, timesteps=timesteps, backend=backend)
    x = unpack(x, opts.height, opts.width)
    x = decoder(decoder_params, {"input": x.type(backend.float32)})["output"]  # type: ignore
    x = x.clamp(-1, 1)  # TODO: add to backend
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray(np.array(127.5 * (x.cpu() + 1.0)).astype(np.uint8))
    img.save("qwe123.png")


if __name__ == "__main__":
    run()
