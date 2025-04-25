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
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from clip import download_clip_encoder_weights, load_clip_encoder, load_clip_tokenizer
from PIL import Image
from sampling import (
    get_schedule,
    prepare_logical,
    unpack_logical,
)
from t5 import download_t5_encoder_weights, load_t5_encoder, load_t5_tokenizer
from tqdm import tqdm
from util import configs, load_decoder, load_flow_model

import mithril as ml
from mithril.models import (
    IOKey,
    Model,
    Multiply,
    Ones,
    Transpose,
)


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None = None


def run(
    model_name: str = "flux-dev",
    backend_name: str = "torch",
    width: int = 1024,
    height: int = 1024,
    prompt: str = "A girl with green eyes on the a wooden bridge",
    device: str = "cpu",
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
        num_steps = 4 if model_name == "flux-schnell" else 28

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

    backend_cls = ml.TorchBackend if backend_name == "torch" else ml.JaxBackend
    backend: ml.Backend = backend_cls(device=device, dtype=ml.bfloat16)
    backend.seed = seed

    # Create models

    flux_pipeline = Model()

    print("Loading T5 encoder")
    t5_lm = load_t5_encoder(
        name=model_name, max_len=256 if "schnell" in model_name else 512
    )
    t5_tokenizer = load_t5_tokenizer(backend, name=model_name)
    t5_weights = download_t5_encoder_weights(backend, name=model_name)
    t5_lm.name = "t5"

    print("Loading CLIP encoder")
    clip_lm = load_clip_encoder(name=model_name)
    clip_tokenizer = load_clip_tokenizer(backend, name=model_name)
    clip_weights = download_clip_encoder_weights(backend, name=model_name)
    clip_lm.name = "clip"

    prepare_logical(flux_pipeline, t5_lm, clip_lm, 1, opts.width, opts.height)

    decoder_lm, decoder_params = load_decoder(model_name, backend=backend)
    decoder_lm.name = "decoder"

    timesteps = get_schedule(
        opts.num_steps,
        flux_pipeline.shapes["img"][1],  # type: ignore
        shift=(model_name != "flux-schnell"),
        backend=backend,
    )

    # Build denoise pipeline
    print("Building denoise pipeline")

    flow_lm, flow_params = load_flow_model(model_name, backend=backend)

    img = IOKey("img")
    if "schnell" not in model_name:
        flux_pipeline |= Ones(shape=(1,)).connect(output="guidance_vec")
        flux_pipeline |= Multiply().connect(
            left="guidance_vec", right=guidance, output="guidance"
        )

    kwargs: dict = {}

    _flow_model: ml.models.Model
    for idx, (t_curr, t_prev) in tqdm(
        enumerate(zip(timesteps[:-1], timesteps[1:], strict=False))
    ):
        flux_pipeline |= Ones(shape=(1,)).connect(output=f"t_vec{idx}")
        t_vec = getattr(flux_pipeline, f"t_vec{idx}")
        t_vec *= t_curr

        flow_out = f"output_{idx}"
        if idx != 0:
            kwargs = {
                key: value
                for key, value in zip(
                    _flow_model.input_keys,  # noqa F821
                    [key for key in _flow_model.conns.input_connections],  # noqa F821
                    strict=False,
                )
                if "$" in key
            }

        _flow_model = deepcopy(flow_lm)
        kwargs |= {
            "img": img,
            "txt": "txt",
            "img_ids": "img_ids",
            "txt_ids": "txt_ids",
            "timesteps": t_vec,
            "y": "y",
            "guidance": "guidance",
            "output": flow_out,
        }
        flux_pipeline |= _flow_model.connect(**kwargs)

        img = img + (t_prev - t_curr) * getattr(flux_pipeline, flow_out)

    unpacked_img = unpack_logical(flux_pipeline, img, opts.height, opts.width)

    flux_pipeline |= decoder_lm(input=unpacked_img, output="decoded")
    flux_pipeline |= Transpose(axes=(0, 2, 3, 1)).connect(
        input="decoded", output=IOKey("output")
    )

    # Sanitize param names
    flow_params = {f"model_0_{key}": value for key, value in flow_params.items()}
    decoder_params = {f"decoder_{key}": value for key, value in decoder_params.items()}
    t5_params = {f"t5_{key}": value for key, value in t5_weights.items()}
    clip_params = {f"clip_{key}": value for key, value in clip_weights.items()}

    params = {**flow_params, **decoder_params, **t5_params, **clip_params}

    print("Compiling Pipeline")
    s_time = time.perf_counter()
    denoise_pm = ml.compile(
        flux_pipeline, backend, inference=True, jit=True, use_short_namings=False
    )
    e_time = time.perf_counter()
    print(f"Time taken for compilation: {e_time - s_time} seconds")

    clip_inp = clip_tokenizer.encode(opts.prompt)
    t5_inp = t5_tokenizer.encode(opts.prompt)

    inp = {"clip_tokens": clip_inp, "t5_tokens": t5_inp}

    # Warmup
    for _ in tqdm(range(5)):
        x = denoise_pm.evaluate(params, inp)["output"]

    # Actual inference
    s_time = time.perf_counter()
    for _ in tqdm(range(10)):
        x = denoise_pm.evaluate(params, inp)["output"]
    e_time = time.perf_counter()
    print(f"Time taken: {e_time - s_time} seconds")

    img_pil = Image.fromarray(
        np.array(127.5 * (x.float().cpu()[0] + 1.0)).clip(0, 255).astype(np.uint8)  # type: ignore
    )
    img_pil.save("img.png")


if __name__ == "__main__":
    run()
