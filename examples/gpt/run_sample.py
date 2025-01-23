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

import argparse
import sys
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import tiktoken
from model import create_gpt
from transformers import GPT2LMHeadModel

import mithril as ml
from mithril import Backend
from mithril.models import PhysicalModel

warnings.filterwarnings("ignore")

backend_map: dict[str, type[Backend]] = {
    "torch": ml.TorchBackend,
    "jax": ml.JaxBackend,
    "numpy": ml.NumpyBackend,
    "mlx": ml.MlxBackend,
}


def run_sample(
    backend: str,
    start: str = "\n",
    num_samples: int = 10,
    max_new_tokens: int = 500,
    top_k: int = 200,
    seed: int = 42,
    temperature: float = 0.8,
):
    # TODO: This recursion limit is minimum we can have for now.
    # We may further improve this limit in the future.
    sys.setrecursionlimit(692)
    # Model Configuration
    block_size = 100
    gpt = create_gpt(
        block_size=block_size,
        vocab_size=50304,
        num_layers=12,
        num_heads=12,
        dims=768,
        bias=True,
    )

    # Create backend.
    backend_obj = backend_map[backend](device="cpu")
    # Set seed.
    backend_obj.set_seed(seed)
    # Compile gpt model.
    compiled_model = ml.compile(
        gpt, backend_obj, data_keys={"input"}, jit=False, use_short_namings=False
    )

    # Get weights in corresponding backend array type.
    trainables = get_weights(backend_obj)

    # Get gpt's default encoder decoder
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})  # noqa: E731
    decode = lambda tokens: enc.decode(tokens)  # noqa: E731

    # Encode input
    start_ids = encode(start)
    x = backend_obj.array(start_ids, dtype=ml.int64)[None, ...]

    # Run generation
    for _ in range(num_samples):
        generate(
            compiled_model,
            block_size,
            trainables,
            x,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            decode=decode,
        )
        print("\n---------------")


def get_weights(backend: Backend):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()

    # The OpenAI checkpoint use Conv1D instead of Linear for projection layers
    # We need to transpose the weights to match the shape of the OpenAI checkpoint
    transposed = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]
    params = {}
    for key in sd_hf:
        ml_key = key.replace(".", "_")
        if any(key.endswith(w) for w in transposed):
            params[ml_key] = backend.array(np.array(sd_hf[key].T))
        else:
            params[ml_key] = backend.array(np.array(sd_hf[key]))

    return params


def generate(
    model: PhysicalModel[Any],
    block_size: int,
    weights: dict[str, ml.DataType],
    idx: ml.DataType,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    decode: Callable | None = None,
):
    for _ in range(max_new_tokens):
        # If the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.shape[1] <= block_size else idx[:, -block_size:]
        # Forward the model to get the logits for the index in the sequence
        outputs = model.evaluate(weights, data={"input": idx_cond})
        logits = outputs["output"]
        # Pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature  # type: ignore
        # Optionally crop the logits to only the top k options
        if top_k is not None:
            v = model.backend.topk(logits, min(top_k, logits.shape[-1]))  # type: ignore
            logits = model.backend.where(
                logits < v[:, [-1]], -model.backend.inf, logits
            )
        # Apply softmax to convert logits to (normalized) probabilities
        probs = model.backend.softmax(logits, dim=-1)
        # Sample from the distribution
        idx_next = model.backend.multinomial(probs, num_samples=1)
        # Append sampled index to the running sequence and continue
        idx = model.backend.cat([idx, idx_next], axis=1)
        assert decode is not None, "decode function must be provided"
        decoded_result = decode(idx_next[0].tolist())
        print(decoded_result, end="", flush=False)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--start", default="What is the answer to life, the universe, and everything?"
    )
    ap.add_argument("--num_samples", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--backend", type=str, default="torch")
    args = ap.parse_args()
    run_args = {k: v for k, v in vars(args).items() if k != "save"}
    run_sample(**run_args)


if __name__ == "__main__":
    main()
