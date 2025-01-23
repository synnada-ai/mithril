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
import platform
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Protocol

import pytest

from mithril import JaxBackend, MlxBackend, NumpyBackend, TorchBackend

AllBackendsType = (
    type[TorchBackend] | type[JaxBackend] | type[MlxBackend] | type[NumpyBackend]
)

installed_backends: list[AllBackendsType] = []
try:
    import torch  # noqa F401

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1337)

    installed_backends.append(TorchBackend)
except ImportError:
    pass

try:
    import jax  # noqa F401
    import jax.numpy as jnp  # noqa F401

    installed_backends.append(JaxBackend)
except ImportError:
    pass

try:
    import numpy  # noqa F401

    installed_backends.append(NumpyBackend)
except ImportError:
    pass

try:
    import mlx.core as mx  # noqa F401

    if platform.system() != "Darwin" or os.environ.get("CI") == "true":
        raise ImportError
    installed_backends.append(MlxBackend)

except ImportError:
    pass

backend_strings: list[str] = [backend.backend_type for backend in installed_backends]

backend_tokens = {"torch": 30, "jax": 25, "numpy": 22, "mlx": 30}

result_prompts = {
    "torch": (
        " The answer is a little more complicated than that, because,"
        " in the book, Platonism states that all creation is made by God "
    ),
    "jax": ("How does that matter? How does that matter matter that much?"),
    "numpy": (
        "That's the question you should ask yourself when you're "
        "designing your own homes. Maybe you want to create something"
    ),
    "mlx": ("There is light, there is temperature, and there is time."),
}

if os.environ.get("CI") == "true":
    result_prompts["torch"] = (
        " It's easy to say that the answer to life is in heaven. "
        "There is no heaven, nor there is there any universe to be found."
    )

    result_prompts["jax"] = (
        "As my own students have noticed,"
        " religion seems to be the only way in which you can"
    )


class TestGPT:
    class RunSampleType(Protocol):
        def __call__(
            self,
            backend: str,
            start: str = "\n",
            num_samples: int = 10,
            max_new_tokens: int = 500,
            top_k: int = 200,
            seed: int = 1337,
            temperature: float = 0.8,
        ) -> None: ...

    @pytest.fixture(scope="class")
    def run_sample_fn(self) -> RunSampleType:
        file_path = os.path.join("examples", "gpt", "run_sample.py")
        sys.path.append(os.path.dirname(file_path))
        import examples.gpt.run_sample as run_sample

        return run_sample.run_sample

    @pytest.mark.parametrize("backend", backend_strings)
    def test_run_sample(self, backend: str, run_sample_fn: RunSampleType):
        with redirect_stdout(StringIO()) as prompt_output:
            run_sample_fn(
                backend,
                start="What is the answer to life, the universe?",
                max_new_tokens=backend_tokens[backend],
                num_samples=1,
            )
        output = prompt_output.getvalue()
        assert result_prompts[backend] in output
