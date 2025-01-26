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

result_prompt = "They went up by the stair because they thought this would make a better impression. They stood in a row in front of Mrs. Darling with their hats off and wishing they were not wearing their pirate clothes. They said nothing but their eyes asked her to have them."


class TestGPT:
    class RunSampleType(Protocol):
        def __call__(
            self,
            backend: str,
            file_path: str,
        ) -> None: ...

    @pytest.fixture(scope="class")
    def run_sample_fn(self) -> RunSampleType:
        file_path = os.path.join("examples", "whisper", "run_whisper.py")
        sys.path.append(os.path.dirname(file_path))
        import examples.whisper.run_whisper as run_whisper

        return run_whisper.run_inference

    @pytest.mark.parametrize("backend", backend_strings)
    def test_run_sample(self, backend: str, run_sample_fn: RunSampleType):
        with redirect_stdout(StringIO()) as prompt_output:
            run_sample_fn(
                file_path="examples/whisper/1040-133433-0001.flac", backend=backend
            )
        output = prompt_output.getvalue()
        assert result_prompt in output
