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

import mithril as ml

installed_backends: list[type[ml.Backend]] = [ml.TorchBackend, ml.JaxBackend, ml.NumpyBackend]

if platform.system() == "Darwin":
    installed_backends.append(ml.MlxBackend)


prompt = "translate English to German: I am not in danger, I am the danger."
expected_response = 'Prompt: translate English to German: I am not in danger, I am the danger.\nGenerated text: Ich bin nicht in Gefahr, ich bin die Gefahr.\n'



class TestT5:
    class RunType(Protocol):
        def __call__(
            self,
            prompt: str,
            backend: ml.Backend,
        ) -> None: ...

    @pytest.fixture(scope="class")
    def run_fn(self) -> RunType:
        file_path = os.path.join("examples", "gpt", "run_sample.py")
        sys.path.append(os.path.dirname(file_path))
        from examples import t5

        return t5.run

    @pytest.mark.parametrize("backend", installed_backends)
    def test_run_sample(self, backend: type[ml.Backend], run_fn: RunType):
        with redirect_stdout(StringIO()) as prompt_output:
            run_fn(
                "translate English to German: I am not in danger, I am the danger.",
                backend()
            )
        output = prompt_output.getvalue()
        
        assert output == expected_response
