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


from ...backends.backend import Backend
from .code_gen import CodeGen
from .python_gen import PythonCodeGen

code_gen_map: dict[type[Backend], type[CodeGen]] = {}

try:
    from ...backends.with_autograd.jax_backend import JaxBackend

    code_gen_map[JaxBackend] = PythonCodeGen
except ImportError:
    pass
try:
    from ...backends.with_autograd.mlx_backend import MlxBackend

    code_gen_map[MlxBackend] = PythonCodeGen
except ImportError:
    pass
try:
    from ...backends.with_autograd.torch_backend import TorchBackend
    from .torch_gen import TorchCodeGen

    code_gen_map[TorchBackend] = TorchCodeGen
except ImportError:
    pass
try:
    from ...backends.with_manualgrad.c_backend import CBackend
    from .c_gen import CGen

    code_gen_map[CBackend] = CGen
except Exception:  # type: ignore
    pass
try:
    from ...backends.with_manualgrad.numpy_backend import NumpyBackend
    from .numpy_gen import NumpyCodeGen

    code_gen_map[NumpyBackend] = NumpyCodeGen
except ImportError:
    pass


__all__ = [
    "CodeGen",
    "code_gen_map",
    "CGen",
    "PythonCodeGen",
    "NumpyCodeGen",
    "TorchCodeGen",
]
