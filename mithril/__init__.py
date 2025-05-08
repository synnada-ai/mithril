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

import builtins
import platform
from collections.abc import Iterable

from .backends.backend import Backend, UnavailableBackend
from .framework.codegen import code_gen_map
from .framework.common import TBD, Tensor
from .framework.logical import Connection, IOKey
from .framework.physical.model import PhysicalConstantType, PhysicalShapeType
from .models import Model, PhysicalModel
from .models.train_model import TrainModel
from .types import (
    Constant,
    DataType,
    bfloat16,
    bool,
    double,
    epsilon_table,
    float,
    float16,
    float32,
    float64,
    int,
    int16,
    int32,
    int64,
    short,
)

__all__ = [
    "JaxBackend",
    "MlxBackend",
    "TorchBackend",
    "CBackend",
    "GGMLBackend",
    "NumpyBackend",
    "compile",
    "DataType",
    "bool",
    "float",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "int",
    "double",
    "int16",
    "int32",
    "int64",
    "short",
    "Backend",
    "IOKey",
    "TBD",
    "Constant",
    "epsilon_table",
    "Tensor",
]

# Load backends
try:
    from .backends.with_autograd.jax_backend.backend import JaxBackend
except ImportError:
    JaxBackend = UnavailableBackend  # type: ignore

try:
    if platform.system() != "Darwin":
        raise ImportError
    from .backends.with_autograd.mlx_backend.backend import MlxBackend
except ImportError:
    MlxBackend = UnavailableBackend  # type: ignore

try:
    from .backends.with_autograd.torch_backend.backend import TorchBackend
except ImportError:
    TorchBackend = UnavailableBackend  # type: ignore

try:
    from .backends.with_manualgrad.c_backend.backend import CBackend
except Exception:
    CBackend = UnavailableBackend  # type: ignore

try:
    from .backends.with_manualgrad.ggml_backend.backend import GGMLBackend
except ImportError:
    GGMLBackend = UnavailableBackend  # type: ignore

try:
    from .backends.with_manualgrad.numpy_backend.backend import NumpyBackend
except ImportError:
    NumpyBackend = UnavailableBackend  # type: ignore


def compile(
    model: Model,
    backend: Backend[DataType],
    *,
    constant_keys: PhysicalConstantType[DataType] | None = None,
    data_keys: Iterable[str | Connection] | None = None,
    discard_keys: Iterable[str | Connection] | None = None,
    trainable_keys: Iterable[str | Connection] | None = None,
    shapes: PhysicalShapeType | None = None,
    inference: builtins.bool = False,
    jit: builtins.bool = True,
    file_path: str | None = None,
    safe_shapes: builtins.bool = True,
    safe_names: builtins.bool = True,
    use_short_namings: builtins.bool = True,
) -> PhysicalModel[DataType]:
    """Compilation of Logical Model.

    Parameters
    ----------
    shapes : Optional[IOShapeType], optional
        _description_, by default None
    static_keys : dict[str: DataType] | None, optional
        _description_, by default None
    discard_keys : set[str] | None, optional
        _description_, by default None
    """

    # TrainModel model requires to be finalized before compilation.
    if isinstance(model, TrainModel):
        model.finalize()

    if model.parent is not None:
        raise ValueError("Model with a parent could not be compiled!")

    # Convert keys to required types.
    constant_keys = constant_keys if constant_keys is not None else dict()
    data_keys = set(data_keys) if data_keys is not None else set()
    discard_keys = set(discard_keys) if discard_keys is not None else set()
    shapes = shapes if shapes is not None else dict()
    trainable_keys = set(trainable_keys) if trainable_keys is not None else set()

    # Initialize Physical Model.
    pm = PhysicalModel[DataType](
        model=model,
        backend=backend,
        data_keys=data_keys,
        constant_keys=constant_keys,
        trainable_keys=trainable_keys,
        discard_keys=discard_keys,
        shapes=shapes,
        inference=inference,
        safe_shapes=safe_shapes,
        safe_names=safe_names,
        use_short_namings=use_short_namings,
        jit=jit,
    )

    # Pick code generator based on backend and generate code.
    CodeGen_Cls = code_gen_map[backend.__class__]
    codegen = CodeGen_Cls(pm)
    codegen.generate_code(file_path=file_path)
    evaluate, evaluate_all = codegen.compile_code(jit=jit)

    pm.generate_functions(evaluate, evaluate_all)
    return pm
