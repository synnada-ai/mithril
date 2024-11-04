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
from collections.abc import Mapping, Sequence

from .backends.backend import Backend, UnavailableBackend
from .core import (
    DataType,
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
from .framework.codegen import code_gen_map
from .framework.common import TBD, Connect, Connection, Constant, IOKey, MainValueType
from .models import BaseModel, Model, PhysicalModel
from .models.train_model import TrainModel

__all__ = [
    "BaseModel",
    "Model",
    "PhysicalModel",
    "JaxBackend",
    "MlxBackend",
    "TorchBackend",
    "CBackend",
    "NumpyBackend",
    "compile",
    "DataType",
    "bool",
    "float",
    "float16",
    "float32",
    "float64",
    "int",
    "double",
    "int16",
    "int32",
    "int64",
    "short",
    "TrainModel",
    "IOKey",
    "TBD",
    "Connect",
    "Constant",
    "epsilon_table",
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
    from .backends.with_manualgrad.numpy_backend.backend import NumpyBackend
except ImportError:
    NumpyBackend = UnavailableBackend  # type: ignore


def compile(
    model: BaseModel,
    backend: Backend[DataType],
    inference: builtins.bool = False,
    static_keys: Mapping[str, DataType | MainValueType] | None = None,
    discard_keys: set[str] | tuple[str] | list[str] | str | None = None,
    jacobian_keys: set[str] | None = None,
    shapes: Mapping[str, Sequence[builtins.int | None]]
    | Mapping[Connection, Sequence[builtins.int | None]]
    | Mapping[str | Connection, Sequence[builtins.int | None]]
    | None = None,
    jit: builtins.bool = True,
    file_path: str | None = None,
    safe: builtins.bool = True,
    safe_shapes: builtins.bool | None = None,
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

    if jit and not model.jittable:
        raise Exception("Model is not jittable. Can only be compiled with jit = False.")
    # TrainModel model requires to be finalized before compilation.
    if safe_shapes is None:
        safe_shapes = safe
    if isinstance(model, TrainModel):
        model._finalize()

    # Generate Physical Model.
    if not isinstance(model, BaseModel):
        raise Exception("Unsupported model type!")
    if model.parent is not None:
        raise ValueError("Model with a parent could not be compiled!")

    if discard_keys is None:
        discard_keys = set()
    elif isinstance(discard_keys, tuple | list):
        discard_keys = set(discard_keys)
    elif isinstance(discard_keys, str):
        discard_keys = set([discard_keys])

    assert isinstance(discard_keys, set), (
        f"Expected discard_keys to be of type 'set', but got type "
        f"'{type(discard_keys).__name__}' instead."
    )

    pm = PhysicalModel[DataType](
        model=model, inference=inference, backend=backend, safe_shapes=safe_shapes
    )

    if static_keys is None:
        static_keys = {}

    # If given keys are logical internal keys, convert them to physical keys.
    static_keys = {
        pm.key_mappings.get(key, key): value for key, value in static_keys.items()
    }
    shapes = (
        {pm.key_mappings.get(key, key): value for key, value in shapes.items()}
        if shapes is not None
        else None
    )

    # First part of the pm with all the inferences.
    pm.pre_compile(
        static_keys=static_keys,
        discard_keys=discard_keys,
        jacobian_keys=jacobian_keys,
        shapes=shapes,
    )
    if safe:
        # statics = static_keys | model.non_differentiables
        # TODO: check if we need to find model.non_differentiables
        output_set = set(model.conns.output_keys)
        statics = static_keys.keys() | {
            key
            for key, con in model.conns.all.items()
            if con.metadata.data.is_non_diff and key not in output_set
        }
        if (model._canonical_input.key not in statics) and ("input" not in statics):
            raise KeyError(
                "Requires model's canonical input key to be a static key! You can set "
                "False to safe flag to use trainable canonical key"
            )

        for key in pm._input_keys:
            if key not in statics and "target" in key:
                raise KeyError(
                    f"Requires '{key}' key to be a static key! You can set False to "
                    f"safe flag to use trainable '{key}' key!"
                )

    # if model is not context:
    #     context.insert_geomean_values(pm, backend)
    if jit and file_path is not None:
        # TODO Fix warning
        raise RuntimeError(
            "Cannot create file while 'jit' flag is true. To write model file set "
            "'jit' flag to 'False'"
        )

    CodeGen_Cls = code_gen_map[backend.__class__]
    codegen = CodeGen_Cls(pm)
    codegen.generate_code(file_path=file_path)
    evaluate, evalute_grad, evaluate_all = codegen.compile_code(jit=jit)

    pm.generate_functions(evaluate, evalute_grad, evaluate_all)
    return pm
