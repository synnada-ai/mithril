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

import ast
from collections.abc import Callable

from ...backends.with_autograd.torch_backend import TorchBackend
from ..logical import PrimitiveModel
from ..physical.model import PhysicalModel
from .python_gen import PythonCodeGen


class TorchCodeGen(PythonCodeGen):
    def __init__(self, pm: PhysicalModel) -> None:
        super().__init__(pm)
        self.is_parallel_defined = False

        assert isinstance(self.pm.backend, TorchBackend)
        self.backend: TorchBackend = self.pm.backend

    def call_primitive(
        self,
        model: PrimitiveModel,
        fn: Callable,
        l_input_keys: list[str],
        g_input_keys: list[str],
        output_key: str,
        formula_key: str,
    ):
        generated_fn, used_keys = self.create_primitive_call(
            fn, l_input_keys, g_input_keys
        )
        targets, _used_keys = self.create_primitive_call_targets(
            output_key, model, self.pm.inference
        )

        if formula_key in self.backend.array_creation_funcs:
            self.add_partial_function(formula_key)

        if (
            formula_key in self.backend.array_creation_funcs
            and self.backend._raw_device_mesh is not None
        ):
            # Import device mesh and create base device mesh only once!
            if not self.is_parallel_defined:
                parallel = ast.ImportFrom(
                    module="mithril.backends.with_autograd.torch_backend.parallel",
                    names=[ast.alias(name="TorchParallel", asname="Parallel")],
                    level=0,
                )

                base_device_mesh = ast.Subscript(
                    ast.Attribute(
                        value=ast.Name(id="Parallel", ctx=ast.Load()),
                        attr="device_meshes",
                        ctx=ast.Load(),
                    ),
                    slice=ast.Constant(value=self.backend._raw_device_mesh),
                    ctx=ast.Load(),
                )
                base_device_mesh_assgn = ast.Assign(
                    targets=[ast.Name(id="device_mesh", ctx=ast.Store())],
                    value=base_device_mesh,
                )

                self.imports.append(parallel)
                self.globals.append(base_device_mesh_assgn)
                self.is_parallel_defined = True

            generated_fn = ast.Call(
                func=ast.Name(id="to_parallel", ctx=ast.Load()),
                args=[generated_fn, ast.Name(id="device_mesh", ctx=ast.Load())],
                keywords=[],
            )

        return ast.Assign(targets, generated_fn), used_keys | _used_keys  # type: ignore
