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

import ctypes
import os
import subprocess
import tempfile
from functools import partial

from ...backends.with_manualgrad.c_backend import CBackend, backend
from ...backends.with_manualgrad.c_backend.src import array
from ...backends.with_manualgrad.c_backend.src.array import Array, PyArray
from ...framework.common import Tensor
from ...utils.type_utils import is_list_int
from ..physical.model import PhysicalModel
from . import c_ast
from .code_gen import CodeGen

FinalCost = "final_cost"


class CGen(CodeGen):
    BACKWARD_FN_SUFFIX = "_grad"

    def __init__(self, pm: PhysicalModel) -> None:
        super().__init__(pm)

        assert isinstance(self.pm.backend, CBackend)
        self.backend: CBackend = self.pm.backend

        self.imports: list[c_ast.AST] = []
        self.globals: list[c_ast.AST] = []
        self.functions: list[c_ast.AST] = []

        # This will be used to store the keys of the argument of the functions
        self.func_arg_keys: dict[str, list[str]] = {}

    def generate_imports(self) -> list[c_ast.Include]:
        header_path = os.path.join(self._get_backend_path(), "src", "cbackend.h")
        return [c_ast.Include(header_path, system=False)]

    def generate_code(self, file_path: str | None = None):
        self.file_path = file_path

        self.imports = self.generate_imports()  # type: ignore
        eval_fn, eval_used_keys = self.generate_evaluate()
        self.functions.append(eval_fn)
        self.func_arg_keys["evaluate"] = sorted(eval_used_keys)
        if not self.pm.inference:
            eval_grad_fn, eval_grad_used_keys = self.generate_evaluate_gradients()
            self.functions.append(eval_grad_fn)
            self.func_arg_keys["evaluate_gradients"] = sorted(eval_grad_used_keys)

        generated_code = c_ast.FILE(self.imports, self.globals, self.functions).to_str()  # type: ignore

        if file_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".c") as tmp_file:
                self.file_path = tmp_file.name
        else:
            self.file_path = file_path

        with open(self.file_path, "w") as f:
            f.write(generated_code)

        self.code = generated_code

    def compile_code(self, jit: bool = False, compile_flags: list[str] | None = None):
        assert not jit, "JIT is not yet supported for CBackend"
        assert self.file_path is not None, "Code has not been generated yet!"

        eval_arg_keys = self.func_arg_keys["evaluate"]
        if not self.pm.inference:
            eval_grad_arg_keys = self.func_arg_keys["evaluate_gradients"]
        so_file_path = self.file_path.replace(".c", ".so")

        default_compile_flags = ["cc", self.file_path, "-shared", "-fPIC"]
        if compile_flags:
            default_compile_flags = compile_flags

        subprocess.check_output(
            [
                *default_compile_flags,
                f"-L{self._get_backend_path()}/src",
                "-lmithrilc",
                f"-Wl,-rpath,{self._get_backend_path()}/src",
                "-o",
                so_file_path,
            ]
        )

        if so_file_path[0] != "/":
            so_file_path = "./" + so_file_path

        # We need backend subtype
        lib = ctypes.CDLL(so_file_path)
        lib.evaluate.argtypes = [ctypes.POINTER(Array)] * len(eval_arg_keys)
        if not self.pm.inference:
            lib.evaluate_gradients.argtypes = [ctypes.POINTER(Array)] * len(
                eval_grad_arg_keys
            )

        # we need backend data types!
        # include_internals flag is used for get internal values for backpropagation
        def evaluate_wrapper(
            params: dict[str, PyArray],
            data: dict[str, PyArray],
            cache: dict[str, PyArray] | None = None,
            include_internals: bool = False,
        ) -> dict[str, PyArray]:
            inputs: dict[str, PyArray] = {}
            if isinstance(params, dict):
                inputs |= params
            if isinstance(data, dict):
                inputs |= data
            if isinstance(cache, dict):
                inputs |= cache

            # Allocate output arrays
            for arg_key in eval_arg_keys:
                if arg_key in inputs:
                    continue

                arr_shape = self._get_array_shape(arg_key)
                inputs[arg_key] = self.backend.empty(*arr_shape)

            inputs_ordered = [inputs[arg].arr for arg in eval_arg_keys]
            lib.evaluate(*inputs_ordered)

            if not include_internals:
                return {key: inputs[key] for key in self.pm.output_keys}
            else:
                return inputs

        def evaluate_gradients_wrapper(
            params: dict[str, PyArray],
            data: dict[str, PyArray] | None = None,
            output_gradients: dict[str, PyArray] | None = None,
            include_output: bool = False,
        ):
            if data is None:
                data = {}

            if output_gradients is None and FinalCost not in self.pm._output_keys:
                raise ValueError(
                    "Requires output gradients if final loss is not attached!"
                )
            elif output_gradients is None:
                output_gradients = {FinalCost: array.ones((1,))}

            gradients = {key: value for key, value in output_gradients.items()}
            forward_pass = evaluate_wrapper(
                params=params, data=data, cache={}, include_internals=True
            )

            # Create gradients for all params
            for key in (
                self.pm._flat_graph.all_source_keys
                - self.pm.data_store.all_static_keys
                - self.pm.data_store.unused_keys
                - self.pm.ignore_grad_keys
            ):
                # In CBackend we are creating all internal gradients with zeros.
                if key not in gradients:
                    arr_shape = self._get_array_shape(key)
                    gradients[key] = self.backend.zeros(*arr_shape)

            gradients = {key + "_grad": value for key, value in gradients.items()}

            inputs = params | data | gradients | forward_pass

            inputs_ordered = [inputs[arg].arr for arg in sorted(inputs.keys())]
            lib.evaluate_gradients(*inputs_ordered)

            return {key: inputs[key + "_grad"] for key in params}

        return (
            evaluate_wrapper,
            evaluate_gradients_wrapper,
            partial(evaluate_gradients_wrapper, include_output=True),
        )

    def create_primitive_call(self, formula_name: str, args: list[str]) -> c_ast.Expr:
        return c_ast.Call(formula_name, args)

    def generate_evaluate(self) -> tuple[c_ast.FunctionDef, set[str]]:
        fn_body = []
        used_keys = set()

        unused_keys = self.pm.data_store.unused_keys
        cached_data_keys = self.pm.data_store.cached_data.keys()

        for output_key in self.pm._flat_graph.topological_order:
            # Staticly infered and unused model will not be added
            if output_key in (cached_data_keys | unused_keys):
                continue

            model = self.pm._flat_graph.get_model(output_key)
            inputs = self.pm._flat_graph.get_source_keys(output_key)

            assert model is not None

            # In C backend we need to pass output array as first argument
            inputs = [output_key] + inputs

            # Create primitive call
            p_call = self.create_primitive_call(model.formula_key, inputs)
            fn_body.append(p_call)

            used_keys.add(output_key)
            used_keys |= set(inputs)

        arguments = []
        for used_key in sorted(used_keys):
            arguments.append(c_ast.Parameter("Array *", used_key))

        evaluate_fn = c_ast.FunctionDef("void", "evaluate", arguments, fn_body)

        return evaluate_fn, used_keys

    def generate_evaluate_gradients(self):
        fn_body = []
        used_keys = set()

        all_ignored_keys = (
            self.pm.ignore_grad_keys
            | self.pm.data_store.all_static_keys
            | self.pm.data_store.unused_keys
        )
        all_ignored_keys, _ = self.pm.infer_ignore(
            set(), self.pm._output_keys, all_ignored_keys, update_graph=False
        )

        for output_key in reversed(self.pm._flat_graph.topological_order):
            # Staticly infered and unused model will not be added
            if output_key in all_ignored_keys:
                continue

            model = self.pm._flat_graph.get_model(output_key)
            inputs = self.pm._flat_graph.get_source_keys(output_key)

            assert model is not None

            # Assume all inputs are Array
            grad_inputs = [input_key + "_grad" for input_key in inputs]
            for idx in range(len(grad_inputs)):
                fn_inputs = (
                    [output_key + "_grad", c_ast.Constant(idx), output_key]
                    + inputs
                    + grad_inputs
                )

                # Create primitive call
                p_call = self.create_primitive_call(
                    model.formula_key + self.BACKWARD_FN_SUFFIX, fn_inputs
                )
                fn_body.append(p_call)

            used_keys.add(output_key)
            used_keys.add(output_key + "_grad")
            used_keys |= set(inputs)
            used_keys |= set(grad_inputs)

        arguments = []

        for used_key in sorted(used_keys):
            arguments.append(c_ast.Parameter("Array *", used_key))

        evaluate_grad_fn = c_ast.FunctionDef(
            "void", "evaluate_gradients", arguments, fn_body
        )

        return evaluate_grad_fn, used_keys

    def _get_backend_path(self):
        backend_path = backend.__file__
        return backend_path[: backend_path.rindex("/")]

    def _get_array_shape(self, key: str) -> tuple[int, ...]:
        array_data = self.pm.data[key]
        assert isinstance(array_data, Tensor)
        shape = array_data.shape.reprs[0].get_shapes()

        if is_list_int(shape):
            return tuple(shape)
        else:
            raise ValueError(f"Unexpected shape: {shape}")
