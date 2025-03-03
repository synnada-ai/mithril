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
from ...backends.with_manualgrad.ggml_backend import GGMLBackend
from ...common import CGenConfig
from ...cores.c.array import PyArray
from ...cores.c.raw_c import array
from ...framework.common import (
    EvaluateAllType,
    EvaluateGradientsType,
    EvaluateType,
)
from ...utils.type_utils import is_list_int
from ..physical.model import PhysicalModel
from . import c_ast
from .code_gen import CodeGen

FinalCost = "final_cost"


class CGen(CodeGen[PyArray]):
    BACKWARD_FN_SUFFIX = "_grad"
    EVALUATE_INPUT_STRUCT_NAME = "eval_inputs"
    EVALUATE_GRAD_INPUT_STRUCT_NAME = "eval_grad_inputs"
    EVALUATE_OUTPUT_STRUCT_NAME = "eval_outputs"
    EVALUATE_GRAD_OUTPUT_STRUCT_NAME = "eval_grad_outputs"
    CACHE_STRUCT_NAME = "cache_keys"
    GRAD_STRUCT_NAME = "grad_keys"
    CACHE_NAME = "cache"

    dynamic_links: list[str] = []

    def __init__(self, pm: PhysicalModel[PyArray]) -> None:
        super().__init__(pm)

        assert isinstance(
            self.pm.backend, CBackend | GGMLBackend
        ), f"Invalid backend '{self.pm.backend.backend_type}'! Must be CBackend or GGMLBackend"

        self.backend: CBackend | GGMLBackend = self.pm.backend

        self.imports: list[c_ast.AST] = []
        self.globals: list[c_ast.AST] = []
        self.functions: list[c_ast.AST] = []

        # This will be used to store the keys of the argument of the functions
        self.func_arg_keys: dict[str, list[str]] = {}
        self.configs: CGenConfig = self.backend.CODEGEN_CONFIG

        # Ignored grad keys
        self.ignored_grad_keys: set[str] = self._infer_ignored_grad_keys()

        # Determine struct keys
        self.determined_struct_keys: dict[str, list[str]] = (
            self._determine_struct_keys()
        )

    def generate_code(self, file_path: str | None = None) -> None:
        self.file_path = file_path

        self.imports += self.generate_imports()

        # Functions
        eval_fn, eval_used_keys = self.generate_evaluate()
        self.functions.append(eval_fn)
        self.func_arg_keys["evaluate"] = sorted(self.pm.input_keys)

        if not self.pm.inference:
            eval_grad_fn, eval_grad_used_keys = self.generate_evaluate_gradients()
            self.functions.append(eval_grad_fn)
            self.func_arg_keys["evaluate_gradients"] = sorted(eval_grad_used_keys)

        # Structs
        self.generate_structs()

        # Init cache struct
        cache_struct = c_ast.StructInit(
            f"{self.CACHE_STRUCT_NAME} {self.CACHE_NAME}",
            {key: "NULL" for key in self.determined_struct_keys["eval_cache_keys"]},
            static=True,
        )
        self.globals.append(cache_struct)

        # Init grad struct
        grad_struct = c_ast.StructInit(
            f"{self.EVALUATE_GRAD_OUTPUT_STRUCT_NAME} {self.GRAD_STRUCT_NAME}",
            {
                key: "NULL"
                for key in self.determined_struct_keys["eval_grad_output_keys"]
            },
            static=True,
        )
        self.globals.append(grad_struct)

        generated_code = c_ast.FILE(self.imports, self.globals, self.functions).to_str()  # type: ignore

        if file_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".c") as tmp_file:
                self.file_path = tmp_file.name
        else:
            self.file_path = file_path

        with open(self.file_path, "w") as f:
            f.write(generated_code)

        self.code = generated_code

    def compile_code(
        self, jit: bool = False, compile_flags: list[str] | None = None
    ) -> tuple[
        EvaluateType[PyArray],
        EvaluateGradientsType[PyArray] | None,
        EvaluateAllType[PyArray] | None,
    ]:
        assert not jit, "JIT is not yet supported for CBackend"
        assert self.file_path is not None, "Code has not been generated yet!"

        so_file_path = self.file_path.replace(".c", ".so")

        default_compile_flags = ["cc", self.file_path, "-shared", "-fPIC"]
        if compile_flags:
            default_compile_flags = compile_flags

        subprocess.check_output(
            [
                *default_compile_flags,
                f"-L{self.backend.SRC_PATH}",
                *self.dynamic_links,
                f"-Wl,-rpath,{self.backend.SRC_PATH}",
                "-o",
                so_file_path,
            ]
        )

        if so_file_path[0] != "/":
            so_file_path = "./" + so_file_path

        # We need backend subtype
        lib = ctypes.CDLL(so_file_path)

        # Input and output structs
        class Inputs(ctypes.Structure):
            _fields_ = [
                (key, ctypes.POINTER(self.backend.get_struct_cls()))
                for key in self.determined_struct_keys["eval_input_keys"]
            ]

        class Outputs(ctypes.Structure):
            _fields_ = [
                (key, ctypes.POINTER(self.backend.get_struct_cls()))
                for key in self.determined_struct_keys["eval_output_keys"]
            ]

        class GradInputs(ctypes.Structure):
            _fields_ = [
                (key, ctypes.POINTER(self.backend.get_struct_cls()))
                for key in self.determined_struct_keys["eval_grad_input_keys"]
            ]

        class GradOutputs(ctypes.Structure):
            _fields_ = [
                (key, ctypes.POINTER(self.backend.get_struct_cls()))
                for key in self.determined_struct_keys["eval_grad_output_keys"]
            ]

        # Set the return type and argument types
        lib.evaluate.argtypes = [ctypes.POINTER(Inputs)]
        lib.evaluate.restype = Outputs

        if not self.pm.inference:
            lib.evaluate_gradients.argtypes = [ctypes.POINTER(GradInputs)]
            lib.evaluate_gradients.restype = GradOutputs

        # we need backend data types!
        # include_internals flag is used for get internal values for backpropagation
        def evaluate_wrapper(
            params: dict[str, PyArray] | None,
            data: dict[str, PyArray] | None,
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

            if self.configs.ALLOCATE_INTERNALS:
                # Allocate output arrays
                for arg_key in self.determined_struct_keys["eval_input_keys"]:
                    if arg_key in inputs:
                        continue

                    arr_shape = self._get_array_shape(arg_key)
                    inputs[arg_key] = self.backend.empty(*arr_shape)

            inputs_struct = Inputs(
                **{
                    key: ctypes.pointer(inputs[key].arr)
                    for key in self.determined_struct_keys["eval_input_keys"]
                }
            )
            inputs_struct_ptr = ctypes.pointer(inputs_struct)

            output_struct = lib.evaluate(inputs_struct_ptr)

            outputs = {}
            return_keys = (
                self.determined_struct_keys["eval_output_keys"]
                if include_internals
                else self.pm.output_keys
            )
            for key in return_keys:
                array_ptr = getattr(output_struct, key)
                outputs[key] = PyArray(
                    array_ptr.contents, shape=self.get_tensor_shape(key)
                )

            return outputs

        def evaluate_gradients_wrapper(
            params: dict[str, PyArray],
            data: dict[str, PyArray] | None = None,
            output_gradients: dict[str, PyArray] | None = None,
            include_output: bool = False,
        ) -> dict[str, PyArray]:
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
                params=params,
                data=data,
                cache={},
                include_internals=self.configs.ALLOCATE_INTERNALS,
            )

            # Create gradients for all params
            if self.configs.ALLOCATE_INTERNALS:
                for key in (
                    self.pm.flat_graph.all_source_keys
                    - self.pm.flat_graph.all_static_keys
                    - self.pm.flat_graph.unused_keys
                    - self.pm.ignore_grad_keys
                ):
                    # In CBackend we are creating all internal gradients with zeros.
                    if key not in gradients:
                        arr_shape = self._get_array_shape(key)
                        gradients[key] = self.backend.zeros(*arr_shape)

            gradients = {key + "_grad": value for key, value in gradients.items()}

            inputs = params | data | gradients | forward_pass

            inputs_struct = GradInputs(
                **{
                    key: ctypes.pointer(inputs[key].arr)
                    for key in self.determined_struct_keys["eval_grad_input_keys"]
                }
            )
            inputs_struct_ptr = ctypes.pointer(inputs_struct)

            output_struct = lib.evaluate_gradients(inputs_struct_ptr)
            outputs = {}
            for key in self.determined_struct_keys["eval_grad_output_keys"]:
                array_ptr = getattr(output_struct, key)
                outputs[key] = PyArray(
                    array_ptr.contents, shape=self.get_tensor_shape(key)
                )

            return outputs

        return (  # type: ignore
            evaluate_wrapper,
            evaluate_gradients_wrapper,
            partial(evaluate_gradients_wrapper, include_output=True),  # type: ignore
        )

    def generate_imports(self) -> list[c_ast.Include]:
        header_path = os.path.join(self.backend.SRC_PATH, self.configs.HEADER_NAME)
        return [c_ast.Include(header_path, system=False)]

    def create_primitive_call(
        self, formula_name: str, args: list[str], context: str
    ) -> c_ast.Expr:
        return c_ast.Call(
            formula_name, [self.create_key_ref(key, context=context) for key in args]
        )

    def create_key_ref(
        self, key: str, context: str, load: bool = True
    ) -> c_ast.Variable:
        if key in self.determined_struct_keys["eval_cache_keys"]:
            return c_ast.Variable(f"{self.CACHE_NAME}.{key}")

        elif (
            context == "eval" and key in self.determined_struct_keys["eval_input_keys"]
        ):
            return c_ast.Variable(f"inputs->{key}")

        elif context == "eval_grad":
            if key in self.determined_struct_keys["eval_grad_input_keys"]:
                return c_ast.Variable(f"inputs->{key}")

            if (
                key in self.pm.flat_graph.all_keys
                or key.replace("_grad", "") in self.pm.flat_graph.all_keys
            ) and not load:
                return c_ast.Variable(f"{self.configs.ARRAY_NAME} *{key}")

        return c_ast.Variable(key)

    def assign_array(self, target: c_ast.Variable, source: c_ast.Expr) -> c_ast.Assign:
        return c_ast.Assign(target, source)

    def define_function(
        self,
        return_type: str,
        name: str,
        params: list[c_ast.Parameter],
        pre_process: list[c_ast.Stmt]
        | list[c_ast.Expr]
        | list[c_ast.Stmt | c_ast.Expr],
        operations: list[c_ast.Stmt] | list[c_ast.Expr] | list[c_ast.Stmt | c_ast.Expr],
        post_process: list[c_ast.Stmt]
        | list[c_ast.Expr]
        | list[c_ast.Stmt | c_ast.Expr],
    ) -> c_ast.FunctionDef:
        body = pre_process + operations + post_process
        return c_ast.FunctionDef(return_type, name, params, body)

    def create_output_struct(self, context: str) -> c_ast.StructInit:
        output_keys = (
            self.determined_struct_keys["eval_output_keys"]
            if context == "eval"
            else self.determined_struct_keys["eval_grad_output_keys"]
        )
        output_struct_init: dict[str, c_ast.Expr] = {
            key: self.create_key_ref(key, context=context) for key in output_keys
        }

        output_struct_name = (
            self.EVALUATE_OUTPUT_STRUCT_NAME
            if context == "eval"
            else self.EVALUATE_GRAD_OUTPUT_STRUCT_NAME
        )

        return c_ast.StructInit(
            f"{output_struct_name} output_struct", output_struct_init
        )

    def generate_evaluate(self) -> tuple[c_ast.FunctionDef, set[str]]:
        # Function body
        pre_process: (
            list[c_ast.Stmt] | list[c_ast.Expr] | list[c_ast.Stmt | c_ast.Expr]
        ) = []
        operations: (
            list[c_ast.Stmt] | list[c_ast.Expr] | list[c_ast.Stmt | c_ast.Expr]
        ) = []
        post_process: (
            list[c_ast.Stmt] | list[c_ast.Expr] | list[c_ast.Stmt | c_ast.Expr]
        ) = []

        # Define function arguments
        arguments = [
            c_ast.Parameter(f"struct {self.EVALUATE_INPUT_STRUCT_NAME} *", "inputs")
        ]

        # Utility structures
        used_keys: set[str] = set()

        for output_key in self.pm.flat_graph.topological_order:
            model = self.pm.flat_graph.get_model(output_key)
            inputs = self.pm.flat_graph.get_source_keys(output_key)

            if self.configs.USE_OUTPUT_AS_INPUT:
                # In raw_c backend we need to pass output array as first argument
                used_keys.add(output_key)
                inputs = [output_key] + inputs

            # Create primitive call
            p_call = self.create_primitive_call(
                model.formula_key, inputs, context="eval"
            )
            p_call_stmt: c_ast.Stmt
            if self.configs.RETURN_OUTPUT:
                p_call_stmt = self.assign_array(
                    self.create_key_ref(output_key, context="eval", load=False), p_call
                )
            else:
                p_call_stmt = c_ast.MakeStmt(p_call)

            operations.append(p_call_stmt)
            used_keys |= set(inputs)

        # Prepare output
        post_process.append(self.create_output_struct(context="eval"))
        post_process.append(c_ast.Return(c_ast.Variable("output_struct")))

        evaluate_fn = self.define_function(
            f"struct {self.EVALUATE_OUTPUT_STRUCT_NAME}",
            "evaluate",
            arguments,
            pre_process,
            operations,
            post_process,
        )

        return evaluate_fn, used_keys

    def generate_evaluate_gradients(self) -> tuple[c_ast.FunctionDef, set[str]]:
        # Function body
        pre_process: (
            list[c_ast.Stmt] | list[c_ast.Expr] | list[c_ast.Stmt | c_ast.Expr]
        ) = []
        operations: (
            list[c_ast.Stmt] | list[c_ast.Expr] | list[c_ast.Stmt | c_ast.Expr]
        ) = []
        post_process: (
            list[c_ast.Stmt] | list[c_ast.Expr] | list[c_ast.Stmt | c_ast.Expr]
        ) = []

        used_keys: set[str] = set()

        # Define function arguments
        arguments = [
            c_ast.Parameter(
                f"struct {self.EVALUATE_GRAD_INPUT_STRUCT_NAME} *", "inputs"
            )
        ]

        for output_key in reversed(self.pm.flat_graph.topological_order):
            # Staticly infered and unused model will not be added
            if output_key in self.ignored_grad_keys:
                continue

            model = self.pm.flat_graph.get_model(output_key)
            inputs = self.pm.flat_graph.get_source_keys(output_key)

            # Assume all inputs are Array
            grad_inputs = [input_key + "_grad" for input_key in inputs]
            for idx in range(len(grad_inputs)):
                fn_inputs: list[str] = [
                    output_key + "_grad",
                    c_ast.Constant(idx).to_str(),
                    output_key,
                ] + inputs

                if self.configs.USE_OUTPUT_AS_INPUT:
                    fn_inputs += grad_inputs

                # Create primitive call
                p_call = self.create_primitive_call(
                    model.formula_key + self.BACKWARD_FN_SUFFIX,
                    fn_inputs,
                    context="eval_grad",
                )

                p_call_stmt: c_ast.Stmt
                if self.configs.RETURN_OUTPUT:
                    p_call_stmt = self.assign_array(
                        self.create_key_ref(
                            grad_inputs[idx], context="eval_grad", load=False
                        ),
                        p_call,
                    )
                else:
                    p_call_stmt = c_ast.MakeStmt(p_call)

                operations.append(p_call_stmt)

            used_keys.add(output_key)
            used_keys.add(output_key + "_grad")
            used_keys |= set(inputs)
            used_keys |= set(grad_inputs)

        # Prepare output
        post_process.append(self.create_output_struct(context="eval_grad"))
        post_process.append(c_ast.Return(c_ast.Variable("output_struct")))

        evaluate_grad_fn = self.define_function(
            f"struct {self.EVALUATE_GRAD_OUTPUT_STRUCT_NAME}",
            "evaluate_gradients",
            arguments,
            pre_process,
            operations,
            post_process,
        )

        return evaluate_grad_fn, used_keys

    def _get_backend_path(self) -> str:
        backend_path = backend.__file__
        return backend_path[: backend_path.rindex("/")]

    def _get_array_shape(self, key: str) -> tuple[int, ...]:
        array_data = self.pm.data[key]
        assert array_data.shape is not None
        shape = array_data.shape.reprs[0].get_shapes()

        if is_list_int(shape):
            return tuple(shape)
        else:
            raise ValueError(f"Unexpected shape: {shape}")

    def generate_structs(self) -> None:
        # Generate structs
        eval_input_struct = self._generate_struct(
            self.EVALUATE_INPUT_STRUCT_NAME,
            self.determined_struct_keys["eval_input_keys"],
        )
        eval_outputs_struct = self._generate_struct(
            self.EVALUATE_OUTPUT_STRUCT_NAME,
            self.determined_struct_keys["eval_output_keys"],
        )

        cache_struct = self._generate_struct(
            self.CACHE_STRUCT_NAME, self.determined_struct_keys["eval_cache_keys"]
        )

        structs = [eval_input_struct, eval_outputs_struct, cache_struct]

        if not self.pm.inference:
            eval_grad_input_struct = self._generate_struct(
                self.EVALUATE_GRAD_INPUT_STRUCT_NAME,
                self.determined_struct_keys["eval_grad_input_keys"],
            )

            eval_grad_outputs_struct = self._generate_struct(
                self.EVALUATE_GRAD_OUTPUT_STRUCT_NAME,
                self.determined_struct_keys["eval_grad_output_keys"],
            )

            structs += [eval_grad_input_struct, eval_grad_outputs_struct]

        self.globals = structs + self.globals

    def _generate_struct(self, name: str, field_keys: list[str]) -> c_ast.Stmt:
        fields = [
            c_ast.StructField(f"{self.configs.ARRAY_NAME} *", key)
            for key in sorted(field_keys)
        ]
        struct = c_ast.StructDef(name, fields)
        return struct

    def _infer_ignored_grad_keys(self) -> set[str]:
        all_ignored_keys = (
            self.pm.ignore_grad_keys
            | self.pm.flat_graph.all_static_keys
            | self.pm.flat_graph.unused_keys
        )
        all_ignored_keys, _ = self.pm.flat_graph.infer_ignore(
            set(), self.pm._output_keys, all_ignored_keys, update_graph=False
        )

        return all_ignored_keys

    def _determine_struct_keys(self) -> dict[str, list[str]]:
        eval_input_keys = sorted(self.pm.input_keys)
        if self.configs.USE_OUTPUT_AS_INPUT:
            eval_input_keys = sorted(self.pm.flat_graph.all_keys)

        eval_output_keys = sorted(self.pm.output_keys)
        eval_cache_keys = sorted(self.pm.flat_graph.all_keys - self.pm.input_keys)

        eval_grad_input_keys = sorted(
            (
                self.pm.input_keys
                | set(self.pm.output_keys)
                | {
                    key + "_grad"
                    for key in set(self.pm.output_keys) - self.pm.ignore_grad_keys
                }
            )
            - self.pm.ignore_grad_keys
            - set(eval_cache_keys)
        )

        if self.configs.USE_OUTPUT_AS_INPUT:
            pass

        eval_grad_output_keys = sorted(
            [
                key + "_grad"
                for key in set(self.pm.input_keys) - self.pm.ignore_grad_keys
            ]
        )

        determined_struct_keys = {
            "eval_input_keys": eval_input_keys,
            "eval_output_keys": eval_output_keys,
            "eval_cache_keys": eval_cache_keys,
            "eval_grad_input_keys": eval_grad_input_keys,
            "eval_grad_output_keys": eval_grad_output_keys,
        }

        return determined_struct_keys

    def get_tensor_shape(self, key: str) -> tuple[int, ...]:
        if key in self.pm.shapes:
            return self.pm.shapes[key]  # type: ignore
        elif key.replace("_grad", "") in self.pm.shapes:
            return self.pm.shapes[key.replace("_grad", "")]  # type: ignore
        else:
            raise ValueError(f"Shape for key {key} not found")
