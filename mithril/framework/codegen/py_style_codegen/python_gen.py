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
import importlib
import keyword
from collections.abc import Callable
from functools import partial
from posixpath import basename, splitext
from typing import Any, Generic, Protocol

from ....backends.backend import ParallelBackend
from ....common import PythonGenConfig
from ....types import DataType
from ....utils.func_utils import prepare_function_args
from ...common import (
    DataEvalType,
    EvaluateAllType,
    EvaluateType,
    FinalCost,
    ParamsEvalType,
)
from ...logical import Operator
from ...physical.model import PhysicalModel
from ...utils import GeneratedFunction
from ..code_gen import CodeGen
from ..utils import (
    convert_to_ast_arg,
    convert_to_ast_kwarg,
    partial_array_creation_func,
)


class RawEvaluateType(Protocol, Generic[DataType]):
    def __call__(
        self,
        params: ParamsEvalType[DataType] | None,
        data: DataEvalType[DataType] | None,
        cache: DataEvalType[DataType] | None,
    ) -> DataEvalType[DataType]: ...


class RawGradientType(Protocol, Generic[DataType]):
    def __call__(
        self,
        params: ParamsEvalType[DataType],
        gradients: ParamsEvalType[DataType],
        data: DataEvalType[DataType],
        cache: DataEvalType[DataType],
    ) -> ParamsEvalType[DataType]: ...


class ManualGradWrapperFn(Protocol, Generic[DataType]):
    def __call__(
        self,
        params: ParamsEvalType[DataType],
        data: DataEvalType[DataType],
        output_gradients: ParamsEvalType[DataType],
    ) -> (
        ParamsEvalType[DataType]
        | tuple[DataEvalType[DataType], ParamsEvalType[DataType]]
    ): ...


class PythonCodeGen(CodeGen[Any], Generic[DataType]):
    def __init__(self, pm: PhysicalModel[DataType]) -> None:
        super().__init__(pm)

        self.module = ast.parse("")

        # Tracks generated partial functions (e.g., for array creation)
        # to avoid passing redundant device/dtype arguments in the generated code.
        self.defined_partial_fns: set[str] = set()

        # Subsections of the generated file
        self.imports: list[ast.stmt] = []
        self.globals: list[ast.stmt] = []
        self.functions: list[ast.stmt] = []

        self.backend = self.pm.backend

        assert isinstance(self.backend.CODEGEN_CONFIG, PythonGenConfig)
        self.configs = self.backend.CODEGEN_CONFIG

    def generate_code(self, file_path: str | None = None) -> None:
        self.file_path = file_path

        self.imports += self.generate_imports()
        self.functions += self.generate_functions()

        self.module.body = self.imports
        self.module.body += self.globals
        self.module.body += self.functions

        self.module = ast.fix_missing_locations(self.module)
        self.code = ast.unparse(self.module)

        if file_path is not None:
            self.write_code(file_path)

    def generate_functions(self) -> list[ast.FunctionDef]:
        return [self.generate_evaluate()]

    def write_code(self, file_path: str) -> None:
        if self.code is None:
            raise Exception(
                "Code is not generated yet! Please call generate_code() first."
            )
        with open(file_path, "w") as file:
            file.write(self.code)

    def compile_code(
        self, jit: bool = False
    ) -> tuple[EvaluateType[DataType], EvaluateAllType[DataType] | None]:
        eval_fn, grad_fn = self.exec_generated_code()
        return self.post_process_fns(eval_fn, grad_fn, jit)

    def exec_generated_code(
        self,
    ) -> tuple[Callable[..., Any], Callable[..., Any] | None]:
        if self.code is None:
            raise Exception(
                "Code is not generated yet! Please call generate_code() first."
            )

        if self.file_path is not None:
            # We are loading the generated code from a file
            # This allows to debug from the generated code
            module_name = splitext(basename(self.file_path))[0]

            module_spec = importlib.util.spec_from_file_location(
                module_name, self.file_path
            )
            module = importlib.util.module_from_spec(module_spec)  # type: ignore
            module_spec.loader.exec_module(module)  # type: ignore
            eval_fn: EvaluateType[DataType] = module.evaluate
            eval_grad_fn = (
                module.evaluate_gradients
                if hasattr(module, "evaluate_gradients")
                else None
            )
            return eval_fn, eval_grad_fn

        # If the file path is not provided, we compile the code
        # and execute it to define the function

        compiled_code = compile(self.code, "<string>", "exec")
        result: dict[str, Any] = {}
        exec(compiled_code, result)
        evaluate_fn = result["evaluate"]
        evaluate_grad_fn = result.get("evaluate_gradients")

        evaluate_metadata = {
            "fn_name": "evaluate",
            "source": self.code,
        }

        evaluate_grad_metadata = {
            "fn_name": "evaluate_gradients",
            "source": self.code,
        }

        # Wrap the generated function in a class that can be pickled
        eval_fn = GeneratedFunction(evaluate_fn, evaluate_metadata)
        grad_fn = (
            GeneratedFunction(evaluate_grad_fn, evaluate_grad_metadata)
            if evaluate_grad_fn is not None
            else None
        )

        return eval_fn, grad_fn

    def post_process_fns(
        self,
        raw_eval_fn: RawEvaluateType[DataType],
        raw_grad_fn: ManualGradWrapperFn[DataType] | None,
        jit: bool,
    ) -> tuple[EvaluateType[DataType], EvaluateAllType[DataType] | None]:
        """In this function going to wrap the raw functions with some additional
        functionalities.

        1. If the backend is parallel, going to register the functions to the backend.
        2. If the backend is manualgrad, going to wrap the eval_grad function.
        3. If jit is True, going to compile the functions with jit fn.
        """

        eval_fn: EvaluateType[DataType] | partial[Any] = partial(
            self.compute_evaluate,
            fn=raw_eval_fn,
            cache=self.pm.flat_graph.cached_data,
        )
        evaluate_all_fn = None
        if not self.pm.inference:
            evaluate_all_fn = self.create_gradient_fn(
                raw_eval_fn, raw_evaluate_grad_fn=raw_grad_fn
            )

        if (
            isinstance(self.pm.backend, ParallelBackend)
            and self.pm.backend.n_devices > 1
        ):
            self.pm.backend.register_callable(eval_fn, "eval_fn", jit)
            if not self.pm.inference:
                assert (
                    evaluate_all_fn is not None
                ), "Evaluate all function is not defined!"
                self.pm.backend.register_callable(evaluate_all_fn, "eval_all_fn", jit)

        elif jit and not self.pm.backend.is_manualgrad:
            eval_fn = self.pm.backend.jit(eval_fn)
            if not self.pm.inference:
                assert (
                    evaluate_all_fn is not None
                ), "Evaluate all function is not defined!"
                evaluate_all_fn = self.pm.backend.jit(evaluate_all_fn)

        return eval_fn, evaluate_all_fn  # type: ignore

    def import_backend(self) -> ast.ImportFrom:
        backend = ast.ImportFrom(
            module="mithril",
            names=[
                ast.alias(
                    name=f"{self.pm.backend.backend_type.capitalize()}Backend",
                    asname="Backend",
                )
            ],
            level=0,
        )

        return backend

    def _add_registered_primitives(self, func_name: str) -> ast.stmt:
        assignment_target = ast.Name(id=func_name, ctx=ast.Store())
        assignment_value = ast.Subscript(
            value=ast.Attribute(
                value=ast.Name(id="Backend", ctx=ast.Load()),
                attr="registered_primitives",
                ctx=ast.Load(),
            ),
            slice=ast.Constant(value=func_name),
            ctx=ast.Load(),
        )
        return ast.Assign(targets=[assignment_target], value=assignment_value)

    def generate_imports(self) -> list[ast.stmt]:
        imports: list[ast.stmt] = []
        # Add import primitive functions
        imports.append(
            ast.ImportFrom(
                module=self.pm.backend.primitive_fn_path,
                names=[ast.alias(name="*", asname=None)],
                level=0,
            )
        )

        # To be able to use the registered primitives, we need to import the backend
        if len(self.pm.backend.registered_primitives.keys()) > 0:
            imports.append(self.import_backend())

        # User can register a custom primitive into the backend
        # we are using the registered primitive in the generated code
        # by creating an object of the registered primitive
        for func_name in self.pm.backend.registered_primitives:
            imports.append(self._add_registered_primitives(func_name))

        return imports

    def get_primitive_details(
        self, output_key: str
    ) -> tuple[Operator, list[str], list[str]]:
        model = self.pm.flat_graph.get_op(output_key)

        global_input_keys = self.pm.flat_graph.get_source_keys(output_key)
        local_input_keys = list(model.input_keys)

        return model, global_input_keys, local_input_keys

    def call_primitive(
        self,
        model: Operator,
        fn: Callable[..., Any],
        l_input_keys: list[str],
        g_input_keys: list[str],
        output_key: str,
        formula_key: str,
    ) -> tuple[ast.Assign, set[str]]:
        # We divided operation generation into two parts:
        # 1. Create the primitive call
        # 2. Create the targets
        # This is done because we need to add partial function for array creation
        # but we need to do it after creating the targets
        generated_fn, used_keys = self.create_primitive_call(
            fn, l_input_keys, g_input_keys
        )
        targets, _used_keys = self.create_primitive_call_targets(
            output_key, model, self.pm.inference
        )
        if formula_key in self.pm.backend.array_creation_funcs:
            self.add_partial_function(formula_key)

        return ast.Assign(targets, generated_fn), used_keys | _used_keys

    def generate_evaluate(self) -> ast.FunctionDef:
        input_body: list[ast.stmt] = []
        function_body: list[ast.stmt] = []
        return_values: list[ast.expr] = []

        used_keys: set[str] = set()
        used_keys |= set(self.pm.flat_graph.output_dict.values())

        unused_keys = self.pm.flat_graph.unused_keys
        cached_data_keys = self.pm.flat_graph.cached_data.keys()
        discarded_keys = self.pm.discarded_keys  # TODO: Consider is this necessary?

        deleted_vars: set[str] = set()
        assigned_output_keys: set[str] = set()

        determined_keys = cached_data_keys | unused_keys | discarded_keys

        # Iterate over ops in topological order to add their formula.
        for output_key in self.pm.flat_graph.topological_order:
            # Get operator details
            op, g_input_keys, l_input_keys = self.get_op_details(output_key)
            formula_key = op.formula_key

            if formula_key in self.pm.backend.op_function_dict:
                primitive_function = self.pm.backend.op_function_dict[formula_key]
            elif formula_key in self.pm.backend.registered_primitives:
                primitive_function = self.pm.backend.registered_primitives[formula_key]
            else:
                raise ValueError(
                    f"Formula key {formula_key} not found in primitive function dict or"
                    " registered primitives"
                )

            # Create primitive call
            primitive_call, _used_keys = self.call_primitive(
                op,
                primitive_function,
                l_input_keys,
                g_input_keys,
                output_key,
                formula_key,
            )

            used_keys |= _used_keys
            used_keys.add(output_key)
            assigned_output_keys.add(output_key)
            function_body.append(primitive_call)

            # Add deletion logic for intermediate variables
            for used_key in g_input_keys:
                if not self._check_deletable(
                    used_key,
                    deleted_vars,
                    determined_keys,
                    assigned_output_keys,
                ):
                    continue

                delete_stmt = ast.Delete(
                    targets=[self._var_ref_ast(used_key, ast.Del())]
                )
                function_body.append(delete_stmt)
                deleted_vars.add(used_key)

        for key in sorted(used_keys):
            if key in cached_data_keys:
                dict_type = "cache"
            elif key in self.pm.flat_graph.runtime_static_keys:
                dict_type = "data"
            elif key not in self.pm.flat_graph.all_target_keys:
                dict_type = "params"
            else:
                continue

            # If cached value is not a tensor, do not append it to code
            if not self.is_static_scalar(key):
                self.append_inputs(input_body, key, dict_type)

        # If output node is pruned adjust key name
        for output_key in self.pm.output_keys:
            # TODO: give an api to get outputdict
            if self.is_static_scalar(output_key):
                return_values.append(
                    ast.Constant(self.pm.flat_graph.cached_data[output_key])
                )
            else:
                return_values.append(
                    ast.Name(self.pm.flat_graph.output_dict[output_key], ast.Load())
                )

        return_body: list[ast.stmt] = [
            (
                ast.Return(
                    value=ast.Dict(
                        keys=[
                            ast.Constant(output_key)
                            for output_key in self.pm.output_keys
                        ],
                        values=return_values,
                    )
                )
            )
        ]
        # Define function arguments
        ast_args = [ast.arg("params"), ast.arg("data"), ast.arg("cache")]
        final_args = ast.arguments(
            args=ast_args,
            defaults=[],
            kwonlyargs=[],
            posonlyargs=[],
            kw_defaults=[],
            vararg=None,
            kwarg=None,
        )

        func_def = ast.FunctionDef(
            name="evaluate",
            args=final_args,
            body=input_body + function_body + return_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
            type_params=[],
            lineno=1,
            col_offset=0,
        )
        return ast.fix_missing_locations(func_def)

    def add_partial_function(self, formula_key: str) -> None:
        # Simply creates partial functions for array creation fns
        # To avoid redundant argument passing for array creation fns
        # We are creating a partial function and adding it to the global scope
        # This partial function will be used in the generated code
        # instead of the original function definition

        if formula_key in self.defined_partial_fns:
            return

        self.defined_partial_fns.add(formula_key)
        self.globals.append(partial_array_creation_func(self.pm.backend, formula_key))

    def append_inputs(
        self, input_body: list[ast.stmt], key: str, dict_type: str
    ) -> None:
        # In manual_grad type backends, cache contains all the required
        # data (local variables and outputs) for the corresponding function.
        # So if the key is not directly an output of a function get it from
        # cache with the key itself.
        if keyword.iskeyword(key) or key in self.pm.backend.op_function_dict:
            val = f"_{key}"
        else:
            val = key
        if dict_type != "cache" or (key not in self.pm.flat_graph.all_target_keys):
            input_body.append(
                ast.Assign(
                    targets=[self._var_ref_ast(val, ast.Store())],
                    value=ast.Subscript(
                        value=ast.Name(id=dict_type, ctx=ast.Load()),
                        slice=ast.Constant(value=key),
                        ctx=ast.Load(),
                    ),
                )
            )
        else:
            # If key is an output of a function, then get the corresponding
            # function cache from general cache and then get "output" from there.
            cached_data = self.pm.flat_graph.cached_data
            data_dict: ast.Subscript | ast.Name
            if key not in cached_data:
                cache_name = key + "_cache"

                data_dict = ast.Subscript(
                    value=ast.Name(id=dict_type, ctx=ast.Load()),
                    slice=ast.Constant(value=cache_name),
                    ctx=ast.Load(),
                )
            else:
                data_dict = ast.Name(id=dict_type, ctx=ast.Load())

            slc = ast.Constant(value="output" if key not in cached_data else key)
            input_body.append(
                ast.Assign(
                    targets=[self._var_ref_ast(val, ast.Store())],
                    value=ast.Subscript(value=data_dict, slice=slc, ctx=ast.Load()),
                )
            )

    def create_primitive_call(
        self,
        function: Callable[..., Any],
        local_keys: list[str],
        global_keys: list[str],
        default_args: dict[str, ast.expr] | None = None,
    ) -> tuple[ast.Call, set[str]]:
        """Generates a single function call AST (Abstract Syntax Tree)."""
        if default_args is None:
            default_args = {}
        cache = self.pm.flat_graph.cached_data
        formula_key = function.__name__
        inputs = {
            key: value for key, value in zip(local_keys, global_keys, strict=False)
        }
        # Prepare function arguments
        fn_args_mapping, fn_kwarg_dict = prepare_function_args(
            self.pm.flat_graph.cached_data,
            function,
            inputs,
            self.pm.backend.array_creation_funcs,
        )
        fn_arg_keys = [
            arg_key for arg_keys in fn_args_mapping.values() for arg_key in arg_keys
        ]
        # Convert to AST nodes
        """Types that should be added inline are defined and appended 
        to code with their corresponding value."""

        # Create args and kwargs
        args = []
        for arg_key in fn_arg_keys:
            if self.is_static_scalar(arg_key):
                args.append(ast.Constant(cache[arg_key]))
            else:
                args.append(
                    convert_to_ast_arg(
                        arg_key,
                        self._var_ref_ast(arg_key, ast.Load()),
                        defaults=default_args,  # type:ignore
                    )
                )

        kwargs = []
        for key, name in fn_kwarg_dict.items():
            if self.is_static_scalar(name):
                value = ast.Constant(cache[fn_kwarg_dict[key]])
                kwargs.append(ast.keyword(arg=key, value=value))
            else:
                kwargs.append(
                    convert_to_ast_kwarg(
                        key, self._var_ref_ast(name, ast.Load()), defaults=default_args
                    )
                )

        generated_fn = ast.Call(
            func=ast.Name(id=formula_key, ctx=ast.Load()),
            args=args,  # type:ignore
            keywords=kwargs,
        )
        used_keys = set(fn_arg_keys) | set(fn_kwarg_dict.values())

        return generated_fn, used_keys

    def create_primitive_call_targets(
        self, output_key: str, model: Operator, inference: bool
    ) -> tuple[list[ast.expr], set[str]]:
        if (
            keyword.iskeyword(output_key)
            or output_key in self.pm.backend.op_function_dict
        ):
            target_name = "_" + output_key
        else:
            target_name = output_key

        targets: list[ast.expr] = [self._var_ref_ast(target_name, ast.Store())]

        return targets, {target_name}

    def compute_evaluate(
        self,
        params: ParamsEvalType[DataType] | None = None,
        data: DataEvalType[DataType] | None = None,
        cache: DataEvalType[DataType] | None = None,
        *,
        fn: RawEvaluateType[DataType],
    ) -> DataEvalType[DataType]:
        return fn(params, data, cache)

    def create_gradient_fn(
        self,
        raw_evaluate_fn: RawEvaluateType[DataType],
        raw_evaluate_grad_fn: ManualGradWrapperFn[DataType] | None,
    ) -> ManualGradWrapperFn[DataType]:
        if not self.pm.backend.is_manualgrad:
            return partial(
                self.compute_gradients,
                raw_evaluate_fn=raw_evaluate_fn,
                cache=self.pm.flat_graph.cached_data,
            )
        else:
            assert raw_evaluate_grad_fn is not None, "Gradient function is not defined!"
            return raw_evaluate_grad_fn

    def compute_gradients(
        self,
        params: ParamsEvalType[DataType],
        data: DataEvalType[DataType] | None = None,
        output_gradients: ParamsEvalType[DataType] | None = None,
        cache: DataEvalType[DataType] | None = None,
        *,
        raw_evaluate_fn: RawEvaluateType[DataType],
    ) -> (
        tuple[DataEvalType[DataType], ParamsEvalType[DataType]]
        | ParamsEvalType[DataType]
    ):
        # Initialize loss output gradients as None. If FinalCost is
        # contained in the compiled model, initialize its gradient
        # with ones. If somehow one wants to set it to another gradient
        # value, must provide it in output_gradients argument which
        # overrides out initial fill.

        if data is None:
            data = {}
        if output_gradients is None:
            output_gradients = {}
        if cache is None:
            cache = {}

        loss_grad = {}
        if FinalCost in self.pm._output_keys:
            loss_grad = {FinalCost: self.pm.backend.ones()}
        elif len(output_gradients) == 0 and len(self.pm._output_keys) == 1:
            (out_key,) = self.pm._output_keys
            out_edge = self.pm.data[self.pm.flat_graph.output_dict[out_key]]
            if not out_edge.is_tensor or out_edge.shape.get_shapes() == []:  # type: ignore
                loss_grad = {out_key: self.pm.backend.ones()}

        # NOTE: FinalCost gradient and output_gradients can not exist at the same time.
        if output_gradients and loss_grad:
            raise Exception(
                "Models with any losses can not take any gradients for other outputs!"
            )

        # Sort gradients with output order
        output_gradients = {
            key: output_gradients[key]
            for key in self.pm.output_keys
            if key in output_gradients
        }

        total_output_gradients = loss_grad | output_gradients
        total_ignore_grad_keys = self.pm._output_keys - total_output_gradients.keys()

        partial_fn: Callable[
            [ParamsEvalType[DataType]],
            tuple[DataEvalType[DataType], DataEvalType[DataType]],
        ] = partial(
            self.filter_ignored_outputs,
            data=data,
            cache=cache,
            ignore_grad_keys=total_ignore_grad_keys,
            raw_evaluate_fn=raw_evaluate_fn,
        )
        output, input_gradients, aux = self.pm.backend.vjp(
            partial_fn,  # type: ignore
            params,
            cotangents=total_output_gradients,
            has_aux=True,
        )
        all_outputs: DataEvalType[DataType] = output | aux
        return all_outputs, input_gradients

    def filter_ignored_outputs(
        self,
        params: ParamsEvalType[DataType] | None = None,
        data: DataEvalType[DataType] | None = None,
        cache: DataEvalType[DataType] | None = None,
        ignore_grad_keys: set[str] | None = None,
        *,
        raw_evaluate_fn: RawEvaluateType[DataType],
    ) -> tuple[ParamsEvalType[DataType], ParamsEvalType[DataType]]:
        if params is None:
            params = {}
        if data is None:
            data = {}
        if cache is None:
            cache = {}
        # Remove outputs for which gradients are to be ignored.
        if ignore_grad_keys is None:
            ignore_grad_keys = set()

        outputs = raw_evaluate_fn(params, data=data, cache=cache)
        aux = {
            key: outputs.pop(key)  # type: ignore
            for key in list(outputs.keys())
            if key in ignore_grad_keys
        }

        if len(outputs) == 0:
            raise ValueError(
                "To evaluate gradients you must provide gradients for"
                f" at least one of the {list(aux.keys())}"
            )

        return outputs, aux  # type: ignore

    def get_op_details(self, output_key: str) -> tuple[Operator, list[str], list[str]]:
        model = self.pm.flat_graph.get_op(output_key)

        global_input_keys = self.pm.flat_graph.get_source_keys(output_key)
        local_input_keys = list(model.input_keys)

        return model, global_input_keys, local_input_keys

    # Variable references will be created with this function
    def _var_ref_ast(self, name: str, ctx: ast.expr_context) -> ast.Name:
        # Make non keyword
        if keyword.iskeyword(name) or name in self.backend.op_function_dict:
            name = "_" + name

        return ast.Name(id=name, ctx=ctx)

    def _check_deletable(
        self,
        used_key: str,
        deleted_vars: set[str],
        determined_keys: set[str],
        assigned_output_keys: set[str],
    ) -> bool:
        # Skip if the key is essential or already deleted
        if (
            used_key in self.pm.flat_graph.output_dict.values()
            or used_key in deleted_vars
            or used_key in self.pm.input_keys
            or self.pm.flat_graph.is_key_static(used_key)
        ):
            return False

        # Find consumers of the key
        target_keys = set(self.pm.flat_graph.get_target_keys(used_key, False))
        remaining_consumers = target_keys - determined_keys

        return remaining_consumers.issubset(assigned_output_keys)
