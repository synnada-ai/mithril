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
import enum
import importlib
import keyword
from collections.abc import Callable
from functools import partial
from posixpath import basename, splitext
from typing import Any, Generic, Literal, Protocol, overload

from ...backends.backend import ParallelBackend
from ...core import DataType, Dtype
from ...utils.func_utils import prepare_function_args
from ..common import (
    DataEvalType,
    EvaluateAllType,
    EvaluateGradientsType,
    EvaluateType,
    ParamsEvalType,
)
from ..logical import PrimitiveModel
from ..physical.model import PhysicalModel
from ..utils import GeneratedFunction
from .code_gen import CodeGen
from .utils import (
    convert_to_ast_arg,
    convert_to_ast_kwarg,
    partial_array_creation_func,
)

FinalCost = "final_cost"


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
    @overload
    def __call__(
        self,
        params: ParamsEvalType[DataType],
        data: DataEvalType[DataType],
        output_gradients: ParamsEvalType[DataType],
        include_output: Literal[True],
    ) -> tuple[DataEvalType[DataType], ParamsEvalType[DataType]]: ...

    @overload
    def __call__(
        self,
        params: ParamsEvalType[DataType],
        data: DataEvalType[DataType],
        output_gradients: ParamsEvalType[DataType],
        include_output: Literal[False],
    ) -> ParamsEvalType[DataType]: ...

    def __call__(
        self,
        params: ParamsEvalType[DataType],
        data: DataEvalType[DataType],
        output_gradients: ParamsEvalType[DataType],
        include_output: bool,
    ) -> (
        ParamsEvalType[DataType]
        | tuple[DataEvalType[DataType], ParamsEvalType[DataType]]
    ): ...


class PythonCodeGen(CodeGen[Any], Generic[DataType]):
    def __init__(self, pm: PhysicalModel[DataType]) -> None:
        super().__init__(pm)

        self.module = ast.parse("")
        self.defined_partial_fns: set[str] = set()

        self.imports: list[ast.stmt] = []
        self.globals: list[ast.stmt] = []
        self.functions: list[ast.stmt] = []
        self.backend = self.pm.backend

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
    ) -> tuple[
        EvaluateType[DataType],
        EvaluateGradientsType[DataType] | None,
        EvaluateAllType[DataType] | None,
    ]:
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

        compiled_code = compile(self.code, "<string>", "exec")
        # Execute the compiled code to define the function
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
    ) -> tuple[
        EvaluateType[DataType],
        EvaluateGradientsType[DataType] | None,
        EvaluateAllType[DataType] | None,
    ]:
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
        grad_fn = None
        evaluate_all_fn = None
        if not self.pm.inference:
            grad_fn, evaluate_all_fn = self.create_gradient_fn(
                raw_eval_fn, raw_evaluate_grad_fn=raw_grad_fn
            )

        if (
            isinstance(self.pm.backend, ParallelBackend)
            and self.pm.backend.n_devices > 1
        ):
            self.pm.backend.register_callable(eval_fn, "eval_fn", jit)
            if not self.pm.inference:
                assert grad_fn is not None, "Gradient function is not defined!"
                assert (
                    evaluate_all_fn is not None
                ), "Evaluate all function is not defined!"

                self.pm.backend.register_callable(grad_fn, "eval_grad_fn", jit)
                self.pm.backend.register_callable(evaluate_all_fn, "eval_all_fn", jit)

        elif jit and not self.pm.backend.is_manualgrad:
            eval_fn = self.pm.backend.jit(eval_fn)
            if not self.pm.inference:
                assert grad_fn is not None, "Gradient function is not defined!"
                assert (
                    evaluate_all_fn is not None
                ), "Evaluate all function is not defined!"

                grad_fn = self.pm.backend.jit(grad_fn)  # type: ignore
                evaluate_all_fn = self.pm.backend.jit(evaluate_all_fn)

        return eval_fn, grad_fn, evaluate_all_fn  # type: ignore

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

        # Add registered primitives
        if len(self.pm.backend.registered_primitives.keys()) > 0:
            backend = self.import_backend()
            imports.append(backend)

        for func_name in self.pm.backend.registered_primitives:
            # Add function definition
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
            imports.append(
                ast.Assign(targets=[assignment_target], value=assignment_value)
            )

        return imports

    def is_static_scalar(self, key: str) -> bool:
        return (
            key in self.pm.flat_graph.cached_data
            and not self.pm.data[key].is_tensor
            and self.pm.data[key].edge_type != Dtype
            and not isinstance(self.pm.flat_graph.cached_data[key], enum.Enum)
        )

    def get_primitive_details(
        self, output_key: str
    ) -> tuple[PrimitiveModel, list[str], list[str]]:
        model = self.pm.flat_graph.get_model(output_key)

        global_input_keys = self.pm.flat_graph.get_source_keys(output_key)
        local_input_keys = list(model.input_keys)

        return model, global_input_keys, local_input_keys

    def call_primitive(
        self,
        model: PrimitiveModel,
        fn: Callable[..., Any],
        l_input_keys: list[str],
        g_input_keys: list[str],
        output_key: str,
        formula_key: str,
    ) -> tuple[ast.Assign, set[str]]:
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

        # Iterate over Primitive models in topological order to add their formula.
        for output_key in self.pm.flat_graph.topological_order:
            model, g_input_keys, l_input_keys = self.get_primitive_details(output_key)
            formula_key = model.formula_key

            primitive_function = (
                self.pm.backend.primitive_function_dict[formula_key]
                if formula_key in self.pm.backend.primitive_function_dict
                else self.pm.backend.registered_primitives[formula_key]
            )

            # Create primitive call
            primitive_call, _used_keys = self.call_primitive(
                model,
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

            for used_key in g_input_keys:
                if (
                    used_key in self.pm.flat_graph.output_dict.values()
                    or used_key in deleted_vars
                    or (
                        used_key in self.pm.input_keys  # Inputs shouldn't deleted
                        or used_key in self.pm.flat_graph.all_static_keys
                    )
                ):
                    continue

                keys = set(self.pm.flat_graph.get_target_keys(used_key, False)) - (
                    cached_data_keys | unused_keys | discarded_keys
                )
                if keys.issubset(assigned_output_keys):
                    function_body.append(
                        ast.Delete(targets=[ast.Name(id=used_key, ctx=ast.Del())])
                    )
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

    def append_inputs(
        self, input_body: list[ast.stmt], key: str, dict_type: str
    ) -> None:
        # In manual_grad type backends, cache contains all the required
        # data (local variables and outputs) for the corresponding function.
        # So if the key is not directly an output of a function get it from
        # cache with the key itself.
        if keyword.iskeyword(key) or key in self.pm.backend.primitive_function_dict:
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
        self, output_key: str, model: PrimitiveModel, inference: bool
    ) -> tuple[list[ast.expr], set[str]]:
        if (
            keyword.iskeyword(output_key)
            or output_key in self.pm.backend.primitive_function_dict
        ):
            target_name = "_" + output_key
        else:
            target_name = output_key

        targets: list[ast.expr] = [self._var_ref_ast(target_name, ast.Store())]

        return targets, {target_name}

    def add_partial_function(self, formula_key: str) -> None:
        if formula_key in self.defined_partial_fns:
            return

        self.defined_partial_fns.add(formula_key)
        self.globals.append(partial_array_creation_func(self.pm.backend, formula_key))

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
        # raw_evaluate_fn: RawEvaluateType[DataType],
        # raw_evaluate_grad_fn: ManualGradWrapperFn[DataType] | None,
        raw_evaluate_fn: RawEvaluateType[DataType],
        raw_evaluate_grad_fn: ManualGradWrapperFn[DataType] | None,
    ) -> tuple[ManualGradWrapperFn[DataType], RawEvaluateType[DataType]]:
        fn_all: EvaluateAllType[DataType]
        grad_fn: EvaluateGradientsType[DataType]
        if not self.pm.backend.is_manualgrad:
            grad_fn = partial(
                self.compute_gradients,
                raw_evaluate_fn=raw_evaluate_fn,
                cache=self.pm.flat_graph.cached_data,
                include_output=False,
            )
            # Fix fn_all for mlx support!!
            fn_all = partial(
                self.compute_gradients,
                raw_evaluate_fn=raw_evaluate_fn,
                cache=self.pm.flat_graph.cached_data,
                include_output=True,
            )
            return grad_fn, fn_all  # type: ignore
        else:
            assert raw_evaluate_grad_fn is not None, "Gradient function is not defined!"

            fn_all = partial(raw_evaluate_grad_fn, include_output=True)  # type: ignore
            grad_fn = partial(raw_evaluate_grad_fn, include_output=False)  # type: ignore

            return grad_fn, fn_all  # type: ignore

    @overload
    def compute_gradients(
        self,
        params: ParamsEvalType[DataType],
        data: DataEvalType[DataType] | None,
        output_gradients: ParamsEvalType[DataType] | None,
        cache: DataEvalType[DataType] | None,
        include_output: Literal[True],
        *,
        raw_evaluate_fn: RawEvaluateType[DataType],
    ) -> tuple[DataEvalType[DataType], ParamsEvalType[DataType]]: ...

    @overload
    def compute_gradients(
        self,
        params: ParamsEvalType[DataType],
        data: DataEvalType[DataType] | None,
        output_gradients: ParamsEvalType[DataType] | None,
        cache: DataEvalType[DataType] | None,
        include_output: Literal[False],
        *,
        raw_evaluate_fn: RawEvaluateType[DataType],
    ) -> ParamsEvalType[DataType]: ...

    def compute_gradients(
        self,
        params: ParamsEvalType[DataType],
        data: DataEvalType[DataType] | None = None,
        output_gradients: ParamsEvalType[DataType] | None = None,
        cache: DataEvalType[DataType] | None = None,
        include_output: bool = False,
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
        total_ignore_grad_keys = self.pm.ignore_grad_keys.union(
            {
                key
                for key in self.pm.output_keys
                if (key not in total_output_gradients) and key != FinalCost
            }
        )
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

        if include_output:
            return all_outputs, input_gradients
        else:
            return input_gradients

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
                " at least one of the {list(aux.keys())}"
            )

        return outputs, aux  # type: ignore

    # Variable references will be created with this function
    def _var_ref_ast(self, name: str, ctx: ast.expr_context) -> ast.Name:
        # Make non keyword
        if keyword.iskeyword(name) or name in self.backend.primitive_function_dict:
            name = "_" + name

        return ast.Name(id=name, ctx=ctx)
