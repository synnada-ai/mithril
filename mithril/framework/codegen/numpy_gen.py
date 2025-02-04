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
import keyword
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, overload

import numpy as np

from ...backends.with_manualgrad.numpy_backend import NumpyBackend
from ...framework.physical.model import PhysicalModel
from ...utils.func_utils import is_make_array_required, prepare_function_args
from ..common import (
    DataEvalType,
    EvaluateAllType,
    EvaluateGradientsType,
    EvaluateType,
    FinalCost,
    IOHyperEdge,
    LossKey,
    ParamsEvalType,
    find_intersection_type,
    is_type_adjustment_required,
)
from ..logical import PrimitiveModel
from .python_gen import PythonCodeGen, RawGradientType
from .utils import check_repr_inequality


class NumpyCodeGen(PythonCodeGen[np.ndarray[Any, Any]]):
    BACKWARD_FN_SUFFIX = "_grad"

    def __init__(self, pm: PhysicalModel[np.ndarray[Any, Any]]) -> None:
        super().__init__(pm)

        assert isinstance(self.pm.backend, NumpyBackend)
        self.backend: NumpyBackend = self.pm.backend

    def generate_functions(self) -> list[ast.FunctionDef]:
        functions: list[ast.FunctionDef] = []
        functions.append(self.generate_evaluate())
        if not self.pm.inference:
            functions.append(self.generate_evaluate_gradients(self.pm.ignore_grad_keys))
        return functions

    def generate_imports(self) -> list[ast.stmt]:
        imports = super().generate_imports()

        # Import grad functions
        imports.append(
            ast.ImportFrom(
                module=self.backend.primitive_grad_fn_path,
                names=[ast.alias(name="*", asname=None)],
                level=0,
            )
        )

        for func_name in self.backend.registered_primitives:
            # Add grad registered function definition
            assignment_target = ast.Name(
                id=func_name + self.BACKWARD_FN_SUFFIX, ctx=ast.Store()
            )
            assignment_value = ast.Subscript(
                value=ast.Attribute(
                    value=ast.Name(id="Backend", ctx=ast.Load()),
                    attr="registered_primitives_grad_fn",
                    ctx=ast.Load(),
                ),
                slice=ast.Constant(value=func_name + self.BACKWARD_FN_SUFFIX),
                ctx=ast.Load(),
            )
            imports.append(
                ast.Assign(targets=[assignment_target], value=assignment_value)
            )

        return imports

    def compile_code(
        self, jit: bool = False
    ) -> tuple[
        EvaluateType[np.ndarray[Any, Any]],
        EvaluateGradientsType[np.ndarray[Any, Any]] | None,
        EvaluateAllType[np.ndarray[Any, Any]] | None,
    ]:
        eval_fn, grad_fn = self.exec_generated_code()

        # TODO: Not looks good, and looks over complicated!

        @overload
        def evaluate_gradients_wrapper_manualgrad(
            params: ParamsEvalType[np.ndarray[Any, Any]] | None,
            data: DataEvalType[np.ndarray[Any, Any]] | None,
            output_gradients: ParamsEvalType[np.ndarray[Any, Any]] | None,
            *,
            grad_fn: RawGradientType[np.ndarray[Any, Any]],
            include_output: Literal[False],
        ) -> DataEvalType[np.ndarray[Any, Any]]: ...

        @overload
        def evaluate_gradients_wrapper_manualgrad(
            params: ParamsEvalType[np.ndarray[Any, Any]] | None,
            data: DataEvalType[np.ndarray[Any, Any]] | None,
            output_gradients: ParamsEvalType[np.ndarray[Any, Any]] | None,
            *,
            grad_fn: RawGradientType[np.ndarray[Any, Any]],
            include_output: Literal[True],
        ) -> tuple[
            DataEvalType[np.ndarray[Any, Any]],
            ParamsEvalType[np.ndarray[Any, Any]],
        ]: ...

        def evaluate_gradients_wrapper_manualgrad(
            params: ParamsEvalType[np.ndarray[Any, Any]] | None = None,
            data: DataEvalType[np.ndarray[Any, Any]] | None = None,
            output_gradients: ParamsEvalType[np.ndarray[Any, Any]] | None = None,
            *,
            grad_fn: RawGradientType[np.ndarray[Any, Any]],
            include_output: bool = False,
        ) -> (
            DataEvalType[np.ndarray[Any, Any]]
            | tuple[
                DataEvalType[np.ndarray[Any, Any]],
                ParamsEvalType[np.ndarray[Any, Any]],
            ]
        ):
            if params is None:
                params = {}
            if data is None:
                data = {}
            # TODO: Consider not unioning batch data (data) into self.data
            # If evaluate_gradients called directly, first call evaluate.
            cached_data = self.pm.flat_graph.cached_data

            output: dict[str, np.ndarray[Any, Any]] = eval_fn(
                params=params, data=data, cache=cached_data
            )
            # Initialize gradients as zero with corresponding shapes.
            gradients: dict[str, np.ndarray[Any, Any]] = {}
            for key in (
                self.pm.flat_graph.all_keys
                - self.pm.flat_graph.all_static_keys
                - self.pm.flat_graph.unused_keys
                - self.pm.ignore_grad_keys
            ):
                key_cache = cached_data.get(key + "_cache", {})
                assert isinstance(key_cache, dict)
                out_data: np.ndarray[Any, Any] | None = None
                if key in params:
                    out_data = params[key]
                elif "output" in key_cache:
                    out_data = key_cache["output"]
                else:
                    # Removed primitives, to take shape of output take input shape
                    _key = self.pm.flat_graph.get_source_keys(
                        key, include_outputs=True
                    )[0]
                    _key_cache = cached_data.get(_key + "_cache", {})
                    assert isinstance(_key_cache, dict)
                    if _key in self.pm.input_keys:
                        out_data = params[_key]
                    else:
                        out_data = _key_cache["output"]

                assert isinstance(out_data, np.ndarray)

                gradients[key] = self.backend.zeros_like(
                    out_data, dtype=self.backend._dtype
                )

            if output_gradients is None:
                if FinalCost not in self.pm._output_keys:
                    raise ValueError(
                        "Requires output gradients if final loss is not attached!"
                    )
                else:
                    # Set "1.0" to output gradient if loss is attached
                    # and output_gradients is not given.
                    gradients |= {FinalCost: np.array(1.0)}
            else:
                gradients |= output_gradients
                if (FinalCost in self.pm._output_keys) and (
                    FinalCost not in output_gradients
                ):
                    gradients |= {FinalCost: np.array(1.0)}

            # Fill self.gradients with all input gradients.
            grad_fn(params=params, gradients=gradients, data=data, cache=cached_data)

            # Return only gradient values of trainable input keys.
            if include_output:
                return output, {key: gradients[key] for key in params}

            return {key: gradients[key] for key in params}

        if grad_fn is not None:
            grad_fn = partial(evaluate_gradients_wrapper_manualgrad, grad_fn=grad_fn)

        return self.post_process_fns(eval_fn, grad_fn, jit)  # type: ignore

    def get_primitive_details(
        self, output_key: str
    ) -> tuple[PrimitiveModel, list[str], list[str]]:
        model = self.pm.flat_graph.get_model(output_key)

        global_input_keys = self.pm.flat_graph.get_source_keys(output_key)
        global_input_keys += [self.get_cache_name(output_key, model)]
        local_input_keys = list(model.input_keys) + ["cache"]

        return model, global_input_keys, local_input_keys

    def is_static_scalar(self, key: str) -> bool:
        is_static = super().is_static_scalar(key)
        return is_static and not key.endswith(
            "_cache"
        )  # temporarily added until cache removed

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

        if formula_key in self.backend.array_creation_funcs:
            self.add_partial_function(formula_key)

        if is_make_array_required(
            self.pm.data[output_key]
        ) or is_type_adjustment_required(self.pm.data, g_input_keys):
            generated_fn = ast.Call(
                func=ast.Name(id="make_array", ctx=ast.Load()),
                args=[generated_fn],
                keywords=[],
            )
            self.add_partial_function("make_array")

        return ast.Assign(targets, generated_fn), used_keys | _used_keys

    def create_primitive_call_targets(
        self, output_key: str, model: PrimitiveModel, inference: bool
    ) -> tuple[list[ast.expr | ast.Name], set[str]]:
        targets: list[ast.expr | ast.Name] = []

        fn_targets, used_keys = super().create_primitive_call_targets(
            output_key, model, inference
        )

        targets += fn_targets

        if not self.pm.inference:
            # TODO: Change this with cache refactor
            cache_name = output_key + f"_{model.cache_name}"
            used_keys.add(cache_name)
            targets.append(
                ast.Subscript(
                    value=ast.Name(id=cache_name, ctx=ast.Load()),
                    slice=ast.Constant(value=PrimitiveModel.output_key),
                    ctx=ast.Store(),
                )
            )

        return targets, used_keys

    def get_cache_name(self, output_key: str, model: PrimitiveModel) -> str:
        cache_name = "_".join([output_key, model.cache_name])
        if cache_name not in self.pm.flat_graph.all_data:
            self.add_cache(model, output_key)

        return cache_name

    def add_cache(self, model: PrimitiveModel, output_key: str) -> None:
        cache_name = "_".join([output_key, model.cache_name])
        cache_value: dict[str, Any] | None = None if self.pm.inference else {}
        # Create a scalar for caches in manualgrad backend.
        self.pm.flat_graph.update_data(
            {cache_name: IOHyperEdge(dict | None, cache_value)}
        )

    def generate_evaluate_gradients(
        self, ignore_grad_keys: set[str]
    ) -> ast.FunctionDef:
        input_body: list[ast.stmt] = []
        function_body: list[ast.stmt] = []
        used_keys: set[str] = set()

        all_ignored_keys = ignore_grad_keys | self.pm.flat_graph.all_static_keys

        # TODO: Is this should be here?
        # Seperate ignored keys into two types of weak and strict ignored keys.
        weak_ignored_keys = {
            key
            for key in all_ignored_keys
            if key in self.pm.data
            and self.pm.data[key].is_tensor
            and find_intersection_type(self.pm.data[key].value_type, float)
        }

        strict_ignored_keys = all_ignored_keys - weak_ignored_keys

        ignore_grad_keys, _ = self.pm.flat_graph.infer_ignore(
            weak_ignored_keys,
            self.pm._output_keys,
            strict_ignored_keys,
            update_graph=False,
        )

        possible_loss_keys = {FinalCost, LossKey}
        if possible_loss_keys & self.pm._output_keys:
            ignore_grad_keys.update(
                self.pm._output_keys
                - possible_loss_keys
                - (
                    self.pm.flat_graph.all_source_keys
                    | {
                        value
                        for key, value in self.pm.flat_graph.output_dict.items()
                        if key != value
                    }
                )
            )

        # Move gradients back for keys in alias_map(pruned or optimized out keys)
        for target_key, source_key in self.pm.flat_graph.output_dict.items():
            if target_key == source_key:
                continue
            if target_key not in ignore_grad_keys:
                source = ast.Subscript(
                    value=ast.Name(id="gradients", ctx=ast.Load()),
                    slice=ast.Constant(
                        "_" + target_key
                        if keyword.iskeyword(target_key)
                        or target_key in self.backend.primitive_function_dict
                        else target_key
                    ),
                    ctx=ast.Load(),
                )

                target = ast.Subscript(
                    value=ast.Name(id="gradients", ctx=ast.Load()),
                    slice=ast.Constant(
                        "_" + source_key
                        if keyword.iskeyword(source_key)
                        or source_key in self.backend.primitive_function_dict
                        else source_key
                    ),
                    ctx=ast.Load(),
                )

                assign = ast.AugAssign(target=target, op=ast.Add(), value=source)
                function_body.append(assign)

        for output_key in reversed(self.pm.flat_graph.topological_order):
            # Iterate over Primitive models in topological order to add their formula.
            model = self.pm.flat_graph.get_model(output_key)

            output_key = self.pm.flat_graph.connections[output_key].key
            inputs = list(self.pm.flat_graph.get_source_keys(output_key))

            # Check if the model is disposable.
            if model.disposable:
                raise Exception(
                    f"{model.__class__.__name__} is a disposable model."
                    " Disposable models have no grad formulas!"
                )

            # Get primitive function inputs order
            primitive_function = (
                self.backend.primitive_function_dict[model.formula_key]
                if model.formula_key in self.backend.primitive_function_dict
                else self.backend.registered_primitives[model.formula_key]
            )
            local_to_global_dict = {
                key: value
                for key, value in zip(
                    list(model.input_keys) + ["cache"], inputs, strict=False
                )
            }
            args, kwargs = prepare_function_args(
                self.pm.flat_graph.cached_data,
                primitive_function,
                local_to_global_dict,
                self.backend.array_creation_funcs,
                False,
            )

            # Get local keys in ordered
            global_to_local_dict: dict[str, list[str]] = {}
            for key, value in zip(
                list(model.input_keys) + ["cache"], inputs, strict=False
            ):
                global_to_local_dict.setdefault(value, [])
                global_to_local_dict[value].append(key)
            primitive_global_inputs = [
                key for keys in args.values() for key in keys if "cache" not in key
            ]
            primitive_global_inputs += [
                key for key in kwargs.values() if "cache" not in key
            ] + [local_to_global_dict["cache"]]
            primitive_local_inputs: list[str] = [
                global_to_local_dict[key].pop(0) for key in primitive_global_inputs
            ]

            # Reorder global keys wrt primitive evaluate function local keys order
            model_local_inputs = list(model.input_keys) + ["cache"]
            _inputs = [
                inputs[model_local_inputs.index(local_key)]
                for local_key in primitive_local_inputs
            ]
            local_input_keys = [*primitive_local_inputs, "output_gradient", "idx"]
            global_input_keys = _inputs + ["output_gradient", "idx"]

            # TODO: Handle ignore gradient keys (models) and
            for idx, global_input_key in enumerate(global_input_keys[:-2]):
                if (
                    global_input_key
                    in ignore_grad_keys | self.pm.flat_graph.runtime_static_keys
                ):
                    continue

                grad_fn = self.backend.primitive_grad_function_dict.get(
                    model.grad_formula
                )
                if grad_fn is None:
                    grad_fn = self.backend.registered_primitives_grad_fn.get(
                        model.grad_formula
                    )

                if grad_fn is None:
                    raise NotImplementedError(
                        f"Primitive {model.formula_key} does not have vjp "
                        "implementation!"
                    )

                grad_arg = ast.Subscript(
                    value=ast.Name(id="gradients", ctx=ast.Load()),
                    slice=ast.Constant(
                        "_" + output_key
                        if keyword.iskeyword(output_key)
                        or output_key in self.backend.primitive_function_dict
                        else output_key
                    ),
                    ctx=ast.Load(),
                )
                idx_arg = ast.Constant(value=idx, kind=None)

                default_args: dict[str, ast.expr] = {
                    "output_gradient": grad_arg,
                    "idx": idx_arg,
                }
                generated_fn, _used_keys = self.create_primitive_call(
                    grad_fn, local_input_keys, global_input_keys, default_args
                )

                if self.backend.is_manualgrad and is_make_array_required(
                    self.pm.data[output_key]
                ):
                    generated_fn = ast.Call(
                        func=ast.Name(id="make_array", ctx=ast.Load()),
                        args=[generated_fn],
                        keywords=[],
                    )
                    self.add_partial_function("make_array")

                if (
                    keyword.iskeyword(global_input_key)
                    or global_input_key in self.backend.primitive_function_dict
                ):
                    manipulated_key = "_" + global_input_key
                else:
                    manipulated_key = global_input_key

                if (
                    (in_shape := self.pm.data[global_input_key].shape) is not None
                    and (out_shape := self.pm.data[output_key].shape) is not None
                    and check_repr_inequality(in_shape, out_shape)
                ):
                    generated_fn = ast.Call(
                        func=ast.Name(id="accumulate_grads", ctx=ast.Load()),
                        args=[
                            generated_fn,
                            ast.Name(manipulated_key),
                            ast.Name(global_input_keys[-3]),
                            idx_arg,
                        ],
                        keywords=[],
                    )

                target = ast.Subscript(
                    value=ast.Name(id="gradients", ctx=ast.Load()),
                    slice=ast.Constant(global_input_key),
                    ctx=ast.Load(),
                )

                function_body.append(
                    ast.AugAssign(target=target, op=ast.Add(), value=generated_fn)
                )

                used_keys |= _used_keys - {"output_gradient", "idx"}

        for key in sorted(used_keys):
            if (
                key
                in self.pm.flat_graph.all_target_keys
                | self.pm.flat_graph.cached_data.keys()
            ):
                dict_type = "cache"
            elif key in self.pm.flat_graph.runtime_static_keys:
                dict_type = "data"
            else:
                dict_type = "params"
            """If cached value is not a tensor, do not append it to code"""
            if not self.is_static_scalar(key):
                self.append_inputs(input_body, key, dict_type)

        ast_args = [
            ast.arg("params"),
            ast.arg("gradients"),
            ast.arg("data"),
            ast.arg("cache"),
        ]
        arguments = ast.arguments(
            posonlyargs=[],
            args=ast_args,
            defaults=[],
            kwonlyargs=[],
            kw_defaults=[],
            vararg=None,
            kwarg=None,
        )

        if len(function_body) == 0:
            function_body = [ast.Pass()]

        func_def = ast.FunctionDef(
            name="evaluate_gradients",
            args=arguments,
            body=input_body + function_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
            type_params=[],
            lineno=1,
            col_offset=0,
        )

        return ast.fix_missing_locations(func_def)
