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

from ...backends.backend import Backend
from ...types import DataType
from ..common import ShapeNode


# TODO: This name misleads
def partial_array_creation_func(
    backend: Backend[DataType], formula_key: str
) -> ast.stmt:
    kwargs = [
        ast.keyword(arg="default_dtype", value=ast.Constant(value=backend._dtype.name))
    ]

    if backend.CODEGEN_CONFIG.SPECIFY_DEVICE:
        kwargs.append(
            ast.keyword(arg="device", value=ast.Constant(value=backend.get_device()))
        )

    partial_fn_call = ast.Call(
        func=ast.Name(id="partial", ctx=ast.Load()),
        args=[ast.Name(formula_key)],
        keywords=kwargs,
    )
    partial_fn = ast.Assign(
        targets=[ast.Name(id=formula_key, ctx=ast.Store())], value=partial_fn_call
    )

    return partial_fn


def convert_to_ast_arg(
    key: str, arg_name: ast.Name, defaults: dict[str, ast.Name] | None = None
) -> ast.Name:
    if defaults is not None and key in defaults:
        return defaults[key]

    return arg_name


def convert_to_ast_kwarg(
    arg_key: str, value: ast.Name, defaults: dict[str, ast.expr]
) -> ast.keyword:
    _value = defaults.get(arg_key, value)

    kwarg = ast.keyword(arg=arg_key, value=_value)
    return kwarg


# TODO: Move these are to common
def check_repr_inequality(shape_1: ShapeNode, shape_2: ShapeNode) -> bool:
    assert shape_1 is not None and shape_2 is not None

    key1_shape = next(iter(shape_1.reprs))
    key2_shape = next(iter(shape_2.reprs))
    meta_list_1 = [key.metadata for key in key1_shape.prefix + key1_shape.suffix]
    meta_list_2 = [key.metadata for key in key2_shape.prefix + key2_shape.suffix]
    return (meta_list_1 != meta_list_2) or (key1_shape.root != key2_shape.root)
