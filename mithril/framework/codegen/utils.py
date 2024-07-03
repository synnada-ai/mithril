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

from ...backends.backend import Backend
from ..common import ShapeNode

key_map_type = dict[str, str]


# TODO: This name misleads
def partial_array_creation_func(backend: Backend, formula_key: str) -> ast.stmt:
    kwargs = [ast.keyword(arg="precision", value=ast.Constant(value=backend.precision))]

    # We don't need device in manulgrad(Numpy)
    if not backend.is_manualgrad:
        kwargs.append(
            ast.keyword(arg="device", value=ast.Constant(value=backend._device))
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
    arg_key: str, reserved_keys: list[str], defaults: dict[str, ast.Name] | None = None
) -> ast.Name:
    if defaults is not None and arg_key in defaults:
        return defaults[arg_key]

    if keyword.iskeyword(arg_key) or arg_key in reserved_keys:
        arg_key = f"_{arg_key}"

    return ast.Name(arg_key)


def convert_to_ast_kwarg(
    arg_key: str, value: str, defaults: dict[str, ast.expr]
) -> ast.keyword:
    if arg_key in defaults:
        _value = defaults[arg_key]
    else:
        _value = ast.Name(id=value, ctx=ast.Load())

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
