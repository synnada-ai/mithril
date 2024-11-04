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

from collections.abc import Callable
from copy import deepcopy

from ..framework.common import Scalar, Tensor

key_map_type = dict[str, str]


def prepare_function_args(
    data: dict[str, Tensor | Scalar],
    function: Callable,
    inputs: key_map_type,
    array_creation_funcs: list[str],
    reduce_with_defaults: bool = True,
) -> tuple[dict[str, list[str]], key_map_type]:
    formula_key = function.__name__
    code_obj = function.__code__

    fn_args_dict = {
        arg_name: False for arg_name in code_obj.co_varnames[: code_obj.co_argcount]
    }

    # Check if variadic positional argument exists
    if code_obj.co_flags & 0x04 == 0x04:
        idx = code_obj.co_argcount + code_obj.co_kwonlyargcount
        fn_args_dict[code_obj.co_varnames[idx]] = True

    fn_kwarg_keys = list(
        code_obj.co_varnames[
            code_obj.co_argcount : code_obj.co_argcount + code_obj.co_kwonlyargcount
        ]
    )

    # Array creation functions requires device and precision to properly create tensor
    if formula_key in array_creation_funcs:
        # Remove precision and device from kwarg_keys we will partially provide them
        fn_args_dict.pop("precision", None)
        if "precision" in fn_kwarg_keys:
            fn_kwarg_keys.remove("precision")

        fn_args_dict.pop("device", None)
        if "device" in fn_kwarg_keys:
            fn_kwarg_keys.remove("device")

    # Prepare arguments
    fn_kwarg_dict, removed_kwarg_dict = create_kwarg_dict(
        data, fn_kwarg_keys, function, inputs, reduce_with_defaults
    )
    fn_args_mapping = reorganize_args(
        data,
        fn_args_dict,
        set(fn_kwarg_dict.values()) | set(removed_kwarg_dict.values()),
        function,
        inputs,
        reduce_with_defaults,
    )

    return fn_args_mapping, fn_kwarg_dict


def create_kwarg_dict(
    data: dict[str, Tensor | Scalar],
    kwarg_keys: list[str],
    function: Callable,
    inputs: key_map_type,
    reduce_with_defaults: bool,
) -> tuple[key_map_type, key_map_type]:
    kwarg_keys_dict: key_map_type = {
        kwarg_key: inputs[kwarg_key] for kwarg_key in kwarg_keys
    }
    removed_kwargs_dict: key_map_type = {}

    kwdefaults = function.__kwdefaults__

    if kwdefaults is not None and reduce_with_defaults:
        for key, value in kwdefaults.items():
            provided_value = data[kwarg_keys_dict[key]].value
            if value == provided_value and type(value) is type(provided_value):
                removed_kwargs_dict[key] = kwarg_keys_dict[key]
                kwarg_keys_dict.pop(key)

    return kwarg_keys_dict, removed_kwargs_dict


def reorganize_args(
    data: dict[str, Tensor | Scalar],
    arg_keys: dict[str, bool],
    kwarg_keys: list[str] | set[str],
    function: Callable,
    inputs: key_map_type,
    reduce_with_defaults: bool,
) -> dict[str, list[str]]:
    defaults = function.__defaults__
    formula_key = function.__name__

    local_input_keys = list(inputs.keys())
    inputs = deepcopy(inputs)
    organized_arguments: dict[str, list[str]] = {}

    for idx, (name, is_variadic) in enumerate(arg_keys.items()):
        if "cache" in name:
            # TODO: Refactor here
            provided_value = data[inputs[name]].value
            if (
                reduce_with_defaults
                and idx == len(arg_keys) - 1
                and defaults
                and provided_value == defaults[-1]
            ):
                continue

            outer_names = [inputs[name]]
            inputs.pop(name)

        # If the argument variadic, then it takes rest of the inputs
        elif is_variadic:
            outer_names = [
                input for input in inputs.values() if input not in kwarg_keys
            ]

        elif name not in local_input_keys:
            raise RuntimeError(
                f"Primitive '{formula_key}' input keys:'{local_input_keys}' and"
                f" backend function input keys: '{arg_keys.keys()}' are not matching!"
            )

        else:
            outer_names = [inputs[name]]
            inputs.pop(name)

        organized_arguments[name] = outer_names

    return organized_arguments


def is_make_array_required(data: Tensor | Scalar):
    if isinstance(data, Tensor):
        _temp_shape = next(iter(data.shape.reprs))
        # It is needed to guarantee that Tensor is at least one dimensional.
        # Note that having variadic field does not imply greater dimensionality
        # as variadic field could also include no uniadics.
        return not (_temp_shape.prefix or _temp_shape.suffix)
    else:
        return False
