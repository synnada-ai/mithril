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
from itertools import product
from types import FunctionType, GenericAlias, UnionType
from typing import Any


def align_shapes(all_dicts: list[dict[Any, Any]]) -> None:
    """Align all shapes given in the list

    Examples:

    >>> list1 = [{"input": [3, 4, 5], "output": [12, 3, 1]}, {"left": [21], "right":
        [1, 14], "output": [1,121, 43, 12]}]
    >>> align_shapes(list1)
    >>> list1
    [
        {
            "input": "[ 3,   4,  5]",
           "output": "[12,   3,  1]"
        },
        {
             "left": "[21]",
            "right": "[ 1,  14]",
           "output": "[ 1, 121, 43, 12]"
        }
    ]

    Args:
        all_dicts (_type_): list of dictionaries that include key to shape information
    """
    if all_dicts:
        for shape_dict in all_dicts:
            for key, value in shape_dict.items():
                if value and isinstance(value[0], list):
                    shape_dict[key] = [[str(val) for val in lst] for lst in value]
                elif value is None:
                    shape_dict[key] = "--"
                else:
                    shape_dict[key] = [[str(val) for val in value]]
        max_val_dict = {}
        all_values: list[list[str]] = [
            val
            for shape_dict in all_dicts
            for lst in shape_dict.values()
            for val in lst
            if lst != "--"
        ]
        reversed_all_values: list[list[str]] = []
        for value in all_values:
            reversed_all_values.append(value[::-1])
        if all_values:
            for idx in range(len(max(all_values, key=len))):
                max_val_dict[idx] = len(
                    max(
                        [
                            value[idx]
                            for value in reversed_all_values
                            if idx < len(value)
                        ],
                        key=len,
                    )
                )
        for shape_dict in all_dicts:
            for key, value in shape_dict.items():
                if value != "--":
                    t_list: list[str] = []
                    for lst in value:
                        reversed_lst = lst[::-1]
                        reversed_lst = [
                            val.rjust(max_val_dict[idx])
                            for idx, val in enumerate(reversed_lst)
                        ]
                        t_list.append(str(reversed_lst[::-1]).replace("'", ""))
                    shape_dict[key] = t_list
                else:
                    shape_dict[key] = ["--"]


class GeneratedFunction:
    """We are generating evaluate and evaluate_gradient functions using the Abstract
    Syntax Tree (AST) module, but these functions are not picklable by default. To
    address this, the following class helps to pickle these dynamically generated
    functions by converting the AST into a serializable format and providing custom
    serialization and deserialization methods.
    """

    def __init__(self, func: FunctionType, metadata: dict[str, str]):
        self.func = func
        self.metadata: dict[str, str] = metadata

    def __reduce__(self) -> tuple[Callable[[str, str], Any], tuple[str, str]]:
        # Serialize the function code and metadata
        fn_name = self.metadata["fn_name"]
        source_code = self.metadata["source"]
        return (self._unpickle, (source_code, fn_name))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    @staticmethod
    def _unpickle(source_code: str, fn_name: str) -> FunctionType:
        # Compile the code string back to a code object
        code = compile(source_code, "<string>", "exec")
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        func = namespace[fn_name]
        return func


def infer_all_possible_types(
    type_def: type | UnionType | GenericAlias,
) -> set[type | UnionType | GenericAlias]:
    possible_types = {type_def}
    if type(type_def) is UnionType:
        sub_types = list(type_def.__args__)
        for sub_type in sub_types:
            possible_types.update(infer_all_possible_types(sub_type))
    elif isinstance(type_def, GenericAlias):
        seq_type: type[tuple[Any, ...]] | type[list[Any]] = type_def.__origin__
        possible_seq_type: type
        sub_types = list(type_def.__args__)
        if seq_type is tuple:
            possible_seq_type = tuple
            if len(sub_types) == 2 and sub_types[-1] == ...:
                for typ in infer_all_possible_types(sub_types[0]):
                    _types: set[Any] = {
                        possible_seq_type[typ, ...],
                        possible_seq_type[typ],
                    }
                    possible_types.update(_types)
            else:
                type_probs = [infer_all_possible_types(typ) for typ in sub_types]
                all_possible_types = {
                    possible_seq_type[i] for i in product(*type_probs)
                }
                possible_types.update(all_possible_types)
        else:
            possible_seq_type = list
            arg = sub_types[0]
            all_args = infer_all_possible_types(arg)
            for arg in all_args:
                possible_types.add(possible_seq_type[arg])
    return possible_types


def find_list_base_type(
    type_def: type[list[Any]]
    | type[float]
    | type[int]
    | type[bool]
    | UnionType
    | GenericAlias,
) -> set[type | UnionType]:
    result: set[type | UnionType] = set()
    if isinstance(type_def, GenericAlias):
        origin: type[list[Any]] | type[tuple[Any, ...]] = type_def.__origin__
        if origin is list:
            # Means there exists recursive list type.
            for arg in type_def.__args__:
                result.update(find_list_base_type(arg))
    elif isinstance(type_def, UnionType):
        for arg in type_def.__args__:
            result.update(find_list_base_type(arg))
    elif type_def in (int, float, bool):
        result.add(type_def)
    else:
        raise Exception(
            f"{type_def} type is not supported in recursive list. Only int, float or "
            "bool types are supported."
        )

    return result


def find_list_depth(arg_type: type | UnionType | GenericAlias) -> int:
    max_depth = 0
    if (
        origin := getattr(arg_type, "__origin__", None)
    ) is not None and origin is not list:
        return 0
    elif (args := getattr(arg_type, "__args__", None)) is not None:
        initial_depth = 1 if getattr(arg_type, "__origin__", None) is list else 0
        for arg in args:
            arg_depth = find_list_depth(arg)
            max_depth = max(max_depth, arg_depth) + initial_depth
    return max_depth


def is_union(typ: type | UnionType | GenericAlias) -> bool:
    if isinstance(typ, GenericAlias):
        if ... in typ.__args__:
            return True
        return any(is_union(subtype) for subtype in typ.__args__)
    else:
        return typ is tuple or typ is list or type(typ) is UnionType


def merge_dicts(
    dict1: dict[str, set[int]], dict2: dict[str, set[int]]
) -> dict[str, set[int]]:
    base_dict = [dict1, dict2][len(dict1) < len(dict2)]
    iter_dict = dict1 if base_dict == dict2 else dict2
    for key, value in iter_dict.items():
        base_dict.setdefault(key, set()).update(value)
    return base_dict


def sort_type(
    type1: type | UnionType | GenericAlias,
) -> type | UnionType | GenericAlias:
    """
    Returns the sorted type of UnionTypes
    """

    def sort_fn(type: type | UnionType | GenericAlias) -> str:
        if isinstance(type, GenericAlias):
            return type.__origin__.__name__
        else:
            return str(type)

    if isinstance(type1, UnionType):
        types = []
        for sub_type in type1.__args__:
            types.append(sort_type(sub_type))
        sorted_types = sorted(types, key=sort_fn, reverse=True)
        current_type = sorted_types.pop()
        while sorted_types:
            current_type |= sorted_types.pop()
        return current_type

    elif isinstance(type1, GenericAlias):
        main_type = type1.__origin__
        types = []
        for sub_type in type1.__args__:
            types.append(sort_type(sub_type))
        return main_type[*types]  # type: ignore

    else:
        return type1
