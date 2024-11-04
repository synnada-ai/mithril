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

from functools import reduce
from itertools import product
from types import FunctionType, GenericAlias, UnionType
from typing import Any


class NestedListType:
    """
    Represents a nested list type.

    Attributes:
        base_type (type): The base type of the nested list.
    """

    __slots__ = "base_type"
    base_type: type

    def __init__(self, base_type):
        self.base_type = base_type


def define_unique_names(models):
    model_name_dict = {}
    single_model_dict = {}
    model_count_dict: dict[str, int] = {}

    for model in models:
        class_name = model.__class__.__name__
        if model_count_dict.setdefault(class_name, 0) == 0:
            single_model_dict[class_name] = model
        else:
            single_model_dict.pop(class_name, None)
        model_name_dict[model] = (
            str(class_name) + "_" + str(model_count_dict[class_name])
        )
        model_count_dict[class_name] += 1

    for m in single_model_dict.values():
        model_name_dict[m] = str(m.__class__.__name__)
    return model_name_dict


def list_shape(ndarray: list[float | int] | float | int) -> list[int]:
    if isinstance(ndarray, list | tuple):
        # More dimensions, so make a recursive call
        outermost_size = len(ndarray)
        row_shape = list_shape(ndarray[0])
        for item in ndarray[1:]:
            shape = list_shape(item)
            if row_shape != shape:
                raise ValueError(
                    f"Shape mismatch: expected {row_shape}, but got {shape}. The list "
                    "should not be ragged."
                )

        return [outermost_size, *row_shape]
    else:
        # No more dimensions, so we're done
        return []


def get_unique_types(arg: list | tuple):
    # Recursively looks all items in nested sequence
    # and returns all unique types in it.
    ...


def align_shapes(all_dicts: list[dict]) -> None:
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
        all_values = [
            val
            for shape_dict in all_dicts
            for lst in shape_dict.values()
            for val in lst
            if lst != "--"
        ]
        reversed_all_values = []
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
                    t_list = []
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

    def __init__(self, func: FunctionType, metadata: dict):
        self.func = func
        self.metadata = metadata

    def __reduce__(self):
        # Serialize the function code and metadata
        fn_name = self.metadata["fn_name"]
        source_code = self.metadata["source"]
        return (self._unpickle, (source_code, fn_name))

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @staticmethod
    def _unpickle(source_code, fn_name):
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
        seq_type: type[tuple] | type[list] = type_def.__origin__
        possible_seq_type: type
        sub_types = list(type_def.__args__)
        if seq_type is tuple:
            possible_seq_type = tuple
            if len(sub_types) == 2 and sub_types[-1] == ...:
                for typ in infer_all_possible_types(sub_types[0]):
                    _types = {possible_seq_type[typ, ...], possible_seq_type[typ]}
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
    type_def: type[list]
    | type[float]
    | type[int]
    | type[bool]
    | UnionType
    | NestedListType
    | GenericAlias,
) -> set[type]:
    result = set()
    if isinstance(type_def, NestedListType):
        result.add(type_def.base_type)
    elif isinstance(type_def, GenericAlias):
        origin: type[list] | type[tuple] = type_def.__origin__
        if origin is list:
            # Means there exists recursive list type.
            for arg in type_def.__args__:
                result.update(find_list_base_type(arg))
    elif isinstance(type_def, UnionType):
        for arg in type_def.__args__:
            result.update(find_list_base_type(arg))
    elif type_def not in (int, float, bool, list):
        raise Exception(
            f"{type_def} type is not supported in recursive list. Only int, float or "
            "bool types are supported."
        )
    elif type_def in (int, float, bool):
        result.add(type_def)
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


def find_intersection_type(
    type_1: type | UnionType | GenericAlias | NestedListType,
    type_2: type | UnionType | GenericAlias | NestedListType,
) -> type | UnionType | None | NestedListType:
    # If NestedListTtype type exists in any arguments
    # first unroll NestedListTtype type into a certain
    # depth same as the other type if other is non-NestedListType.
    # Else if both are NestedListType, constraint their base type
    # into their intersection.
    nested_types: list[NestedListType] = []
    if isinstance(type_1, NestedListType):
        nested_types.append(type_1)
    if isinstance(type_2, NestedListType):
        nested_types.append(type_2)

    if len(nested_types) == 1:
        regular_type, nested_type = (
            (type_2, type_1) if isinstance(type_1, NestedListType) else (type_1, type_2)
        )
        assert not isinstance(regular_type, NestedListType)
        assert isinstance(nested_type, NestedListType)
        # If regular type is list, we can not know its depth, so simply
        # return nested type.
        if regular_type is list:
            return nested_type

        # Find depth of the list type and unroll nested type to that depth.
        depth = find_list_depth(regular_type)
        base: type | UnionType = nested_type.base_type
        final = base
        for _ in range(depth):
            base = list[base]  # type: ignore[valid-type] # mypy does not accept list[arg]
            final |= base
        # Assign new type representation to the corresponding argument.
        if type(type_1) is NestedListType:
            type_1 = final
        else:
            type_2 = final
    elif len(nested_types) == 2:
        # Means both types are NestedListType.
        # Constrain larger union type to the smaller one.
        base_type_1 = nested_types[0].base_type
        base_type_2 = nested_types[1].base_type
        return NestedListType(find_intersection_type(base_type_1, base_type_2))

    # First find direct intersections.
    subtypes_1 = set(type_1.__args__) if type(type_1) is UnionType else {type_1}
    subtypes_2 = set(type_2.__args__) if type(type_2) is UnionType else {type_2}
    intersect = subtypes_1 & subtypes_2

    # if one of the subtypes have list or tuple without an origin (without square
    # brackets, ex: tuple), look for other set if it contains corresponding type
    # with origin (ex: tuple[int, int]) if the set contains it, add that type with
    # origin (since it contains more information)

    for s_types in (subtypes_1, subtypes_2):
        other_set = subtypes_2 if s_types == subtypes_1 else subtypes_1
        for orig_type in (list, tuple):
            if orig_type in s_types:
                for typ in other_set:
                    if isinstance(typ, GenericAlias) and typ.__origin__ == orig_type:
                        intersect.add(typ)

    # Take tuple types from remaining sets and find intesection types
    # of all consistent pairs of cartesian product.
    for typ_1 in subtypes_1.difference(intersect):
        if not isinstance(typ_1, GenericAlias):
            continue

        args_1 = typ_1.__args__
        assert typ_1.__origin__ is tuple or typ_1.__origin__ is list
        for typ_2 in subtypes_2.difference(intersect):
            if not isinstance(typ_2, GenericAlias):
                continue

            args_2 = typ_2.__args__
            assert typ_2.__origin__ is tuple or typ_2.__origin__ is list
            if typ_1.__origin__ == typ_2.__origin__:
                if len(args_1) == 0 or len(args_2) == 0:
                    # if one of the lengths of the args_1 and args_2 are zero,
                    # this means one of the types with origin are empty list or tuple,
                    # in that case, take the empty one (tuple[()], or list[()]) as
                    # intersection type
                    common = typ_1.__origin__[()]  # type: ignore

                elif typ_1.__origin__ is tuple:
                    ellipsis_1 = ... in args_1
                    ellipsis_2 = ... in args_2
                    common = False
                    if ellipsis_1 and ellipsis_2:
                        common = find_intersection_type(args_1[0], args_2[0])
                        if common:
                            common = [common, ...]
                    elif ellipsis_1:
                        # Remove ellipsis and replace it with base type
                        # as many times as length of args_2
                        common = [
                            find_intersection_type(args_1[0], args_2[i])
                            for i in range(len(args_2))
                        ]
                    elif ellipsis_2:
                        # Remove ellipsis and replace it with base type
                        # as many times as length of args_1
                        common = [
                            find_intersection_type(args_1[i], args_2[0])
                            for i in range(len(args_1))
                        ]
                    elif len(args_1) == len(args_2):
                        common = [
                            find_intersection_type(args_1[i], args_2[i])
                            for i in range(len(args_1))
                        ]
                    if common and None not in common:
                        intersect.add(tuple[*common])  # type: ignore

                elif typ_1.__origin__ is list:
                    if len(args_2) > 1 or len(args_1) > 1:
                        raise TypeError(
                            "args of type list cannot take more than 1 element"
                        )
                    else:
                        common = find_intersection_type(args_1[0], args_2[0])
                    if common:
                        intersect.add(list[common])  # type: ignore[valid-type] # mypy does not accept list[arg]
    if intersect:
        result = reduce(lambda x, y: x | y, intersect)
        return result
    return None


def find_type(connection) -> type:
    if isinstance(connection, tuple | list):
        element_types: list[Any] = [find_type(elem) for elem in connection]
        if isinstance(connection, tuple):
            return tuple[*element_types]  # type: ignore
        else:
            result: UnionType | type = reduce(lambda x, y: x | y, element_types)
            return list[result]  # type: ignore
    else:
        return type(connection)


def is_union(typ):
    if hasattr(typ, "__origin__"):
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


def sort_type(type1: type | UnionType | GenericAlias):
    """
    Returns the sorted type of UnionTypes
    """

    def sort_fn(type):
        if hasattr(type, "__origin__"):
            return type.__origin__.__name__
        else:
            return type.__name__

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
