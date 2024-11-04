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

import math
from collections.abc import Callable, Sequence
from functools import reduce
from itertools import product, zip_longest
from types import EllipsisType, GenericAlias, NoneType, UnionType
from typing import get_origin

from ..utils.type_utils import (
    is_axis_reduce_type,
    is_axis_reverse_type,
    is_index_type,
    is_list_int,
    is_list_int_or_none,
    is_padding_type,
    is_tuple_int,
    is_tuple_int_or_none,
    is_tuple_of_two_ints,
)
from ..utils.utils import PaddingType, find_dominant_type
from .common import (
    DNF,
    TBD,
    Constant,
    ConstrainResultType,
    NestedListType,
    PossibleValues,
    Scalar,
    ShapeRepr,
    Tensor,
    ToBeDetermined,
    Uniadic,
    UniadicRecord,
    Updates,
    UpdateType,
    Variadic,
)
from .utils import (
    find_intersection_type,
    find_list_base_type,
    is_union,
    list_shape,
)

__all__ = [
    "general_tensor_type_constraint",
    "floor_divide_type_constraint",
    "scalar_slice_type_constraint",
    "scalar_item_type_constraint",
    "bcast",
    "bcast_matrix_mult",
    "sliding_window_1d_constraints",
    "sliding_window_2d_constraints",
    "flatten_constrains",
    "concat_constraints",
    "reduce_constraints",
    "reverse_constraints",
    "polynomial_features_constraints",
    "arange_constraints",
    "broadcast_to_constraints",
    "reshape_constraints",
    "squeeze_constraints",
    "size_constraints",
    "shape_constraints",
    "swap_axes_constraints",
    "to_tensor_constraints",
    "tensor_to_list_constraints",
    "to_list_constraints",
    "where_constrains",
    "validate_bcast",
    "eye_constraints",
    "item_constraints",
    "scalar_item_constraints",
    "to_tuple_constraints",
    "tensor_item_constraints",
    "tensor_slice_constraints",
    "tensor_to_list_type_constraint",
    "reduce_type_constraint",
    "type_constraints",
    "post_process_map",
    "padding_1d_constraint",
    "padding_2d_constraint",
    "stride_constraint",
    "tuple_converter_constraint",
    "conv_1d_constraints",
    "conv_2d_constraints",
]


# Below functions are used in various constraints.
def prod_fn(a, b):
    return (a if isinstance(a, int) else a.value) * (
        b if isinstance(b, int) else b.value
    )


def is_repr_known(repr) -> bool:
    return (
        repr.root is None
        and repr.prefix
        and all([uni.value is not None for uni in repr.prefix])
    )


def create_union_type(
    *types: type | UnionType | GenericAlias,
) -> type | UnionType | GenericAlias:
    if len(types) > 0:
        result = types[0]
        for typ in types[1:]:
            result |= typ
        return result
    else:
        raise TypeError("At least one type should be given!")


def _reduce_union_type(
    output_type: type | UnionType | GenericAlias | NestedListType,
    arg_type: UnionType,
) -> None | set[type]:
    eliminated: set[type[int] | type[float] | type[bool]] | None = {float}
    if output_type is float:
        eliminated = None
    if output_type is bool:
        eliminated = {float, int}
    if not eliminated:
        return None
    new_type: set[type] = set(arg_type.__args__) - eliminated
    if not new_type:
        # Means arg can not take any bool value which is an error.
        raise TypeError(
            f"One of arguments is of type {arg_type} which is not possible for {type} "
            "type output!"
        )
    return new_type


def general_tensor_type_constraint(*args: Scalar | Tensor):
    # NOTE: Assumes first argument is always output as other constraints.
    # Also requires all types of args consists of any combination of
    # float, int and bool. For instance, int | float is an acceptable type
    # but tuple or list[int] are not.
    status = False
    updates = Updates()
    output, *inputs = args
    arg_types: set[type | UnionType | NestedListType | GenericAlias] = set()
    all_possible_types: set[type | UnionType | NestedListType | GenericAlias] = set()
    union_types: set[tuple[Scalar | Tensor, UnionType]] = set()
    # Set all different types and also Union types in input args.
    for arg in inputs:
        typ = arg._type
        arg_types.add(typ)
        if isinstance(typ, UnionType):
            union_types.add((arg, typ))
            all_possible_types.update(set(typ.__args__))
        else:
            all_possible_types.add(typ)

    # Check existance of any possible unsupported types
    if unsupported := (all_possible_types - {int, float, bool}):
        raise TypeError(f"Possible Unsupported type(s) ({unsupported}) detected!")

    # Try reverse type inference first.
    if not isinstance(output._type, UnionType):
        # Means output has a definite type (int, float or bool).
        out_exists = output._type in arg_types
        related_unions = {
            pair for pair in union_types if output._type in pair[1].__args__
        }
        if not (out_exists or related_unions):
            # At least one of arg_types or UnionTypes must contain
            # output type.
            raise TypeError(
                f"None of arguments consist of type {output._type} which is the "
                "exact output type!"
            )
        elif not out_exists and len(related_unions) == 1:
            # If only one of them contains output type, enforce this union
            # type to be same as output type.
            arg = related_unions.pop()[0]
            updates |= arg.set_type(output._type)
            status = True
        # elif not out_exists:
        # Update Union type arguments.
        for pair in related_unions:
            arg, arg_type = pair
            new_type = _reduce_union_type(output._type, arg_type)
            if new_type is not None:
                updates |= arg.set_type(create_union_type(*new_type))
        if not out_exists:
            # If any one of inputs became same type as output, set
            # status True.
            for pair in related_unions:
                if pair[0]._type == output._type:
                    status = True
                    break
    elif output._type == int | bool:
        if float in arg_types:
            raise TypeError(
                "One of arguments value is float which is not possible when output "
                "type is int | bool"
            )
        else:
            # We can eliminate any float possibility from Union type args.
            for pair in union_types:
                arg, arg_type = pair
                new_type = _reduce_union_type(output._type, arg_type)
                if new_type is not None:
                    updates |= arg.set_type(create_union_type(*new_type))

    # Try forward type inference.
    out_type = None
    if not status:
        if len(arg_types) == 1:
            # All arguments are of same type, so is the output.
            out_type = arg_types.pop()
        elif float in arg_types:
            # Float type is dominant in type coercion for tensors.
            out_type = float
        elif int in arg_types:
            # If there is no union float type, output type is simply int.

            out_type = int
            if float in all_possible_types:
                # Means there are some union types which can be float.
                out_type = int | float

        elif bool in arg_types:
            # If there is no union type, output type is simply bool.
            if not union_types:
                out_type = bool
            elif all_possible_types.issuperset({float, int}):
                if output._type != float | int | bool:
                    out_type = float | int | bool
            elif all_possible_types.issuperset({float}):
                if output._type != float | bool:
                    out_type = float | bool
            elif all_possible_types.issuperset({int}) and output._type != int | bool:
                out_type = int | bool
        elif int | float in arg_types:
            out_type = int | float

    # Set output type if inferred.
    if out_type is not None:
        updates |= output.set_type(out_type)
        # If out_type became non-union type, set status to True.
        if not isinstance(out_type, UnionType):
            status = True

    return status, updates


def floor_divide_type_constraint(
    output: Tensor, numerator: Tensor, denominator: Tensor
):
    status = False
    updates = Updates()
    # First be sure that output can only be of type float | int. So
    # constrain its type to float | int.
    updates |= output.set_type(int | float)
    # Try reverse type inference first.
    if output._type is int:
        # Only possible when numerator and denominator are integers or booleans.
        updates |= numerator.set_type(int | bool)
        updates |= denominator.set_type(int | bool)
        status = True
    elif output._type is float:
        # At least one of inputs is float.
        if (
            isinstance(numerator._type, UnionType)
            and float not in numerator._type.__args__
        ):
            updates |= denominator.set_type(float)
            status = True
        elif (
            isinstance(denominator._type, UnionType)
            and float not in denominator._type.__args__
        ):
            updates |= numerator.set_type(float)
            status = True
    return status, updates


def scalar_slice_type_constraint(
    output: Scalar, input: Scalar, start: Scalar, stop: Scalar, step: Scalar
):
    updates = Updates()
    output_type = output._type
    input_type = input._type

    assert (
        isinstance(start.value, ToBeDetermined)
        or type(start.value) is int
        or start.value is None
    )
    assert (
        isinstance(stop.value, ToBeDetermined)
        or type(stop.value) is int
        or stop.value is None
    )
    assert (
        isinstance(step.value, ToBeDetermined)
        or type(step.value) is int
        or step.value is None
    )

    if (
        isinstance(input_type, GenericAlias)
        and input_type.__origin__ is tuple
        and input_type.__args__[1] is not ...
    ):
        # if input is tuple and its values are exactly determined in terms of types
        # (ex: tuple[int, float, float, int]),
        # find the type of output
        if (
            not isinstance(start.value, ToBeDetermined)
            and not isinstance(step.value, ToBeDetermined)
            and not isinstance(stop.value, ToBeDetermined)
        ):
            # if all values of indexes are given, find exactly the type of output.
            out_args = input_type.__args__[start.value : stop.value : step.value]
            out_type = tuple[*out_args]  # type: ignore
            if (
                intersection_type := find_intersection_type(output_type, out_type)
            ) is not None:
                updates |= output.set_type(intersection_type)
            else:
                raise TypeError("Inferred types does not match in slice constraints!")
        else:
            # if all values are not given, match the output with origin of input
            updates |= output.set_type(input_type.__origin__)

    elif (
        isinstance(output_type, GenericAlias)
        and output_type.__origin__ is tuple
        and output_type.__args__[1] is not ...
    ):
        # if output is tuple and its values are exactly determined in terms of types
        # (ex: tuple[int, float, float, int]),
        # try to infer type of input by using this information
        if (start.value is None) and (stop.value is None) and (step.value is None):
            # if all of the values of index is None, this means input should be exactly
            # equal to output, find intersection of these types and update accordingly
            intersection_type = find_intersection_type(output_type, input_type)
            if intersection_type is not None:
                updates |= input.set_type(intersection_type)
                updates |= output.set_type(intersection_type)
            else:
                raise TypeError("Inferred types does not match in slice constraints!")

        else:
            # if the condition is not True, try to infer input's type
            # with output's origin's type.
            updates |= input.set_type(output_type.__origin__)

    else:
        # Above conditions are only conditions in type inference that is also give
        # info about atomic types of input's and output's. If it is not satisfied,
        # directly intersect types of inputs and outputs as they should have same type.
        intersection_type = find_intersection_type(output_type, input_type)
        if intersection_type is not None:
            updates |= input.set_type(intersection_type)
            updates |= output.set_type(intersection_type)
        else:
            raise TypeError("Inferred types does not match in slice constraints!")

    status = not is_union(output._type)
    return status, updates


def scalar_item_type_constraint_forward_helper(
    input_type: GenericAlias | UnionType | type, index_val: int | ToBeDetermined
) -> type | UnionType | GenericAlias:
    # forward inference of scalar item type constraint:
    # Examples:
    # > scalar_item_type_constraint_forward_helper(list[list[int]], 3) -> list[int]
    # > scalar_item_type_constraint_forward_helper(list[int | float], 3) -> int | float

    new_type = input_type
    if isinstance(input_type, GenericAlias):
        if input_type.__origin__ is tuple:
            if ... in input_type.__args__:
                # if second value is ellipsis, directly take first value
                # (tuple[int, ...] -> int)
                new_type = input_type.__args__[0]
            else:
                # case when type of tuple is exact (ex: tuple[int, float])
                if not isinstance(index_val, ToBeDetermined):
                    # if index val is specified, directly take the corresponding item
                    # of type
                    new_type = input_type.__args__[index_val]
                else:
                    # if not specified this means it can be all of them,
                    # take union of all types inside tuple
                    new_type = create_union_type(*input_type.__args__)

        elif input_type.__origin__ is list:
            # if list, directly take first argument (list[int | float] -> int | float)
            new_type = input_type.__args__[0]
    elif input_type is list or input_type is tuple:
        new_type = input_type | int | float | list

    return new_type


def check_index_type_compatibility(
    _type: type,
    index: int | ToBeDetermined,
    is_variadic: bool,
    raise_error: bool = False,
) -> bool:
    if (
        isinstance(_type, GenericAlias)
        and _type.__origin__ is tuple
        and not isinstance(index, ToBeDetermined)
        and not is_variadic
    ):
        args_len = len(_type.__args__)
        if not (-args_len <= index <= args_len - 1):
            if raise_error:
                raise TypeError(
                    f"Index value {index} is out of range for type {_type}!"
                )
            return False
    return True


def scalar_item_reduce_input_type(
    output_type: type | UnionType | GenericAlias,
    input_type: type | UnionType | GenericAlias,
    index,
):
    possible_types = []
    out_origin: type[list] | type[tuple] | type[UnionType] | None = get_origin(
        output_type
    )
    input_origin: type[list] | type[tuple] | type[UnionType] | None = None
    # Look for compatible types in __args__ of input type with the output_type.
    if isinstance(input_type, UnionType):
        input_origin = UnionType
        for arg in input_type.__args__:
            origin_type = get_origin(arg)
            is_variadic = ... in arg.__args__ if origin_type is not None else False
            if check_index_type_compatibility(origin_type, index, is_variadic):
                if origin_type is not None:
                    # Search sub_args since input_type is UnionType
                    for sub_arg in arg.__args__:
                        if find_intersection_type(output_type, sub_arg):
                            possible_types.append(
                                origin_type[sub_arg, ...]
                                if is_variadic
                                else origin_type[sub_arg]
                            )
                elif arg is tuple or arg is list:
                    # If arg is list or tuple, directly take "arg" as origin type
                    # and origin of "output_type" as inner type if exists.
                    inner_type: list[
                        type | type[UnionType] | EllipsisType | GenericAlias | UnionType
                    ] = []
                    if out_origin is not None and not isinstance(
                        output_type, UnionType
                    ):
                        inner_type.append(out_origin)
                    else:
                        inner_type.append(output_type)
                    if arg is tuple:
                        inner_type.append(...)
                    possible_types.append(arg[*inner_type])  # type: ignore
        return create_union_type(*possible_types)
    elif isinstance(input_type, GenericAlias):
        input_origin = input_type.__origin__

        is_variadic = ... in input_type.__args__ if input_origin is not None else False
        if check_index_type_compatibility(
            input_origin, index, is_variadic, raise_error=True
        ):
            if index == ... or input_origin is list:
                for arg in input_type.__args__:
                    if find_intersection_type(output_type, arg):
                        return input_type
            elif input_origin is tuple:
                possible_types = [
                    arg if idx != index else find_intersection_type(arg, output_type)
                    for idx, arg in enumerate(input_type.__args__)
                ]
                return (
                    input_origin[*possible_types, ...]  # type: ignore
                    if is_variadic
                    else input_origin[*possible_types]  # type: ignore
                )
    else:
        return input_type


def scalar_item_type_constraint(output: Scalar, input: Scalar, index: Scalar):
    updates = Updates()
    assert not isinstance(input._type, NestedListType)
    assert not isinstance(output._type, NestedListType)
    input_type = input._type
    output_type = output._type
    index_value = index.value
    assert isinstance(index_value, ToBeDetermined) or type(index_value) is int

    if not (
        isinstance(input_type, UnionType)
        or hasattr(input_type, "__origin__")
        or input_type in [tuple, list]
    ):
        raise TypeError("Input type should be list, tuple or UnionType!")

    if (
        inferred_input_type := scalar_item_reduce_input_type(
            output_type, input_type, index_value
        )
    ) is None:
        raise TypeError(
            f"Output type {output_type} is not compatible with input type {input_type}!"
        )

    updates |= input.set_type(inferred_input_type)

    # extract all possibilites and put it in to a list
    # TODO: This part should take NestedListType into account.
    args = input._type.__args__ if isinstance(input._type, UnionType) else [input._type]

    # Do the forward inference in all types in args, then make Union
    types = [
        scalar_item_type_constraint_forward_helper(arg, index_value) for arg in args
    ]
    inferred_out_type = create_union_type(*types)

    updates |= output.set_type(inferred_out_type)

    status = not is_union(output._type)
    return status, updates


def tensor_to_list_type_constraint(output: Scalar, input: Tensor):
    status = not is_union(output._type)
    updates = Updates()
    assert input._temp_shape is not None
    in_shape: ShapeRepr = input._temp_shape
    assert (
        output._type is list
        or output._type is float
        or output._type is int
        or output._type is bool
        or isinstance(output._type, NestedListType | UnionType)
        or (isinstance(output._type, GenericAlias) and output._type.__origin__ is list)
    )

    # If input type is UnionType, try to constrain it using output type
    if get_origin(input._type) == UnionType and (
        out_types := find_list_base_type(output._type)  # type: ignore  # (MyPy bug)
    ):
        possible_input_types = find_intersection_type(
            input._type, create_union_type(*out_types)
        )
        if not possible_input_types:
            raise TypeError(
                f"Input type {input._type} is not compatible with output type "
                f"{output._type}!"
            )
        updates |= input.set_type(possible_input_types)

    # Create the base same as input type
    base = input._type
    if in_shape.root is None:
        for _ in range(len(in_shape.prefix + in_shape.suffix)):
            # recursively cover list with base equal to number of all determined
            # uniadics
            base = list[base]  # type: ignore
    else:
        # if input has variadic, add also list['NestedFloatOrIntOrBoolList']
        base = NestedListType(base)  # type: ignore

    updates |= output.set_type(base)

    if in_shape.root is not None:
        status = not (
            is_union(output._type) or isinstance(output._type, NestedListType)
        )
    else:
        status = True

    return status, updates


def reduce_type_constraint(output: Tensor, input: Tensor):
    updates = Updates()
    input_type = input._type

    possible_output_types: list[type[int] | type[float] | type[bool]] = []

    ### Forward Inference ###
    if find_intersection_type(input_type, int | bool):
        # if input has type of int or bool, int is one of the possible output types
        possible_output_types.append(int)

    if find_intersection_type(input_type, float):
        # if input has type of float, float is one of the possible output types
        possible_output_types.append(float)

    union_output_types = create_union_type(*possible_output_types)
    updates |= output.set_type(union_output_types)

    ### Reverse Inference ###
    if output._type is float:
        # if output type is float, it is guaranteed that input will be float
        updates |= input.set_type(float)
    elif output._type is int:
        # if output type is int, input should either be int or bool
        updates |= input.set_type(bool | int)

    status = not isinstance(output._type, UnionType)

    return status, updates


def bcast_get_pos_vals(
    input: ShapeRepr, other: ShapeRepr, output: ShapeRepr
) -> list[PossibleValues]:
    from .common import AND, DNF

    # TODO: guarantee len(output) == len(other) with no roots
    output_unis = output.prefix[len(input.prefix) : len(output) - len(input.suffix)]

    if len(output) != len(other):
        uniadics: list[Uniadic] = []
        for idx, uni in enumerate(output_unis):
            if idx < (len(output_unis) - len(other)):
                uniadics.append(uni)
            elif (
                input.root is not None
                and input.root.possibles is not None
                and len(output_unis) in input.root.possibles
            ):
                uniadics.append(input.root.possibles[len(output_unis)].uniadics[idx])
            else:
                uniadics.append(Uniadic())
        return [PossibleValues(tuple(uniadics), [])]

    # Revert output_unis to match with other.
    output_unis = output_unis[::-1]

    _range = list(range(len(output_unis) + 1))
    other_unis = other.prefix[len(input.prefix) : len(other) - len(input.suffix)][::-1]

    pos_vals: list[PossibleValues] = []
    for idx in _range:
        uniadics = []
        for _idx in range(idx):
            if other_unis[_idx].value == 1:
                uniadics.append(output_unis[_idx])
            elif (
                input.root is not None
                and input.root.possibles is not None
                and idx in input.root.possibles
            ):
                uniadics.append(input.root.possibles[idx].uniadics[::-1][_idx])
            else:
                uniadics.append(Uniadic())

        max_len = len(output_unis) - idx
        dnf_list = [
            DNF([AND({other_uni: out_uni})])
            for other_uni, out_uni in zip(
                other_unis[::-1][:max_len], output_unis[::-1][:max_len], strict=False
            )
        ]
        existing_pos = None
        if input.root is not None and input.root.possibles is not None:
            existing_pos = input.root.possibles.get(idx)
        if existing_pos:
            for _idx in range(idx):
                # TODO: check also equivalences
                ex_uni: Uniadic = existing_pos.uniadics[::-1][_idx]
                other_uni: Uniadic = other_unis[_idx]
                if other_uni.metadata == ex_uni.metadata or (
                    ex_uni in existing_pos.dnf_lookup_table
                    and other_uni in existing_pos.dnf_lookup_table[ex_uni].uniadics
                ):
                    # dnf_list.append(DNF([AND({output_unis[_idx]: other_unis[_idx]})]))
                    dnf_list += [DNF([AND({output_unis[_idx]: other_unis[_idx]})])]
                # elif output_unis[_idx].possible_values is not None:
                #     ex_uni.update_possible_values(
                # {1} | output_unis[_idx].possible_values
                # )
        pos_vals.append(PossibleValues(tuple(uniadics[::-1]), dnf_list))
    return pos_vals


def bcast_update_possible_values_of_input(
    input: ShapeRepr, other: ShapeRepr, output: ShapeRepr
) -> Updates:
    if input.root is None:
        return Updates()

    elif output.root is None:
        if other.root is None:
            pos_vals = bcast_get_pos_vals(input, other, output)
        else:
            # TODO: also check other.root.max_len!
            # lengths = [idx for idx in range(len(output) - len(input) + 1)]
            # pos_vals = bcast_get_pos_vals(input, other, output)
            unis = [Uniadic() for _ in range(len(output) - len(input))]
            pos_vals = [
                PossibleValues(
                    input.root.possibles[length].uniadics
                    if input.root.possibles is not None
                    and length in input.root.possibles
                    # else tuple(Uniadic() for _ in range(length))
                    else tuple(unis[:length][::-1])
                )
                for length in range(len(output) - len(input) + 1)
            ]

    elif output.root.possibles is not None:
        # TODO: also check output.root.max_len!
        return Updates()
        # len_output = len(output) + output.root.max_len
        # lengths = [idx for idx in range(len_output - len(input) + 1)]
    else:
        return Updates()

    return input.root.update_possible_values(*pos_vals)
    # if input.root.possibles is None:
    # return input.root.update_possible_values(
    #     PossibleValues(tuple(Uniadic() for _ in range(length)))
    #     for length in lengths
    # )
    # else:
    #     return input.root.update_possible_values(
    #         input.root.possibles[length]
    #         for length in lengths
    #         if length in input.root.possibles
    #     )


def get_possibles(input: ShapeRepr) -> list[PossibleValues | None]:
    if input.root is None or input.root.possibles is None:
        return [None]
    return [pos_val for pos_val in input.root.possibles.values()]


def bcast_get_list(input: ShapeRepr, pos: PossibleValues | None) -> list[Uniadic]:
    if input.root is None:
        return input.prefix
    elif pos is None:
        return input.suffix
    else:
        return input.prefix + list(pos.uniadics) + input.suffix


def bcast_check_uniadics(
    left: ShapeRepr,
    right: ShapeRepr,
    output: ShapeRepr,
    pos_vals: tuple[
        PossibleValues | None, PossibleValues | None, PossibleValues | None
    ],
    index: int,
) -> bool:
    left_pos, right_pos, output_pos = pos_vals
    # Check compatibility
    left_list = bcast_get_list(left, left_pos)[-index - 1 :: -1]
    right_list = bcast_get_list(right, right_pos)[-index - 1 :: -1]
    output_list = bcast_get_list(output, output_pos)[-index - 1 :: -1]

    for uni_group in zip_longest(output_list, left_list, right_list):
        # Check given uniadic group is valid for bcast or not and check DNF
        # compatibility
        dnf_list: list[DNF] = []
        if left_pos is not None:
            dnf_list += left_pos.dnf_list
        if right_pos is not None:
            dnf_list += right_pos.dnf_list
        if output_pos is not None:
            dnf_list += output_pos.dnf_list
        pos = PossibleValues((), dnf_list)
        is_unadics_applicable = bcast_check_uniadic_group(uni_group, pos_vals)
        if not (is_unadics_applicable and pos.is_applicable):
            return False
    return True


def bcast_update_all_possibilites(
    output: ShapeRepr, left: ShapeRepr, right: ShapeRepr, index: int
) -> Updates:
    updates = Updates()
    updates |= bcast_update_possible_values_of_input(left, right, output)
    updates |= bcast_update_possible_values_of_input(right, left, output)

    valid_left_possibles: set[int] = set()
    valid_right_possibles: set[int] = set()
    valid_output_possibles: set[int] = set()

    for pos_vals in product(
        get_possibles(left), get_possibles(right), get_possibles(output)
    ):
        left_pos, right_pos, output_pos = pos_vals
        if bcast_check_uniadics(left, right, output, pos_vals, index):
            if left_pos is not None:
                valid_left_possibles.add(len(left_pos.uniadics))
            if right_pos is not None:
                valid_right_possibles.add(len(right_pos.uniadics))
            if output_pos is not None:
                valid_output_possibles.add(len(output_pos.uniadics))

    if (
        valid_left_possibles
        and left.root is not None
        and left.root.possibles is not None
    ):
        possible_list = [left.root.possibles[idx] for idx in valid_left_possibles]
        updates |= left.root.update_possible_values(*possible_list)

    if (
        valid_right_possibles
        and right.root is not None
        and right.root.possibles is not None
    ):
        possible_list = [right.root.possibles[idx] for idx in valid_right_possibles]
        updates |= right.root.update_possible_values(*possible_list)

    if (
        valid_output_possibles
        and output.root is not None
        and output.root.possibles is not None
    ):
        possible_list = [output.root.possibles[idx] for idx in valid_output_possibles]
        updates |= output.root.update_possible_values(*possible_list)

    # # TODO: Make this update up to fix point!
    updates |= bcast_uniadics(output, left, right, index)
    updates |= bcast_update_possible_values_of_input(left, right, output)
    updates |= bcast_update_possible_values_of_input(right, left, output)

    # Update output's possibles accordingly.
    max_right_len: int | None = len(right)
    max_left_len: int | None = len(left)
    if left.root is not None:
        if left.root.possibles is not None:
            max_left_len += left.root.max_len
        else:
            max_left_len = None

    if right.root is not None:
        if right.root.possibles is not None:
            max_right_len += right.root.max_len
        else:
            max_right_len = None
    if (
        output.root is not None
        and max_left_len is not None
        and max_right_len is not None
    ):
        max_len = max(max_right_len, max_left_len) - len(output)
        min_len = max(0, min(max_right_len, max_left_len) - len(output))
        updates |= output.root.update_possible_values(
            *[
                PossibleValues(tuple(Uniadic() for _ in range(idx)))
                for idx in range(min_len, max_len + 1)
            ]
        )

    return updates


def bcast_uniadics(
    output: ShapeRepr, left: ShapeRepr, right: ShapeRepr, index: int = 0
) -> Updates:
    updates = Updates()
    fn_keys_set: set[tuple[Uniadic, ...]] = set()
    uni_keys_dict: dict[Uniadic, set[tuple[Uniadic, ...]]] = {}
    is_uniadics = (left.root or right.root or output.root) is None
    # check if there is a mismatch in list of uniadics
    if is_uniadics and max(len(left), len(right)) != len(output):
        raise ValueError("Shape mismatch for output!")

    left_unis = (left.suffix, left.prefix)[left.root is None][::-1][index:]
    right_unis = (right.suffix, right.prefix)[right.root is None][::-1][index:]
    output_unis = (output.suffix, output.prefix)[output.root is None][::-1][index:]
    for idx, (left_uni, right_uni, out_uni) in enumerate(
        zip_longest(left_unis, right_unis, output_unis)
    ):
        # iterate through uniadics
        if bool(out_uni) and bool(left_uni) and bool(right_uni):
            # Check value consistency between left and right uniadics.
            if (
                left_uni.value not in (None, 1)
                and right_uni.value not in (None, 1)
                and left_uni.value != right_uni.value
            ):
                raise ValueError(
                    f"Inputs shape mismatch at dimension "
                    f"{len(output) - index - 1 - idx}. Shapes are inconsistent."
                )

            # If same uniadic for both left and right, update output uniadic with that.
            if left_uni.metadata == right_uni.metadata:
                if out_uni.metadata != left_uni.metadata:
                    updates |= out_uni.match(left_uni)
                continue
            # if both left_uni and right uni is exist, put left_uni, right_uni
            # and right uni int fn_keys_set.
            fn_keys = (left_uni, right_uni, out_uni)
            fn_keys_set.add(fn_keys)
            uni_keys_dict.setdefault(left_uni, set()).add(fn_keys)
            uni_keys_dict.setdefault(right_uni, set()).add(fn_keys)
            uni_keys_dict.setdefault(out_uni, set()).add(fn_keys)
        elif is_uniadics:
            # If one of the uniadics is None, update output uniadic with the other one.
            existing_uni = left_uni if left_uni else right_uni
            if out_uni.metadata != existing_uni.metadata:
                updates |= out_uni.match(existing_uni)
        elif bool(out_uni) and (bool(left_uni) or bool(right_uni)):
            existing_uni = left_uni if left_uni else right_uni
            if existing_uni.value not in {None, 1} or out_uni.value == 1:
                updates |= out_uni.match(existing_uni)
            elif out_uni.possible_values is not None:
                updates |= existing_uni.update_possible_values(
                    out_uni.possible_values | {1}
                )

    while fn_keys_set:
        # run the inference algorithm. Note that fn_keys_set contains inputs
        # of bcast_uniadic_group function. For each set of uniadics, run the
        # function
        _fn_keys = fn_keys_set.pop()
        _updates = bcast_uniadic_group(_fn_keys)
        for uni in _updates.uniadic_updates:
            fn_keys_set.update(uni_keys_dict[uni])
        updates |= _updates

    return updates


def bcast_uniadic_group_per_input(
    in1: Uniadic, in2: Uniadic, output: Uniadic
) -> Updates:
    updates = Updates()
    if in1.value == 1 or output.value == 1:
        updates |= output.match(in2)

    if output.possible_values is not None:
        updates |= in2.update_possible_values(output.possible_values | {1})

    if in1.possible_values is not None and 1 not in in1.possible_values:
        updates |= in2.update_possible_values(in1.possible_values | {1})
        updates |= output.update_possible_values(in1.possible_values)

    return updates


def bcast_check_uniadic_group_per_input(
    in1: set[int] | None, in2: set[int] | None, output: set[int] | None
) -> bool:
    if in1 is None or 1 in in1:
        if output is not None and in2 is not None and output & in2 == set():
            return False
    else:
        if not (in2 is None or 1 in in2) and in1 & in2 == set():
            return False
    return True


def bcast_check_uniadic_group(
    uniadics: tuple[Uniadic | None, ...], pos_vals: tuple[PossibleValues | None, ...]
) -> bool:
    output, left, right = uniadics
    left_pos, right_pos, output_pos = pos_vals
    remainings = {right, left, output}
    remainings.discard(None)
    if len(remainings) < 2:
        return True
    if left is None and right is not None:
        return not (
            output is not None
            and output.possible_values is not None
            and right.possible_values is not None
            and output.possible_values & right.possible_values == set()
        )

    elif right is None and left is not None:
        return not (
            output is not None
            and output.possible_values is not None
            and left.possible_values is not None
            and output.possible_values & left.possible_values == set()
        )

    if left is not None:
        if left_pos is not None and left in left_pos.dnf_lookup_table:
            left_vals = left_pos.dnf_lookup_table[left].values
        else:
            left_vals = left.possible_values

    if right is not None:
        if right_pos is not None and right in right_pos.dnf_lookup_table:
            right_vals = right_pos.dnf_lookup_table[right].values
        else:
            right_vals = right.possible_values

    if output_pos is not None and output in output_pos.dnf_lookup_table:
        output_vals = output_pos.dnf_lookup_table[output].values
    elif output is not None:
        output_vals = output.possible_values
    else:
        output_vals = None

    is_valid = bcast_check_uniadic_group_per_input(left_vals, right_vals, output_vals)
    is_valid &= bcast_check_uniadic_group_per_input(right_vals, left_vals, output_vals)
    return is_valid


def bcast_uniadic_group(uniadics: tuple[Uniadic, ...]) -> Updates:
    left, right, output = uniadics
    updates = bcast_uniadic_group_per_input(left, right, output)
    updates |= bcast_uniadic_group_per_input(right, left, output)
    return updates


def bacast_align_output(
    output: ShapeRepr, left: ShapeRepr, right: ShapeRepr, index: int
) -> Updates:
    updates = Updates()
    # Output will have same shape structure as the longer one.
    if len(left) == len(right):
        longer_repr = (left, right)[right.root is not None]
        shorter_repr = (left, right)[left == longer_repr]
    else:
        longer_repr = (left, right)[len(left) < len(right)]
        shorter_repr = (left, right)[left == longer_repr]

    # Handle special case of right and left has same Variadic object
    if (
        right.root == left.root
        and left.root is not None
        and output.root is not None
        and len(right.prefix) == len(left.prefix)
    ):
        output.inner_match([Uniadic() for _ in range(len(left.prefix))], Variadic())
        for unis in zip(right.prefix, left.prefix, output.prefix, strict=False):
            bcast_uniadic_group(unis)
    # Align output shape structure with the longer one if available.
    # if (len(output.prefix) != (longer_repr.prefix)) or (len(output.suffix) !=
    # len(longer_repr.suffix)): First (len(longer) - len(shorter)) uniadics of output
    # prefix will be same as the longer one.
    equal_uni_length = (
        len(longer_repr) - len(shorter_repr)
        if shorter_repr.root is None or shorter_repr.root == longer_repr.root
        else 0
    )

    prefix: list[Uniadic] = []
    for idx, uni in enumerate(longer_repr.prefix):
        if (
            (idx < equal_uni_length)
            or ((idx < (len(longer_repr) - index)) and uni.value not in (None, 1))
            and (output.root is None or right.root is None or left.root is None)
        ):
            prefix.append(uni)
        else:
            _uni = Uniadic()
            # if all inputs have uniadic as their first element, then it could be
            # deduced that output will also have uniadic as a first element (union
            # of possible_values of left uni and right).
            if (
                idx == 0
                and len(shorter_repr.prefix) > 0
                and shorter_repr[0].possible_values is not None
                and uni.possible_values is not None
            ):
                updates |= _uni.update_possible_values(
                    shorter_repr[0].possible_values | uni.possible_values
                )
            prefix.append(_uni)

    suffix = [
        uni
        if (idx < (equal_uni_length - len(longer_repr.prefix)))
        or ((idx < (len(longer_repr.suffix) - index)) and uni.value not in (None, 1))
        else Uniadic()
        for idx, uni in enumerate(longer_repr.suffix)
    ]

    # Shorter repr has no Variadic field and longer_repr suffix longer than
    # shorter_repr or variadic fields are same objects and length of suffixes
    # are equal, then root will be same as longer_repr.
    if (
        shorter_repr.root is None
        and (len(longer_repr.suffix) >= len(shorter_repr))
        or (shorter_repr.root == longer_repr.root)
        and len(longer_repr.suffix) == len(shorter_repr.suffix)
    ):
        root = longer_repr.root
    else:
        root = Variadic()

    # If longer repr has not Variadic field make calculated prefix as suffix.
    if longer_repr.root is None and shorter_repr.root is not None:
        suffix = prefix
        prefix = []

    # Final check to be sure about we are not re-matching output with
    # the same shape structure.
    # if (len(output.prefix) != len(prefix)) or (len(output.suffix) != len(suffix)):
    updates |= output.inner_match(prefix=prefix, root=root, suffix=suffix)
    return updates


def bcast_align_input(output: ShapeRepr, left: ShapeRepr, right: ShapeRepr):
    # Example: output: [V1, 2, 3], left: [V2, 1, 1], right: [V3]
    # Result:  output: [V1, 2, 3], left: [V2, 1, 1], right: [V4, 2, 3]

    # TODO: Handle following and add its test
    # Example: output: [3, V1], left: [1, V2], right: [V3]
    # Result:  output: [3, V1], left: [1, V2], right: [3, V4]

    # TODO: Handle following and add its test
    # Example: output: [V1, 2, u2], left: [V2, 1, 1], right: [V3]
    # Result:  output: [V1, 2, u2], left: [V2, 1, 1], right: [V4, 2, u3]
    updates = Updates()
    uniadics: list[Uniadic] = []

    _left = (left.suffix, left.prefix)[left.root is None]
    _right = (right.suffix, right.prefix)[right.root is None]
    _output = (output.suffix, output.prefix)[output.root is None]

    for l_, r, o in zip_longest(_left[::-1], _right[::-1], _output[::-1]):
        if o is not None:
            if (
                r is not None
                and r.value == 1
                and l_ is None
                and o.value not in {None, 1}
            ):
                uniadics.append(o)
            else:
                break
    if uniadics != []:
        uniadics = uniadics[::-1]
        if output.root is None and len(output) == len(uniadics) != len(right):
            updates |= left.inner_match(uniadics)
        else:
            updates |= left.inner_match(root=Variadic(), suffix=uniadics)
    return updates


def bcast_helper(
    output: ShapeRepr, left: ShapeRepr, right: ShapeRepr, index: int
) -> ConstrainResultType:
    """_summary_

    Parameters
    ----------
    output : ShapeRepr
        _description_
    left : ShapeRepr
        _description_
    right : ShapeRepr
        _description_
    index : int
        The number of last Uniadics not to be processed/inferred.

    Returns
    -------
    ConstrainResultType
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    updates = Updates()

    # First align output shape structure with the longer one if available.
    updates |= bacast_align_output(output, left, right, index)
    updates |= bcast_align_input(output, left, right)
    updates |= bcast_align_input(output, right, left)
    updates |= bcast_update_all_possibilites(output, left, right, index)
    updates |= bcast_uniadics(output, left, right, index)

    return bcast_exit_condition(output, left, right, index), updates


def bcast(output: Tensor, left: Tensor, right: Tensor) -> ConstrainResultType:
    assert output._temp_shape is not None, "Output shape of broadcast is not set!"
    assert left._temp_shape is not None, "Left shape of broadcast is not set!"
    assert right._temp_shape is not None, "Right shape of broadcast is not set!"
    return bcast_helper(output._temp_shape, left._temp_shape, right._temp_shape, 0)


def bcast_matrix_mult(
    output: Tensor, left: Tensor, right: Tensor
) -> ConstrainResultType:
    assert output._temp_shape is not None, "Output shape of broadcast is not set!"
    assert left._temp_shape is not None, "Left shape of broadcast is not set!"
    assert right._temp_shape is not None, "Right shape of broadcast is not set!"
    return bcast_helper(output._temp_shape, left._temp_shape, right._temp_shape, 2)


def check_reverse(left: list[Uniadic], right: list[Uniadic], output: list[Uniadic]):
    status = True
    left_reverse = left[::-1]
    right_reverse = right[::-1]
    output_reverse = output[::-1]

    for idx, symbol in enumerate(output_reverse):
        if (
            idx < len(left_reverse)
            and symbol.metadata != left_reverse[idx].metadata
            and ((symbol.value != left_reverse[idx].value) or symbol.value is None)
        ):
            _status = False
        # elif idx >= len(left_reverse):
        #     _status = False
        else:
            _status = True
        if (
            idx < len(right_reverse)
            and symbol.metadata != right_reverse[idx].metadata
            and ((symbol.value != right_reverse[idx].value) or symbol.value is None)
        ):
            _status |= False
        # elif idx >= len(right_reverse):
        #     _status |= False
        else:
            _status = True
        status &= _status

    return status


def bcast_exit_condition(
    output: ShapeRepr, left: ShapeRepr, right: ShapeRepr, index: int
):
    return (
        output.root is None
        and left.root is None
        and right.root is None
        and check_reverse(left.prefix, right.prefix, output.prefix)
    )


def bcast_error_check(
    output: Tensor,
    left: Tensor,
    right: Tensor,
    index: int = 0,
) -> ConstrainResultType:
    assert left._temp_shape is not None, "Left shape of broadcast is not set!"
    assert right._temp_shape is not None, "Right shape of broadcast is not set!"
    assert output._temp_shape is not None, "Output shape of broadcast is not set!"

    status = True
    left_list: list[Uniadic] = (left._temp_shape.prefix + left._temp_shape.suffix)[
        -index - 1 :: -1
    ]
    right_list: list[Uniadic] = (right._temp_shape.prefix + right._temp_shape.suffix)[
        -index - 1 :: -1
    ]
    output_list: list[Uniadic] = (
        output._temp_shape.prefix + output._temp_shape.suffix
    )[-index - 1 :: -1]
    # Proceed to check only if any uniadic occurs in left or right
    # that has not same metadata with the corresponding index in output.
    for out_uni, left_uni, right_uni in zip_longest(output_list, left_list, right_list):
        # TODO: Below if added as a guard for the ShapeRepr's combinations
        # which are not solved by bcast constraint since it sets the status
        # to True after the first solved combination. Other repr's remain
        # unsolved whose output may not be consisting of metadata same
        # with left or right. This should be fixed???

        if out_uni is None or out_uni.value is None:
            status = False
            break

        for uni in (left_uni, right_uni):
            if uni is not None and uni.metadata != out_uni.metadata:
                if uni.value is not None:
                    if uni.value not in (out_uni.value, 1):
                        raise ValueError(
                            f"Shape mismatch for broadcast. Dimensionalities for the "
                            f"corresponding shape index are left: {left_uni.value}, "
                            f"right: {right_uni.value}, output: {out_uni.value}"
                        )
                else:
                    status = False
                    break
        else:
            continue
        break

    return status, Updates()


def bcast_is_compatible(
    output: ShapeRepr, left: ShapeRepr, right: ShapeRepr, index: int = 0
) -> bool:
    left_list = (left.suffix, left.prefix)[left.root is None][-index - 1 :: -1]
    right_list = (right.suffix, right.prefix)[right.root is None][-index - 1 :: -1]
    output_list = (output.suffix, output.prefix)[output.root is None][-index - 1 :: -1]

    if (
        output.root is None
        and left.root is None
        and right.root is None
        and (len(output) != len(left_list) and len(output) != len(right_list))
    ):
        return False

    # TODO: include possible values to check!
    # TODO: check first uniadics from left
    for out_uni, left_uni, right_uni in zip_longest(output_list, left_list, right_list):
        inputs = {
            left_uni.value if left_uni is not None else None,
            right_uni.value if right_uni is not None else None,
        }
        inputs -= {None, 1}
        if (
            len(inputs) > 1
            or len(inputs) == 1
            and out_uni.value is not None
            and out_uni.value != next(iter(inputs))
        ):
            return False
    return True


def bcast_mat_mul_check(
    output: Tensor, left: Tensor, right: Tensor
) -> ConstrainResultType:
    return bcast_error_check(output, left, right, index=2)


def reduce_constraints(
    output: Tensor, input: Tensor, axis: Scalar, keepdim: Scalar | None = None
) -> ConstrainResultType:
    updates = Updates()
    assert input._temp_shape is not None, "Input shape of reduce is not set!"
    assert output._temp_shape is not None, "Output shape of reduce is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    axis_val = axis.value
    keepdim_val = keepdim.value if keepdim is not None else False
    assert is_axis_reduce_type(axis_val) or isinstance(
        axis_val, ToBeDetermined
    ), f"given axis value {axis_val} is not valid!"
    assert isinstance(
        keepdim_val, bool | ToBeDetermined
    ), f"given keepdim value {keepdim_val} is not valid!"
    replacement = Uniadic(1) if keepdim_val else None

    if axis_val is not TBD:
        if isinstance(axis_val, int):
            axis_val = (axis_val,)
        elif axis_val is None:
            if not keepdim_val:
                updates |= input_shape._update_uniadics(input_shape.prefix, [])
                updates |= output_shape._update_uniadics(output_shape.reverse, [])
                if output_shape.root is not None:
                    updates |= output_shape.remove_variadic([])
        elif not isinstance(axis_val, tuple):
            raise ValueError("Requires valid axis type!")

        if isinstance(axis_val, tuple):
            if len(axis_val) != len(set(axis_val)):
                raise ValueError("Duplicate value in reduce 'axis'")
            if (
                input_shape.root is not None
                and output_shape.root is not None
                and input_shape.root != output_shape.root
            ):
                positive_axes = [val for val in axis_val if val >= 0]
                negative_axes = [val for val in axis_val if val not in positive_axes]
                pos_idx = max(positive_axes) + 1 if positive_axes else None
                neg_idx = abs(min(negative_axes)) if negative_axes else None
                # If input already has corresponding axes as uniadics, simply match
                # corresponding part of input shape_map with output shape_map.
                if (pos_idx is None or len(input_shape.prefix) >= pos_idx) and (
                    neg_idx is None or len(input_shape.suffix) >= neg_idx
                ):  # pos_idx and neg_idx can not be None at the same time.
                    repr_prefix: list[Uniadic] = [
                        uni
                        for idx, uni in enumerate(input_shape.prefix)
                        if idx not in positive_axes
                    ]
                    repr_suffix: list[Uniadic] = [
                        uni
                        for idx, uni in enumerate(input_shape.reverse)
                        if -(idx + 1) not in negative_axes
                    ][::-1]
                    repr_root = input_shape.root
                    updates |= output_shape.inner_match(
                        prefix=repr_prefix, root=repr_root, suffix=repr_suffix
                    )

                else:
                    prefix = []
                    suffix = []

                    if pos_idx is not None and len(input_shape.prefix) < pos_idx:
                        prefix = input_shape.prefix + [
                            Uniadic() for _ in range(pos_idx - len(input_shape.prefix))
                        ]
                        if len(input_shape.suffix) < (
                            amount := max(
                                pos_idx, neg_idx if neg_idx is not None else 0
                            )
                            - pos_idx
                        ):
                            suffix = [
                                Uniadic() for _ in range(amount)
                            ] + input_shape.suffix
                    elif neg_idx is not None and len(input_shape.suffix) < neg_idx:
                        suffix = [
                            Uniadic() for _ in range(neg_idx - len(input_shape.suffix))
                        ] + input_shape.suffix
                        prefix = []

                    # Determine minimum length for given axis values such that they
                    # are guaranteed not to coincide.
                    for ax in negative_axes:
                        # Check positive counterpart exists in axis.
                        if (len(prefix) + len(suffix) + ax) in axis_val:
                            suffix.insert(0, Uniadic())

                    # Align input shape structure with minimum requirements using
                    # prefix and suffix.
                    if prefix or suffix:
                        updates |= input_shape.inner_match(
                            prefix=prefix, root=Variadic(), suffix=suffix
                        )

                    # Try to infer output shape structure from input shape structure.
                    # First initialize out_prefix and out_suffix with the Uniadics
                    # which may be transferred to the output.
                    out_prefix = []
                    for idx, uni in enumerate(input_shape.prefix):
                        if idx not in axis_val:
                            if not neg_idx or idx < (len(input_shape) - neg_idx):
                                out_prefix.append(uni)
                            else:
                                out_prefix.append(Uniadic())
                        elif replacement:
                            out_prefix.append(replacement)

                    out_suffix = []
                    for idx, uni in enumerate(input_shape.suffix):
                        if (idx - len(input_shape.suffix)) not in axis_val:
                            if not positive_axes or (
                                idx + len(input_shape.prefix)
                            ) > max(positive_axes):
                                out_suffix.append(uni)
                            else:
                                out_suffix.append(Uniadic())
                        elif replacement:
                            out_suffix.append(replacement)

                    # Now remove residual uniadics from input shape structure
                    # in order to guarantee min length of output shape.
                    if not keepdim_val and (
                        diff := (
                            (len(out_prefix) + len(out_suffix))
                            - (len(input_shape) - len(axis_val))
                        )
                    ):
                        for _ in range(diff):
                            if out_prefix:
                                out_prefix.pop()
                            else:
                                out_suffix.pop(0)

                    if out_prefix or out_suffix:
                        pos_len = pos_idx if pos_idx is not None else 0
                        neg_len = neg_idx if neg_idx is not None else 0
                        if (
                            len(input_shape.prefix) >= pos_len
                            and len(input_shape.suffix) >= neg_len
                        ):
                            var = input_shape.root
                        else:
                            var = Variadic()
                        updates |= output_shape.inner_match(
                            prefix=out_prefix, root=var, suffix=out_suffix
                        )

        if input_shape.root is None:
            if axis_val is None:
                axis_val = tuple([idx for idx in range(len(input_shape.prefix))])
            # Min rank of input must be  max(axis) + 1.
            if len(axis_val) > 0 and (in_rank := len(input_shape)) < (
                max_axis := max(axis_val) + 1
            ):
                raise ValueError(
                    f"Input rank is {in_rank}. Minimum rank {max_axis} input is "
                    f"required for axis = {axis_val}."
                )
            # Convert all negative axis values into corresponding positive ones.
            axis_list: list[int] = list()
            for idx in axis_val:
                real_idx = idx if idx >= 0 else idx + in_rank
                if real_idx not in axis_list:
                    axis_list.append(real_idx)
                else:
                    raise ValueError(
                        f"Dim {real_idx} appears multiple times in the reduce axes"
                    )
            axis_val = tuple(axis_list)
            if output_shape.root is not None:
                var_replacement = [
                    input_shape.prefix[idx] if idx not in axis_val else replacement
                    for idx in range(len(input_shape.prefix))
                ]
                filtered_var_replacement: list[Uniadic] = list(
                    filter(None, var_replacement)
                )
                updates |= output_shape._update_uniadics(
                    output_shape.prefix, filtered_var_replacement
                )
                updates |= output_shape._update_uniadics(
                    output_shape.reverse, filtered_var_replacement[::-1]
                )
                updates |= output_shape.remove_variadic(filtered_var_replacement)
            # Transfer available values using input and output.
            else:
                # Check rank consistency.
                if (in_rank := len(input_shape)) != (
                    (out_rank := len(output_shape))
                    + (0 if keepdim_val else len(axis_val))
                ):
                    # axis_val = None if len(axis_val) == len(input_shape) else axis_val
                    raise ValueError(
                        f"Shape mismatch, output rank = {out_rank}. Output rank must "
                        f"be exactly {in_rank - len(axis_val)} where "
                        f"input rank = {in_rank} "
                        f"and axis = {axis_val}. Axis numbers printed as their "
                        "counterparts."
                    )
                if out_rank != 0:
                    # Create an iterator for output.
                    out_iter = iter(output_shape.prefix)
                    for idx, in_uni in enumerate(input_shape.prefix):
                        # Transfer uniadics if applicable.
                        if idx not in axis_val:
                            out_uni = next(out_iter)
                            updates |= in_uni.match(out_uni)
                        elif keepdim_val:
                            out_uni = next(out_iter)
                            if in_uni.value is not None and out_uni.set_value(1):
                                updates.add(out_uni)

        elif output_shape.root is None and axis_val is not None:
            # Convert all negative axis values into corresponding positive ones.
            in_rank = (
                len(output_shape) if keepdim_val else len(axis_val) + len(output_shape)
            )
            axis_val = tuple([idx if idx > 0 else idx + in_rank for idx in axis_val])
            out_iter = iter(output_shape.prefix)
            input_uniadics = []
            for idx in range(in_rank):
                if idx in axis_val:
                    input_uniadics.append(Uniadic())
                    if keepdim_val:
                        assert isinstance(replacement, Uniadic)
                        updates |= next(out_iter).match(replacement)
                else:
                    input_uniadics.append(next(out_iter))
            updates |= input_shape._update_uniadics(input_shape.prefix, input_uniadics)
            updates |= input_shape._update_uniadics(
                input_shape.reverse, input_uniadics[::-1]
            )
            updates |= input_shape.remove_variadic(input_uniadics)

    return input_shape.root == output_shape.root, updates


def concat_constraints(
    output: Tensor, axis: Scalar, *inputs: Tensor
) -> ConstrainResultType:
    status = False
    updates = Updates()
    keys: list[ShapeRepr] = []
    for input in inputs:
        assert input._temp_shape is not None, "Input shape of concat is not set!"
        keys.append(input._temp_shape)
    assert output._temp_shape is not None, "Output shape of concat is not set!"
    output_shape: ShapeRepr = output._temp_shape

    reprs = keys + [output_shape]
    axis_val = axis.value
    assert (
        isinstance(axis_val, int)
        or axis_val is None
        or isinstance(axis_val, ToBeDetermined)
    ), "Invalid axis value!"
    # look if all reprs have different variadics
    if (
        not isinstance(axis_val, ToBeDetermined)
        and len(set(repr.root for repr in reprs)) == len(reprs)
        and output_shape.root is not None
    ):
        if axis_val is not None:
            # if axis is determined and is positive, we can know that tensors have
            # shape at least value of dimensions. Also we know that All shapes of
            # all reprs must be same except shape at axis same applies for when axis
            # is negative
            var = Variadic()
            if axis_val >= 0:
                uniadics = [Uniadic() for _ in range(axis_val)]
                for repr in reprs:
                    updates |= repr.inner_match(prefix=uniadics + [Uniadic()], root=var)
            elif axis_val < 0:
                uniadics = [Uniadic() for _ in range(-axis_val - 1)]
                for repr in reprs:
                    updates |= repr.inner_match(root=var, suffix=[Uniadic()] + uniadics)
        else:
            updates |= output_shape.inner_match(prefix=[Uniadic()])

    if not isinstance(axis_val, ToBeDetermined):
        if axis_val is not None:
            # If axis is determined and not None, at first take all the uniadic
            # values at axis. shape formula of output of axis must be out =
            # sum(all ins). Therefore, if there is only one unknown, we can
            # infer unknown uniadic's shape by algebra.
            uniadics, uniadic_values, pruned_uni_values = [], [], []
            for repr in reprs:
                if (
                    repr.root is None
                    or (axis_val >= 0 and len(repr.prefix) >= axis_val + 1)
                    or (axis_val < 0 and len(repr.suffix) >= abs(axis_val))
                ):
                    uniadics.append(uni := repr[axis_val])
                    uniadic_values.append(uni_value := uni.value)
                    if uni_value is not None:
                        pruned_uni_values.append(uni_value)
            if len(pruned_uni_values) + 1 == len(reprs):
                status = True
                if uniadic_values[-1] is None:
                    if uniadics[-1].set_value(sum(pruned_uni_values)):
                        updates.add(uniadics[-1])
                else:
                    idx = uniadic_values.index(None)
                    if uniadics[idx].set_value(
                        pruned_uni_values[-1] - sum(pruned_uni_values[:-1])
                    ):
                        updates.add(uniadics[idx])
            elif len(pruned_uni_values) == len(reprs):
                status = True
        else:
            if output_shape.prefix[0].value is None:
                output_size = 0
                for key in keys:
                    if key.root is None:
                        values = [item.value for item in key.prefix]
                        if is_list_int(values):
                            output_size += math.prod(values)
                        else:
                            break
                    else:
                        break
                else:
                    status = True
                    if output_shape.prefix[0].set_value(output_size):
                        updates.add(output_shape.prefix[0])
            else:
                dividing_factor = 1
                substract_factor = 0
                none_values = []
                for key in keys:
                    if key.root is None:
                        unis_without_value = [
                            uni for uni in key.prefix if uni.value is None
                        ]
                        unis_with_value = [
                            uni.value for uni in key.prefix if uni.value is not None
                        ]
                        none_values += unis_without_value
                        if len(none_values) > 1:
                            break
                        if unis_without_value:
                            dividing_factor *= math.prod(unis_with_value)
                        else:
                            substract_factor += math.prod(unis_with_value)
                    else:
                        break
                else:
                    if len(none_values) == 1:
                        status = True
                        if none_values[0].set_value(
                            (output_shape.prefix[0].value - substract_factor)
                            // dividing_factor
                        ):
                            updates.add(none_values[0])
                    elif len(none_values) == 0:
                        status = True
    return status, updates


def reverse_constraints(
    output: Tensor, input: Tensor, axes: Scalar
) -> ConstrainResultType:
    status = False
    assert input._temp_shape is not None, "Input shape of reverse is not set!"
    assert output._temp_shape is not None, "Output shape of reverse is not set!"

    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    axes_val = axes.value
    assert is_axis_reverse_type(axes_val) or isinstance(
        axes_val, ToBeDetermined
    ), "Invalid axis value!"
    status = False
    updates = Updates()

    if axes_val is None:
        if output_shape.root is None:
            # TODO Maybe we should embed uniadic updates in remove_variadic
            updates |= input_shape._update_uniadics(
                input_shape.prefix, output_shape.reverse
            )
            updates |= input_shape._update_uniadics(
                input_shape.reverse, output_shape.prefix
            )
            if input_shape.root is not None:
                updates |= input_shape.remove_variadic(output_shape.reverse)
                if len(input_shape.prefix) != len(output_shape.prefix):
                    raise ValueError("Shape mismatch in Transpose model")
            status = True
        if input_shape.root is None:
            updates |= output_shape._update_uniadics(
                output_shape.prefix, input_shape.reverse
            )
            updates |= output_shape._update_uniadics(
                output_shape.reverse, input_shape.prefix
            )
            if output_shape.root is not None:
                updates |= output_shape.remove_variadic(input_shape.reverse)
                if len(input_shape.prefix) != len(output_shape.prefix):
                    raise ValueError("Shape mismatch in Transpose model")
            status = True

    elif isinstance(axes_val, int | Sequence):
        axes_val = [axes_val] if isinstance(axes_val, int) else axes_val
        in_unis = [Uniadic() for idx in range(len(axes_val))]
        out_unis = [in_unis[axis] for axis in axes_val]

        updates |= input_shape._update_uniadics(input_shape.prefix, in_unis)
        updates |= input_shape._update_uniadics(input_shape.reverse, in_unis[::-1])

        updates |= output_shape._update_uniadics(output_shape.prefix, out_unis)
        updates |= output_shape._update_uniadics(output_shape.reverse, out_unis[::-1])

        if input_shape.root is not None:
            updates |= input_shape.remove_variadic(in_unis)
        if output_shape.root is not None:
            updates |= output_shape.remove_variadic(out_unis)

        status = True

    return status, updates


def polynomial_features_constraints(
    output: Tensor, input: Tensor, degree: Scalar
) -> ConstrainResultType:
    status = False
    updates = Updates()
    assert (
        input._temp_shape is not None
    ), "Input shape of Polynomial Features is not set!"
    assert (
        output._temp_shape is not None
    ), "Output shape of Polynomial Features is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    degree_val = degree.value
    assert isinstance(degree_val, int | ToBeDetermined), "Invalid degree value!"
    # First, check prefix lengths!
    if (
        not isinstance(degree_val, ToBeDetermined)
        and len(input_shape.prefix) == 2
        and len(output_shape.prefix) == 2
        and input_shape.root is None
        and output_shape.root is None
    ):
        output_uniadic = output_shape[1]
        input_uniadic = input_shape[1]
        if input_uniadic.value is not None:
            dim = input_uniadic.value
            value = (
                int(
                    math.factorial(dim + degree_val)
                    / (math.factorial(degree_val) * math.factorial(dim))
                )
                - 1
            )
            if output_uniadic.set_value(value):
                updates.add(output_uniadic)
            status = True
        elif (
            input_uniadic.value is None
            and output_uniadic.value is not None
            and degree is not None
        ):
            # Increment input dimensionality by one up to
            # satisfying the equation: (dim + degree).(dim + degree - 1)....(dim + 1) =
            # value * factorial(degree).
            # This equation comes from total_terms = dim! / (degree! * (dim - degree)!)
            target = (output_uniadic.value + 1) * math.factorial(degree_val)
            # NOTE: We exclude bias term from total terms so add 1 to the output term.
            dim = 1
            while True:
                value = int(math.factorial(dim + degree_val) / math.factorial(dim))
                if value < target:
                    dim += 1
                elif value > target:
                    raise ValueError(
                        "Something went wrong while calculating Polynomial Features "
                        "shapes!"
                    )
                else:
                    if input_uniadic.set_value(dim):
                        updates.add(input_uniadic)
                    status = True
                    break
    return status, updates


def sliding_window_constraint_helper(
    output: Uniadic,
    input: Uniadic,
    stride: int,
    padding: tuple[int, int] | int,
    dilation: int,
    kernel_size: int,
) -> ConstrainResultType:
    status = False
    updates = Updates()
    if isinstance(padding, Sequence):
        padding = sum(padding)
        padding_factor = padding
    else:
        padding_factor = 2 * padding
    # TODO: Is Uniadic type kernel_size possible?
    if input.value is not None:
        if (
            val := (input.value + padding_factor - (kernel_size - 1) * dilation - 1)
            // stride
            + 1
        ) <= 0:
            raise ValueError(
                "Dimension Error: Output dimension calculated to be lesser than zero!"
            )
        if output.set_value(val):
            updates.add(output)
        status = True

    return status, updates


def sliding_window_1d_constraints(
    output: Tensor,
    input: Tensor,
    stride: Scalar,
    padding: Scalar,
    dilation: Scalar,
    kernel_size: Scalar,
) -> ConstrainResultType:
    updates = Updates()
    status = False
    assert input._temp_shape is not None, "Input shape of sliding window is not set!"
    assert output._temp_shape is not None, "Output shape of sliding window is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape

    stride_val = stride.value
    padding_val = padding.value
    dilation_val = dilation.value
    kernel_size_val = kernel_size.value
    assert isinstance(stride_val, int | ToBeDetermined), "Invalid stride value!"
    assert (
        is_tuple_of_two_ints(padding_val) or type(padding_val) is ToBeDetermined
    ), "Invalid padding value!"
    assert type(dilation_val) is int or isinstance(
        dilation_val, ToBeDetermined
    ), "Invalid dilation value!"
    assert type(kernel_size_val) is int or isinstance(
        kernel_size_val, ToBeDetermined
    ), "Invalid kernel_size value!"
    is_input_propagatable = len(input_shape.suffix) >= 1 or (
        input_shape.root is None and len(input_shape.prefix) > 1
    )
    is_output_propagatable = len(output_shape.suffix) >= 1 or (
        output_shape.root is None and len(output_shape.prefix) > 1
    )

    if (
        not isinstance(stride_val, ToBeDetermined)
        and not isinstance(padding_val, ToBeDetermined)
        and not isinstance(dilation_val, ToBeDetermined)
        and not isinstance(kernel_size_val, ToBeDetermined)
        and is_input_propagatable
        and is_output_propagatable
    ):
        status, _updates = sliding_window_constraint_helper(
            output_shape[-1],
            input_shape[-1],
            stride_val,
            padding_val,
            dilation_val,
            kernel_size_val,
        )
        updates |= _updates
    return status, updates


def conv_1d_constraints(
    output: Tensor,
    input: Tensor,
    stride: Scalar,
    padding: Scalar,
    dilation: Scalar,
    kernel: Tensor,
) -> ConstrainResultType:
    updates = Updates()
    status = False
    assert input._temp_shape is not None, "Input shape of Convolution1D is not set!"
    assert output._temp_shape is not None, "Output shape of Convolution1D is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape

    stride_val = stride.value
    padding_val = padding.value
    dilation_val = dilation.value

    assert (
        type(stride_val) is int or type(stride_val) is ToBeDetermined
    ), "Invalid stride value!"
    assert (
        is_tuple_of_two_ints(padding_val)
        or type(padding_val) is int
        or type(padding_val) is ToBeDetermined
    ), "Invalid padding value!"
    assert (
        type(dilation_val) is int or type(dilation_val) is ToBeDetermined
    ), "Invalid dilation value!"
    kernel_size_val: ToBeDetermined | int = TBD
    if len(kernel_shp := kernel.shape.get_shapes()) == 3 and isinstance(
        kernel_shp[-1], int
    ):
        kernel_size_val = kernel_shp[-1]
    is_input_propagatable = len(input_shape.suffix) >= 1 or (
        input_shape.root is None and len(input_shape.prefix) > 1
    )
    is_output_propagatable = len(output_shape.suffix) >= 1 or (
        output_shape.root is None and len(output_shape.prefix) > 1
    )

    if (
        is_input_propagatable
        and is_output_propagatable
        and not isinstance(stride_val, ToBeDetermined)
        and not isinstance(padding_val, ToBeDetermined)
        and not isinstance(dilation_val, ToBeDetermined)
        and not isinstance(kernel_size_val, ToBeDetermined)
    ):
        status, _updates = sliding_window_constraint_helper(
            output_shape[-1],
            input_shape[-1],
            stride_val,
            padding_val,
            dilation_val,
            kernel_size_val,
        )
        updates |= _updates
    return status, updates


# TODO: Change name (Conv also uses the constraint below)
def sliding_window_2d_constraints(
    output: Tensor,
    input: Tensor,
    stride: Scalar,
    padding: Scalar,
    dilation: Scalar,
    kernel_size: Scalar,
) -> ConstrainResultType:
    status = False
    updates = Updates()
    assert input._temp_shape is not None, "Input shape of Convolution2D is not set!"
    assert output._temp_shape is not None, "Output shape of Convolution2D is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape

    stride_val = stride.value
    padding_val = padding.value
    dilation_val = dilation.value
    kernel_size_val = kernel_size.value

    assert is_tuple_of_two_ints(stride_val) or isinstance(
        stride_val, ToBeDetermined
    ), "Invalid stride value!"
    assert is_tuple_of_two_ints(dilation_val) or isinstance(
        dilation_val, ToBeDetermined
    ), "Invalid stride value!"
    assert is_tuple_of_two_ints(kernel_size_val) or isinstance(
        kernel_size_val, ToBeDetermined
    ), "Invalid stride value!"
    assert is_padding_type(padding_val) or isinstance(
        padding_val, ToBeDetermined
    ), "Invalid padding value!"

    is_input_propagatable = len(input_shape.suffix) >= 2 or (
        input_shape.root is None and len(input_shape.prefix) > 2
    )
    is_output_propagatable = len(output_shape.suffix) >= 2 or (
        output_shape.root is None and len(output_shape.prefix) > 2
    )

    # To calculate maxpool constraint we need to know ... and last 2 dimension of
    # the input
    if (
        not isinstance(stride_val, ToBeDetermined)
        and not isinstance(padding_val, ToBeDetermined)
        and not isinstance(dilation_val, ToBeDetermined)
        and not isinstance(kernel_size_val, ToBeDetermined)
        and is_input_propagatable
        and is_output_propagatable
    ):
        status_height, symbols_height = sliding_window_constraint_helper(
            output_shape[-2],
            input_shape[-2],
            stride_val[0],
            padding_val[0],
            dilation_val[0],
            kernel_size_val[0],
        )
        status_width, symbols_width = sliding_window_constraint_helper(
            output_shape[-1],
            input_shape[-1],
            stride_val[1],
            padding_val[1],
            dilation_val[1],
            kernel_size_val[1],
        )
        status = (
            status_height and status_width and input_shape.root == output_shape.root
        )
        updates |= symbols_height
        updates |= symbols_width

    return status, updates


def conv_2d_constraints(
    output: Tensor,
    input: Tensor,
    stride: Scalar,
    padding: Scalar,
    dilation: Scalar,
    kernel: Tensor,
) -> ConstrainResultType:
    status = False
    updates = Updates()
    assert input._temp_shape is not None, "Input shape of Convolution2D is not set!"
    assert output._temp_shape is not None, "Output shape of Convolution2D is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape

    stride_val = stride.value
    padding_val = padding.value
    dilation_val = dilation.value

    assert is_tuple_of_two_ints(stride_val) or isinstance(
        stride_val, ToBeDetermined
    ), "Invalid stride value!"
    assert is_tuple_of_two_ints(dilation_val) or isinstance(
        dilation_val, ToBeDetermined
    ), "Invalid stride value!"
    assert is_padding_type(padding_val) or isinstance(
        padding_val, ToBeDetermined
    ), "Invalid padding value!"

    kernel_size_0: ToBeDetermined | int = TBD
    kernel_size_1: ToBeDetermined | int = TBD
    if (
        len(kernel_shp := kernel.shape.get_shapes()) == 4
        and isinstance(kernel_shp[-1], int)
        and isinstance(kernel_shp[-2], int)
    ):
        kernel_size_0 = kernel_shp[-2]
        kernel_size_1 = kernel_shp[-1]

    is_input_propagatable = len(input_shape.suffix) >= 2 or (
        input_shape.root is None and len(input_shape.prefix) > 2
    )
    is_output_propagatable = len(output_shape.suffix) >= 2 or (
        output_shape.root is None and len(output_shape.prefix) > 2
    )

    # To calculate maxpool constraint we need to know ... and last 2 dimension of
    # the input
    if (
        not isinstance(stride_val, ToBeDetermined)
        and not isinstance(padding_val, ToBeDetermined)
        and not isinstance(dilation_val, ToBeDetermined)
        and not isinstance(kernel_size_0, ToBeDetermined)
        and not isinstance(kernel_size_1, ToBeDetermined)
        and is_input_propagatable
        and is_output_propagatable
    ):
        status_height, symbols_height = sliding_window_constraint_helper(
            output_shape[-2],
            input_shape[-2],
            stride_val[0],
            padding_val[0],
            dilation_val[0],
            kernel_size_0,
        )
        status_width, symbols_width = sliding_window_constraint_helper(
            output_shape[-1],
            input_shape[-1],
            stride_val[1],
            padding_val[1],
            dilation_val[1],
            kernel_size_1,
        )
        status = (
            status_height and status_width and input_shape.root == output_shape.root
        )
        updates |= symbols_height
        updates |= symbols_width

    return status, updates


def flatten_constrains(
    output: Tensor, input: Tensor, start_dim: Scalar, end_dim: Scalar
) -> ConstrainResultType:
    status = False
    updates = Updates()
    new_shape_items = set()
    assert input._temp_shape is not None, "Input shape of Flatten is not set!"
    assert output._temp_shape is not None, "Output shape of Flatten is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    start_dim_val = start_dim.value
    end_dim_val = end_dim.value
    assert type(start_dim_val) is int or type(start_dim_val) is ToBeDetermined
    assert type(end_dim_val) is int or type(end_dim_val) is ToBeDetermined

    if (
        isinstance(start_dim_val, ToBeDetermined)
        and not isinstance(end_dim_val, ToBeDetermined)
        and end_dim_val >= 0
    ):
        input_prefix = [Uniadic() for _ in range(end_dim_val + 1)]
        updates |= input_shape.inner_match(prefix=input_prefix, root=Variadic())
        updates |= output_shape.inner_match(prefix=[Uniadic()], root=Variadic())

    elif (
        isinstance(start_dim_val, ToBeDetermined)
        and not isinstance(end_dim_val, ToBeDetermined)
        and end_dim_val < 0
    ):
        input_suffix = [Uniadic() for _ in range(abs(end_dim_val))]
        updates |= input_shape.inner_match(suffix=input_suffix, root=(Variadic()))
        uni_from_input = input_shape.suffix[len(input_shape.suffix) + end_dim_val + 1 :]
        updates |= output_shape.inner_match(
            root=Variadic(), suffix=[Uniadic()] + uni_from_input
        )

    elif (
        not isinstance(start_dim_val, ToBeDetermined)
        and start_dim_val >= 0
        and isinstance(end_dim_val, ToBeDetermined)
    ):
        input_prefix = [Uniadic() for _ in range(start_dim_val + 1)]
        updates |= input_shape.inner_match(prefix=input_prefix, root=(Variadic()))
        updates |= output_shape.inner_match(
            prefix=input_prefix[:-1] + [Uniadic()], root=Variadic()
        )

    elif (
        not isinstance(start_dim_val, ToBeDetermined)
        and start_dim_val >= 0
        and not isinstance(end_dim_val, ToBeDetermined)
        and end_dim_val >= 0
    ):
        input_prefix = [Uniadic() for _ in range(end_dim_val + 1)]
        output_prefix = input_prefix[:start_dim_val] + [
            Uniadic() if start_dim_val != end_dim_val else input_prefix[start_dim_val]
        ]
        new_var = Variadic()
        updates |= input_shape.inner_match(prefix=input_prefix, root=new_var)
        updates |= output_shape.inner_match(prefix=output_prefix, root=new_var)

    elif (
        not isinstance(start_dim_val, ToBeDetermined)
        and start_dim_val >= 0
        and not isinstance(end_dim_val, ToBeDetermined)
        and end_dim_val < 0
    ):
        input_prefix = [Uniadic() for _ in range(start_dim_val + 1)]
        input_suffix = [Uniadic() for _ in range(abs(end_dim_val) - 1)]
        updates |= input_shape.inner_match(
            prefix=input_prefix, root=Variadic(), suffix=input_suffix
        )
        updates |= output_shape.inner_match(
            prefix=input_prefix[:start_dim_val] + [Uniadic()] + input_suffix
        )

    elif (
        not isinstance(start_dim_val, ToBeDetermined)
        and start_dim_val < 0
        and isinstance(end_dim_val, ToBeDetermined)
    ):
        input_suffix = [Uniadic() for _ in range(abs(start_dim_val))]
        updates |= input_shape.inner_match(suffix=input_suffix, root=Variadic())
        # Output should have at least 1 dimension (i.e. end_dim = -1).
        updates |= output_shape.inner_match(prefix=[Uniadic()], root=Variadic())

    elif (
        not isinstance(start_dim_val, ToBeDetermined)
        and start_dim_val < 0
        and not isinstance(end_dim_val, ToBeDetermined)
        and end_dim_val < 0
    ):
        input_suffix = [Uniadic() for _ in range(abs(start_dim_val))]
        suffix = input_suffix[end_dim_val + 1 :] if end_dim_val != -1 else []
        output_suffix = [
            Uniadic() if start_dim_val != end_dim_val else input_suffix[start_dim_val]
        ] + suffix
        new_var = Variadic()
        updates |= input_shape.inner_match(suffix=input_suffix, root=new_var)
        updates |= output_shape.inner_match(suffix=output_suffix, root=new_var)

    if not isinstance(start_dim_val, ToBeDetermined) and not isinstance(
        end_dim_val, ToBeDetermined
    ):
        prod = 1
        if input_shape.root is None:
            input_shapes = input_shape.prefix
            abs_start_dim = (
                start_dim_val
                if start_dim_val >= 0
                else len(input_shapes) - abs(start_dim_val)
            )
            abs_end_dim = (
                end_dim_val
                if end_dim_val >= 0
                else len(input_shapes) - abs(end_dim_val)
            )
            if abs_start_dim >= abs_end_dim:
                raise ValueError("Start_dim cannot be greater or equal to end dim!")
            if not (0 <= abs_start_dim <= len(input_shapes)):
                raise ValueError(
                    "value of start dim out of boundary (start dim needs to be in "
                    "range of ({-len(input_shapes)}, {len(input_shapes) - 1}). But "
                    "given start dim is {start_dim_val}"
                )
            if not (0 <= abs_end_dim <= len(input_shapes)):
                raise ValueError(
                    "value of end dim out of boundary (end dim needs to be in range of "
                    "({-len(input_shapes)}, {len(input_shapes) - 1}). But given end dim"
                    " is {end_dim_val}"
                )
            keys = [key.value for key in input_shapes[abs_start_dim : abs_end_dim + 1]]
            if is_list_int(keys):
                prod = math.prod(keys)
                status = True
                suffix = input_shapes[end_dim_val + 1 :] if end_dim_val != -1 else []
                prefix = input_shapes[:start_dim_val]
                updates |= output_shape.inner_match(
                    prefix=prefix + [(new_uni := Uniadic(prod))] + suffix
                )
                new_shape_items.add(new_uni)
    return status, updates


def where_constrains(
    output: Tensor, cond: Tensor, input1: Tensor, input2: Tensor
) -> ConstrainResultType:
    # TODO: Find a way to implement this constraint without creating a Tensor and
    # ShapeRepr
    assert output._temp_shape is not None, "Output shape of Where is not set!"
    assert cond._temp_shape is not None, "Condition shape of Where is not set!"
    assert input1._temp_shape is not None, "Input1 shape of Where is not set!"
    assert input2._temp_shape is not None, "Input2 shape of Where is not set!"
    status = False
    updates = Updates()
    new_shape_items = set()

    broadcast_shp = ShapeRepr(root=(new_var := Variadic()))
    new_shape_items.add(new_var)

    _, local_updates = bcast_helper(
        broadcast_shp, input1._temp_shape, input2._temp_shape, 0
    )
    updates |= local_updates
    status, local_updates = bcast_helper(
        output._temp_shape, broadcast_shp, cond._temp_shape, 0
    )
    updates |= local_updates
    return status, updates


def arange_constraints(
    output: Tensor, start: Scalar, stop: Scalar, step: Scalar
) -> ConstrainResultType:
    assert output._temp_shape is not None, "Output shape of Arange is not set!"
    output_shape: ShapeRepr = output._temp_shape
    status = False
    updates = Updates()
    start_val = start.value
    stop_val = stop.value
    step_val = step.value
    assert (
        type(start_val) is int
        or type(start_val) is ToBeDetermined
        or type(start_val) is float
    )
    assert (
        type(stop_val) is int
        or type(stop_val) is ToBeDetermined
        or type(stop_val) is float
    )
    assert (
        type(step_val) is int
        or type(step_val) is ToBeDetermined
        or type(step_val) is float
    )

    if (
        not isinstance(start_val, ToBeDetermined)
        and not isinstance(stop_val, ToBeDetermined)
        and not isinstance(step_val, ToBeDetermined)
    ):
        # Check consistencies.
        if start_val > stop_val and step_val > 0:
            raise ValueError(
                f"Start number ({start_val}) can not be "
                f"higher than stop number ({stop_val}) "
                f"while step = {step_val}"
            )
        elif start_val < stop_val and step_val < 0:
            raise ValueError(
                f"Start number ({start_val}) can not be "
                f"lower than stop number ({stop_val}) "
                f"while step = {step_val}"
            )
        # Set value.
        val = (start_val - stop_val) / step_val
        # If value has decimal part take absolute of integer part of it
        # and add 1.
        val = abs(int(val)) if int(val) == val else abs(int(val)) + 1

        if output_shape.root is None:
            # Check output length is consistent with val.
            if len(output_shape) != (val != 0):
                raise ValueError(
                    f"Arange output shape can only have {[0, 1][val != 0]} dim in this "
                    f"setting. Got {len(output_shape)} dim(s) here."
                )
            elif val > 0 and (uni := output_shape.prefix[0]).set_value(val):
                updates.add(uni)
            status = True
        elif (min_dims := len(output_shape)) <= 1:
            if val > 0:
                out_uniadic = [Uniadic()]
                updates |= output_shape._update_uniadics(
                    output_shape.prefix, out_uniadic
                )
                updates |= output_shape._update_uniadics(
                    output_shape.reverse, out_uniadic
                )
                updates |= output_shape.remove_variadic(out_uniadic)
            elif min_dims != 1:
                updates |= output_shape.remove_variadic([])  # Simply empty list.
            else:
                raise ValueError(
                    f"Arange output shape has minimum {min_dims} dim(s) where it is a "
                    "rank-0 array."
                )
            status = True
        else:
            raise ValueError(
                f"Shape mismatch. Output has at least {min_dims} dim(s) where it can "
                "have at most 1 dim."
            )
    # TODO: Should we try to infer step if start, stop value and output shape is known?
    # updated_symbols -= new_shape_items
    return status, updates


def broadcast_to_constraints(
    output: Tensor, shape: Scalar, input: Tensor
) -> ConstrainResultType:
    status = False
    updates = Updates()
    assert input._temp_shape is not None, "Input shape of BroadcastTo is not set!"
    assert output._temp_shape is not None, "Output shape of BroadcastTo is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    shape_val = shape.value
    assert is_tuple_int(shape_val) or isinstance(
        shape_val, ToBeDetermined
    ), "Invalid shape value!"

    if not isinstance(shape_val, ToBeDetermined):
        if output_shape.root is not None:
            # Check shape consistency.
            if (
                min_dims := (len(output_shape.prefix) + len(output_shape.suffix))
            ) > len(shape_val):
                raise ValueError(
                    f"Shape mismatch. Output has minimum {min_dims} dim(s) where it "
                    f"must have exactly {len(shape_val)} dim(s)."
                )
            out_uniadics = [Uniadic(dim) for dim in shape_val]
            updates |= output_shape._update_uniadics(output_shape.prefix, out_uniadics)
            updates |= output_shape._update_uniadics(
                output_shape.reverse, out_uniadics[::-1]
            )
            updates |= output_shape.remove_variadic(out_uniadics)

        else:
            # Check shape consistency.
            if len(output_shape) != len(shape_val):
                raise ValueError(
                    f"Shape mismatch. Output has {len(output_shape)} dim(s) "
                    f"where it must "
                    f"have {len(shape_val)} dim(s)."
                )
            for idx, shp in enumerate(shape_val):
                if (uni := output_shape.prefix[idx]).set_value(shp):
                    updates.add(uni)

        if input_shape.root is None:
            # if input is uniadic, look for if every input is determined,
            # if determined, validate its shape (whether if it matches
            # to output's shape based on bcast rule). If it is validated,
            # set status to True.
            for uni in input_shape.prefix:
                if uni.value is None:
                    break
            else:
                validate_bcast(input_shape, shape_val)
                status = True

    return status, updates


def validate_bcast(input: ShapeRepr, shape: tuple[int, ...]):
    if input.root is None:
        if len(input) > len(shape):
            raise ValueError("Cannot broadcast to lower dimension")
        for idx, in_uni in enumerate(input.reverse):
            out_value = shape[-idx - 1]
            if in_uni.value != 1 and in_uni.value != out_value:
                raise ValueError("Shape mismatch in broadcast_to model")


def reshape_constraints(
    output: Tensor, input: Tensor, shape: Scalar
) -> ConstrainResultType:
    # TODO: We can add inference for the case where
    # shape = (1,2,3,4), input_shape = (1, 2, 4, "u1") for example.
    # Last dimension of input is obviously 3.
    status = False
    updates = Updates()
    assert input._temp_shape is not None, "Input shape of Reshape is not set!"
    assert output._temp_shape is not None, "Output shape of Reshape is not set!"

    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    shape_val = shape.value
    assert (
        is_tuple_int_or_none(shape_val)
        or isinstance(shape_val, ToBeDetermined)
        or is_list_int_or_none(shape_val)
    ), "Invalid shape value!"
    if not isinstance(shape_val, ToBeDetermined):
        known_input = False
        shp_prod = 1
        if input_shape.root is None and input_shape.prefix:
            input_shape_values = [uni.value for uni in input_shape.prefix]
            if known_input := is_list_int(input_shape_values):
                input_prod = math.prod(input_shape_values)
                if is_list_int(shape_val) or is_tuple_int(shape_val):
                    shp_prod = math.prod(shape_val)
                    # Check original shape and reshaped one are consistent.

                    if [
                        shp_prod != input_prod,
                        not (input_prod / shp_prod).is_integer(),
                    ][-1 in shape_val]:
                        raise ValueError(
                            f"Input {tuple(uni.value for uni in input_shape.prefix)}"
                            f" can not be"
                            f" reshaped to {shape_val}"
                        )
        if output_shape.root is not None:
            if (min_out := len(output_shape)) > len(shape_val):
                raise ValueError(
                    f"Shape mismatch! Output has mimimum {min_out} dim(s) while "
                    f"reshaped one has {len(shape_val)} dim(s)."
                )
            out_uniadics = [
                Uniadic(val) if val != -1 else Uniadic() for val in shape_val
            ]
            updates |= output_shape._update_uniadics(output_shape.prefix, out_uniadics)
            updates |= output_shape._update_uniadics(
                output_shape.reverse, out_uniadics[::-1]
            )
            updates |= output_shape.remove_variadic(out_uniadics)
        # Infer towards output.
        if len(output_shape) != len(shape_val):
            raise ValueError(
                f"Shape mismatch! Output has {len(output_shape)} dim(s) "
                f"while reshaped one "
                f"has {len(shape_val)} dim(s)."
            )

        for idx, shp in enumerate(shape_val):
            if shp != -1 and (uni := output_shape.prefix[idx]).set_value(shp):
                # TODO: Here we're adding uniadic symbol without checking
                # if it was created in this call or already contained in
                # output. Normally, we do not add newly created symbols into the
                # updated symbols set.
                updates.add(uni)
        # Handle the case when shape_val contains -1 value.
        if -1 in shape_val and known_input:
            idx = shape_val.index(-1)
            value = int(input_prod / (-shp_prod))
            if (uni := output_shape.prefix[idx]).set_value(value):
                updates.add(uni)

        if (-1 not in shape_val) and (
            is_list_int(shape_val) or is_tuple_int(shape_val)
        ):
            # Handle the inference where, there is only one unknown shape in
            # input shapes and/or output shapes. If it is the case,
            # shape of the last unknown shape can be simply found as:
            # (product of given shape values) / (product of known tensor shapes)
            # Note that there should be not -1 in shape values

            # TODO: add also this inference between input shape and output shape.
            # Same logic still holds
            if input_shape.root is None:
                input_values = [uni.value for uni in input_shape.prefix]
                if input_values.count(None) == 1:
                    none_index = input_values.index(None)
                    uni_val = reduce(prod_fn, shape_val) // reduce(
                        prod_fn, filter(None, input_values)
                    )
                    if (uni := input_shape.prefix[none_index]).set_value(uni_val):
                        updates.add(uni)

            if output_shape.root is None:
                output_values = [uni.value for uni in output_shape.prefix]
                if output_values.count(None) == 1:
                    none_index = output_values.index(None)
                    uni_val = reduce(prod_fn, shape_val) // reduce(
                        prod_fn, filter(None, output_values)
                    )
                    if (uni := output_shape.prefix[none_index]).set_value(uni_val):
                        updates.add(uni)

    # Try to infer shape value.
    elif is_repr_known(output_shape):
        if is_repr_known(input_shape) and reduce(prod_fn, input_shape.prefix) != reduce(
            prod_fn, output_shape.prefix
        ):
            out_shape = tuple(uni.value for uni in output_shape.prefix)
            in_shape = tuple(uni.value for uni in input_shape.prefix)
            raise ValueError(
                f"Shape mismatch! output {out_shape} and input {in_shape} have "
                "incompatible shapes"
            )
        values: list[int | None] | tuple[int | None, ...] = [
            uni.value for uni in output_shape.prefix
        ]
        assert isinstance(shape._type, GenericAlias)
        if shape._type.__origin__ is tuple:
            values = tuple(values)
        # TODO: This update assumes no -1 is given in shapes. However,
        # situations may occur where shape is given with -1.

        # EX: if output shape is [1, 2, 3, 4, 5], this part of the code
        # directly assumes shape scalar will be also [1, 2, 3, 4, 5].
        # However, shape scalar may be given with [1, 2, 3, 4, -1], [1, 2, -1, 4, 5],
        # etc, and still can be valid reshape
        updates |= shape.set_value(values)
    status = is_repr_known(input_shape) and is_repr_known(output_shape)
    return status, updates


def squeeze_constraints(output: Tensor, input: Tensor) -> ConstrainResultType:
    updates = Updates()
    assert input._temp_shape is not None, "Input shape of Squeeze is not set!"
    assert output._temp_shape is not None, "Output shape of Squeeze is not set!"

    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    status = output_shape.root is None

    if input_shape.root is None and len(input_shape) < len(output_shape):
        raise ValueError(
            f"Output shape can not have higher number of dimensions"
            f" (min {len(output_shape)})"
            f" than input ({len(input_shape)})"
        )
    if any(
        [
            uni.value == 1
            for uni in output_shape.prefix + output_shape.suffix
            if uni.value is not None
        ]
    ):
        raise ValueError(
            "Squeeze output shape can not have any dimensionality as 1, got output "
            f"shape as {output_shape.get_shapes()}"
        )
    if output_shape.root is None:
        # TODO: Handle the case where output is None. Fill all places
        # with ones in input shape other than the values in output shape.
        # For example: input -> [4, Var, 2, u], output -> [4, 2], then
        # u = 1
        ...
    new_prefix, new_suffix = [], []
    variadic_required = False

    for uni in input_shape.prefix:
        if uni.value is None:
            variadic_required = True
            break
        elif uni.value != 1:
            new_prefix.append(uni)

    # If Variadic input, iterate over reverse suffix else
    # reverse prefix.
    reverse_uni_list = list()
    for uni in (
        input_shape.suffix[::-1]
        if input_shape.root is not None
        else input_shape.prefix[::-1]
    ):
        if uni.value is None:
            variadic_required = True
            break
        elif uni.value != 1:
            reverse_uni_list.append(uni)

    new_var = None
    if variadic_required or input_shape.root is not None:
        new_var = Variadic()
        new_suffix = reverse_uni_list[::-1]

    # Match shape representation.
    updates |= output_shape.inner_match(
        prefix=new_prefix, root=new_var, suffix=new_suffix
    )

    if output_shape.root is None:
        status = True

    return status, updates


def size_constraints(output: Scalar, input: Tensor, dim: Scalar) -> ConstrainResultType:
    assert input._temp_shape is not None, "Input shape of Size is not set!"
    input_shape: ShapeRepr = input._temp_shape

    status = False
    updates = Updates()
    dim_val = dim.value
    output_val = output.value
    assert (
        type(dim_val) is int
        or is_tuple_int(dim_val)
        or isinstance(dim_val, ToBeDetermined)
        or dim_val is None
    )
    assert (
        type(output_val) is int
        or is_tuple_int(output_val)
        or isinstance(output_val, ToBeDetermined)
    )
    if not isinstance(dim_val, ToBeDetermined):
        is_int = False
        if isinstance(dim_val, int):
            is_int = True
            dim_val = [dim_val]
        if dim_val is None:
            max_dim = -float("inf")
        else:
            pos_dims = [item for item in dim_val if item >= 0]
            neg_dims = [item for item in dim_val if item < 0]
            max_dim = (
                max(max(pos_dims) + 1, abs(min(neg_dims)))
                if pos_dims and neg_dims
                else (max(pos_dims) + 1 if pos_dims else abs(min(neg_dims)))
            )
        if input_shape.root is None:
            if len(input_shape) < (max_dim):
                # Check if input shape has at least (dim + 1) dimensions
                # if dim is not None, else raise ValueError.
                raise ValueError(
                    f"Input has dimensionality of {len(input_shape)}. "
                    f"Should be at least "
                    "{max_dim} dimensional when dim = {original_dim}"
                )
        elif dim_val is not None and len(input_shape) < (max_dim):
            prefix = [Uniadic() for _ in range(int(max_dim))]
            updates |= input_shape.inner_match(prefix=prefix, root=Variadic())
            # TODO: Is it required to do the below check here? Is it possible to
            # have len(input) < (max_dim + 1) after above inner_match operation???

        if dim_val is not None:
            is_all_int = False
            if input_shape.root is not None:
                if pos_dims and neg_dims:
                    if len(input_shape.prefix) >= (max(pos_dims) + 1) and len(
                        input_shape.suffix
                    ) >= abs(min(neg_dims)):
                        is_all_int = all(
                            isinstance(input_shape.prefix[idx].value, int)
                            for idx in pos_dims
                        ) and all(
                            isinstance(input_shape.suffix[idx].value, int)
                            for idx in neg_dims
                        )
                elif pos_dims:
                    if len(input_shape.prefix) >= (max(pos_dims) + 1):
                        is_all_int = all(
                            isinstance(input_shape.prefix[idx].value, int)
                            for idx in pos_dims
                        )
                elif len(input_shape.suffix) >= abs(min(neg_dims)):
                    is_all_int = all(
                        isinstance(input_shape.suffix[idx].value, int)
                        for idx in neg_dims
                    )
            else:
                is_all_int = all(
                    isinstance(input_shape[idx].value, int) for idx in dim_val
                )

            if is_all_int:
                if is_int:
                    updates |= output.set_value(input_shape[dim_val[0]].value)
                else:
                    updates |= output.set_value(
                        tuple(input_shape[idx].value for idx in dim_val)
                    )
                status = True

            elif not isinstance(output_val, ToBeDetermined):
                if isinstance(output_val, int):
                    output_val = (output_val,)
                output_value = tuple(output_val)
                max_pos_dim = max(pos_dims) + 1 if pos_dims else 0
                max_neg_dim = -min(neg_dims) if neg_dims else 0

                input_prefix = []
                for idx, _ in enumerate(range(max_pos_dim)):
                    if len(input_shape.prefix) > idx:
                        input_prefix.append(input_shape.prefix[idx])
                    else:
                        input_prefix.append(Uniadic())

                input_suffix = []
                rev_suffix = input_shape.suffix[::-1]
                for idx, _ in enumerate(range(max_neg_dim)):
                    if len(rev_suffix) > idx:
                        input_suffix.append(rev_suffix[idx])
                    else:
                        input_suffix.append(Uniadic())
                input_suffix = input_suffix[::-1]

                for dim_value, out_val in zip(dim_val, output_value, strict=False):
                    if dim_value >= 0:
                        if len(input_shape.prefix) >= dim_value:
                            if input_shape.prefix[dim_value].set_value(out_val):
                                updates.add(input_shape.prefix[dim_value])
                        else:
                            input_prefix[dim_value].set_value(out_val)
                    else:
                        if len(input_shape.suffix) > abs(dim_value):
                            if input_shape.suffix[dim_value].set_value(out_val):
                                updates.add(input_shape.suffix[dim_value])
                        else:
                            input_suffix[dim_value].set_value(out_val)
                updates |= input_shape.inner_match(
                    prefix=input_prefix, root=(Variadic())
                )
                updates |= input_shape.inner_match(
                    root=(Variadic()), suffix=input_suffix
                )
                if input_shape.root is None:
                    status = all(
                        isinstance(input_shape[idx].value, int) for idx in dim_val
                    )
                else:
                    status = (
                        len(input_shape.prefix) >= max_pos_dim
                        and len(input_shape.suffix) >= max_neg_dim
                    )

        elif input_shape.root is None:
            input_shape_values = [uni.value for uni in input_shape.prefix]
            if is_list_int(input_shape_values):
                updates |= output.set_value(math.prod(input_shape_values))
                status = True
    return status, updates


def shape_constraints(output: Scalar, input: Tensor) -> ConstrainResultType:
    assert input._temp_shape is not None, "Input shape of Shape is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_val = output.value
    assert isinstance(output_val, ToBeDetermined) or is_tuple_int(output_val)
    status = False
    updates = Updates()

    if input_shape.root is None:
        in_shape = input_shape.get_shapes({}, {})
        if all(isinstance(x, int) for x in in_shape):
            updates |= output.set_value(tuple(in_shape))
            # NOTE: Should we add output.scalar into the updated_symbols???
            status = True
    elif not isinstance(output_val, ToBeDetermined):
        input_prefix = [Uniadic(val) for val in output_val]
        updates |= input_shape.inner_match(prefix=input_prefix)
        status = True

    return status, updates


def eye_constraints(output: Tensor, N: Scalar, M: Scalar) -> ConstrainResultType:
    updates = Updates()
    assert output._temp_shape is not None, "Output shape of Eye is not set!"
    output_shape: ShapeRepr = output._temp_shape
    n_uni, m_uni = output_shape.prefix[0], output_shape.prefix[1]
    n_valued = isinstance(N.value, int)
    m_valued = isinstance(M.value, int | NoneType)
    n_uni_valued = isinstance(n_uni.value, int)
    m_uni_valued = isinstance(m_uni.value, int)

    if n_valued and not n_uni_valued:
        n_uni.set_value(N.value)
        updates.add(n_uni)
    elif n_uni_valued and not n_valued:
        N.set_value(n_uni.value)
        updates.add(N)

    if m_valued and not m_uni_valued:
        m_uni.set_value(M.value)
        updates.add(m_uni)
    elif m_uni_valued and not m_valued:
        M.set_value(m_uni.value)
        updates.add(M)
    all_items: list[Scalar | Uniadic] = [N, M, n_uni, m_uni]
    return all(isinstance(s.value, int) for s in all_items), updates


def swap_axes_constraints(
    output: Tensor, input: Tensor, axis1: Scalar, axis2: Scalar
) -> ConstrainResultType:
    assert input._temp_shape is not None, "Input shape of SwapAxes is not set!"
    assert output._temp_shape is not None, "Output shape of SwapAxes is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    axis1_val = axis1.value
    axis2_val = axis2.value
    assert type(axis1_val) is int or isinstance(axis1_val, ToBeDetermined)
    assert type(axis2_val) is int or isinstance(axis2_val, ToBeDetermined)
    status = False
    updates = Updates()

    if not isinstance(axis1_val, ToBeDetermined) and not isinstance(
        axis2_val, ToBeDetermined
    ):
        if input_shape.root is not None and output_shape.root is not None:
            # Find minimum required prefix and suffx length
            # for input/output for given axis values taking
            # corresponding signs into account.
            min_len = axis1_val + 1 if axis1_val > 0 else -axis1_val
            min_pre_len = 0 if axis1_val < 0 else min_len
            min_suf_len = 0 if axis1_val > 0 else min_len
            if axis2_val >= 0 and min_len < (axis2_val + 1):
                min_len = axis2_val + 1
                min_pre_len = min_len - min_suf_len
            elif axis2_val < 0 and min_len < abs(axis2_val):
                min_len = -axis2_val
                min_suf_len = min_len - min_pre_len
            # Create a repr which has minimum length of min_len
            # and then match input/output repr's with this new repr.

            prefix: list[Uniadic] = []
            for idx, _ in enumerate(range(min_pre_len)):
                if len(input_shape.prefix) > idx:
                    prefix.append(input_shape.prefix[idx])
                else:
                    prefix.append(Uniadic())

            suffix: list[Uniadic] = []
            rev_suffix = input_shape.suffix[::-1]
            for idx, _ in enumerate(range(min_suf_len)):
                if len(rev_suffix) > idx:
                    suffix.append(rev_suffix[idx])
                else:
                    suffix.append(Uniadic())
            suffix = suffix[::-1]
            updates |= input_shape.inner_match(
                prefix=prefix, root=(new_var := Variadic()), suffix=suffix
            )

            if (axis1_val < 0 and axis2_val < 0) or (axis1_val >= 0 and axis2_val >= 0):
                # Swap corresponding axes and match with output if axis indices
                # are available for corresponding prefix or suffix.
                if axis1_val >= 0 and axis2_val >= 0:
                    prefix[axis1_val], prefix[axis2_val] = (
                        prefix[axis2_val],
                        prefix[axis1_val],
                    )
                if axis1_val < 0 and axis2_val < 0:
                    suffix[axis1_val], suffix[axis2_val] = (
                        suffix[axis2_val],
                        suffix[axis1_val],
                    )
                updates |= output_shape.inner_match(
                    prefix=prefix, root=new_var, suffix=suffix
                )
                status = True

            else:
                positive_axis = max(axis1_val, axis2_val)
                negative_axis = min(axis1_val, axis2_val)
                # Find minimum common length for input and output.
                min_common_len = min(positive_axis, len(input_shape) + negative_axis)
                # Add common ones and non-common ones to output prefix.
                out_pre = prefix[:min_common_len]
                out_pre += [Uniadic() for _ in range(len(prefix) - min_common_len)]
                # Output suffix length is equal to input suffix length
                # but may have different values.
                out_suf = [Uniadic() for _ in range(len(suffix))]
                updates |= output_shape.inner_match(
                    prefix=out_pre, root=Variadic(), suffix=out_suf
                )
        else:
            # We can use non-variadic one to match with another (Variadic or not)
            # and then swap corresponding axes.
            non_variadic = [input_shape, output_shape][output_shape.root is None]
            len_prefix = len(non_variadic.prefix)
            if not -len_prefix <= axis1_val <= len_prefix - 1:
                raise ValueError(
                    "axis1 exceeds the shape bounds in swapaxes model (axis1 "
                    "should be in range of ({-len_prefix}, {len_prefix -1}) but "
                    "given axis1 is {axis1_val})"
                )
            if not -len_prefix <= axis2_val <= len_prefix - 1:
                raise ValueError(
                    "axis2 exceeds the shape bounds in swapaxes model (axis2 "
                    "should be in range of ({-len_prefix}, {len_prefix -1}) but "
                    "given axis2 is {axis1_val})"
                )
            other = input_shape if non_variadic == output_shape else output_shape
            if other.root is None:
                updates |= other[axis1_val].match(non_variadic[axis2_val])
                updates |= other[axis2_val].match(non_variadic[axis1_val])

            else:
                updates |= other._match(non_variadic)
                other[axis1_val], other[axis2_val] = other[axis2_val], other[axis1_val]
            status = True

    elif isinstance(axis1_val, ToBeDetermined) ^ isinstance(axis2_val, ToBeDetermined):
        # If only one of the axes are given. Find the given axis.
        # create uniadics with the same amount of this axis and match it
        # with input
        if not isinstance(axis1_val, ToBeDetermined):
            given_axis = axis1_val
        elif not isinstance(axis2_val, ToBeDetermined):
            given_axis = axis2_val
        if given_axis >= 0:
            unis = [Uniadic() for _ in range(given_axis + 1)]
        elif given_axis < 0:
            unis = [Uniadic() for _ in range(abs(given_axis))]
        updates |= input_shape.inner_match(prefix=unis, root=Variadic())

    return status, updates


def to_tensor_constraints(output: Tensor, input: Scalar) -> ConstrainResultType:
    updates = Updates()
    status = False
    assert output._temp_shape is not None, "Output shape of ToTensor is not set!"
    output_shape: ShapeRepr = output._temp_shape
    input_val = input.value
    assert (
        type(input_val) is list
        or type(input_val) is tuple
        or type(input_val) is int
        or type(input_val) is float
        or type(input_val) is Constant
        or isinstance(input_val, ToBeDetermined)
    ), "Invalid input value!"

    if not isinstance(input_val, ToBeDetermined):
        shape = []
        if isinstance(input_val, list | tuple):
            # list_shape function takes only list type input.
            shape = [idx for idx in list_shape(list(input_val))]
            typ = find_dominant_type(input_val)
            updates |= output.set_type(typ)
            updates.add(output, update_type=UpdateType.TYPE)
        elif isinstance(input_val, float | int):
            shape = []
            updates |= output.set_type(input._type)
            updates.add(output, update_type=UpdateType.TYPE)
        if output_shape.root is None:
            if len(shape) != len(output_shape.prefix):
                raise ValueError("Shape dimensions does not match")
            else:
                for uni_out, uni_in in zip(output_shape.prefix, shape, strict=False):
                    if (uni_out.value is not None) and (uni_in != uni_out.value):
                        raise ValueError("Shape representations does not match")

        for uni, value in zip(output_shape.prefix, shape, strict=False):
            if uni.set_value(value):
                updates.add(uni)

        if output_shape.root is not None:
            for uni, value in zip(output_shape.reverse, shape[::-1], strict=False):
                if uni.set_value(value):
                    updates.add(uni)
            replacement = [Uniadic(uni) for uni in shape]
            updates |= output_shape.remove_variadic(replacement)

        status = True
    return status, updates


def tensor_to_list_constraints(output: Scalar, input: Tensor) -> ConstrainResultType:
    assert input._temp_shape is not None, "Input shape of TensorToList is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_val = output.value
    assert (
        isinstance(output_val, ToBeDetermined)
        or type(output_val) is list
        or isinstance(output_val, NestedListType)
    )
    updates = Updates()
    output_value = output.value
    input_shape = input._temp_shape
    status = False
    if not isinstance(input.value, NoneType | ToBeDetermined):
        # NOTE: Only executed for Physical model. Tensor values can only exist
        # in Physical models.
        updates |= output.set_value(input.value.tolist())
        status = True
    elif not isinstance(output_val, ToBeDetermined) and not isinstance(
        output_val, NestedListType
    ):
        if isinstance(output_value, list | tuple):
            shape = [Uniadic(idx) for idx in list_shape(list(output_val))]
        elif isinstance(output_value, float | int):
            shape = []

        updates |= input_shape.inner_match(prefix=shape)
        status = True

    return status, updates


def item_constraints(output: Scalar, input: Tensor) -> ConstrainResultType:
    assert input._temp_shape is not None, "Input shape of Item is not set!"
    input_shape: ShapeRepr = input._temp_shape
    updates = Updates()
    status = False
    for uni in input_shape.prefix + input_shape.suffix:
        val = uni.value
        if val is not None and val != 1:
            raise ValueError(
                f"Only tensors with 1 elements can be converted to scalar, got input "
                f"shape as {input_shape.get_shapes()}"
            )
        elif val is None:
            uni.set_value(1)
            # updated_symbols |= uni
            updates.add(uni)
    # If input is all inferred, set status to True.
    if input_shape.root is None:
        status = True
    return status, updates


def scalar_item_constraints(
    output: Scalar, input: Scalar, index: Scalar
) -> ConstrainResultType:
    assert (
        isinstance(output.value, ToBeDetermined)
        or type(output.value) is int
        or type(output.value) is float
        or type(output.value) is tuple
        or type(output.value) is list
    )

    assert (
        isinstance(input.value, ToBeDetermined)
        or type(input.value) is tuple
        or type(input.value) is list
    )

    assert isinstance(index.value, ToBeDetermined) or type(index.value) is int

    updates = Updates()
    status = False
    # Forward value propagation.
    if not isinstance(input.value, ToBeDetermined) and not isinstance(
        index.value, ToBeDetermined
    ):
        updates |= output.set_value(input.value[index.value])
        status = True
    elif not isinstance(input.value, ToBeDetermined) and isinstance(output.value, int):
        # Try to infer index value from input-output values. If
        # output value appears only once in input sequence, write its
        # index as the value of index argument.
        if input.value.count(output.value) == 1:
            index.set_value(input.value.index(output.value))
            updates._add_scalar(index)
            status = True
    return status, updates


def to_tuple_constraints(output: Scalar, *args: Scalar) -> ConstrainResultType:
    updates = Updates()
    status = False
    assert isinstance(output.value, ToBeDetermined) or type(output.value) is tuple
    # Forward value propagation.
    values = [arg.value for arg in args]
    if all([val is not TBD for val in values]):
        updates |= output.set_value(tuple(values))
        status = True
    # Backward value propagation.
    elif not isinstance(output.value, ToBeDetermined):
        for val, arg in zip(output.value, args, strict=False):
            updates |= arg.set_value(val)
        status = True
    return status, updates


def to_list_constraints(output: Scalar, *args: Scalar) -> ConstrainResultType:
    updates = Updates()
    status = False
    assert isinstance(output.value, ToBeDetermined) or type(output.value) is list
    # Backward value propagation.
    if not isinstance(output.value, ToBeDetermined):
        for val, arg in zip(output.value, args, strict=False):
            updates |= arg.set_value(val)
        status = True
    else:
        # Forward value propagation.
        values = []
        for arg in args:
            if (arg_val := arg.value) is TBD:
                break
            values.append(arg_val)
        else:
            updates |= output.set_value(list(values))
            status = True
    return status, updates


constrain_fn_dict = {key: fn for key, fn in globals().items() if callable(fn)}


def tensor_item_constraints(
    output: Tensor, input: Tensor, index: Scalar
) -> ConstrainResultType:
    assert output._temp_shape is not None, "Output shape of TensorItem is not set!"
    assert input._temp_shape is not None, "Input shape of TensorItem is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    index_val = index.value
    assert (
        isinstance(index_val, ToBeDetermined)
        or type(index_val) is int
        or type(index_val) is slice
        or type(index_val) is NoneType
        or type(index_val) is EllipsisType
        or is_index_type(index_val)
    )

    status = False
    updated_symbols = Updates()
    if not isinstance(index_val, ToBeDetermined):
        index_prefix: tuple[int | slice | EllipsisType | None, ...]
        index_suffix: tuple[int | slice | EllipsisType | None, ...]
        if not isinstance(index_val, tuple):
            index_val = (index_val,)
        if ... in index_val:
            # Firstly, find if there is an ellipsis in,
            # index, then find the loaction of index and,
            # seperate the index to two tuples.
            assert (
                index_val.count(...) == 1
            ), "an index can only have one ellipsis (...)"
            ellipsis_idx = index_val.index(...)
            index_prefix, index_suffix = (
                index_val[:ellipsis_idx],
                index_val[ellipsis_idx + 1 :],
            )
        else:
            # if there is no index in model, treat
            # the ellipsis placed in the last dimension,
            # (e.g output[3,2:4, None] = output[3, 2:4, None, ...])
            index_prefix, index_suffix = index_val, tuple()

        valued_prefix_items = [item for item in index_prefix if item is not None]
        valued_suffix_items = [item for item in index_suffix if item is not None]

        if len(valued_prefix_items) > len(input_shape.prefix) or len(
            valued_suffix_items
        ) > len(input_shape.reverse):
            # If this condition happens, this means there is more
            # information in index value than current input's prefix
            # or suffix, In this case, inner match the input with
            # minimum shapes in prefix and suffix
            input_prefix = []
            input_suffix = []
            if len(valued_prefix_items) > len(input_shape.prefix):
                input_prefix = [Uniadic() for _ in valued_prefix_items]
            if len(valued_suffix_items) > len(input_shape.suffix):
                input_suffix = [Uniadic() for _ in valued_suffix_items]
            updated_symbols |= input_shape.inner_match(
                prefix=input_prefix, root=Variadic(), suffix=input_suffix
            )

        # try to infer output prefix and suffix with given index
        output_prefix = tensor_item_constraint_helper(index_prefix, input_shape.prefix)
        output_reverse = tensor_item_constraint_helper(
            index_suffix[::-1], input_shape.reverse
        )

        if input_shape.root is not None:
            updated_symbols |= output_shape.inner_match(
                prefix=output_prefix, root=input_shape.root, suffix=output_reverse[::-1]
            )
        else:
            remaining_input_unis = input_shape.prefix[
                len(valued_prefix_items) : None
                if len(valued_suffix_items) == 0
                else -len(valued_suffix_items)
            ]
            updated_symbols |= output_shape.inner_match(
                prefix=output_prefix + remaining_input_unis + output_reverse[::-1]
            )

    unsolved_input_symbols: set[UniadicRecord | Variadic] = set()
    unsolved_output_symbols: set[UniadicRecord | Variadic] = set()
    for symbol in input_shape.prefix + input_shape.suffix:
        if symbol.value is None:
            unsolved_input_symbols.add(symbol.metadata)
    if input_shape.root is not None:
        unsolved_input_symbols.add(input_shape.root)

    unsolved_output_symbols = set()
    for symbol in output_shape.prefix + output_shape.suffix:
        if symbol.value is None:
            unsolved_output_symbols.add(symbol.metadata)
    if output_shape.root is not None:
        unsolved_output_symbols.add(output_shape.root)

    if unsolved_output_symbols - unsolved_input_symbols == set() and index is not TBD:
        status = True
    else:
        status = False

    return status, updated_symbols


def tensor_item_constraint_helper(
    item_values: tuple | list, input_unis: list[Uniadic]
) -> list[Uniadic]:
    # calculates output uniadics based on given item values and
    # input uniadics.

    # Example:
    # item_values = (3, slice(2, 4, None), None, None, slice(0, None, None))
    # input_unis = [Uniadic(10), Uniadic(5), Uniadic(2)] --> items = [Uniadic(2),
    # Uniadic(1), Uniadic(1), Uniadic(2)]

    items = []
    idx = 0
    for item in item_values:
        if item is None:
            items.append(Uniadic(1))
        else:
            if isinstance(item, slice):
                uni = input_unis[idx]
                if uni.value is not None:
                    out_value = len(list(range(uni.value))[item])
                    items.append(Uniadic(out_value))
                else:
                    items.append(Uniadic())
            idx += 1
    return items


def tensor_slice_constraints(
    output: Tensor, input: Tensor, start: Scalar, stop: Scalar, step: Scalar
) -> ConstrainResultType:
    assert output._temp_shape is not None, "Output shape of TensorSlice is not set!"
    assert input._temp_shape is not None, "Input shape of TensorSlice is not set!"
    output_shape: ShapeRepr = output._temp_shape
    input_shape: ShapeRepr = input._temp_shape
    updated_symbols = Updates()
    status = False
    if input_shape.prefix and output_shape.prefix:
        in_uni, out_uni = input_shape[0], output_shape[0]
        if in_uni.value is not None and out_uni.value is not None:
            status = True
        else:
            if (
                start.value is not TBD
                and stop.value is not TBD
                and step.value is not TBD
                and in_uni.value is not None
            ):
                slc = slice(start.value, stop.value, step.value)
                out_val = len(list(range(in_uni.value))[slc])
                out_uni.set_value(out_val)
                updated_symbols.add(out_uni)
                status = True

    return status, updated_symbols


def padding_1d_constraint(
    output: Scalar, input: Scalar, kernel_size: Scalar
) -> ConstrainResultType:
    status = False
    updates = Updates()
    input_value = input.value
    kernel_size_value = kernel_size.value
    if isinstance(input_value, PaddingType):
        if input_value == PaddingType.VALID:
            updates |= output.set_value((0, 0))
            status = True
        else:
            if isinstance(kernel_size_value, int):
                if kernel_size_value % 2 == 0:
                    raise RuntimeError(
                        "'same' padding is not supported when the kernel size is even!"
                    )
                updates |= output.set_value((kernel_size_value // 2,) * 2)
                status = True
            elif kernel_size_value is not TBD:
                raise RuntimeError("Kernel size must be 'tuple[int, int]' or 'int'!")

    elif isinstance(input_value, int):
        updates |= output.set_value((input_value, input_value))
        status = True

    elif isinstance(input_value, Sequence):
        if isinstance(input_value[0], Sequence) or isinstance(input_value[1], Sequence):
            raise RuntimeError(f"Given input value '{input_value}' is not valid!")
        updates |= output.set_value(tuple(input_value))
        status = True

    return status, updates


def padding_2d_constraint(
    output: Scalar, input: Scalar, kernel_size: Scalar
) -> ConstrainResultType:
    status = False
    updates = Updates()
    input_value = input.value
    if isinstance(input_value, PaddingType):
        if input_value == PaddingType.VALID:
            updates |= output.set_value((0, 0))
            status = True
        else:
            if isinstance(kernel_size, tuple):
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    raise RuntimeError(
                        "'same' padding is not supported when the kernel size is even!"
                    )
                _padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            elif isinstance(kernel_size, int):
                if kernel_size % 2 == 0:
                    raise RuntimeError(
                        "'same' padding is not supported when the kernel size is even!"
                    )
                updates |= output.set_value([(kernel_size // 2, kernel_size // 2)] * 2)
                status = True
            elif kernel_size.value is not TBD:
                raise RuntimeError("Kernel size must be 'tuple[int, int]' or 'int'!")
    elif isinstance(input_value, int):
        updates |= output.set_value((input_value, input_value))
        status = True
    elif is_padding_type(input_value):
        updated_padding: list[tuple[int, ...]] = []
        for p in input_value:
            if isinstance(p, int):
                updated_padding.append((p, p))
            elif isinstance(input_value, Sequence) and len(p) == 2:
                updated_padding.append(tuple(p))
            else:
                raise RuntimeError(f"Given padding '{input_value}' is not valid!")
        final_padding = (
            (updated_padding[0][0], updated_padding[0][1]),
            (updated_padding[1][0], updated_padding[1][1]),
        )
        updates |= output.set_value(final_padding)
        status = True
    return status, updates


def stride_constraint(
    output: Scalar, input: Scalar, kernel_size: Scalar
) -> ConstrainResultType:
    status = False
    updates = Updates()
    input_value = input.value
    assert (
        isinstance(input_value, ToBeDetermined)
        or is_padding_type(input_value)
        or type(input_value) is int
        or input_value is None
    )

    assert (
        is_tuple_of_two_ints(output.value)
        or isinstance(output.value, ToBeDetermined)
        or type(output.value) is int
    )

    assert (
        is_tuple_of_two_ints(kernel_size.value)
        or isinstance(kernel_size.value, ToBeDetermined)
        or type(kernel_size.value) is int
    )
    kernel_size_value = kernel_size.value
    if input_value is None:
        if not isinstance(kernel_size_value, ToBeDetermined):
            updates |= output.set_value(kernel_size_value)
            status = True
    elif input_value is not TBD:
        updates |= output.set_value(input_value)
        status = True
    elif output.value is not TBD:
        status = True
    return status, updates


def tuple_converter_constraint(output: Scalar, input: Scalar) -> ConstrainResultType:
    status = False
    updates = Updates()
    input_value = input.value
    if input_value is not TBD:
        if isinstance(input_value, int):
            updates |= output.set_value((input_value, input_value))
            status = True
        if isinstance(input_value, tuple):
            updates |= output.set_value(input_value)
            status = True
    if output.value is not TBD:
        status = True
    return status, updates


type_constraints = {
    general_tensor_type_constraint,
    floor_divide_type_constraint,
    scalar_slice_type_constraint,
    scalar_item_type_constraint,
    tensor_to_list_type_constraint,
    reduce_type_constraint,
}

post_process_map: dict[Callable, set[Callable]] = {
    bcast: {bcast_error_check},
    bcast_matrix_mult: {bcast_mat_mul_check},
}
