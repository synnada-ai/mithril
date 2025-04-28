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
from itertools import combinations_with_replacement, permutations, product, zip_longest
from operator import or_
from types import EllipsisType, GenericAlias, NoneType, UnionType
from typing import Any, Literal, get_args, get_origin

from ..common import PaddingType
from ..types import Constant
from ..utils.type_utils import (
    is_generic_alias_type,
    is_list_int,
    is_tuple_int,
    is_tuple_of_two_ints,
    is_union_type,
)
from .common import (
    DNF,
    TBD,
    ConstrainResultType,
    ConstraintFunctionType,
    IOHyperEdge,
    MaxNestedListDepth,
    PossibleValues,
    ScalarValueType,
    ShapeRepr,
    Tensor,
    ToBeDetermined,
    Uniadic,
    Updates,
    UpdateType,
    VariableSequenceType,
    Variadic,
    _TensorTypes,
    enhanced_isinstance,
    find_intersection_type,
    find_type,
    is_index_type,
    is_tensor_type,
    is_valued,
    process_value,
    squash_tensor_types,
)
from .utils import find_list_base_type, is_union

__all__ = [
    "scalar_slice_type_constraint",
    "indexer_initial_type_constraint",
    "indexer_type_constraint",
    "slice_constraints",
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
    "eye_constraints",
    "item_constraints",
    "indexer_constraints",
    "to_tuple_constraints",
    "tensor_to_list_type_constraint",
    "reduce_type_constraint",
    "constraint_type_map",
    "padding_1d_constraint",
    "padding_2d_constraint",
    "stride_constraint",
    "tuple_converter_constraint",
    "pad_constraints",
    "randn_constraints",
    "buffer_constraint",
    "polynomial_kernel_constraint",
    "general_forward_constraint",
    "general_type_constraint",
    "sum_fn",
    "distance_matrix_const",
]


def to_sequence_helper(
    type_def: type[list[Any]] | type[tuple[Any, ...]],
    output: IOHyperEdge,
    *args: IOHyperEdge,
) -> ConstrainResultType:
    updates = Updates()
    forward_list = type_def([arg._value for arg in args])
    backward_list = output._value
    assert isinstance(backward_list, type_def | ToBeDetermined)

    updates |= output.set_value(forward_list)
    if isinstance(backward_list, list | tuple):
        for arg, val in zip(args, backward_list, strict=False):
            updates |= arg.set_value(val)
    status = all(arg.is_valued for arg in [output, *args])
    return status, updates


def generate_nested_list_type(
    base_type: _TensorTypes, min_depth: int = 0, max_depth: int = MaxNestedListDepth
) -> type[list[Any]] | _TensorTypes:
    types = set()
    typ: type[list[Any]] | _TensorTypes = base_type
    for depth in range(max_depth + 1):
        if depth >= min_depth:
            types.add(typ)
        typ = list[typ]  # type: ignore
    return reduce(or_, types)


# Below functions are used in various constraints.
def prod_fn(a: int | Uniadic, b: int | Uniadic) -> int:
    if isinstance(a, Uniadic):
        value_a = a.value
        assert value_a is not None
    else:
        value_a = a

    if isinstance(b, Uniadic):
        value_b = b.value
        assert value_b is not None
    else:
        value_b = b

    return value_a * value_b


def is_repr_known(repr: ShapeRepr) -> bool:
    return repr.root is None and all([uni.value is not None for uni in repr.prefix])


def sum_fn(*inputs: Any) -> Any:
    return sum(inputs)


def distance_matrix_const(input1: Any, input2: Any, input3: Any) -> Any:
    return (input1 - input2) ** input3


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


def set_edge_type(edge: IOHyperEdge, new_type: Any) -> Updates:
    # Used for type setting for polymorphic IOHyperEdges.
    # Simply wraps new type into Tensor if edge_type is Tensor,
    # else sets directly.
    type = new_type
    if edge.is_tensor:
        type = Tensor[new_type]
    return edge.set_type(type)


def general_forward_constraint(
    *keys: IOHyperEdge, callable: Callable[..., Any]
) -> ConstrainResultType:
    updates = Updates()
    status = True
    if all(io.is_scalar for io in keys):
        output, *inputs = keys
        input_values = [input.value for input in inputs]
        if TBD not in input_values:
            output_value = callable(*input_values)
            updates |= output.set_value(output_value)
        else:
            status = False
    return status, updates


### General type utils ###

type_map: dict[type[int] | type[float] | type[bool], int | float | bool] = {
    int: 2,
    float: 2.0,
    bool: True,
}

all_possible_types: set[type | GenericAlias] = {
    int,
    float,
    bool,
    Tensor[int],
    Tensor[float],
    Tensor[bool],
}


def unsquash_tensor_types(
    value_type: type | UnionType | GenericAlias,
) -> type | UnionType | GenericAlias:
    # TODO: use match case when Python unifies all UnionType and GenericAlias types.
    new_type: type | UnionType | GenericAlias = value_type
    if is_generic_alias_type(value_type) and get_origin(value_type) is Tensor:
        # Example: Tensor[int | float] -> Tensor[int] | Tensor[float]
        sub_type = get_args(value_type)[0]
        sub_types = (
            sub_type.__args__ if isinstance(sub_type, UnionType) else (sub_type,)
        )
        new_type = reduce(or_, (Tensor[typ] for typ in sub_types))  # type: ignore

    elif is_union_type(value_type):
        # Example: Tensor[int | float] | bool -> Tensor[int] | Tensor[float] | bool
        new_type = reduce(
            or_,
            (unsquash_tensor_types(typ) for typ in get_args(value_type)),
        )

    return new_type


def process_list_types(
    left_args: set[type | GenericAlias],
    right_args: set[type | GenericAlias],
    output_args: set[type | GenericAlias],
) -> set[tuple[type | GenericAlias, ...]]:
    # TODO: Implement this function
    return set(product(left_args, right_args, output_args))


def process_tuple_types(
    left_args: set[type | GenericAlias],
    right_args: set[type | GenericAlias],
    output_args: set[type | GenericAlias],
) -> set[tuple[type | GenericAlias, ...]]:
    # TODO: Implement this function
    return set(product(left_args, right_args, output_args))


def process_tensor_op_types(
    operation: Callable[..., Any] | None,
    is_bitwise: bool,
    output_args: set[type | GenericAlias],
    *input_args: set[type | GenericAlias],
) -> set[tuple[type | GenericAlias, ...]]:
    # TODO: uncomment type ignores when all TypeGuards switched to TypeIs.

    # extract all possible input output type combinations
    possible_arg_types: set[tuple[type | GenericAlias, ...]] = set(
        product(output_args, *input_args)
    )

    # diminish all possible types based on available input types
    available_possible_types = all_possible_types & reduce(or_, input_args)

    possible_type_set: set[tuple[type | GenericAlias, ...]] = set()

    for sample_types in combinations_with_replacement(
        available_possible_types, len(input_args)
    ):
        has_tensor = False
        input_types: list[type[int] | type[float] | type[bool]] = []

        for sample_type in sample_types:
            if is_generic_alias_type(sample_type):
                # extract built-in type from Tensor type
                input_types.append(get_args(sample_type)[0])
                has_tensor = True
            else:
                input_types.append(sample_type)  # type: ignore

        if has_tensor and all(typ is bool for typ in input_types) and not is_bitwise:
            # handle edge case occured in some operations.

            # bool + bool -> int (Python)
            # Tensor[bool] + Tensor[bool] = Tensor[bool] (backends, (e.g. NumPy))

            out_type: type = Tensor[bool]
        else:
            values = [type_map[typ] for typ in input_types]
            if operation is None:
                out_type = type(values[0])
            else:
                out_type = type(operation(*values))
            out_type = Tensor[out_type] if has_tensor else out_type  # type: ignore

        possible_type_set.update((out_type,) + p for p in permutations(sample_types))

    return possible_arg_types & possible_type_set


def general_type_constraint(
    *keys: IOHyperEdge,
    fn: Callable[..., Any] | None = None,
    is_bitwise: bool = False,
    is_edge: bool = False,
) -> ConstrainResultType:
    updates = Updates()
    all_output_types: set[tuple[type | GenericAlias, ...]] = set()

    args: list[Any] = []

    for key in keys:
        key_type = unsquash_tensor_types(key._type)
        key_args = key_type.__args__ if is_union_type(key_type) else (key_type,)
        key_types = {
            arg for arg in key_args if is_tensor_type(arg) or arg in (int, float, bool)
        }
        args.append(key_types)

    all_output_types |= process_tensor_op_types(fn, is_bitwise, *args)

    max_possible_results = 1
    for result, key in zip(zip(*all_output_types, strict=False), keys, strict=False):
        res_type = reduce(or_, result)
        max_possible_results *= len(set(result))
        res_type = squash_tensor_types(res_type)
        updates |= key.set_type(res_type)

    if is_edge:
        status = not any(io.is_polymorphic for io in keys)
    else:
        status = not len(all_output_types) < max_possible_results

    return status, updates


def scalar_slice_type_constraint(
    output: IOHyperEdge,
    input: IOHyperEdge,
    start: IOHyperEdge,
    stop: IOHyperEdge,
    step: IOHyperEdge,
) -> ConstrainResultType:
    updates = Updates()

    output_type = output.value_type
    input_type = input.value_type

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

    status = not is_union(output.value_type)
    return status, updates


def scalar_item_type_constraint_helper(
    input_type: GenericAlias | UnionType | type, index_val: int | slice | ToBeDetermined
) -> type | UnionType | GenericAlias:
    # forward inference of scalar item type constraint:
    # Examples:
    # > scalar_item_type_constraint_helper(list[list[int]], 3) -> list[int]
    # > scalar_item_type_constraint_helper(list[int | float], 3) -> int | float

    new_type = input_type
    if isinstance(input_type, GenericAlias):
        origin = get_origin(input_type)
        if origin is tuple:
            if ... in input_type.__args__:
                variadic_required = True
                # if second value is ellipsis, directly take first value
                # (tuple[int, ...] -> int)
                new_type = input_type.__args__[0]
            else:
                # case when type of tuple is exact (ex: tuple[int, float])
                if not isinstance(index_val, ToBeDetermined):
                    variadic_required = False
                    # if index val is specified, directly take the corresponding item
                    # of type
                    if isinstance(index_val, int):
                        new_type = input_type.__args__[index_val]
                    else:
                        new_type = tuple[*input_type.__args__[index_val]]  # type: ignore
                else:
                    variadic_required = True
                    # if not specified this means it can be all of them,
                    # take union of all types inside tuple
                    new_type = create_union_type(*input_type.__args__)

            if variadic_required:
                if isinstance(index_val, slice):
                    new_type = tuple[new_type, ...]  # type: ignore
                else:
                    new_type = new_type | tuple[new_type, ...]  # type: ignore

        elif origin is list:
            if isinstance(index_val, slice):
                new_type = input_type
            else:
                new_type = input_type.__args__[0]
    elif input_type is list or input_type is tuple:
        if isinstance(index_val, slice):
            new_type = input_type
        else:
            new_type = input_type | int | float | list

    return new_type


def check_index_type_compatibility(
    _type: type,
    index: int | ToBeDetermined | slice,
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
        if isinstance(index, int) and not (-args_len <= index <= args_len - 1):
            if raise_error:
                raise TypeError(
                    f"Index value {index} is out of range for type {_type}!"
                )
            return False
    return True


def scalar_item_reduce_input_type(  # type: ignore
    output_type: type | UnionType | GenericAlias,
    input_type: type | UnionType | GenericAlias,
    index: int | slice | ToBeDetermined,
) -> type | UnionType | GenericAlias | None:
    possible_types = []
    out_origin: type[list] | type[tuple] | type[UnionType] | None = get_origin(  # type: ignore
        output_type
    )
    input_origin: type[list] | type[tuple] | type[UnionType] | None = None  # type: ignore
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
                    possible_types.append(arg[*inner_type])
        return create_union_type(*possible_types)
    elif isinstance(input_type, GenericAlias):
        input_origin = input_type.__origin__

        is_variadic = ... in input_type.__args__ if input_origin is not None else False
        if check_index_type_compatibility(
            input_origin, index, is_variadic, raise_error=True
        ):
            if index == ... or input_origin is list:
                if isinstance(index, int):
                    for arg in input_type.__args__:
                        if find_intersection_type(output_type, arg):
                            return input_type
                        else:
                            return None
                else:
                    return input_type

            elif input_origin is tuple:
                if isinstance(index, int):
                    possible_types = [
                        arg
                        if idx != index
                        else find_intersection_type(arg, output_type)
                        for idx, arg in enumerate(input_type.__args__)
                    ]
                    return (
                        input_origin[*possible_types, ...]  # type: ignore
                        if is_variadic
                        else input_origin[*possible_types]  # type: ignore
                    )
                else:
                    return input_type
    else:
        return input_type


def indexer_initial_type_constraint(
    output: IOHyperEdge, input: IOHyperEdge, index: IOHyperEdge
) -> ConstrainResultType:
    status = False
    updates = Updates()
    if input.is_scalar or output.is_scalar:
        updates |= index.set_type(
            int
            | EllipsisType
            | None
            | slice
            | tuple[int | EllipsisType | None | slice, ...]
        )
        if input.edge_type is not ToBeDetermined:
            # For this constraint, the content of sequence is not important. So
            # we set output type to the most general case which includes Tensor
            # also since it is possible to have Tensor types in input sequence.
            updates |= output.set_type(ScalarValueType | Tensor[int | float | bool])
        elif output.edge_type is not ToBeDetermined:
            # Set input to the most general Sequence type.
            updates |= input.set_type(Sequence[Any])
        status = True
    elif input.is_tensor or output.is_tensor:
        if input.is_tensor and output.is_tensor:
            intersection_type = find_intersection_type(
                output.value_type, input.value_type
            )
            updates |= input.set_type(Tensor[intersection_type])  # type: ignore
            updates |= output.set_type(Tensor[intersection_type])  # type: ignore
            status = True
        else:
            (tensor_edge, other_edge) = (
                (input, output) if input.is_tensor else (output, input)
            )
            updates |= other_edge.set_type(Tensor[tensor_edge.value_type])  # type: ignore
            status = True
    return status, updates


def indexer_type_constraint(
    output: IOHyperEdge, input: IOHyperEdge, index: IOHyperEdge
) -> ConstrainResultType:
    status = False
    updates = Updates()
    if input.is_scalar:
        # Input is a non-tensor type.
        input_type = input.value_type
        # output_type = output.value_type
        output_type = output.edge_type
        index_value = index.value
        assert (
            isinstance(index_value, ToBeDetermined)
            or type(index_value) is int
            or type(index_value) is slice
        )

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
                f"Output type {output_type} is not compatible with "
                f"input type {input_type}!"
            )

        updates |= input.set_type(inferred_input_type)

        # extract all possibilites and put it in to a list
        # TODO: This part should take NestedListType into account.
        args = (
            input.value_type.__args__
            if isinstance(input.value_type, UnionType)
            else [input.value_type]
        )

        # Do the forward inference in all types in args, then make Union
        types = [scalar_item_type_constraint_helper(arg, index_value) for arg in args]
        inferred_out_type = create_union_type(*types)

        updates |= output.set_type(inferred_out_type)

        status = not is_union(output.edge_type)
    elif input.is_tensor:
        status = True
    return status, updates


def slice_constraints(
    output: IOHyperEdge, start: IOHyperEdge, stop: IOHyperEdge, step: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    output_value = output.value
    start_value = start.value
    stop_value = stop.value
    step_value = step.value
    status = False

    assert isinstance(start_value, ToBeDetermined | int | None)
    assert isinstance(stop_value, ToBeDetermined | int | None)
    assert isinstance(step_value, ToBeDetermined | int | None)
    assert isinstance(output_value, ToBeDetermined | slice)

    if (
        not isinstance(start_value, ToBeDetermined)
        and not isinstance(step_value, ToBeDetermined)
        and not isinstance(stop_value, ToBeDetermined)
    ):
        updates |= output.set_value(slice(start_value, stop_value, step_value))
        status = True

    elif not isinstance(output_value, ToBeDetermined):
        start_val = output_value.start
        stop_val = output_value.stop
        step_val = output_value.step

        updates |= start.set_value(start_val)
        updates |= stop.set_value(stop_val)
        updates |= step.set_value(step_val)
        status = True

    return status, updates


def tensor_to_list_type_constraint(
    output: IOHyperEdge, input: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    input_type = input.value_type
    output_type = output.value_type
    assert input.shape is not None
    in_shape: ShapeRepr = input.shape.reprs[0]
    assert (
        output_type is list
        or output_type is float
        or output_type is int
        or output_type is bool
        or isinstance(output_type, UnionType)
        or (isinstance(output_type, GenericAlias) and output_type.__origin__ is list)
    )

    # If input type is UnionType, try to constrain it using output type
    if get_origin(input_type) == UnionType and (
        out_types := find_list_base_type(output_type)  # type: ignore
    ):
        possible_input_types = find_intersection_type(
            input_type, create_union_type(*out_types)
        )
        if not possible_input_types:
            raise TypeError(
                f"Input type {input_type} is not compatible with output type "
                f"{output_type}!"
            )
        updates |= set_edge_type(input, possible_input_types)

    # Create output nested type using the input type as base type.
    # Uniadic numbers define min depth of nested list. If input
    # has no variadic shape then uniadic numbers also define
    # max depth of nested list.
    min_depth = len(in_shape.prefix + in_shape.suffix)
    max_depth = min_depth if in_shape.root is None else MaxNestedListDepth
    assert isinstance(
        input.value_type, type(int) | type(float) | type(bool) | UnionType
    )
    base = generate_nested_list_type(
        input.value_type,
        min_depth=len(in_shape.prefix + in_shape.suffix),
        max_depth=max_depth,
    )

    updates |= set_edge_type(output, base)

    status = not is_union(output.value_type) if in_shape.root is not None else True

    return status, updates


def reduce_type_constraint(
    output: IOHyperEdge, input: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()

    assert isinstance(
        input.value_type, type(int) | type(float) | type(bool) | UnionType
    )

    input_type = input.value_type

    possible_output_types: list[type[int] | type[float] | type[bool]] = []

    ### Forward Inference ###
    if find_intersection_type(input_type, int | bool):
        # if input has type of int or bool, int is one of the possible output types
        possible_output_types.append(int)

    if find_intersection_type(input_type, float):
        # if input has type of float, float is one of the possible output types
        possible_output_types.append(float)

    union_output_types = create_union_type(*possible_output_types)
    assert not isinstance(union_output_types, GenericAlias)
    updates |= output.set_type(Tensor[union_output_types])  # type: ignore

    ### Reverse Inference ###
    if output.value_type is float:
        # if output type is float, it is guaranteed that input will be float
        updates |= input.set_type(Tensor[float])
    elif output.value_type is int:
        # if output type is int, input should either be int or bool
        updates |= input.set_type(Tensor[bool | int])

    status = not isinstance(output.value_type, UnionType)

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
            assert max_left_len is not None
            max_left_len += left.root.max_len
        else:
            max_left_len = None

    if right.root is not None:
        if right.root.possibles is not None:
            assert max_right_len is not None
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


def bcast_align_input(output: ShapeRepr, left: ShapeRepr, right: ShapeRepr) -> Updates:
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


def _bcast(
    output: IOHyperEdge, left: IOHyperEdge, right: IOHyperEdge, index: int
) -> ConstrainResultType:
    if left.is_tensor and right.is_tensor:
        assert output._temp_shape is not None, "Output shape of broadcast is not set!"
        assert left._temp_shape is not None, "Left shape of broadcast is not set!"
        assert right._temp_shape is not None, "Right shape of broadcast is not set!"
        return bcast_helper(
            output._temp_shape, left._temp_shape, right._temp_shape, index
        )
    elif left.is_scalar and right.is_scalar:
        # Means all edges are scalar types. Simply return True
        # without any updates.
        return True, Updates()

    else:
        merge_edge = left if left.is_tensor else right
        assert isinstance(merge_edge._value, Tensor)
        assert output.shape is not None
        return True, merge_edge._value.match_shapes(output.shape)


def bcast(
    output: IOHyperEdge, left: IOHyperEdge, right: IOHyperEdge
) -> ConstrainResultType:
    return _bcast(output, left, right, 0)


def bcast_matrix_mult(
    output: IOHyperEdge, left: IOHyperEdge, right: IOHyperEdge
) -> ConstrainResultType:
    return _bcast(output, left, right, 2)


def check_reverse(
    left: list[Uniadic], right: list[Uniadic], output: list[Uniadic]
) -> bool:
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
) -> bool:
    return (
        output.root is None
        and left.root is None
        and right.root is None
        and check_reverse(left.prefix, right.prefix, output.prefix)
    )


def bcast_error_check(
    output: IOHyperEdge,
    left: IOHyperEdge,
    right: IOHyperEdge,
    index: int = 0,
) -> ConstrainResultType:
    if left.edge_type is not Tensor or right.edge_type is not Tensor:
        return True, Updates()
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
    output: IOHyperEdge, left: IOHyperEdge, right: IOHyperEdge
) -> ConstrainResultType:
    return bcast_error_check(output, left, right, index=2)


def _reduce_sanity_check(
    input_shape: ShapeRepr,
    output_shape: ShapeRepr,
    axis_val: int | tuple[int | ToBeDetermined, ...] | ToBeDetermined | None,
    keepdim_val: bool | ToBeDetermined,
) -> None:
    axes = (axis_val,) if isinstance(axis_val, int) else axis_val
    if input_shape.root is output_shape.root:
        if keepdim_val is True:
            if len(input_shape) != len(output_shape):
                raise ValueError(
                    f"Input shape {input_shape} and output shape {output_shape} "
                    f"must have same length when keepdim is True in reduce model"
                )
        elif (
            keepdim_val is False
            and not isinstance(axes, ToBeDetermined | None)
            and (len(input_shape) != len(output_shape) + len(axes))
        ):
            raise ValueError(
                f"Shape mismatch, output rank = {len(output_shape)}. "
                f"Output rank must be exactly {len(input_shape) - len(axes)} "
                f"where input rank = {len(input_shape)} and axis = {axis_val}. "
                "Axis numbers printed as their counterparts."
            )
    if not isinstance(axes, ToBeDetermined | None):
        axes = tuple(val for val in axes if not isinstance(val, ToBeDetermined))
        if input_shape.root is not None and output_shape.root is not None:
            pos_axes = list(axes)
        else:
            if input_shape.root is None:
                input_len = len(input_shape)
            else:
                input_len = len(output_shape.prefix) + len(axes)
            pos_axes = [idx if idx >= 0 else idx + input_len for idx in axes]
            if (max_axis := (max(pos_axes, default=-1) + 1)) > input_len:
                raise ValueError(
                    f"Input rank is {input_len}. "
                    f"Minimum rank {max_axis} input is required for "
                    f"axis = {axis_val}."
                )
        seen_axes: set[int] = set()
        for val in pos_axes:
            if val in seen_axes:
                raise ValueError(f"Dim {val} appears multiple times in the reduce axes")
            seen_axes.add(val)


def reduce_constraints(
    output: IOHyperEdge, input: IOHyperEdge, axis: IOHyperEdge, keepdim: IOHyperEdge
) -> ConstrainResultType:
    output_shape = output._temp_shape
    input_shape = input._temp_shape
    axis_value = axis.value
    keepdim_val = keepdim.value

    assert input_shape is not None
    assert output_shape is not None
    assert isinstance(axis_value, int | ToBeDetermined | None) or enhanced_isinstance(
        axis_value, tuple[int | ToBeDetermined, ...]
    )
    assert isinstance(keepdim_val, bool | ToBeDetermined)
    _reduce_sanity_check(input_shape, output_shape, axis_value, keepdim_val)

    updates = Updates()

    if isinstance(axis_value, int | tuple):
        _axis_value = (axis_value,) if isinstance(axis_value, int) else axis_value
        is_axis_valued = is_valued(_axis_value)
        _axis_value = tuple(val for val in _axis_value if val is not TBD)

        pos_vals = {val for val in _axis_value if val >= 0}
        neg_vals = {val for val in _axis_value if val < 0}

        max_pos_val = max(pos_vals, default=-1) + 1
        max_neg_val = abs(min(neg_vals, default=0))

        if input_shape.root is not None and output_shape.root is not None:
            # try to infer minimum input rank from axis values.
            min_input_rank = max(max_pos_val, abs(max_neg_val))
            while True:
                # look if any overlapping axis values. If any,
                # increment minimum input rank
                # (e.g, minimum input rank cannot be 2 when axis values are
                # (0, 1, -1). It is because values 1 and -1 would overlap.
                # In this case, increment minimum input rank to 3 so that
                # no axis values would overlap)
                for pos_val in pos_vals:
                    neg_counterpart = pos_val - min_input_rank
                    if neg_counterpart in neg_vals:
                        min_input_rank += 1
                        break
                else:
                    break

            # calculate additional rank to be added to input shape
            additional_rank = min_input_rank - len(input_shape)
            prefix: list[Uniadic] = input_shape.prefix[:]
            suffix: list[Uniadic] = input_shape.suffix[::-1]

            for _ in range(additional_rank):
                # add all additional ranks to prefix
                if len(prefix) < max_pos_val:
                    prefix.append(Uniadic())
                else:
                    suffix.append(Uniadic())

            updates |= input_shape.inner_match(
                prefix=prefix, root=Variadic(), suffix=suffix[::-1]
            )

        if keepdim_val is not TBD and is_axis_valued:
            match (input_shape.root, output_shape.root):
                case (Variadic(), Variadic()):
                    prefix = []
                    suffix = []
                    root = input_shape.root

                    for idx, uni in enumerate(input_shape.prefix):
                        if idx not in pos_vals:
                            if idx < len(input_shape) - max_neg_val:
                                # check if negative axis values can overlap with
                                # positive shape values. If it is not possible,
                                # directly add Uniadic to output prefix.
                                prefix.append(uni)
                            else:
                                # create new uniadic and add it since
                                # there is a possibility of overlap
                                prefix.append(Uniadic())
                        elif keepdim_val:
                            prefix.append(Uniadic(1))

                    for idx, uni in enumerate(input_shape.suffix[::-1]):
                        if -(idx + 1) not in neg_vals:
                            if idx < len(input_shape) - max_pos_val:
                                suffix.append(uni)
                            else:
                                suffix.append(Uniadic())
                        elif keepdim_val:
                            suffix.append(Uniadic(1))

                    for idx in _axis_value:
                        # check if axis value is in the range of input shape
                        # if not in range, remove last uniadic (closest to
                        # Variadic) from output prefix or suffix.
                        if idx >= len(input_shape.prefix) or idx < -len(
                            input_shape.suffix
                        ):
                            root = Variadic()
                            if not keepdim_val:
                                prefix.pop() if prefix else suffix.pop()

                    updates |= output_shape.inner_match(
                        prefix=prefix,
                        root=root,
                        suffix=suffix[::-1],
                    )

                case (None, Variadic()) | (None, None):
                    # forward reduce inference. directly infer
                    # all shapes symbolically
                    pos_axes = [
                        idx if idx >= 0 else len(input_shape.prefix) + idx
                        for idx in _axis_value
                    ]
                    output_unis: list[Uniadic] = []
                    for idx, uni in enumerate(input_shape.prefix):
                        if idx not in pos_axes:
                            output_unis.append(uni)
                        elif keepdim_val:
                            output_unis.append(Uniadic(1))
                    updates |= output_shape.inner_match(output_unis)

                case (Variadic(), None):
                    # Backward reduce inference. insert dimensions to
                    # input shape to specified axes.
                    in_rank = (
                        len(_axis_value) + len(output_shape.prefix)
                        if not keepdim_val
                        else len(output_shape.prefix)
                    )
                    input_unis = output_shape.prefix[:]
                    output_unis = output_shape.prefix[:]
                    sorted_axis = sorted(
                        idx if idx > 0 else idx + in_rank for idx in _axis_value
                    )
                    for _axis in sorted_axis:
                        if keepdim_val:
                            input_unis[_axis] = Uniadic()
                            output_unis[_axis] = Uniadic(1)
                        else:
                            input_unis[_axis:_axis] = [Uniadic()]
                    updates |= input_shape.inner_match(input_unis)
                    updates |= output_shape.inner_match(output_unis)

        return input_shape.root == output_shape.root, updates

    elif axis_value is None and keepdim_val is not TBD:
        if keepdim_val:
            # case where keepdim is True and axis is None
            # output shape rank will be the same as input rank
            # and all dimensions will be 1 in output
            output_prefix = [Uniadic(1) for _ in range(len(input_shape.prefix))]
            output_suffix = [Uniadic(1) for _ in range(len(input_shape.suffix))]
            _output_root = None if input_shape.root is None else Variadic()
            updates |= output_shape.inner_match(
                output_prefix, _output_root, output_suffix
            )
            return input_shape.root is None, updates
        else:
            # set output rank to zero directly, nothing else to infer.
            return True, output_shape.inner_match()

    else:
        # case where axis is None or TBD and keepdim is TBD
        # we cannot infer anything in this case as keepdim is TBD
        return False, updates


def concat_constraints(
    output: IOHyperEdge, input: IOHyperEdge, axis: IOHyperEdge
) -> ConstrainResultType:
    status = False
    updates = Updates()
    keys: list[ShapeRepr] = []
    assert isinstance(output._value, Tensor)
    if input._value is TBD or (
        isinstance(input._value, Sequence) and TBD in input._value
    ):
        return False, updates

    input_val = input._value
    assert isinstance(input_val, list | tuple)

    for arg in input_val:
        assert isinstance(arg, Tensor)
        assert arg._temp_shape is not None, "Input shape of concat is not set!"
        keys.append(arg._temp_shape)
    if keys:
        assert (
            output._value._temp_shape is not None
        ), "Output shape of concat is not set!"
        output_shape: ShapeRepr = output._value._temp_shape

        reprs = keys + [output_shape]
        axis_val = axis.value
        assert (
            isinstance(axis_val, int)
            or axis_val is None
            or isinstance(axis_val, ToBeDetermined)
        ), "Invalid axis value!"
        # look if all reprs have different variadics
        if not isinstance(axis_val, ToBeDetermined) and (
            (
                (len(repr_set := set(repr.root for repr in reprs)) == len(reprs))
                or None in repr_set
            )
            or output_shape.root is not None
        ):
            if axis_val is not None:
                # if axis is determined and is positive, we can know that tensors have
                # shape at least value of dimensions. Also we know that All shapes of
                # all reprs must be same except shape at axis same applies for when axis
                # is negative
                var = Variadic()
                if axis_val >= 0:
                    uniadics: list[Uniadic] = [Uniadic() for _ in range(axis_val)]
                    for repr in reprs:
                        updates |= repr.inner_match(
                            prefix=uniadics + [Uniadic()], root=var
                        )
                elif axis_val < 0:
                    uniadics = [Uniadic() for _ in range(-axis_val - 1)]
                    for repr in reprs:
                        updates |= repr.inner_match(
                            root=var, suffix=[Uniadic()] + uniadics
                        )
            else:
                updates |= output_shape.inner_match(prefix=[Uniadic()])

        if not isinstance(axis_val, ToBeDetermined):
            if axis_val is not None:
                # If axis is determined and not None, at first take all the uniadic
                # values at axis. shape formula of output of axis must be out =
                # sum(all ins). Therefore, if there is only one unknown, we can
                # infer unknown uniadic's shape by algebra.
                if (
                    non_var_repr := next(
                        (repr for repr in reprs if repr.root is None), None
                    )
                ) is not None:
                    # Match all uniadics in all reprs with same Uniadics in non_var_repr
                    # except the axis_val.
                    pos_axis_val = (
                        axis_val
                        if axis_val >= 0
                        else len(non_var_repr.prefix) + axis_val
                    )
                    for repr in reprs:
                        if repr is not non_var_repr:
                            updates |= repr.inner_match(
                                prefix=[
                                    uni if idx != pos_axis_val else Uniadic()
                                    for idx, uni in enumerate(non_var_repr.prefix)
                                ]
                            )

                uniadics = []
                uniadic_values: list[int | None] = []
                pruned_uni_values: list[int] = []
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
                    none_values: list[Uniadic] = []
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


def pad_constraints(
    output: IOHyperEdge, input: IOHyperEdge, pad_width: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    pad_value: tuple[tuple[int, int], ...] | ToBeDetermined = pad_width.value  # type: ignore
    input_shape = input._temp_shape
    output_shape = output._temp_shape
    assert input_shape is not None
    assert output_shape is not None
    status = True
    pad_status = True

    def prcess_single_pad_shape(
        input_uni: Uniadic,
        output_uni: Uniadic,
        pad_value: tuple[int | ToBeDetermined, int | ToBeDetermined],
        updates: Updates,
    ) -> tuple[
        Uniadic, Uniadic, tuple[int | ToBeDetermined, int | ToBeDetermined], bool
    ]:
        # Infers single shape for given pad value,
        # Example:
        #    input_uni.value = 2
        #    pad_value = (1, 2)  -> output_uni.value = 5

        input_value = input_uni.value
        output_value = output_uni.value
        status = False
        match input_value, output_value:
            case (int(), int()):
                left_pad, right_pad = pad_value
                padded_shape = output_value - input_value

                match (left_pad, right_pad):
                    case (int(), int()):
                        if not (input_value + left_pad + right_pad == output_value):
                            raise ValueError(
                                f"Shape mismatch for pad. "
                                f"Input shape = {input_value}, "
                                f"pad value = {pad_value}, "
                                f"output shape = {output_value}"
                            )
                        status = True

                    case (ToBeDetermined(), int()):
                        # case where left pad value is unknown
                        # and can be inferred
                        inferred_pad = padded_shape - right_pad
                        pad_value = (inferred_pad, right_pad)
                        status = True

                    case (int(), ToBeDetermined()):
                        # case where right pad value is unknown
                        # and can be inferred
                        inferred_pad = padded_shape - left_pad
                        pad_value = (left_pad, inferred_pad)
                        status = True

                    case (ToBeDetermined(), ToBeDetermined()):
                        if (
                            input_uni is output_uni
                            or input_uni.value == output_uni.value
                        ):
                            pad_value = (0, 0)
                            status = True

            case (int(), None):
                if is_tuple_of_two_ints(pad_value):
                    # case where output is unknown and can be inferred:
                    inferred_output = input_value + sum(pad_value)
                    if output_uni.set_value(inferred_output):
                        updates.add(output_uni)
                        status = True

            case (None, int()):
                if is_tuple_of_two_ints(pad_value):
                    # case where input is unknown and can be inferred:
                    inferred_input = output_value - sum(pad_value)
                    if input_uni.set_value(inferred_input):
                        updates.add(input_uni)
                        status = True

            case (None, None):
                pass
                # do nothing as nothing to infer

        return input_uni, output_uni, pad_value, status

    if isinstance(pad_value, ToBeDetermined):
        if input_shape is output_shape:
            pad_val = tuple((0, 0) for _ in range(len(input_shape.prefix)))
            updates |= pad_width.set_value(tuple(pad_val))
            return True, updates
        else:
            return False, updates

    # Use pad width
    temp_uniadics = [Uniadic() for _ in range(len(pad_value))]
    updates |= input_shape.inner_match(prefix=temp_uniadics, root=None, suffix=[])

    temp_uniadics = [Uniadic() for _ in range(len(pad_value))]
    updates |= output_shape.inner_match(prefix=temp_uniadics, root=None, suffix=[])

    inferred_pad: list[
        tuple[int | ToBeDetermined, int | ToBeDetermined] | ToBeDetermined
    ] = []
    for input_uni, output_uni, single_pad_value in zip(
        input_shape.prefix, output_shape.prefix, pad_value, strict=False
    ):
        if not isinstance(single_pad_value, ToBeDetermined):
            input_uni, output_uni, inferred_pad_value, _status = (
                prcess_single_pad_shape(
                    input_uni, output_uni, single_pad_value, updates
                )
            )
            status &= _status
            inferred_pad.append(inferred_pad_value)

        elif input_uni is output_uni or (
            input_uni.value == output_uni.value and input_uni.value is not None
        ):
            inferred_pad.append((0, 0))

        else:
            inferred_pad.append(TBD)
            pad_status = False
    updates |= pad_width.set_value(tuple(inferred_pad))

    return status & pad_status, updates


def reverse_constraints(
    output: IOHyperEdge, input: IOHyperEdge, axes: IOHyperEdge
) -> ConstrainResultType:
    status = False
    assert input._temp_shape is not None, "Input shape of reverse is not set!"
    assert output._temp_shape is not None, "Output shape of reverse is not set!"

    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    axes_val = axes.value
    status = False
    updates = Updates()

    if axes_val is None:
        if output_shape.root is None:
            # TODO Maybe we should embed uniadic updates in remove_variadic
            updates |= input_shape.update_uniadics(
                input_shape.prefix, output_shape.reverse
            )
            updates |= input_shape.update_uniadics(
                input_shape.reverse, output_shape.prefix
            )
            if input_shape.root is not None:
                updates |= input_shape.remove_variadic(output_shape.reverse)
                if len(input_shape.prefix) != len(output_shape.prefix):
                    raise ValueError("Shape mismatch in Transpose model")
            status = True
        if input_shape.root is None:
            updates |= output_shape.update_uniadics(
                output_shape.prefix, input_shape.reverse
            )
            updates |= output_shape.update_uniadics(
                output_shape.reverse, input_shape.prefix
            )
            if output_shape.root is not None:
                updates |= output_shape.remove_variadic(input_shape.reverse)
                if len(input_shape.prefix) != len(output_shape.prefix):
                    raise ValueError("Shape mismatch in Transpose model")
            status = True

    elif isinstance(axes_val, int | list | tuple):
        _axes_val: list[int | ToBeDetermined]

        if isinstance(axes_val, int):
            _axes_val = [axes_val]
        elif enhanced_isinstance(axes_val, tuple[int | ToBeDetermined, ...]):
            _axes_val = list(axes_val)
        else:
            assert enhanced_isinstance(axes_val, list[int | ToBeDetermined])
            _axes_val = axes_val

        # check if all values in axes_val is determined
        # if all of them is determined, do the shape updates
        # if only one of them is missing, infer it.
        missing_indices: set[int] = set()
        for idx in range(len(_axes_val)):
            # find all missing indices
            if idx not in _axes_val:
                missing_indices.add(idx)
            if len(missing_indices) >= 2:
                break
        else:
            # Note that length of missing indices is 1 or 0 in there
            if missing_indices:
                # find the index of TBD and replace it
                # with the missing index.
                missing_idx = missing_indices.pop()
                tbd_idx = _axes_val.index(missing_idx)
                _axes_val[tbd_idx] = missing_idx
                updates |= axes.set_value(_axes_val)
            assert is_list_int(_axes_val)
            in_unis = [Uniadic() for _ in range(len(_axes_val))]
            out_unis = [in_unis[axis] for axis in _axes_val]

            updates |= input_shape.update_uniadics(input_shape.prefix, in_unis)
            updates |= input_shape.update_uniadics(input_shape.reverse, in_unis[::-1])

            updates |= output_shape.update_uniadics(output_shape.prefix, out_unis)
            updates |= output_shape.update_uniadics(
                output_shape.reverse, out_unis[::-1]
            )

            if input_shape.root is not None:
                updates |= input_shape.remove_variadic(in_unis)
            if output_shape.root is not None:
                updates |= output_shape.remove_variadic(out_unis)

            status = True

    return status, updates


def polynomial_features_constraints(
    output: IOHyperEdge, input: IOHyperEdge, degree: IOHyperEdge
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
            and degree_val is not None
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
    input: int,
    stride: int,
    padding: tuple[int, int] | int,
    dilation: int,
    kernel_size: int,
) -> int:
    if isinstance(padding, Sequence):
        padding = sum(padding)
        padding_factor = padding
    else:
        padding_factor = 2 * padding
    # TODO: Is Uniadic type kernel_size possible?
    if (
        val := (input + padding_factor - (kernel_size - 1) * dilation - 1) // stride + 1
    ) <= 0:
        raise ValueError(
            "Dimension Error: Output dimension calculated to be lesser than zero!"
        )

    return val


def sliding_window_1d_constraints(
    output: IOHyperEdge,
    input: IOHyperEdge,
    stride: IOHyperEdge,
    padding: IOHyperEdge,
    dilation: IOHyperEdge,
    kernel: IOHyperEdge,
) -> ConstrainResultType:
    status = False
    updates = Updates()
    assert input._temp_shape is not None, "Input shape of Convolution2D is not set!"
    assert output._temp_shape is not None, "Output shape of Convolution2D is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    kernel_shape: ShapeRepr | None = kernel._temp_shape

    input_val = input_shape[-1].value if len(input_shape.reverse) >= 1 else None

    if kernel_shape is not None:
        kernel_val = kernel_shape[-1].value if len(kernel_shape.reverse) >= 1 else None
    else:
        kernel_val = kernel._value if isinstance(kernel._value, int) else None
        assert isinstance(kernel_val, int | None), "Invalid kernel value!"

    stride_val = stride.value
    padding_val = padding.value
    dilation_val = dilation.value

    assert isinstance(stride_val, int | ToBeDetermined), "Invalid stride value!"
    assert isinstance(dilation_val, int | ToBeDetermined), "Invalid dilation value!"
    assert isinstance(padding_val, int | ToBeDetermined) or enhanced_isinstance(
        padding_val, tuple[int | ToBeDetermined, int | ToBeDetermined]
    ), "Invalid padding value!"

    if (
        not isinstance(stride_val, ToBeDetermined)
        and not isinstance(padding_val, ToBeDetermined)
        and not isinstance(dilation_val, ToBeDetermined)
    ):
        output_shape_val: int | None = None
        if (
            isinstance(input_val, int)
            and (
                isinstance(padding_val, int)
                or enhanced_isinstance(padding_val, tuple[int, int])
            )
            and isinstance(kernel_val, int)
        ):
            output_shape_val = sliding_window_constraint_helper(
                input_val,
                stride_val,
                padding_val,
                dilation_val,
                kernel_val,
            )
        updates |= output_shape.inner_match(
            prefix=[], root=Variadic(), suffix=[Uniadic(output_shape_val)]
        )

    status = len(output_shape) >= 1 and output_shape[-1].value is not None
    return status, updates


def sliding_window_2d_constraints(
    output: IOHyperEdge,
    input: IOHyperEdge,
    stride: IOHyperEdge,
    padding: IOHyperEdge,
    dilation: IOHyperEdge,
    kernel: IOHyperEdge,
) -> ConstrainResultType:
    updates = Updates()
    assert input._temp_shape is not None, "Input shape of Convolution2D is not set!"
    assert output._temp_shape is not None, "Output shape of Convolution2D is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    kernel_shape: ShapeRepr | None = kernel._temp_shape

    if len(input_shape.reverse) >= 2:
        input_val = (input_shape[-2].value, input_shape[-1].value)
    elif len(input_shape.reverse) == 1:
        input_val = (None, input_shape[-1].value)
    else:
        input_val = (None, None)

    if kernel_shape is not None:
        if len(kernel_shape.reverse) >= 2:
            kernel_val = (kernel_shape[-2].value, kernel_shape[-1].value)
        elif len(kernel_shape.reverse) == 1:
            kernel_val = (None, kernel_shape[-1].value)
        else:
            kernel_val = (None, None)
    else:
        kernel_val = kernel._value  # type: ignore
        assert enhanced_isinstance(
            kernel_val, tuple[int | ToBeDetermined, int | ToBeDetermined]
        ), "Invalid kernel value!"

    stride_val = stride.value
    padding_val = padding.value
    dilation_val = dilation.value

    assert enhanced_isinstance(
        stride_val, tuple[int | ToBeDetermined, int | ToBeDetermined]
    ) or isinstance(stride_val, ToBeDetermined), "Invalid stride value!"
    assert enhanced_isinstance(
        dilation_val, tuple[int | ToBeDetermined, int | ToBeDetermined]
    ) or isinstance(dilation_val, ToBeDetermined), "Invalid stride value!"
    assert (
        enhanced_isinstance(
            padding_val, tuple[int | ToBeDetermined, int | ToBeDetermined]
        )
        or isinstance(padding_val, ToBeDetermined)
        or enhanced_isinstance(
            padding_val,
            tuple[
                tuple[int | ToBeDetermined, int | ToBeDetermined],
                tuple[int | ToBeDetermined, int | ToBeDetermined],
            ],
        )
    ), "Invalid padding value!"

    # To calculate maxpool constraint we need to know ... and last 2 dimension of
    # the input
    if (
        not isinstance(stride_val, ToBeDetermined)
        and not isinstance(padding_val, ToBeDetermined)
        and not isinstance(dilation_val, ToBeDetermined)
    ):
        input_h, input_w = input_val
        stride_h, stride_w = stride_val
        padding_h, padding_w = padding_val
        dilation_h, dilation_w = dilation_val
        kernel_h, kernel_w = kernel_val

        output_shape_h: int | None = None
        output_shape_w: int | None = None
        if (
            isinstance(input_h, int)
            and isinstance(stride_h, int)
            and (
                isinstance(padding_h, int)
                or enhanced_isinstance(padding_h, tuple[int, int])
            )
            and isinstance(dilation_h, int)
            and isinstance(kernel_h, int)
        ):
            output_shape_h = sliding_window_constraint_helper(
                input_h,
                stride_h,
                padding_h,
                dilation_h,
                kernel_h,
            )

        if (
            isinstance(input_w, int)
            and isinstance(stride_w, int)
            and (
                isinstance(padding_w, int)
                or enhanced_isinstance(padding_w, tuple[int, int])
            )
            and isinstance(dilation_w, int)
            and isinstance(kernel_w, int)
        ):
            output_shape_w = sliding_window_constraint_helper(
                input_w,
                stride_w,
                padding_w,
                dilation_w,
                kernel_w,
            )
        updates |= output_shape.inner_match(
            prefix=[],
            root=Variadic(),
            suffix=[Uniadic(output_shape_h), Uniadic(output_shape_w)],
        )

    status = (
        len(output_shape) >= 2
        and output_shape[-2].value is not None
        and output_shape[-1].value is not None
    )
    return status, updates


def flatten_constrains(
    output: IOHyperEdge,
    input: IOHyperEdge,
    start_dim: IOHyperEdge,
    end_dim: IOHyperEdge,
) -> ConstrainResultType:
    status = False
    updates = Updates()
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
                    prefix=prefix + [Uniadic(prod)] + suffix
                )
    return status, updates


def where_constrains(
    output: IOHyperEdge, cond: IOHyperEdge, input1: IOHyperEdge, input2: IOHyperEdge
) -> ConstrainResultType:
    # TODO: Find a way to implement this constraint without creating a Tensor and
    # ShapeRepr
    assert output._temp_shape is not None, "Output shape of Where is not set!"
    assert cond._temp_shape is not None, "Condition shape of Where is not set!"
    assert input1._temp_shape is not None, "Input1 shape of Where is not set!"
    assert input2._temp_shape is not None, "Input2 shape of Where is not set!"
    status = False
    updates = Updates()

    broadcast_shp = ShapeRepr(root=Variadic())

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
    output: IOHyperEdge, start: IOHyperEdge, stop: IOHyperEdge, step: IOHyperEdge
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
                updates |= output_shape.update_uniadics(
                    output_shape.prefix, out_uniadic
                )
                updates |= output_shape.update_uniadics(
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
    return status, updates


def randn_constraints(output: IOHyperEdge, shape: IOHyperEdge) -> ConstrainResultType:
    updates = Updates()
    assert output._temp_shape is not None, "Output shape of Reshape is not set!"

    output_shape: ShapeRepr = output._temp_shape
    shape_val = shape.value

    assert (
        enhanced_isinstance(shape_val, tuple[int | ToBeDetermined, ...])
        or enhanced_isinstance(shape_val, list[int | ToBeDetermined])
        or isinstance(shape_val, ToBeDetermined)
        or enhanced_isinstance(shape_val, tuple[()])
    ), "Invalid shape value!"

    if not isinstance(shape_val, ToBeDetermined):
        # Check shape consistency.
        if (min_dims := (len(output_shape.prefix) + len(output_shape.suffix))) > len(
            shape_val
        ):
            raise ValueError(
                f"Shape mismatch. Output has minimum {min_dims} dim(s) where it "
                f"must have exactly {len(shape_val)} dim(s)."
            )
        out_uniadics = [
            Uniadic(dim) if isinstance(dim, int) else Uniadic() for dim in shape_val
        ]
        updates |= output_shape.inner_match(out_uniadics)

    return shape.is_valued, updates


def broadcast_to_constraints(
    output: IOHyperEdge, shape: IOHyperEdge, input: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    assert input._temp_shape is not None, "Input shape of BroadcastTo is not set!"
    assert output._temp_shape is not None, "Output shape of BroadcastTo is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    shape_val = shape.value
    assert enhanced_isinstance(
        shape_val, tuple[int | ToBeDetermined, ...]
    ) or isinstance(shape_val, ToBeDetermined), "Invalid shape value!"

    if not isinstance(shape_val, ToBeDetermined):
        # Check shape consistency.
        if (min_dims := (len(output_shape.prefix) + len(output_shape.suffix))) > len(
            shape_val
        ):
            raise ValueError(
                f"Shape mismatch. Output has minimum {min_dims} dim(s) where it "
                f"must have exactly {len(shape_val)} dim(s)."
            )
        out_uniadics = [
            Uniadic(dim) if isinstance(dim, int) else Uniadic() for dim in shape_val
        ]
        updates |= output_shape.inner_match(out_uniadics)

        if input_shape.root is None:
            # if input is uniadic, look for if every input is determined,
            # if determined, validate its shape
            for uni in input_shape.prefix:
                if uni.value is None:
                    break
            else:
                if is_tuple_int(shape_val):
                    validate_bcast(input_shape, shape_val)

    return shape.is_valued and is_repr_known(input_shape), updates


def validate_bcast(input: ShapeRepr, shape: tuple[int, ...]) -> None:
    if input.root is None:
        if len(input) > len(shape):
            raise ValueError("Cannot broadcast to lower dimension")
        for idx, in_uni in enumerate(input.reverse):
            out_value = shape[-idx - 1]
            if in_uni.value != 1 and in_uni.value != out_value:
                raise ValueError("Shape mismatch in broadcast_to model")


def reshape_constraints(
    output: IOHyperEdge, input: IOHyperEdge, shape: IOHyperEdge
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
    if shape.is_valued:
        assert enhanced_isinstance(
            shape_val, tuple[int | None, ...]
        ) or enhanced_isinstance(shape_val, list[int | None]), "Invalid shape value!"

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
            updates |= output_shape.update_uniadics(output_shape.prefix, out_uniadics)
            updates |= output_shape.update_uniadics(
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
    elif is_repr_known(output_shape) and is_repr_known(input_shape):
        if is_repr_known(input_shape) and reduce(prod_fn, input_shape.prefix) != reduce(  # type: ignore
            prod_fn,  # type: ignore
            output_shape.prefix,
        ):
            out_shape = output_shape.get_shapes()
            in_shape = input_shape.get_shapes()
            raise ValueError(
                f"Shape mismatch! output {out_shape} and input {in_shape} have "
                "incompatible shapes"
            )
    status = is_repr_known(input_shape) and is_repr_known(output_shape)
    return status, updates


def squeeze_constraints(output: IOHyperEdge, input: IOHyperEdge) -> ConstrainResultType:
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
    new_prefix: list[Uniadic] = []
    new_suffix: list[Uniadic] = []
    variadic_required = False

    for uni in input_shape.prefix:
        if uni.value is None:
            variadic_required = True
            break
        elif uni.value != 1:
            new_prefix.append(uni)

    # If Variadic input, iterate over reverse suffix else
    # reverse prefix.
    reverse_uni_list: list[Uniadic] = list()
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


def size_constraints(
    output: IOHyperEdge, input: IOHyperEdge, dim: IOHyperEdge
) -> ConstrainResultType:
    assert input._temp_shape is not None, "Input shape of Size is not set!"
    input_shape: ShapeRepr = input._temp_shape
    updates = Updates()
    dim_val = dim.value
    output_val = output.value
    assert isinstance(dim_val, int | None | ToBeDetermined)
    assert isinstance(output_val, int | None | ToBeDetermined)
    if isinstance(dim_val, int):
        prefix: list[Uniadic] = []
        suffix: list[Uniadic] = []
        if dim_val > 0:
            prefix = [Uniadic() for _ in range(dim_val + 1)]
        else:
            suffix = [Uniadic() for _ in range(abs(dim_val))]
        updates |= input_shape.inner_match(
            prefix=prefix, root=Variadic(), suffix=suffix
        )
        if (
            len(input_shape.prefix) > dim_val >= 0
            or -len(input_shape.suffix) <= dim_val < 0
        ):
            uni = input_shape[dim_val]
            if uni.value is not None:
                updates |= output.set_value(uni.value)

    elif dim_val is None:
        if is_repr_known(input_shape):
            shape_values = [val.value for val in input_shape.prefix]
            assert enhanced_isinstance(shape_values, list[int])
            output_val = math.prod(shape_values)
            updates |= output.set_value(output_val)

    return output.is_valued, updates


def shape_constraints(output: IOHyperEdge, input: IOHyperEdge) -> ConstrainResultType:
    assert input._temp_shape is not None, "Input shape of Shape is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_val = output.value
    assert isinstance(output_val, ToBeDetermined) or enhanced_isinstance(
        output_val, tuple[int | ToBeDetermined, ...]
    )
    updates = Updates()

    if input_shape.root is None:
        in_shape = input_shape.get_shapes({}, {})
        inferred_output_value = tuple(
            shp if isinstance(shp, int) else TBD for shp in in_shape
        )
        updates |= output.set_value(inferred_output_value)
        output_val = output.value
        assert isinstance(output_val, tuple)
        input_prefix = [
            Uniadic(val) if isinstance(val, int) else Uniadic() for val in output_val
        ]
        updates |= input_shape.inner_match(prefix=input_prefix)

    elif not isinstance(output_val, ToBeDetermined):
        input_prefix = [
            Uniadic(val) if isinstance(val, int) else Uniadic() for val in output_val
        ]
        updates |= input_shape.inner_match(prefix=input_prefix)
        in_shape = input_shape.get_shapes({}, {})
        output_value = tuple(shp if isinstance(shp, int) else TBD for shp in in_shape)
        updates |= output.set_value(output_value)

    return output.is_valued and is_repr_known(input_shape), updates


def eye_constraints(
    output: IOHyperEdge, N: IOHyperEdge, M: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    assert output._temp_shape is not None, "Output shape of Eye is not set!"
    output_shape: ShapeRepr = output._temp_shape
    n_uni, m_uni = output_shape.prefix[0], output_shape.prefix[1]
    n_valued = isinstance(N.value, int)
    m_valued = isinstance(M.value, int | NoneType)
    n_uni_valued = isinstance(n_uni.value, int)
    m_uni_valued = isinstance(m_uni.value, int)

    if n_valued and not n_uni_valued:
        assert isinstance(N.value, int)
        n_uni.set_value(N.value)
        updates.add(n_uni)

    elif n_uni_valued and not n_valued:
        updates |= N.set_value(n_uni.value)

    if m_valued and not m_uni_valued:
        assert isinstance(M.value, int | NoneType)
        m_uni.set_value(M.value)
        updates.add(m_uni)

    elif m_uni_valued and not m_valued:
        updates |= M.set_value(m_uni.value)

    all_items: list[IOHyperEdge | Uniadic] = [N, M, n_uni, m_uni]
    return all(isinstance(s.value, int) for s in all_items), updates


def swap_axes_constraints(
    output: IOHyperEdge, input: IOHyperEdge, axis1: IOHyperEdge, axis2: IOHyperEdge
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
                updates |= other.match(non_variadic)
                other[axis1_val], other[axis2_val] = other[axis2_val], other[axis1_val]
            status = True

    elif isinstance(axis1_val, ToBeDetermined) ^ isinstance(axis2_val, ToBeDetermined):
        # If only one of the axes are given. Find the given axis.
        # create uniadics with the same amount of this axis and match it
        # with input
        given_axis: int | None = None
        if not isinstance(axis1_val, ToBeDetermined):
            given_axis = axis1_val
        elif not isinstance(axis2_val, ToBeDetermined):
            given_axis = axis2_val
        assert isinstance(given_axis, int)

        unis: list[Uniadic] = []
        if given_axis >= 0:
            unis = [Uniadic() for _ in range(given_axis + 1)]
        elif given_axis < 0:
            unis = [Uniadic() for _ in range(abs(given_axis))]
        updates |= input_shape.inner_match(prefix=unis, root=Variadic())

    return status, updates


def to_tensor_constraints(
    output: IOHyperEdge, input: IOHyperEdge
) -> ConstrainResultType:
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
        shape: list[int] = []
        if isinstance(input_val, list | tuple) and is_valued(input_val):
            _shape, _, typ = process_value(input_val)
            assert _shape is not None
            shape = _shape
            updates |= output.set_type(Tensor[typ])  # type: ignore
            updates.add(output, update_type=UpdateType.TYPE)
        elif isinstance(input_val, float | int):
            assert isinstance(input.value_type, type(int) | type(float))
            shape = []
            updates |= output.set_type(Tensor[input.value_type])  # type: ignore
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


def tensor_to_list_constraints(
    output: IOHyperEdge, input: IOHyperEdge
) -> ConstrainResultType:
    assert input._temp_shape is not None, "Input shape of TensorToList is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_val = output.value
    assert isinstance(output_val, ToBeDetermined) or type(output_val) is list
    updates = Updates()
    output_value = output.value
    input_shape = input._temp_shape
    status = False
    if not isinstance(output_val, ToBeDetermined):
        shape: list[Uniadic] = []
        if isinstance(output_value, list | tuple) and is_valued(output_value):
            shp, *_ = process_value(output_val)
            assert shp is not None
            shape = [Uniadic(idx) for idx in shp]

        updates |= input_shape.inner_match(prefix=shape)
        status = True

    return status, updates


def item_constraints(output: IOHyperEdge, input: IOHyperEdge) -> ConstrainResultType:
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
    output: IOHyperEdge, input: IOHyperEdge, index: IOHyperEdge
) -> ConstrainResultType:
    assert (
        isinstance(output._value, ToBeDetermined)
        or type(output._value) is int
        or type(output._value) is float
        or type(output._value) is tuple
        or type(output._value) is list
        or type(output._value) is Tensor
    )

    assert (
        isinstance(input._value, ToBeDetermined)
        or type(input._value) is tuple
        or type(input._value) is list
    )

    assert (
        isinstance(index._value, ToBeDetermined)
        or type(index._value) is int
        or type(index._value) is slice
    )

    updates = Updates()
    status = False
    # Forward value propagation.
    if not isinstance(input._value, ToBeDetermined) and not isinstance(
        index._value, ToBeDetermined
    ):
        updates |= output.set_value(input._value[index._value])
        status = output.is_valued
    elif (
        not isinstance(input._value, ToBeDetermined)
        and TBD not in input._value
        and isinstance(output._value, int | float | bool | Tensor)
    ):
        # Try to infer index value from input-output values. If
        # output value appears only once in input sequence, write its
        # index as the value of index argument.
        if input._value.count(output._value) == 1:
            updates |= index.set_value(input._value.index(output._value))
            status = True
    return status, updates


def to_tuple_constraints(
    output: IOHyperEdge, *args: IOHyperEdge
) -> ConstrainResultType:
    return to_sequence_helper(tuple, output, *args)


def to_list_constraints(output: IOHyperEdge, *args: IOHyperEdge) -> ConstrainResultType:
    return to_sequence_helper(list, output, *args)


def tensor_item_constraints(
    output: IOHyperEdge, input: IOHyperEdge, index: IOHyperEdge
) -> ConstrainResultType:
    assert output._temp_shape is not None, "Output shape of Item is not set!"
    assert input._temp_shape is not None, "Input shape of Item is not set!"
    input_shape: ShapeRepr = input._temp_shape
    output_shape: ShapeRepr = output._temp_shape
    index_val = index._value

    assert (
        isinstance(index_val, ToBeDetermined)
        or type(index_val) is int
        or type(index_val) is slice
        or type(index_val) is NoneType
        or type(index_val) is EllipsisType
        or type(index_val) is Tensor
        or find_intersection_type(find_type(index_val), VariableSequenceType[int])  # type: ignore
        or is_index_type(index_val)
    )

    status = False
    updated_symbols = Updates()

    if not isinstance(index_val, tuple):
        index_val = (index_val,)
    index_status = True
    for idx_val in index_val:
        if idx_val is TBD:
            index_status = False
            break
        elif isinstance(idx_val, Sequence):
            if not is_valued(idx_val):
                index_status = False
                break

    if index_status:
        # Initially set all status to True,
        # Final status will be intersection of all status

        # slice_status determines if all slice operation inferences are made
        slice_status = True

        # match_status determines if input and
        # output shapes are matched without a problem
        match_status = True

        # bcast status determines if all broadcast operations are made
        bcast_status = True

        assert is_index_type(index_val)

        # keeps all tensor shapes in the index
        tensor_reprs: list[ShapeRepr] = []

        # successive_status:
        # 0: init,
        # 1: successive tensor indices found,
        # 2: tensor indices stopped,
        # 3: non-successive tensor indices found
        successive_status: Literal[0, 1, 2, 3] = 0
        slice_process_list: list[tuple[Uniadic, Uniadic, slice]] = []

        # input shapes inferred from index values
        index_input_prefix: list[Uniadic] = []
        index_input_suffix: list[Uniadic] = []

        # output shapes inferred from index values
        inferred_output_prefix: list[Uniadic] = []
        inferred_output_suffix: list[Uniadic] = []

        # reduce_result_dim:
        # 0: int - keeps where will the result be reduced to
        # 1: list[Uniadic] - inferred_output_prefix or inferred_output_suffix
        reduce_result_dim: tuple[int, list[Uniadic]] = (0, inferred_output_prefix)

        current_index_unis = index_input_prefix
        current_inferred_unis = inferred_output_prefix

        for value in index_val:
            if isinstance(value, int | Tensor | Sequence):
                if successive_status == 0:
                    reduce_result_dim = (
                        len(current_inferred_unis),
                        current_inferred_unis,
                    )
                    successive_status = 1

                if successive_status == 2:
                    reduce_result_dim = (0, inferred_output_prefix)
                    successive_status = 3

                if isinstance(value, Tensor):
                    tensor_reprs.append(value.shape.reprs[0])

                if isinstance(value, Sequence):
                    shp, *_ = process_value(value)
                    assert shp is not None
                    tensor_reprs.append(ShapeRepr(prefix=[Uniadic(idx) for idx in shp]))

                current_index_unis.append(Uniadic())

            else:
                if successive_status == 1:
                    successive_status = 2

                if value is Ellipsis:
                    # if value is ellipsis, change current lists
                    current_index_unis = index_input_suffix
                    current_inferred_unis = inferred_output_suffix

                elif value is None:
                    # add newaxis to output if value is None
                    current_inferred_unis.append(Uniadic(1))

                elif isinstance(value, slice):
                    # if value is slice, add new uniadic to both input and output
                    # also add it to slice process list to infer it later
                    current_inferred_unis.append(output_uni := Uniadic())
                    current_index_unis.append(input_uni := Uniadic())
                    slice_process_list.append((output_uni, input_uni, value))

        if not tensor_reprs:
            # if no tensor found, bcast_result is empty
            reduced_output_shapes = []

        elif all(repr.root is None for repr in tensor_reprs):
            # if all tensors are non-variadic, broadcast them
            all_reprs = (repr for repr in tensor_reprs)
            prev_repr = next(all_reprs)
            for repr in all_reprs:
                next_repr = ShapeRepr(root=Variadic())
                _status, _updates = bcast_helper(next_repr, prev_repr, repr, 0)
                bcast_status &= _status
                updated_symbols |= _updates
                prev_repr = next_repr
            reduced_output_shapes = prev_repr.prefix

        else:
            # TODO: Index tensors could be variadic in some cases
            # handle the case where tensors are variadic
            return False, Updates()

        idx, tensor_affix = reduce_result_dim
        tensor_affix[idx:idx] = reduced_output_shapes

        # finally, match the inferred shapes from index to input shaeps
        updated_symbols |= input_shape.inner_match(
            prefix=index_input_prefix, root=Variadic(), suffix=index_input_suffix
        )

        for output_uni, input_uni, slc in slice_process_list:
            # find values of sliced outputs if possible
            if slc == slice(None, None, None):
                updated_symbols |= output_uni.match(input_uni)
            else:
                if input_uni.value is not None:
                    output_value = len(list(range(input_uni.value))[slc])
                    if output_uni.set_value(output_value):
                        updated_symbols.add(output_uni)
                else:
                    slice_status = False

        if input_shape.root is None:
            out_result_shape = input_shape.prefix[:]

            out_result_shape[: len(index_input_prefix)] = inferred_output_prefix
            out_result_shape[len(out_result_shape) - len(index_input_suffix) :] = (
                inferred_output_suffix
            )

            updated_symbols |= output_shape.inner_match(prefix=out_result_shape)
        else:
            if len(index_input_prefix) > len(input_shape.prefix) or len(
                index_input_suffix
            ) > len(input_shape.reverse):
                # case where inferred inputs with index could not match properly with
                # current input shape
                match_status = False
                updated_symbols |= output_shape.inner_match(
                    inferred_output_prefix, Variadic(), inferred_output_suffix
                )
            else:
                output_prefix = input_shape.prefix[:]
                output_suffix = input_shape.suffix[:]

                output_prefix[: len(index_input_prefix)] = inferred_output_prefix
                output_suffix[len(output_suffix) - len(index_input_suffix) :] = (
                    inferred_output_suffix
                )

                updated_symbols |= output_shape.inner_match(
                    output_prefix, input_shape.root, output_suffix
                )

        status = bcast_status and slice_status and match_status
    return status, updated_symbols


def indexer_constraints(
    output: IOHyperEdge, input: IOHyperEdge, index: IOHyperEdge
) -> ConstrainResultType:
    if input.is_tensor:
        return tensor_item_constraints(output, input, index)
    elif input.is_scalar:
        return scalar_item_constraints(output, input, index)
    return False, Updates()


def padding_1d_constraint(
    output: IOHyperEdge, input: IOHyperEdge, kernel_size: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    input_value = input.value
    kernel_size_value = kernel_size.value
    if isinstance(input_value, PaddingType):
        if input_value == PaddingType.VALID:
            updates |= output.set_value((0, 0))
        else:
            if isinstance(kernel_size_value, int):
                if kernel_size_value % 2 == 0:
                    raise RuntimeError(
                        "'same' padding is not supported when the kernel size is even!"
                    )
                updates |= output.set_value((kernel_size_value // 2,) * 2)
            elif kernel_size_value is not TBD:
                raise RuntimeError("Kernel size must be 'tuple[int, int]' or 'int'!")

    elif isinstance(input_value, int):
        updates |= output.set_value((input_value, input_value))

    elif isinstance(input_value, Sequence):
        if isinstance(input_value[0], Sequence) or isinstance(input_value[1], Sequence):
            raise RuntimeError(f"Given input value '{input_value}' is not valid!")
        updates |= output.set_value(tuple(input_value))

    return output.is_valued, updates


def padding_2d_constraint(
    output: IOHyperEdge, input: IOHyperEdge, kernel_size: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    input_value = input.value
    kernel_size_value = kernel_size._value
    if isinstance(input_value, PaddingType):
        if input_value == PaddingType.VALID:
            updates |= output.set_value((0, 0))
        else:
            if enhanced_isinstance(
                kernel_size_value, tuple[int | ToBeDetermined, int | ToBeDetermined]
            ):
                height = kernel_size_value[0]
                width = kernel_size_value[1]
                if (isinstance(height, int) and height % 2 == 0) or (
                    isinstance(width, int) and width % 2 == 0
                ):
                    raise RuntimeError(
                        "'same' padding is not supported when the kernel size is even!"
                    )
                height_val = height // 2 if isinstance(height, int) else height
                width_val = width // 2 if isinstance(width, int) else width
                updates |= output.set_value(
                    ((height_val, height_val), (width_val, width_val))
                )
            elif isinstance(kernel_size_value, int):
                if kernel_size_value % 2 == 0:
                    raise RuntimeError(
                        "'same' padding is not supported when the kernel size is even!"
                    )
                updates |= output.set_value(
                    (tuple([kernel_size_value // 2, kernel_size_value // 2]),) * 2
                )
            elif kernel_size.value is not TBD:
                raise RuntimeError("Kernel size must be 'tuple[int, int]' or 'int'!")
    elif isinstance(input_value, int):
        updates |= output.set_value((input_value, input_value))
    elif enhanced_isinstance(
        input_value,
        tuple[
            tuple[int | ToBeDetermined, int | ToBeDetermined],
            tuple[int | ToBeDetermined, int | ToBeDetermined],
        ],
    ):
        updates |= output.set_value(input_value)

    elif enhanced_isinstance(
        input_value, tuple[int | ToBeDetermined, int | ToBeDetermined]
    ):
        pad1, pad2 = input_value
        updates |= output.set_value(((pad1, pad1), (pad2, pad2)))
    return output.is_valued, updates


def stride_constraint(
    output: IOHyperEdge, input: IOHyperEdge, kernel_size: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    input_value = input.value
    kernel_size_value = kernel_size.value
    if input_value is None:
        updates |= output.set_value(kernel_size_value)
        updates |= kernel_size.set_value(output._value)
        status = output.is_valued and kernel_size.is_valued
    else:
        updates |= output.set_value(input_value)
        updates |= input.set_value(output._value)
        status = output.is_valued and input.is_valued
    return status, updates


def tuple_converter_constraint(
    output: IOHyperEdge, input: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    input_value = input._value
    if isinstance(input_value, int):
        updates |= output.set_value((input_value, input_value))
    if isinstance(input_value, tuple):
        updates |= output.set_value(input_value)
    return output.is_valued, updates


def cross_entropy_constraint(
    categorical: IOHyperEdge, input: IOHyperEdge, target: IOHyperEdge
) -> ConstrainResultType:
    assert input._temp_shape is not None, "Input shape of reverse is not set!"
    assert target._temp_shape is not None, "Target shape of reverse is not set!"

    status = False
    updates = Updates()
    categorical_value = categorical.value

    input_shape: ShapeRepr = input._temp_shape
    target_shape: ShapeRepr = target._temp_shape

    if categorical_value is not TBD:
        if not categorical_value:
            updates |= target_shape.match(input_shape)
        else:
            N = Uniadic()
            C = Uniadic()
            var = Variadic()
            in_repr = ShapeRepr([N, C], var)
            target_repr = ShapeRepr([N], var)
            updates = input_shape.match(in_repr)
            updates = target_shape.match(target_repr)

        status = True
    return status, updates


def buffer_constraint(output: IOHyperEdge, input: IOHyperEdge) -> ConstrainResultType:
    updates = Updates()
    status = False
    is_input_polymorphic: bool = input.is_polymorphic
    is_output_polymorphic: bool = output.is_polymorphic

    if not (is_input_polymorphic and is_output_polymorphic):
        # at least one of them is not polymorphic

        if is_input_polymorphic ^ is_output_polymorphic:
            # one of them is polymorphic while other is not
            typed, non_typed = (
                (input, output) if is_output_polymorphic else (output, input)
            )
            updates |= non_typed.set_type(typed.edge_type)
            updates |= non_typed.set_value(typed._value)
            if typed.is_tensor or typed.is_valued:
                status = True
        else:
            # both are not polymorphic
            updates |= output.set_type(input.edge_type)
            updates |= input.set_type(output.edge_type)
            if input.is_tensor:
                if input._value is not output._value:
                    updates |= input.set_value(output._value)
                    status = True
            else:
                is_input_valued = input._value is not TBD
                is_output_valued = output._value is not TBD
                if is_input_valued ^ is_output_valued:
                    valued, non_valued = (
                        (input, output) if is_input_valued else (output, input)
                    )
                    updates |= non_valued.set_value(valued._value)
                    status = True

    return status, updates


def polynomial_kernel_constraint(
    poly_coef: IOHyperEdge, degree: IOHyperEdge
) -> ConstrainResultType:
    updates = Updates()
    coef_status = False
    degree_status = False
    # poly_coef update.
    if not poly_coef.is_polymorphic:
        coef_status = True
        if poly_coef.is_tensor:
            assert poly_coef.shape is not None
            updates |= poly_coef.shape.set_values([])
    # degree update.
    if not degree.is_polymorphic:
        degree_status = True
        if degree.is_tensor:
            assert degree.shape is not None
            updates |= degree.shape.set_values([])
    return coef_status & degree_status, updates


constrain_fn_dict = {key: fn for key, fn in globals().items() if callable(fn)}

constraint_type_map: dict[ConstraintFunctionType, list[UpdateType]] = {
    scalar_slice_type_constraint: [UpdateType.TYPE],
    indexer_initial_type_constraint: [UpdateType.TYPE],
    indexer_type_constraint: [UpdateType.TYPE],
    slice_constraints: [UpdateType.VALUE],
    bcast: [UpdateType.SHAPE],
    bcast_matrix_mult: [UpdateType.SHAPE],
    to_tensor_constraints: [UpdateType.SHAPE, UpdateType.TYPE],
    tensor_to_list_constraints: [UpdateType.SHAPE, UpdateType.TYPE],
    to_list_constraints: [UpdateType.VALUE],
    where_constrains: [UpdateType.SHAPE],
    item_constraints: [UpdateType.SHAPE],
    to_tuple_constraints: [UpdateType.VALUE],
    tensor_to_list_type_constraint: [UpdateType.TYPE],
    reduce_type_constraint: [UpdateType.TYPE],
    padding_1d_constraint: [UpdateType.VALUE],
    padding_2d_constraint: [UpdateType.VALUE],
    stride_constraint: [UpdateType.VALUE],
    tuple_converter_constraint: [UpdateType.VALUE],
    buffer_constraint: [UpdateType.TYPE, UpdateType.VALUE],
    general_type_constraint: [UpdateType.TYPE],
}
