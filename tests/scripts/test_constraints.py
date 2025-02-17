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

from collections.abc import Callable, Mapping
from copy import deepcopy
from functools import partial
from types import EllipsisType, NoneType, UnionType
from typing import Any, TypeGuard

import numpy as np
import pytest

from mithril.framework.common import (
    TBD,
    ConstraintFunctionType,
    ConstraintSolver,
    IOHyperEdge,
    PossibleValues,
    ShapeRepr,
    ShapeResultType,
    ShapeTemplateType,
    Tensor,
    TensorToListType,
    ToBeDetermined,
    Uniadic,
    UniadicRecord,
    Updates,
    UpdateType,
    Variadic,
    create_shape_repr,
)
from mithril.framework.constraints import (
    arange_constraints,
    bcast,
    bcast_matrix_mult,
    broadcast_to_constraints,
    concat_constraints,
    constraint_type_map,
    eye_constraints,
    flatten_constrains,
    general_forward_constraint,
    general_tensor_type_constraint,
    generate_nested_list_type,
    indexer_constraints,
    indexer_type_constraint,
    item_constraints,
    pad_constraints,
    polynomial_features_constraints,
    randn_constraints,
    reduce_constraints,
    reduce_type_constraint,
    reshape_constraints,
    reverse_constraints,
    scalar_slice_type_constraint,
    shape_constraints,
    size_constraints,
    slice_constraints,
    sliding_window_2d_constraints,
    split_constraints,
    squeeze_constraints,
    swap_axes_constraints,
    tensor_to_list_constraints,
    tensor_to_list_type_constraint,
    to_list_constraints,
    to_tensor_constraints,
    to_tuple_constraints,
    where_constrains,
)
from mithril.types import GenericDataType

from .test_utils import check_shapes_semantically


def is_type_checker(
    ref_results: dict[str, type] | ShapeResultType,
    constraint_fn: ConstraintFunctionType,
) -> TypeGuard[dict[str, type]]:
    types = constraint_type_map.get(constraint_fn)
    if types:
        return UpdateType.TYPE in types
    else:
        return False


def is_shape_checker(
    ref_results: dict[str, type] | ShapeResultType,
    constraint_fn: ConstraintFunctionType,
) -> TypeGuard[ShapeResultType]:
    types = constraint_type_map.get(constraint_fn)
    if types:
        return bool({UpdateType.SHAPE, UpdateType.VALUE} & set(types))
    else:
        return True


######################### Helper Functions #########################

VariadicPossiblesType = (
    list[tuple[int, ...]] | list[tuple[str, ...]] | list[tuple[int | str, ...]]
)
VariadicTemplateType = tuple[str, EllipsisType]

AssignmentType = (
    Mapping[str, set[int]]
    # | dict[tuple[str, EllipsisType], list[tuple[int | str, ...]]]
    # | dict[tuple[str, EllipsisType], list[tuple[int, ...]]]
    # | dict[tuple[str, EllipsisType], list[tuple[str, ...]]]
    | Mapping[VariadicTemplateType, VariadicPossiblesType]
    | Mapping[VariadicTemplateType, set[int]]
    | Mapping[str, VariadicPossiblesType]
    | Mapping[str | VariadicTemplateType, set[int] | VariadicPossiblesType]
)


def shape_map_to_tensor(
    shape_map: dict[str, ShapeRepr],
) -> Mapping[str, IOHyperEdge]:
    # Simply converts ShapeRepr objects to Tensor types.
    tensor_dict = {}
    for key, value in shape_map.items():
        tensor: Tensor[int | float | bool] = Tensor(
            type=float | int | bool, shape=value.node
        )
        edge = IOHyperEdge(value=tensor, key_origin=key)
        # set temp_shape. Since temp_shape of a Tensor initialized as None in its
        # constructor.
        assert edge.shape is not None
        edge._temp_shape = next(iter(edge.shape.reprs))
        tensor_dict[key] = edge
    return tensor_dict


def uniadic_update_values(
    key: str, values: set[int], used_keys: dict[str, Uniadic | Variadic]
) -> None:
    if (uni := used_keys.get(key)) is None:
        uni = used_keys[key] = Uniadic()
    assert isinstance(uni, Uniadic)
    uni.update_possible_values(values)


def variadic_update_values(
    key: VariadicTemplateType,
    values: VariadicPossiblesType,
    used_keys: dict[str, Uniadic | Variadic],
) -> None:
    var_symbol = used_keys[key[0]]
    assert isinstance(var_symbol, Variadic)
    assert isinstance(values, list)
    all_assignments: list[PossibleValues] = []
    for pos_vals in values:
        assert isinstance(pos_vals, tuple)
        uni_list: list[Uniadic] = []
        for pos_val in pos_vals:
            if isinstance(pos_val, str):
                if (uni := used_keys.get(pos_val)) is None:
                    uni = used_keys[pos_val] = Uniadic()
            else:
                uni = Uniadic(pos_val)
            assert isinstance(uni, Uniadic)
            uni_list.append(uni)
        uni_tuple = tuple(uni_list)
        all_assignments.append(PossibleValues(uni_tuple))
    var_symbol.update_possible_values(*all_assignments)


def extract_uniadic_possibles(
    uni: Uniadic,
    assignments: AssignmentType,
    uni_cache: dict[UniadicRecord, str],
) -> None:
    # Takes an uniadic object and fills the assignments dictionary
    # based on possible values of the uniadic object.
    if (uni_str := uni_cache.get(uni.metadata)) is None:
        uni_str = uni_cache[uni.metadata] = f"u{len(uni_cache) + 1}"
    if uni.possible_values is not None and len(uni.possible_values) > 1:
        assignments[uni_str] = uni.possible_values  # type: ignore


def extract_variadic_possibles(
    var: Variadic,
    assignments: AssignmentType,
    uni_cache: dict[UniadicRecord, str],
    var_cache: dict[Variadic, str],
) -> None:
    assert var.possibles is not None
    all_possible_values: dict[int, PossibleValues] = var.possibles
    possibles_list: list[tuple] = []
    for possible_values in all_possible_values.values():
        single_possible_list: list[int] | list[str] | list[int | str] = []
        for uni in possible_values.uniadics:
            if isinstance(uni.value, int):
                single_possible_list.append(uni.value)  # type: ignore
            else:
                if (uni_str := uni_cache.get(uni.metadata)) is None:
                    uni_str = uni_cache[uni.metadata] = f"u{len(uni_cache) + 1}"
                single_possible_list.append(uni_str)  # type: ignore
                if uni.possible_values is not None and len(uni.possible_values) > 1:
                    assignments[uni_str] = uni.possible_values  # type: ignore
        possibles_list.append(tuple(single_possible_list))
    assignments[(var_cache[var], ...)] = possibles_list  # type: ignore


def assert_shape_results(
    data: dict[str, IOHyperEdge],
    ref_results: ShapeResultType,
    ref_assignments: AssignmentType,
    updated_symbols: Updates,
    expected_updates: set[str],
) -> None:
    # First check shape updates with the expected updates.
    assert {
        data[key] for key in expected_updates
    } == updated_symbols.shape_updates | updated_symbols.value_updates
    # Then check final shapes with the expected ref_results.
    uni_cache: dict[UniadicRecord, str] = {}
    var_cache: dict[Variadic, str] = {}
    shapes = {}
    assignments: AssignmentType = {}
    for key, value in data.items():
        if value.is_tensor:
            assert value.shape is not None
            shapes[key] = value.shape.get_shapes(uni_cache, var_cache, verbose=True)
            shape_repr = value._temp_shape
            assert shape_repr is not None
            all_repr_unis: set[Uniadic] = {*shape_repr.prefix, *shape_repr.suffix}
            for uni in all_repr_unis:
                extract_uniadic_possibles(uni, assignments, uni_cache)
            if (root := shape_repr.root) is not None and root.possibles is not None:
                extract_variadic_possibles(root, assignments, uni_cache, var_cache)
        else:
            shapes[key] = []

    check_shapes_semantically(shapes, ref_results, assignments, ref_assignments)


def assert_type_results(
    data: dict[str, IOHyperEdge],
    ref_results: dict[str, type],
    updated_symbols: Updates,
    expected_updates: set[str],
) -> None:
    # First check type updates with the expected updates.
    updated_constraints = set()
    for key in expected_updates:
        updated_constraints |= data[key].constraints[UpdateType.TYPE]
    assert updated_constraints == {
        constr
        for constr in updated_symbols.constraints
        if UpdateType.TYPE in constr.types
    }
    # Then check final types with the expected ref_results.
    for key, value in data.items():
        assert value.value_type == ref_results[key]


def assert_value_results(
    data: dict[str, IOHyperEdge], ref_results: dict[str, Any]
) -> None:
    for key, value in ref_results.items():
        if isinstance(
            value, int | float | bool | tuple | list | str | ToBeDetermined | slice
        ):
            assert data[key].value == value
        else:
            # If value is a tensor of any supported backend.
            assert data[key].is_tensor
            d_val = data[key].value
            assert GenericDataType.is_tensor_type(d_val)
            assert (d_val == value).all()


def make_assertions(
    constraint_fn: Callable,
    data: dict[str, IOHyperEdge],
    ref_results: dict[str, type] | ShapeResultType,
    ref_assignments: AssignmentType,
    updated_symbols: Updates,
    expected_updates: set[str],
    final_values: dict[str, Any],
) -> None:
    # Check final shapes with the expected ref_shapes. Also check updated symbols.
    if is_shape_checker(ref_results, constraint_fn):
        assert_shape_results(
            data, ref_results, ref_assignments, updated_symbols, expected_updates
        )
    else:
        assert is_type_checker(ref_results, constraint_fn)
        assert_type_results(data, ref_results, updated_symbols, expected_updates)
    # NOTE: There is no other possibilities. Only for type cheking!

    # Check final values with the expected final_values.
    assert_value_results(data, final_values)


def assert_constraint_results(
    # shapes: dict[str, list[str | int | None | tuple[str, EllipsisType]]],
    shapes: Mapping[str, ShapeTemplateType],
    assignments: AssignmentType,
    ref_results: dict,
    ref_assignments: AssignmentType,
    constraint_fn: Callable,
    expected_status: bool,
    expected_updates: set[str],
    scalar_data: Mapping[str, IOHyperEdge] | None = None,
    final_values: dict[str, Any] | None = None,
    initial_values: dict[str, Any] | None = None,
    initial_types: Mapping[str, type | UnionType] | None = None,
    variadic_fn: bool = False,
):
    for _ in range(50):
        args = (
            shapes,
            assignments,
            ref_results,
            ref_assignments,
            constraint_fn,
            expected_status,
            expected_updates,
            scalar_data,
            final_values,
            initial_values,
            initial_types,
            variadic_fn,
        )
        _assert_constraint_results(*deepcopy(args))


def _assert_constraint_results(
    shapes: Mapping[str, ShapeTemplateType],
    assignments: AssignmentType,
    ref_results: dict,
    ref_assignments: AssignmentType,
    constraint_fn: Callable,
    expected_status: bool,
    expected_updates: set[str],
    scalar_data: Mapping[str, IOHyperEdge] | None = None,
    final_values: dict[str, Any] | None = None,
    initial_values: dict[str, Any] | None = None,
    initial_types: Mapping[str, type | UnionType] | None = None,
    variadic_fn: bool = False,
):
    # Create shape maps and corresponding data.
    solver = ConstraintSolver()
    used_keys: dict[str, Uniadic | Variadic] = {}
    shape_map = {
        key: create_shape_repr(shp_list, solver, used_keys)
        for key, shp_list in shapes.items()
    }
    for key, assignment in assignments.items():
        if isinstance(key, tuple):
            assert isinstance(assignment, list)
            variadic_update_values(key, assignment, used_keys)

        else:
            assert isinstance(assignment, set)
            uniadic_update_values(key, assignment, used_keys)

    data = shape_map_to_tensor(shape_map)  # type: ignore
    assert isinstance(data, dict)

    if initial_values is None:
        initial_values = dict()

    # In case there exists Scalar data, add it.
    if scalar_data is not None:
        data |= scalar_data

    # If initial types are given, set them.
    if initial_types is not None:
        for key, type in initial_types.items():
            data[key]._value.set_type(type)

    # If any initial values are given, set them.
    for key, value in initial_values.items():
        data[key].value = value

    if final_values is None:
        final_values = dict()

    # First call for the corresponging constraint solver function.
    status, updated_symbols = (
        constraint_fn(**data) if not variadic_fn else constraint_fn(*data.values())
    )

    # Assert status the expected status.
    assert expected_status == status

    # Make all assertions.
    make_assertions(
        constraint_fn,
        data,
        ref_results,
        ref_assignments,
        updated_symbols,
        expected_updates,
        final_values,
    )

    # In order to check idempotency, call corresponding function again.
    post_status, reupdated_symbols = (
        constraint_fn(**data) if not variadic_fn else constraint_fn(*data.values())
    )

    # Check no change for the status.
    assert post_status == status
    # Make all assertions again.
    make_assertions(
        constraint_fn,
        data,
        ref_results,
        ref_assignments,
        reupdated_symbols,
        set(),
        final_values,
    )


######################### Test Cases #########################
def test_bcast_error_1():
    """Should raise ValueError since left and right shapes
    are not consistent to be broadcasted.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var3", ...), "x", "z"],
        "left": [4, 3],
        "right": [3, 3],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, bcast, True, set())
    # assert str(err_info.value) == 'Shape mismatch in Broadcast!'
    assert (
        str(err_info.value)
        == "Inputs shape mismatch at dimension 0. Shapes are inconsistent."
    )


def test_bcast_error_2():
    """Should raise ValueError since output shape
    is not consistent with the inputs.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "y", "z"],
        "left": [4, 3],
        "right": [4, 3],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, bcast, True, set())
    assert (
        str(err_info.value)
        == "Determined shape representations should have same length."
    )
    # assert str(err_info.value) == "Shape mismatch for output!"


def test_bcast_error_3():
    """Should raise ValueError since output shape
    is not consistent with the inputs.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "y", "z"],
        "left": [4, 3, 3, 3],
        "right": [4, 3],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, bcast, True, set())
    assert (
        str(err_info.value)
        == "Determined shape representations should have same length."
    )
    # assert str(err_info.value) == "Shape mismatch for output!"


def test_bcast_error_4():
    """Should raise ValueError since left input has a shape
    value different than output in second dimension (3 != 4).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5],
        "left": ["a", 3, "b"],
        "right": [1, "c", 5],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, bcast, True, set())
    assert str(err_info.value) == "Possible values mismatch!"


def test_bcast_error_5():
    """Should raise ValueError since symbol "a" is already found to
    be 5 and then updated to be 3 in the inference sequence from
    left to output.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, ("Var", ...), 5],
        "left": ["a", 1, "a"],
        "right": [4, 1],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, bcast, True, set())
    assert str(err_info.value) == "Possible values mismatch!"


def test_bcast_error_8():
    """
    This test should raise an error as left will try to set one of the
    "x"'s to 3 and other one of the "x"'s to 4.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "x", "x", "y"],
        "left": [4, 4, 3, 3],
        "right": [("Var1", ...)],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, bcast, True, set())
    assert str(err_info.value) == "Possible values mismatch!"


def test_bcast_forward_1():
    """Should work with no problem."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var3", ...), "x", "z"],
        "left": [4, 3],
        "right": [3],
    }
    final_shapes = {"output": [4, 3], "left": [4, 3], "right": [3]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"output"})


def test_bcast_forward_2():
    """Should work with no problem."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var3", ...), "x", "z"],
        "left": [3],
        "right": [4, 3],
    }
    final_shapes = {"output": [4, 3], "left": [3], "right": [4, 3]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"output"})


def test_bcast_forward_3():
    """Should work with no problem."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var3", ...), "x", "z"],
        "left": [1],
        "right": [4, 3],
    }
    final_shapes = {"output": [4, 3], "left": [1], "right": [4, 3]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"output"})


def test_bcast_forward_4():
    """Should work with no problem."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", ("Var3", ...), "z"],
        "left": [1],
        "right": [4, 3],
    }
    final_shapes = {"output": [4, 3], "left": [1], "right": [4, 3]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"output"})


def test_bcast_forward_5():
    """Should work with no problem."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", ("Var3", ...), "z"],
        "left": [1, 1, 4, 1],
        "right": [1, 3],
    }
    final_shapes = {"output": [1, 1, 4, 3], "left": [1, 1, 4, 1], "right": [1, 3]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"output"})


def test_bcast_forward_6():
    """Should work with no problem."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "y"],
        "left": [4, 1],
        "right": [1],
    }
    final_shapes = {"output": [4, 1], "left": [4, 1], "right": [1]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"output"})


@pytest.mark.skip("Expected possible values mismatch")
def test_bcast_forward_7():
    """Should work with no problem."""

    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "x", "y", "y"],
        "left": [4, 4, 3, 3],
        "right": [("Var1", ...)],
    }
    assignments: AssignmentType = {}

    final_shapes = {
        "output": [4, 4, 3, 3],
        "left": [4, 4, 3, 3],
        "right": ["(RefVar1, ...)"],
    }
    # dict[tuple[str, EllipsisType], list[tuple[str, ...]]]
    ref_assignments: AssignmentType = {
        ("RefVar1", ...): [
            (),
            ("u1",),
            ("u2", "u1"),
            ("u3", "u2", "u1"),
            ("u4", "u3", "u2", "u1"),
        ],
        # "u1": {1, 3},
        # "u2": {1, 3},
        # "u3": {1, 4},
        # "u4": {1, 4},
    }
    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        ref_assignments,
        bcast,
        False,
        {"output", "right"},
    )


def test_bcast_forward_8():
    """
    This test should raise an error as left will try to set one of the
    "x"'s to 3 and other one of the "x"'s to 4.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "x", "x", "y"],
        "left": [4, 4, 3, 3],
        "right": [1],
    }

    with pytest.raises(Exception):  # noqa :B017
        assert_constraint_results(shapes, {}, {}, {}, bcast, True, {"output"})
    # TODO: Assert error message!!!


def test_bcast_forward_9():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "x", "x", "x"],
        "left": [1],
        "right": [2, ("Var1", ...)],
    }
    final_shapes = {"output": [2, 2, 2, 2], "left": [1], "right": [2, 2, 2, 2]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"output", "right"}
    )


def test_bcast_forward_10():
    """
    This test should work without a problem, alogirhm should infer the rank of right
    from rank of output. Inferring the rank of right, algorithm should set all x's to
    2 as first element of right equal to 2. And algorithm should also infer all sizes of
    right after knowing shape of the output as [2, 2, 2, 2].
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "x", "x", "x"],
        "left": ["x"],
        "right": [2, ("Var1", ...)],
    }

    assignments: AssignmentType = {}

    final_shapes = {"output": [2, 2, 2, 2], "left": [2], "right": [2, 2, 2, "u1"]}

    final_assignments = {"u1": {1, 2}}
    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        final_assignments,
        bcast,
        True,
        {"output", "left", "right"},
    )


@pytest.mark.skip("Expected possible values mismatch")
def test_bcast_forward_11():
    """
    This test tests inference capabilities of bcast algorthm when output is uniadic
    and left and right is variadic. In this test, it is known that first element of
    left is 2. And since also we know that output has two element with unknown values.
    It should infer x = 2. with same logic.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "y"],
        "left": [2, ("Var1", ...)],
        "right": [("Var2", ...), 3],
    }
    assignments: AssignmentType = {}
    final_shapes = {"output": [2, 3], "left": [2, "u1"], "right": ["(V1, ...)", 3]}
    ref_assignments: dict[tuple[str, EllipsisType] | str, list | set] = {
        ("V1", ...): [(), ("u2",)],
        "u1": {1, 3},
        "u2": {1, 2},
    }

    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        ref_assignments,
        bcast,
        False,
        {"output", "left", "right"},
    )


def test_bcast_forward_11_1():
    """
    this test is a variation of test_bcast_forward_11. Output has exactly three
    dimensions. we know thar right most dimension of right is 5 and left is symbolic
    "a". Bcast algorithm should infer z = 5. Also it is known that second right most
    digit of left is 2. Hence, it can be also determined that y = 2. Finally, since
    left most digit of left is 3, x can be nothing but 3. It is also inferable that
    # of uniadics in Var1 field is 1 since output has rank 3. Finding variadic has 1
    # unidadic, only value that uniadic can take is 1. Overall. Bcast algorithm an
    # infer all values except the value of "a" (it can be either 1 or 5) from these
    # given informations.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "y", "z"],
        "left": [("Var1", ...), 2, "a"],
        "right": [3, ("Var1", ...), 5],
    }

    assignments: AssignmentType = {}

    final_shapes = {"output": [3, 2, 5], "left": [1, 2, "u1"], "right": [3, 1, 5]}

    ref_assignments = {"u1": {1, 5}}
    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        ref_assignments,
        bcast,
        True,
        {"output", "left", "right"},
    )


def test_bcast_forward_11_2():
    """
    This test is also a variation of test_bcast_forward_11. It is known that output
    has exactly 5 dimensions. inferring from right most values of right, it can be
    determined c = 3, d = 4, e = 5. However, we cannot say anything about a and b as
    right can be ["u1", "u2", 3, 4, 5] and left can be [3, 4, 1]. It is a valid bcast
    but it does not say anything about a and b.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b", "c", "d", "e"],
        "left": [3, 4, ("Var2", ...)],
        "right": [("Var1", ...), 3, 4, 5],
    }
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2", 3, 4, 5],
        # "left": [3, 4, "u3", "(V1, ...)"],
        "left": [3, 4, "(V1, ...)", "u3"],
        "right": ["(V2, ...)", 3, 4, 5],
    }
    ref_assignments: AssignmentType = {  # type: ignore
        # TODO: Check this part again.
        "u3": {1, 5},
        ("V1", ...): [(), ("u7", "u8")],
        ("V2", ...): [(), ("u9",), ("u10", "u9")],
    }
    expected_updates = {"output", "left", "right"}
    assert_constraint_results(
        shapes, {}, final_shapes, ref_assignments, bcast, False, expected_updates
    )


def test_bcast_forward_11_3():
    """
    In this test, every shape are inferable except right most shape of left. Since we
    know output has exactly 3 output and also we know right has at least three ranks.
    Var1 should contain exactly 0 uniadics. Since also we know right has three
    determined shapes in its left most shapes, We can infer that shape of right is
    exactly [3, 4, 5]. By inferring shape of right, it can be also inferred a = 3,
    b = 4 and c = 5. After inferring the output, we can exactly infer that Var2 has 1
    uniadic as it is the only valid length for left. However, after inferring length
    of left's shape, we cannot say anything exact about right most shape of left
    (it can be either 1 or 5).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b", "c"],
        "left": [3, 4, ("Var2", ...)],
        "right": [3, 4, 5, ("Var1", ...)],
    }
    assignments: AssignmentType = {}
    final_shapes = {"output": [3, 4, 5], "left": [3, 4, "u1"], "right": [3, 4, 5]}
    ref_assignments = {"u1": {1, 5}}
    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        ref_assignments,
        bcast,
        True,
        {"output", "left", "right"},
    )


# @pytest.mark.skip("Expected possible values mismatch")
def test_bcast_forward_11_4():
    """
    From this test, it can be inferred that a = 4, b = 4, and c = 5. Because we know
    that output has exactly 4 rank. However, we cannot say anything about Var1 ,Var2
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b", "c", "d"],
        "left": [4, 4, ("Var1", ...)],
        "right": [4, 4, 4, ("Var2", ...)],
    }
    assignments: AssignmentType = {}
    final_shapes = {
        "output": [4, 4, 4, "u1"],
        "left": [4, 4, "(V1, ...)"],
        "right": [4, 4, 4, "(V2, ...)"],
    }
    ref_assignments = {
        ("V1", ...): [(), ("u2",), ("u3", "u2")],
        ("V2", ...): [(), ("u4",)],
        # "u3": {1, 4},
        # "u4": {1, 4},
    }
    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        ref_assignments,
        bcast,
        False,
        {"output", "left", "right"},
    )


def test_bcast_forward_12():
    """
    Tests the case where;
        output = ["a", "b", ("Var1", ...), "c"]
        left = ["c", "d", ("Var2", ...), "e"]
        right = ["f", ("Var3", ...), "g"]
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b", ("Var1", ...), "c"],
        "left": [2, 1, ("Var2", ...), "e"],
        "right": [2, ("Var3", ...), "g"],
    }
    final_shapes = {
        "output": [2, "b", "(Var1, ...)", "c"],
        "left": [2, 1, "(Var2, ...)", "e"],
        "right": [2, "(Var3, ...)", "g"],
    }
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, False, {"output"})


def test_bcast_forward_13():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "left": ["a", "b", "c", ("Var2", ...), "d", "e"],
        "right": ["f"],
    }
    final_shapes = {
        "output": ["a", "b", "c", "(Var2, ...)", "d", "g"],
        "left": ["a", "b", "c", "(Var2, ...)", "d", "e"],
        "right": ["f"],
    }
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, False, {"output"})


def test_bcast_forward_14():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "left": [],
        "right": [("Var2", ...)],
    }
    final_shapes = {"output": ["(Var2, ...)"], "left": [], "right": ["(Var2, ...)"]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, False, {"right"})


def test_bcast_forward_15():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "left": [("Var2", ...)],
        "right": [("Var1", ...)],
    }
    final_shapes = {
        "output": ["(Var1, ...)"],
        "left": ["(Var2, ...)"],
        "right": ["(Var1, ...)"],
    }
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, False, set())


def test_bcast_forward_16():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var2", ...)],
        "left": [("Var1", ...), "c", "d"],
        "right": [1, 2, ("Var1", ...), "e", "f"],
    }
    final_shapes = {
        "output": [1, 2, "(Var1, ...)", "a", "b"],
        "left": ["(Var1, ...)", "c", "d"],
        "right": [1, 2, "(Var1, ...)", "e", "f"],
    }
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, False, {"output"})


def test_bcast_forward_17():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["c", "d", ("Var2", ...)],
        "left": [("Var1", ...), "c", "d"],
        "right": [1, 2, ("Var1", ...), "e", "f"],
    }
    assignments: AssignmentType = {}
    final_shapes = {
        "output": [1, 2, "(Var1, ...)", "a", 2],
        "left": ["(Var1, ...)", 1, 2],
        "right": [1, 2, "(Var1, ...)", "a", "f"],
    }
    ref_assignments = {
        "f": {1, 2},
    }
    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        ref_assignments,
        bcast,
        False,
        {"output", "left", "right"},
    )


def test_bcast_forward_18():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["c", ("Var2", ...), "d"],
        "left": ["a", "b"],
        "right": ["a", "b"],
    }
    final_shapes = {"output": ["a", "b"], "left": ["a", "b"], "right": ["a", "b"]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"output"})


def test_bcast_backward_12():
    """
    This test is the only case of all shapes are inferable when output's shape is
    exactly determined and both left and right carries no information. Output has
    rank 0. Therefore, both left and right should have rank exactly 0
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "left": [("Var1", ...)],
        "right": [("Var2", ...)],
    }
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "left": [],
        "right": [],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"left", "right"}
    )


def test_bcast_backward_13():
    """
    This test should work without a problem. Since we know shape of left is
    exactly equal to shape of rigth, their shapes should be equal to shape of output
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5, 6],
        "left": [("Var1", ...)],
        "right": [("Var1", ...)],
    }
    final_shapes = {"output": [3, 4, 5, 6], "left": [3, 4, 5, 6], "right": [3, 4, 5, 6]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"left", "right"}
    )


def test_bcast_backward_14():
    """
    Int this test, shape of output is symbolically determined. Since left most shapes
    of left is determined and equal to 3, 4, 5 respectively. we can infer x = 3, y = 4,
    z = 5. Variadics can be also inferred as zero uniadics. Therefore, status should be
    True and algorithm can infer all shapes.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "y", "z"],
        "left": [("Var1", ...), 3, 4, 5],
        "right": [3, 4, 5, ("Var2", ...)],
    }
    final_shapes = {"output": [3, 4, 5], "left": [3, 4, 5], "right": [3, 4, 5]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"output", "left", "right"}
    )


def test_bcast_backward_15():
    """
    This test is similar to test_bcast_backward_14. It should also work
      without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5],
        "left": [("Var1", ...), "x", "y", "z"],
        "right": ["x", "y", "z", ("Var1", ...)],
    }
    final_shapes = {"output": [3, 4, 5], "left": [3, 4, 5], "right": [3, 4, 5]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"left", "right"}
    )


def test_bcast_backward_16():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "left": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "right": [134, 47, 1, 1, 1],
    }
    final_shapes = {
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "left": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "right": [134, 47, 1, 1, 1],
    }
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"output"})


@pytest.mark.skip("Expected possible values mismatch")
def test_bcast_backward_17():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [4, 5, 7, 3, 4],
        "left": [4, 5, 7, 1, 1],
        "right": [("Var1", ...)],
    }
    assignments: AssignmentType = {}
    final_shapes = {
        "output": [4, 5, 7, 3, 4],
        "left": [4, 5, 7, 1, 1],
        "right": ["(Var1, ...)", 3, 4],
    }
    ref_assignments: AssignmentType = {  # type: ignore
        ("Var1", ...): [(), ("u1",), ("u2", "u1"), ("u3", "u2", "u1")],
        "u1": {1, 7},
        "u2": {1, 5},
        "u3": {1, 4},
    }
    assert_constraint_results(
        shapes, assignments, final_shapes, ref_assignments, bcast, False, {"right"}
    )


def test_bcast_backward_18():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [4, 5, 7, 3, 4],
        "left": [("Var1", ...)],
        "right": [("Var1", ...)],
    }
    final_shapes = {
        "output": [4, 5, 7, 3, 4],
        "left": [4, 5, 7, 3, 4],
        "right": [4, 5, 7, 3, 4],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"left", "right"}
    )


def test_bcast_backward_19():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [2, 3, 3],
        "left": ["x", "y", "z"],
        "right": ["y", "z", "z"],
    }
    final_shapes = {"output": [2, 3, 3], "left": [2, 1, 3], "right": [1, 3, 3]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"left", "right"}
    )


def test_bcast_backward_20():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", "y"],
        "left": [1],
        "right": ["a", "b"],
    }
    final_shapes = {"output": ["a", "b"], "left": [1], "right": ["a", "b"]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"output"})


def test_bcast_backward_21():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "e", "c", "d"],
        "left": [1],
        "right": ["d", "c", "a", 4],
    }
    final_shapes = {"output": [4, 4, 4, 4], "left": [1], "right": [4, 4, 4, 4]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"output", "right"}
    )


def test_bcast_backward_22():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [1, "z", "c"],
        "left": ["x", 5, "z"],
        "right": ["y", "x", "y"],
    }
    final_shapes = {"output": [1, 5, 5], "left": [1, 5, 5], "right": [1, 1, 1]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"output", "left", "right"}
    )


def test_bcast_backward_23():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "f", "e", "d", "c"],
        "left": [5],
        "right": ["f", "e", "d", "c", "a"],
    }
    final_shapes = {"output": [5, 5, 5, 5, 5], "left": [5], "right": [5, 5, 5, 5, 5]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"output", "right"}
    )


def test_bcast_backward_24():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2", 1, "u3", "u4", "u5"],
        "left": [("Var1", ...), 3, 4, 5],
        "right": [5, 7, ("Var1", ...), 5],
    }
    final_shapes = {
        "output": [5, 7, 1, 3, 4, 5],
        "left": [1, 1, 1, 3, 4, 5],
        "right": [5, 7, 1, 1, 1, 5],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"output", "left", "right"}
    )


@pytest.mark.skip("Expected possible values mismatch")
def test_bcast_backward_25():
    """
    This test should work without a problem
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2", 1, "u3", "u4", "u5"],
        "left": [("Var1", ...), 3, 4, 5],
        "right": [5, 7, ("Var2", ...), 5],
    }
    assignments: AssignmentType = {}
    final_shapes = {
        "output": [5, 7, 1, 3, 4, 5],
        "left": ["(Var1, ...)", 1, 3, 4, 5],
        "right": [5, 7, 1, "c", "d", 5],
    }
    ref_assignments: dict[
        tuple[str, EllipsisType] | str, list[tuple[str, ...]] | set[int]
    ] = {
        ("Var1", ...): [(), ("u1",), ("u2", "u1")],
        "c": {1, 3},
        "d": {1, 4},
        "u1": {1, 7},
        "u2": {1, 5},
    }
    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        ref_assignments,
        bcast,
        False,
        {"output", "right", "left"},
    )


def test_bcast_backward_1():
    """Should work with no problem."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4],
        "left": ["a", "b"],
        "right": [1, 1],
    }
    final_shapes = {"output": [3, 4], "left": [3, 4], "right": [1, 1]}
    assert_constraint_results(shapes, {}, final_shapes, {}, bcast, True, {"left"})


def test_bcast_backward_2():
    """Should work with no problem but can not infer values
    of all symbols (b can be either 1 or 5). So, status should be
    False while symbols "a" and "c" are inferred as 3 and 4.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5],
        "left": ["a", 1, "b"],
        "right": [1, "c", 5],
    }
    assignments: AssignmentType = {}
    final_shapes = {"output": [3, 4, 5], "left": [3, 1, "a"], "right": [1, 4, 5]}
    ref_assignments = {
        "a": {1, 5},
    }
    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        ref_assignments,
        bcast,
        True,
        {"left", "right"},
    )


def test_bcast_backward_error():
    """Should work with no problem but can not infer values
    of all symbols. So, status should be False while symbols
    "a" and "c" are inferred as 3 and 4.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5],
        "left": ["a", 4, "b"],
        "right": [1, 3, 5],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, bcast, True, set())
    assert (
        str(err_info.value)
        == "Inputs shape mismatch at dimension 1. Shapes are inconsistent."
    )


def test_bcast_forward_backward_1():
    """Should work with no problem and infer values of
    all symbols.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, "c", 5],
        "left": ["a", 1, "b"],
        "right": [1, 4, 1],
    }
    final_shapes = {"output": [3, 4, 5], "left": [3, 1, 5], "right": [1, 4, 1]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"output", "left"}
    )


def test_bcast_forward_backward_2():
    """Should work with no problem and infer values of
    all symbols.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, ("Var", ...), 5],
        "left": ["a", 1, "b"],
        "right": [4, 1],
    }
    final_shapes = {"output": [3, 4, 5], "left": [3, 1, 5], "right": [4, 1]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"output", "left"}
    )


def test_bcast_forward_backward_variadic_1():
    """Tests inference capability when one of inputs have variadic
    field. It is obvious that variadic field of right input would
    evolve to None, "b" = 2 and "a" = 5.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["b", 5],
        "left": [1, "a"],
        "right": [("Var_right", ...), 2, 1],
    }
    final_shapes = {"output": [2, 5], "left": [1, 5], "right": [2, 1]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"output", "left", "right"}
    )


def test_bcast_forward_backward_variadic_2():
    """Tests inference capability when one of inputs have variadic
    field. It is obvious that variadic field of right input would
    evolve to None, "b" = 2 and "a" = 5.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 5],
        "left": ["a"],
        "right": [("Var_right", ...), 1],
    }
    final_shapes = {"output": [3, 5], "left": [5], "right": [3, 1]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, bcast, True, {"left", "right"}
    )


############# BCAST MATRIX MULT #############


def test_bcast_matmul_forward_1():
    """Should work with no problem."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...), "b", "d"],
        "left": ["x", 2, "z", "a", "b", "c"],
        "right": [("right", ...), 3, "c", "d"],
    }
    assignments: AssignmentType = {}
    final_shapes = {
        "output": ["(out, ...)", "k", 2, "l", 3, "b", "d"],
        "left": ["x", 2, "z", "a", "b", "c"],
        "right": ["(right, ...)", 3, "c", "d"],
    }
    ref_assignments = {
        "a": {1, 3},
    }
    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        ref_assignments,
        bcast_matrix_mult,
        False,
        {"output", "left"},
    )


############# REDUCE #############


def test_reduce_forward_1():
    """Should work with no problem with axis = 1."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b"],
        "input": [3, 4, 5],
    }
    final_shapes = {"output": [3, 5], "input": [3, 4, 5], "axis": [], "keepdim": []}
    scalar_info = {
        "axis": IOHyperEdge(type=int, value=1),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"output"}, scalar_info
    )


def test_reduce_forward_2():
    """Should work with no problem with axis = (1, 3)."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b"],
        "input": [3, 4, 5, 6],
    }
    final_shapes = {"output": [3, 5], "input": [3, 4, 5, 6], "axis": [], "keepdim": []}
    scalar_info = {
        "axis": IOHyperEdge(value=(1, 3)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"output"}, scalar_info
    )


def test_reduce_forward_3():
    """Should work with no problem with axis = (-1, 1)."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b"],
        "input": [3, 4, 5, 6],
    }
    final_shapes = {"output": [3, 5], "input": [3, 4, 5, 6], "axis": [], "keepdim": []}
    scalar_info = {
        "axis": IOHyperEdge(value=(-1, 1)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"output"}, scalar_info
    )


def test_reduce_forward_4():
    """Should work with no problem with axis = (-1, 1)."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["x", ("Var1", ...), "y"],
        "input": [3, 4, 5, 6],
    }
    final_shapes = {"output": [3, 5], "input": [3, 4, 5, 6], "axis": [], "keepdim": []}
    scalar_info = {
        "axis": IOHyperEdge(value=(-1, 1)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"output"}, scalar_info
    )


def test_reduce_forward_5():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", ("V1", ...), 10],
        "input": ["u3", ("V2", ...), "u5", 10],
    }
    final_shapes = {
        "output": ["u1", "(V1, ...)", 10],
        "input": ["u3", "(V2, ...)", "u5", 10],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {"axis": IOHyperEdge(value=0), "keepdim": IOHyperEdge(value=False)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, False, set(), scalar_info
    )


# @pytest.mark.skip("Known Bug, update reduce constraints.")
def test_reduce_forward_6():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [("V2", ...)],
    }
    final_shapes = {
        "output": ["x", "(V2, ...)", "u5"],
        "input": ["a", "b", "c", "(V1, ...)", "u5"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(2, -4)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reduce_constraints,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_reduce_forward_7():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [("V2", ...)],
    }
    final_shapes = {
        "output": ["a", "(V2, ...)"],
        "input": ["a", "b", "c", "(V1, ...)"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(2, -2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reduce_constraints,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_reduce_forward_8():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [("V2", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)"],
        "input": ["a", "b", "(V2, ...)"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, -2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, False, {"input"}, scalar_info
    )


def test_reduce_forward_9():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [("V2", ...), "a"],
    }
    final_shapes = {
        "output": ["(V1, ...)"],
        "input": ["x", "(V2, ...)", "a"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, -2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, False, {"input"}, scalar_info
    )


def test_reduce_forward_10():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [("V2", ...), "a", "b"],
    }
    final_shapes = {
        "output": ["(V1, ...)"],
        "input": ["(V2, ...)", "a", "b"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, -2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, False, set(), scalar_info
    )


def test_reduce_forward_11():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [("V2", ...)],
    }
    final_shapes = {
        "output": ["a", "(V1, ...)", "e", "f"],
        "input": ["a", "b", "(V1, ...)", "c", "d", "e", "f"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, -3, -4)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reduce_constraints,
        True,
        {"input", "output"},
        scalar_info,
    )


def test_reduce_forward_12():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": ["a", "b", "c", ("V2", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)", "d", "e"],
        "input": ["a", "b", "c", "(V2, ...)", "d", "e"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, -3, -5)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reduce_constraints,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_reduce_forward_13():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": ["a", "b", "c", ("V2", ...)],
    }
    final_shapes = {
        "output": ["a", "(V2, ...)"],
        "input": ["a", "b", "c", "(V2, ...)"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, 2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"output"}, scalar_info
    )


def test_reduce_forward_14():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": ["a", "b", "c", ("V2", ...)],
    }
    final_shapes = {
        "output": ["a", "(V1, ...)"],
        "input": ["a", "b", "c", "(V2, ...)"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(-1, -2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, False, {"output"}, scalar_info
    )


def test_reduce_forward_15():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": ["a", ("V2", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)"],
        "input": ["a", "b", "(V2, ...)"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(-1, -2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, False, {"input"}, scalar_info
    )


def test_reduce_forward_16():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V2", ...)],
        "input": ["a", ("V1", ...)],
    }
    final_shapes = {
        "output": [1, "(V1, ...)"],
        "input": ["a", "(V1, ...)"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {"axis": IOHyperEdge(value=0), "keepdim": IOHyperEdge(value=True)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"output"}, scalar_info
    )


def test_reduce_forward_17():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V2", ...)],
        "input": [("V1", ...), "a"],
    }
    final_shapes = {
        "output": ["(V1, ...)", 1],
        "input": ["(V1, ...)", "a"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {"axis": IOHyperEdge(value=-1), "keepdim": IOHyperEdge(value=True)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"output"}, scalar_info
    )


def test_reduce_forward_18():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V2", ...)],
        "input": [("V1", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)", 1],
        "input": ["(V1, ...)", "a"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {"axis": IOHyperEdge(value=-1), "keepdim": IOHyperEdge(value=True)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reduce_constraints,
        True,
        {"output", "input"},
        scalar_info,
    )


def test_reduce_forward_error_3():
    """Should work with no problem with axis = (-1, 1)."""
    shapes: dict[str, list[int | str | tuple]] = {"output": [], "input": [3, 4]}
    scalar_info = {
        "axis": IOHyperEdge(value=(-1, 0, 1)),
        "keepdim": IOHyperEdge(value=False),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reduce_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == "Dim 1 appears multiple times in the reduce axes"


def test_reduce_forward_error_1():
    """Should raise ValueError since expected output rank is 2 but got
    3 for axis = 1.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b", "c"],
        "input": [3, 4, 5],
    }
    scalar_info = {"axis": IOHyperEdge(value=1), "keepdim": IOHyperEdge(value=False)}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reduce_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Shape mismatch, output rank = 3. Output rank must be exactly 2 where input "
        "rank = 3 and axis = (1,). Axis numbers printed as their counterparts."
    )


def test_reduce_forward_error_2():
    """Should raise ValueError since expected output rank is 2 but got
    3 for axis = (1, 2).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b"],
        "input": [3, 4, 5],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, 2)),
        "keepdim": IOHyperEdge(value=False),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reduce_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Shape mismatch, output rank = 2. Output rank must be exactly 1 where input "
        "rank = 3 and axis = (1, 2). Axis numbers printed as their counterparts."
    )


def test_reduce_backward_error_1():
    """Should raise ValueError since expected output rank is 2 but got
    3 for axis = (1, 2).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5],
        "input": ["a", "b", "c"],
    }
    scalar_info = {"axis": IOHyperEdge(value=(1)), "keepdim": IOHyperEdge(value=False)}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reduce_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Shape mismatch, output rank = 3. Output rank must be exactly 2 where input "
        "rank = 3 and axis = (1,). Axis numbers printed as their counterparts."
    )


def test_reduce_backward_1():
    """Should work with no problem in backwards when axis = (1, )."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5],
        "input": ["a", "b", "c", "d"],
    }
    final_shapes = {
        "output": [3, 4, 5],
        "input": [3, "b", 4, 5],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {"axis": IOHyperEdge(value=1), "keepdim": IOHyperEdge(value=False)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"input"}, scalar_info
    )


def test_reduce_backward_2():
    """Should work with no problem in backwards when axis = (-2, 2)."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5],
        "input": ["a", "b", "c", "d", "e"],
    }
    final_shapes = {
        "output": [3, 4, 5],
        "input": [3, 4, "c", "d", 5],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(-2, 2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"input"}, scalar_info
    )


def test_reduce_forward_backward_1():
    """Should work bidirectional with no problem when axis = (-2, 2).
    Can infer "b" from input, and "a" and "b" from output.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, "b", 5],
        "input": ["a", 2, "c", "d", "e"],
    }
    final_shapes = {
        "output": [3, 2, 5],
        "input": [3, 2, "c", "d", 5],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(-2, 2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reduce_constraints,
        True,
        {"output", "input"},
        scalar_info,
    )


def test_reduce_forward_backward_error_1():
    """Should raise ValueError since output shape must have exactly 4 dims
    where input has 6 dims and axis = (-2, 2).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, "b", 5],
        "input": ["a", 2, "c", "d", "e", "f"],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(-2, 2)),
        "keepdim": IOHyperEdge(value=False),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reduce_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Shape mismatch, output rank = 3. Output rank must be exactly 4 where "
        "input rank = 6 and axis = (4, 2). Axis numbers printed as their counterparts."
    )


def test_reduce_axis_valued_keep_dim_true():
    """Test multiple positive/negative axis and keepdim"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": [1, 2, 3, 4, 5, 6, 7],
    }
    final_shapes = {
        "output": [1, 1, 1, 4, 1, 1, 7],
        "input": [1, 2, 3, 4, 5, 6, 7],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, 2, -2, -3)),
        "keepdim": IOHyperEdge(value=True),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"output"}, scalar_info
    )


def test_reduce_axis_none_keep_dim_true():
    """Test multiple none axis and keepdim"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": [1, 2, 3, 4, 5, 6, 7],
    }
    final_shapes = {
        "output": [1, 1, 1, 1, 1, 1, 1],
        "input": [1, 2, 3, 4, 5, 6, 7],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {"axis": IOHyperEdge(value=None), "keepdim": IOHyperEdge(value=True)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"output"}, scalar_info
    )


def test_reduce_axis_none_keep_dim_tbd():
    """Test multiple none axis and keepdim"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": [1, 2, 3, 4, 5, 6, 7],
    }
    final_shapes = {
        "output": ["(Var1, ...)"],
        "input": [1, 2, 3, 4, 5, 6, 7],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=None),
        "keepdim": IOHyperEdge(type=int, value=TBD),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, False, set(), scalar_info
    )


def test_reduce_axis_valued_keep_dim_tbd():
    """Test multiple none axis and keepdim"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": [1, 2, 3, 4, 5, 6, 7],
    }
    final_shapes = {
        "output": ["(Var1, ...)"],
        "input": [1, 2, 3, 4, 5, 6, 7],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, 2, -2, -3)),
        "keepdim": IOHyperEdge(type=int),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, False, set(), scalar_info
    )


def test_reduce_axis_valued_keep_dim_false_input_variadic():
    """Test multiple none axis and keepdim"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": [("Var2", ...)],
    }
    final_shapes = {
        "output": ["(Var1, ...)", "f"],
        "input": ["a", "b", "c", "(Var2, ...)", "d", "f"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, 2, -2, -3)),
        "keepdim": IOHyperEdge(type=bool, value=False),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reduce_constraints,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_reduce_axis_valued_keep_dim_tbd_input_variadic():
    """Test multiple none axis and keepdim"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": [("Var2", ...)],
    }
    final_shapes = {
        "output": ["(Var1, ...)"],
        "input": ["a", "b", "c", "(Var2, ...)", "d", "f"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, 2, -2, -3)),
        "keepdim": IOHyperEdge(type=bool),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, False, {"input"}, scalar_info
    )


# @pytest.mark.skip("Fix reduce constraints")
def test_reduce_with_given_axis():
    """Test multiple none axis and keepdim"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": [("Var2", ...)],
    }
    final_shapes = {
        "output": ["(Var1, ...)"],
        "input": ["(Var1, ...)", "u1", "u2"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(-1, -2)),
        "keepdim": IOHyperEdge(type=bool, value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"input"}, scalar_info
    )


def test_reduce_backward_3():
    """Test multiple positive/negative axis"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2", "u3", "u4"],
        "input": [("Var", ...)],
    }
    final_shapes = {
        "" "output": ["u1", "u2", "u3", "u4"],
        "input": ["u1", "a", "u2", "u3", "b", "u4"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, -2)),
        "keepdim": IOHyperEdge(value=False),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"input"}, scalar_info
    )


def test_reduce_backward_4():
    """Test multiple positive/negative axis"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2", "u3", "u4"],
        "input": [("Var", ...)],
    }
    final_shapes = {
        "output": ["u1", 1, 1, "u4"],
        "input": ["u1", "a", "b", "u4"],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, -2)),
        "keepdim": IOHyperEdge(value=True),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reduce_constraints,
        True,
        {"output", "input"},
        scalar_info,
    )


def test_reduce_backward_5():
    """Test multiple positive/negative axis"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [4, 1, 1, 7],
        "input": [("Var", ...)],
    }
    final_shapes = {
        "output": [4, 1, 1, 7],
        "input": [4, "a", "b", 7],
        "axis": [],
        "keepdim": [],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, -2)),
        "keepdim": IOHyperEdge(value=True),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reduce_constraints, True, {"input"}, scalar_info
    )


def test_reduce_backward_5_error():
    """Should raise ValueError since keepdim forces
    output shape to have "1" in dimension 1 but it is "2".
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [4, 2, 1, 7],
        "input": [("Var", ...)],
    }
    scalar_info = {
        "axis": IOHyperEdge(value=(1, -2)),
        "keepdim": IOHyperEdge(value=True),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reduce_constraints, True, {"input"}, scalar_info
        )
    assert str(err_info.value) == "Possible values mismatch!"


############# PAD #############


def test_pad_all_inputs_defined_forward_zero_pad():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [1, 2, 3, 4],
    }
    final_shapes = {
        "output": [1, 2, 3, 4],
        "input": [1, 2, 3, 4],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((0, 0), (0, 0), (0, 0), (0, 0)))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, True, {"output"}, scalar_info
    )


def test_pad_all_inputs_defined_forward_one_pad_symmetric():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [1, 2, 3, 4],
    }
    final_shapes = {
        "output": [3, 4, 5, 6],
        "input": [1, 2, 3, 4],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((1, 1), (1, 1), (1, 1), (1, 1)))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, True, {"output"}, scalar_info
    )


def test_pad_all_inputs_defined_forward_one_pad_asymmetric():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [1, 2, 3, 4],
    }
    final_shapes = {
        "output": [2, 3, 4, 5],
        "input": [1, 2, 3, 4],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((0, 1), (0, 1), (0, 1), (0, 1)))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, True, {"output"}, scalar_info
    )


def test_pad_all_inputs_defined_forward_random_pad():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": [1, 2, 3, 4],
    }
    final_shapes = {
        "output": [2, 7, 8, 25],
        "input": [1, 2, 3, 4],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((0, 1), (2, 3), (5, 0), (9, 12)))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, True, {"output"}, scalar_info
    )


def test_pad_some_inputs_defined_forward_random_pad():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": ["a", 2, 3, 4],
    }
    final_shapes = {
        "output": ["b", 7, 8, 25],
        "input": ["a", 2, 3, 4],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((0, 1), (2, 3), (5, 0), (9, 12)))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, False, {"output"}, scalar_info
    )


def test_pad_input_with_variadic_forward_random_pad():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V2", ...)],
        "input": ["a", ("V1", ...), 3, "b", 4],
    }
    final_shapes = {
        "output": ["d", "e", 8, "f", 25],
        "input": ["a", "b", 3, "c", 4],
        "pad_width": [],
    }
    scalar_info = {
        "pad_width": IOHyperEdge(value=((0, 1), (2, 3), (5, 0), (1, 1), (9, 12)))
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        pad_constraints,
        False,
        {"output", "input"},
        scalar_info,
    )


def test_pad_output_defined_backward_zero_pad():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5, 6],
        "input": [("V1", ...)],
    }
    final_shapes = {
        "output": [3, 4, 5, 6],
        "input": [3, 4, 5, 6],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((0, 0), (0, 0), (0, 0), (0, 0)))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, True, {"input"}, scalar_info
    )


def test_pad_output_defined_backward_one_pad_symmetric():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5, 6],
        "input": [("V1", ...)],
    }
    final_shapes = {
        "output": [3, 4, 5, 6],
        "input": [1, 2, 3, 4],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((1, 1), (1, 1), (1, 1), (1, 1)))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, True, {"input"}, scalar_info
    )


def test_pad_output_defined_backward_one_pad_asymmetric():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3, 4, 5, 6],
        "input": [("V1", ...)],
    }
    final_shapes = {
        "output": [3, 4, 5, 6],
        "input": [2, 3, 4, 5],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((0, 1), (0, 1), (0, 1), (0, 1)))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, True, {"input"}, scalar_info
    )


def test_pad_output_defined_backward_random_pad():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [2, 7, 8, 25],
        "input": [("V1", ...)],
    }
    final_shapes = {
        "output": [2, 7, 8, 25],
        "input": [1, 2, 3, 4],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((0, 1), (2, 3), (5, 0), (9, 12)))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, True, {"input"}, scalar_info
    )


def test_pad_output_with_variadic_forward_random_pad():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V2", ...), 8, "b", 25],
        "input": ["a", ("V1", ...)],
    }
    final_shapes = {
        "output": ["d", "e", 8, "f", 25],
        "input": ["a", "b", 3, "c", 4],
        "pad_width": [],
    }
    scalar_info = {
        "pad_width": IOHyperEdge(value=((0, 1), (2, 3), (5, 0), (1, 1), (9, 12)))
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        pad_constraints,
        False,
        {"output", "input"},
        scalar_info,
    )


def test_pad_input_output_mismatch_error():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [2, 7, 8, 25],
        "input": [1, 2, 3, 5],
    }
    final_shapes = {
        "output": [2, 7, 8, 25],
        "input": [1, 2, 3, 4],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(value=((0, 1), (2, 3), (5, 0), (9, 12)))}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, final_shapes, {}, pad_constraints, True, {"input"}, scalar_info
        )
    assert str(err_info.value) == "Possible values mismatch!"


def test_pad_pad_width_tbd():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V1", ...)],
        "input": ["a", 2, 3, 4],
    }
    final_shapes = {
        "output": ["V1, ..."],
        "input": ["a", 2, 3, 4],
        "pad_width": [],
    }
    scalar_info = {"pad_width": IOHyperEdge(type=tuple | ToBeDetermined)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, pad_constraints, False, set(), scalar_info
    )


############# ARANGE #############


def test_arange_1():
    """Should work with no problem with start, stop, step = (1, 5, 1)"""
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a"]}
    final_shapes = {"output": [4], "start": [], "stop": [], "step": []}
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=5),
        "step": IOHyperEdge(value=1),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, arange_constraints, True, {"output"}, scalar_info
    )


def test_arange_2():
    """Should work with no problem with start, stop, step = (1, 5, 2)"""
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a"]}
    final_shapes = {"output": [2], "start": [], "stop": [], "step": []}
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=5),
        "step": IOHyperEdge(value=2),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, arange_constraints, True, {"output"}, scalar_info
    )


def test_arange_3():
    """Should work with no problem with decimal step."""
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a"]}
    final_shapes = {"output": [4], "start": [], "stop": [], "step": []}
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=5),
        "step": IOHyperEdge(value=1.1),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, arange_constraints, True, {"output"}, scalar_info
    )


def test_arange_4():
    """Should work with no problem with negative values and
    decimal step.
    """
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a"]}
    final_shapes = {"output": [4], "start": [], "stop": [], "step": []}
    scalar_info = {
        "start": IOHyperEdge(value=-1),
        "stop": IOHyperEdge(value=-5),
        "step": IOHyperEdge(value=-1.1),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, arange_constraints, True, {"output"}, scalar_info
    )


def test_arange_5():
    """Should work with no problem with positive values with
    negative decimal step .
    """
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a"]}
    final_shapes = {"output": [4], "start": [], "stop": [], "step": []}
    scalar_info = {
        "start": IOHyperEdge(value=5),
        "stop": IOHyperEdge(value=1),
        "step": IOHyperEdge(value=-1.1),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, arange_constraints, True, {"output"}, scalar_info
    )


def test_arange_6():
    """Should work with no problem with positive decimal values
    with negative decimal step .
    """
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a"]}
    final_shapes = {"output": [5], "start": [], "stop": [], "step": []}
    scalar_info = {
        "start": IOHyperEdge(value=5.2),
        "stop": IOHyperEdge(value=1.34),
        "step": IOHyperEdge(value=-0.9),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, arange_constraints, True, {"output"}, scalar_info
    )


def test_arange_7():
    """Should work with no problem with positive decimal values
    with indefinite step. No change in status and updated_symbols.
    """
    shapes: dict[str, list[int | str | tuple]] = {"output": [5]}
    final_shapes = {"output": [5], "start": [], "stop": [], "step": []}
    scalar_info = {
        "start": IOHyperEdge(value=5.2),
        "stop": IOHyperEdge(value=1.34),
        "step": IOHyperEdge(int | float, value=TBD),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, arange_constraints, False, set(), scalar_info
    )


def test_arange_8():
    """Should work with no problem with start, stop, step = (1, 1, 1).
    In this case output array is empty which means no change occurs in
    output shape map since it's also empty.
    """
    shapes: dict[str, list[int | str | tuple]] = {"output": []}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "start": [],
        "stop": [],
        "step": [],
    }
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=1),
        "step": IOHyperEdge(value=1),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, arange_constraints, True, set(), scalar_info
    )


def test_arange_9():
    """Should work with no problem with start, stop, step = (1, 1, 1).
    In this case output array is empty which means variadic field of
    output will be updated and output shape_map will become [].
    """
    shapes: dict[str, list[int | str | tuple]] = {"output": [("V", ...)]}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "start": [],
        "stop": [],
        "step": [],
    }
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=1),
        "step": IOHyperEdge(value=1),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, arange_constraints, True, {"output"}, scalar_info
    )


def test_arange_error_1():
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a"]}
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=5),
        "step": IOHyperEdge(value=-0.9),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, arange_constraints, False, set(), scalar_info
        )
    assert (
        str(err_info.value)
        == "Start number (1) can not be lower than stop number (5) while step = -0.9"
    )


def test_arange_error_2():
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a"]}
    scalar_info = {
        "start": IOHyperEdge(value=4),
        "stop": IOHyperEdge(value=1),
        "step": IOHyperEdge(value=0.9),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, arange_constraints, False, set(), scalar_info
        )
    assert (
        str(err_info.value)
        == "Start number (4) can not be higher than stop number (1) while step = 0.9"
    )


def test_arange_error_3():
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a", ("V", ...), "b"]}
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=5),
        "step": IOHyperEdge(value=2),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, arange_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Shape mismatch. Output has at least 2 dim(s) where it can have "
        "at most 1 dim."
    )


def test_arange_error_4():
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a", "b"]}
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=5),
        "step": IOHyperEdge(value=2),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, arange_constraints, False, set(), scalar_info
        )
    assert (
        str(err_info.value)
        == "Arange output shape can only have 1 dim in this setting. Got 2 dim(s) here."
    )


def test_arange_error_5():
    """Should raise ValueError since output is zero dimensional
    for start, step, stop = (1, 1, 2).
    """
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a"]}
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=1),
        "step": IOHyperEdge(value=2),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, arange_constraints, False, set(), scalar_info
        )
    assert (
        str(err_info.value)
        == "Arange output shape can only have 0 dim in this setting. Got 1 dim(s) here."
    )


def test_arange_error_6():
    """Should raise ValueError since output shape has variadic field and
    has a minimum length of 2.
    """
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a", ("V", ...), "b"]}
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=3),
        "step": IOHyperEdge(value=1),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, arange_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Shape mismatch. Output has at least 2 dim(s) where it can have "
        "at most 1 dim."
    )


def test_arange_error_7():
    """Should raise ValueError since output shape has variadic field and
    has a minimum length of 1 but it must exactly have 0 dim.
    """
    shapes: dict[str, list[int | str | tuple]] = {"output": [("V", ...), "b"]}
    scalar_info = {
        "start": IOHyperEdge(value=1),
        "stop": IOHyperEdge(value=1),
        "step": IOHyperEdge(value=1),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, arange_constraints, False, set(), scalar_info
        )
    assert (
        str(err_info.value)
        == "Arange output shape has minimum 1 dim(s) where it is a rank-0 array."
    )


############# Randn #############


def test_randn_1():
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a", "b", "c"]}
    final_shapes = {"output": [3, 4, 5], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(3, 4, 5))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, randn_constraints, True, {"output"}, scalar_info
    )


def test_randn_2():
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a", 5, "c"]}
    final_shapes = {"output": [3, 4, 5], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(3, 4, 5))}
    try:
        assert_constraint_results(
            shapes,
            {},
            final_shapes,
            {},
            randn_constraints,
            True,
            {"output"},
            scalar_info,
        )
    except ValueError as e:
        assert str(e) == "Possible values mismatch!"


def test_randn_3():
    shapes: dict[str, list[int | str | tuple]] = {"output": ["a", "b", "c", "d"]}
    final_shapes = {"output": [3, 4, 5], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(3, 4, 5))}
    try:
        assert_constraint_results(
            shapes,
            {},
            final_shapes,
            {},
            randn_constraints,
            True,
            {"output"},
            scalar_info,
        )
    except ValueError as e:
        assert (
            str(e) == "Shape mismatch. Output has 4 dim(s) where it must have 3 dim(s)."
        )


def test_randn_4():
    shapes: dict[str, list[int | str | tuple]] = {"output": [("Var1", ...)]}
    final_shapes = {"output": [3, 4, 5], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(3, 4, 5))}

    assert_constraint_results(
        shapes, {}, final_shapes, {}, randn_constraints, True, {"output"}, scalar_info
    )


def test_randn_5():
    shapes: dict[str, list[int | str | tuple]] = {"output": [2, 3, 4, 5, ("Var1", ...)]}
    final_shapes = {"output": [3, 4, 5], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(3, 4, 5))}
    try:
        assert_constraint_results(
            shapes,
            {},
            final_shapes,
            {},
            randn_constraints,
            True,
            {"output"},
            scalar_info,
        )
    except ValueError as e:
        assert str(e) == (
            "Shape mismatch. Output has minimum 4 dim(s) where it must have exactly "
            "3 dim(s)."
        )


############# MAXPOOL_2D #############


def test_max_pool_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [10, 20, "c", "d"],
        "input": [10, 10, 10, 10],
    }
    final_shapes = {
        "output": [10, 20, 8, 8],
        "input": [10, 10, 10, 10],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(1, 1)),
        "padding": IOHyperEdge(value=(0, 0)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_max_pool_2():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [10, 20, "c", "d"],
        "input": [10, 10, 10, 10],
    }
    final_shapes = {
        "output": [10, 20, 6, 6],
        "input": [10, 10, 10, 10],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(1, 1)),
        "padding": IOHyperEdge(value=(0, 0)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(5, 5)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_max_pool_3_error():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [10, 20, "c", "d"],
        "input": [10, 10, 10, 10],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(1, 1)),
        "padding": IOHyperEdge(value=(0, 0)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(12, 12)),
    }

    with pytest.raises(Exception) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, sliding_window_2d_constraints, True, set(), scalar_info
        )
    assert (
        str(err_info.value)
        == "Dimension Error: Output dimension calculated to be lesser than zero!"
    )


def test_max_pool_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [10, 20, "c", "d"],
        "input": [10, 10, 10, 10],
    }
    final_shapes = {
        "output": [10, 20, 4, 8],
        "input": [10, 10, 10, 10],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(0, 0)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_max_pool_5():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [10, 20, "c", "d"],
        "input": [10, 10, 10, 10],
    }
    final_shapes = {
        "output": [10, 20, 5, 10],
        "input": [10, 10, 10, 10],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(1, 1)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_max_pool_6():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [10, 20, "c", "d"],
        "input": [10, 10, 10, 10],
    }
    final_shapes = {
        "output": [10, 20, 5, 12],
        "input": [10, 10, 10, 10],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(1, 2)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_max_pool_7():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [10, 20, "c", "d"],
        "input": [10, 10, 10, 10],
    }
    final_shapes = {
        "output": [10, 20, 3, 12],
        "input": [10, 10, 10, 10],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(1, 2)),
        "dilation": IOHyperEdge(value=(3, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_max_pool_8():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...), "c", "d"],
        "input": [("Var1", ...), 10, 10],
    }
    final_shapes = {
        "output": ["(Var1, ...)", 3, 12],
        "input": ["(Var1, ...)", 10, 10],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(1, 2)),
        "dilation": IOHyperEdge(value=(3, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_max_pool_9():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...), "c", "d"],
        "input": [("Var1", ...), "c", 10, 10],
    }
    final_shapes = {
        "output": ["(Var1, ...)", 3, 12],
        "input": ["(Var1, ...)", 3, 10, 10],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(1, 2)),
        "dilation": IOHyperEdge(value=(3, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        True,
        {"output", "input"},
        scalar_info,
    )


def test_max_pool_10():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...), "c", "d"],
        "input": ["a", "b", 10, 10],
    }
    final_shapes = {
        "output": ["(Var1, ...)", 4, 8],
        "input": ["a", "b", 10, 10],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(0, 0)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_max_pool_11():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": ["a", "b", "c", "d"],
    }
    final_shapes = {
        "output": ["(Var1, ...)"],
        "input": ["a", "b", "c", "d"],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(0, 0)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        False,
        set(),
        scalar_info,
    )


def test_max_pool_12():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": [("Var2", ...), "d"],
    }
    final_shapes = {
        "output": ["(Var1, ...)"],
        "input": ["(Var2, ...)", "d"],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(0, 0)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        False,
        set(),
        scalar_info,
    )


def test_max_pool_13():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": ["a", ("Var2", ...), "d"],
    }
    final_shapes = {
        "output": ["(Var1, ...)"],
        "input": ["a", "(Var2, ...)", "d"],
        "stride": [],
        "padding": [],
        "dilation": [],
        "kernel_size": [],
    }
    scalar_info = {
        "stride": IOHyperEdge(value=(2, 1)),
        "padding": IOHyperEdge(value=(0, 0)),
        "dilation": IOHyperEdge(value=(1, 1)),
        "kernel_size": IOHyperEdge(value=(3, 3)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        sliding_window_2d_constraints,
        False,
        set(),
        scalar_info,
    )


############# BROADCAST_TO #############


def test_broadcast_to_1():
    """Should work with no problem with variadic output shape
    and shape = (1, 2).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V", ...)],
        "input": [("V2", ...)],
    }
    final_shapes = {"output": [1, 2], "input": ["(V2, ...)"], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(1, 2))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        broadcast_to_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_broadcast_to_2():
    """Should work with no problem with uniadic output shape
    and shape = (1, 2).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b"],
        "input": [("V2", ...)],
    }
    final_shapes = {"output": [1, 2], "input": ["(V2, ...)"], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(1, 2))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        broadcast_to_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_broadcast_to_3():
    """Should work with no problem with variadic output and
    unknown shape.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V", ...)],
        "input": [("V2", ...)],
    }
    final_shapes = {"output": ["(V, ...)"], "input": ["(V2, ...)"], "shape": []}
    scalar_info = {"shape": IOHyperEdge(tuple[int, ...], value=TBD)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        broadcast_to_constraints,
        False,
        set(),
        scalar_info,
    )


def test_broadcast_to_4():
    """Should work with no problem with uniadic output and
    unknown shape.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b"],
        "input": [("V2", ...)],
    }
    final_shapes = {"output": ["a", "b"], "input": ["(V2, ...)"], "shape": []}
    scalar_info = {"shape": IOHyperEdge(tuple[int, ...], value=TBD)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        broadcast_to_constraints,
        False,
        set(),
        scalar_info,
    )


def test_broadcast_to_5():
    """Should work with no problem with variadic output and
    shape = (3, 4, 5).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", ("V", ...), "b"],
        "input": [("V2", ...)],
    }
    final_shapes = {"output": [3, 4, 5], "input": ["(V2, ...)"], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(3, 4, 5))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        broadcast_to_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_broadcast_to_error_1():
    """Should raise ValueError since number of output dims (1) is
    lower than 2.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a"],
        "input": [("V2", ...)],
    }
    scalar_info = {"shape": IOHyperEdge(value=(1, 2))}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, broadcast_to_constraints, False, set(), scalar_info
        )
    assert (
        str(err_info.value)
        == "Shape mismatch. Output has 1 dim(s) where it must have 2 dim(s)."
    )


def test_broadcast_to_error_2():
    """Should raise ValueError since minimum output dims (2) is
    higher than 1 which is length of shape.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", ("V", ...), "b"],
        "input": [("V2", ...)],
    }
    scalar_info = {"shape": IOHyperEdge(value=(3,))}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, broadcast_to_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Shape mismatch. Output has minimum 2 dim(s) where it must "
        "have exactly 1 dim(s)."
    )


############# RESHAPE #############


def test_reshape_1():
    """Should work with no problem with variadic output, known
    input shape and reshape value = (6, ).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V", ...)],
        "input": [1, 2, 3],
    }
    final_shapes = {"output": [6], "input": [1, 2, 3], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(6,))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reshape_constraints, True, {"output"}, scalar_info
    )


def test_reshape_2():
    """Should work with no problem with variadic output, unknown
    input shape and reshape value = (6, 1). Since input can not
    be fully inferred, status is returned as False.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {"output": [6, 1], "input": ["(in, ...)"], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(6, 1))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reshape_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_reshape_3():
    """Should work with no problem with uniadic output, known
    input shape and reshape value = (3, 2).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b"],
        "input": [1, 2, 3],
    }
    final_shapes = {"output": [3, 2], "input": [1, 2, 3], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(3, 2))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reshape_constraints, True, {"output"}, scalar_info
    )


def test_reshape_4():
    """Should work with no problem with uniadic output, unknown
    input shape and reshape value = (2, 3). Since input can not
    be fully inferred, status is returned as False.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b"],
        "input": [("in", ...)],
    }
    final_shapes = {"output": [2, 3], "input": ["(in, ...)"], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(2, 3))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        reshape_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_reshape_5():
    """Should work with no problem with known output shape, unknown
    input shape and unkonwn reshape value. Infers value of reshape
    value.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [1, 2, 3],
        "input": [("in", ...)],
    }
    final_shapes = {"output": [1, 2, 3], "input": ["(in, ...)"], "shape": []}
    scalar_info = {"shape": IOHyperEdge(tuple[int, ...], value=TBD)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reshape_constraints, False, {"shape"}, scalar_info
    )


def test_reshape_6():
    """Should work with no problem with unknown output shape, known
    input shape and known reshape value. Infer value of reshape
    value.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var", ...)],
        "input": [1, 2, 3],
    }
    final_shapes = {"output": [3, 2], "input": [1, 2, 3], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(-1, 2))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reshape_constraints, True, {"output"}, scalar_info
    )


def test_reshape_7():
    """Should work with no problem with unknown output shape, known
    input shape and konwn reshape value. Infer value of reshape
    value.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b"],
        "input": [1, 2, 3],
    }
    final_shapes = {"output": [3, 2], "input": [1, 2, 3], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(-1, 2))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reshape_constraints, True, {"output"}, scalar_info
    )


def test_reshape_8():
    """Should work with no problem with known output shape, unknown
    input shape and unkonwn reshape value. Infer value of reshape
    value.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var", ...)],
        "input": [1, 2, 3, 4, 5, 6],
    }
    final_shapes = {"output": [5, 2, 36, 2], "input": [1, 2, 3, 4, 5, 6], "shape": []}
    scalar_info = {"shape": IOHyperEdge(value=(5, 2, -1, 2))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reshape_constraints, True, {"output"}, scalar_info
    )


def test_reshape_error_1():
    """Should raise ValueError since input shape and reshaped
    shape are not consistent. Variadic output here.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V", ...)],
        "input": [1, 2, 3, 4],
    }
    scalar_info = {"shape": IOHyperEdge(value=(2, 5))}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reshape_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == "Input (1, 2, 3, 4) can not be reshaped to (2, 5)"


def test_reshape_error_2():
    """Should raise ValueError since input shape and reshaped
    shape are not consistent. Uniadic output here.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b", "c"],
        "input": [1, 2, 3, 4],
    }
    scalar_info = {"shape": IOHyperEdge(value=(2, 5))}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reshape_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == "Input (1, 2, 3, 4) can not be reshaped to (2, 5)"


def test_reshape_error_3():
    """Should raise ValueError since output shape has higer
    amount of dimensions than resulting reshaped array.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b", "c"],
        "input": [1, 2, 3, 4],
    }
    scalar_info = {"shape": IOHyperEdge(value=(2, 12))}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reshape_constraints, False, set(), scalar_info
        )
    assert (
        str(err_info.value)
        == "Shape mismatch! Output has 3 dim(s) while reshaped one has 2 dim(s)."
    )


def test_reshape_error_4():
    """Should raise ValueError since output shape has
    minimum 3 dimensions while result has 2.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", ("V", ...), "b", "c"],
        "input": [1, 2, 3, 4],
    }
    scalar_info = {"shape": IOHyperEdge(value=(2, 12))}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reshape_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Shape mismatch! Output has mimimum 3 dim(s) while reshaped "
        "one has 2 dim(s)."
    )


def test_reshape_error_5():
    """Should raise ValueError since input shape and reshaped
    shape are not consistent.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [2, 2],
        "input": [1, 2, 3, 4],
    }
    scalar_info = {"shape": IOHyperEdge(tuple[int, ...], value=TBD)}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, reshape_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Shape mismatch! output (2, 2) and input (1, 2, 3, 4) have "
        "incompatible shapes"
    )


############# SQUEEZE #############


def test_squeeze_1():
    """Should work with no problem with uniadic output shape."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b", "c"],
        "input": [1, 2, 3, 1, 5],
    }
    final_shapes = {"output": [2, 3, 5], "input": [1, 2, 3, 1, 5]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, True, {"output"}
    )


def test_squeeze_2():
    """Should work with no problem with full variadic output shape."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("V", ...)],
        "input": [1, 2, 3, 1, 5],
    }
    final_shapes = {"output": [2, 3, 5], "input": [1, 2, 3, 1, 5]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, True, {"output"}
    )


def test_squeeze_3():
    """Should work with no problem with variadic and uniadic output shape."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", ("V", ...), "b"],
        "input": [1, 2, 3, 1, 5],
    }
    final_shapes = {"output": [2, 3, 5], "input": [1, 2, 3, 1, 5]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, True, {"output"}
    )


def test_squeeze_4():
    """Should work with no problem with variadic input shape and
    variadic output shape.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [1, 2, ("in", ...), 1, 5],
    }
    final_shapes = {"output": [2, "(V1, ...)", 5], "input": [1, 2, "(V2, ...)", 1, 5]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, False, {"output"}
    )


def test_squeeze_5():
    """Should work with no problem with uniadic input shape and
    variadic output shape where output shape has 0 dim.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [1, 1],
    }
    final_shapes = {"output": [], "input": [1, 1]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, True, {"output"}
    )


def test_squeeze_6():
    """Should work with no problem. No changes occur for the configuration
    where there is no 1s in input shape and it has variadic field.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [2, ("in", ...), 4],
    }
    final_shapes = {"output": [2, "(V1, ...)", 4], "input": [2, "(V2, ...)", 4]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, False, {"output"}
    )


def test_squeeze_7():
    """Should work with no problem. No changes occur for the configuration
    where there is no 1s in input shape and it has variadic field.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [2, ("out", ...), 4],
        "input": [("in", ...)],
    }
    final_shapes = {"output": [2, "(out, ...)", 4], "input": ["(in, ...)"]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, False, set()
    )


def test_squeeze_8():
    """Should work with no problem. No changes occur for the configuration
    where there is no 1s in input shape and it has variadic field.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [2, "a", 4],
    }
    final_shapes = {"output": [2, "(V1, ...)", 4], "input": [2, "a", 4]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, False, {"output"}
    )


def test_squeeze_9():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [2, 3, ("in", ...), 4],
    }
    final_shapes = {"output": [2, 3, "(V1, ...)", 4], "input": [2, 3, "(V2, ...)", 4]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, False, {"output"}
    )


def test_squeeze_10():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [2, 1, 3, ("in", ...), 4],
    }
    final_shapes = {
        "output": [2, 3, "(V1, ...)", 4],
        "input": [2, 1, 3, "(V2, ...)", 4],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, False, {"output"}
    )


def test_squeeze_11():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [2, "u1", 3, ("in", ...), 4],
    }
    final_shapes = {
        "output": [2, "(V1, ...)", 4],
        "input": [2, "u1", 3, "(V2, ...)", 4],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, False, {"output"}
    )


def test_squeeze_12():
    """Should work with no problem. No changes occur for the configuration
    where there is no 1s in input shape and it has variadic field.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [2, ("in", ...), 4, 1, 1],
    }
    final_shapes = {"output": [2, "(V1, ...)", 4], "input": [2, "(V2, ...)", 4, 1, 1]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, False, {"output"}
    )


def test_squeeze_13():
    """Should work with no problem. No changes occur for the configuration
    where there is no 1s in input shape and it has variadic field.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [2, ("in", ...), 1, 1, 4],
    }
    final_shapes = {"output": [2, "(V1, ...)", 4], "input": [2, "(V2, ...)", 1, 1, 4]}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, squeeze_constraints, False, {"output"}
    )


def test_squeeze_error_1():
    """Should raise error since output dimensionality is higher than
    input.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [2, ("out", ...), 3],
        "input": [4],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, squeeze_constraints, False, set())
    assert str(err_info.value) == (
        "Output shape can not have higher number of dimensions (min 2) "
        "than input (1)"
    )


def test_squeeze_error_2():
    """Should raise error since first dimension of output
    shape mismaches with first dimension of squeezed array.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [2, ("out", ...), 3],
        "input": [1, 4, ("in", ...), 1, 3],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, squeeze_constraints, False, set())
    assert str(err_info.value) == "Possible values mismatch!"


def test_squeeze_error_3():
    """Should raise error since a dimension of output
    shape has 1 in it. Squeeze model can not have any output
    shape dimensionality as 1.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [2, ("out", ...), 1, 3],
        "input": [1, 4, ("in", ...), 1, 3],
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(shapes, {}, {}, {}, squeeze_constraints, False, set())
    assert str(err_info.value) == (
        "Squeeze output shape can not have any dimensionality as 1, "
        "got output shape as [2, '(V1, ...)', 1, 3]"
    )


############# SIZE #############


def test_size_1():
    """Should work with no problem when dim = 2 and fully variadic input
    shape.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("in", ...)],
    }
    final_shapes = {"input": ["a", "b", "c", "(V1, ...)"], "output": [], "dim": []}
    scalar_info = {
        "dim": IOHyperEdge(value=2),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, size_constraints, False, {"input"}, scalar_info
    )


def test_size_2():
    """Should work with no problem when dim = None and fully variadic input
    and output shape.
    """
    shapes: dict[str, list[int | str | tuple]] = {"input": [("in", ...)]}
    final_shapes = {"input": ["(in, ...)"], "output": [], "dim": []}
    scalar_info = {
        "dim": IOHyperEdge(value=None),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, size_constraints, False, set(), scalar_info
    )


def test_size_3():
    shapes: dict[str, list[int | str | tuple]] = {"input": [2, 3, 4]}
    final_shapes = {"input": [2, 3, 4], "output": [], "dim": []}
    final_values = {"output": 2}
    scalar_info = {
        "dim": IOHyperEdge(value=0),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }

    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_size_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [2, 3, 4],
    }
    final_shapes = {"input": [2, 3, 4], "output": [], "dim": []}
    final_values = {"output": (3, 4)}
    scalar_info = {
        "dim": IOHyperEdge(value=(1, 2)),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_size_5():
    shapes: dict[str, list[int | str | tuple]] = {"input": [2, 3, 4]}
    final_shapes = {"input": [2, 3, 4], "output": [], "dim": []}
    final_values = {"output": (4, 2)}
    scalar_info = {
        "dim": IOHyperEdge(value=(-1, 0)),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_size_6():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [2, 3, 4],
    }
    final_shapes = {"input": [2, 3, 4], "output": [], "dim": []}
    final_values = {"output": (3, 2, 4)}
    scalar_info = {
        "dim": IOHyperEdge(value=(1, 0, 2)),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_size_7():
    shapes: dict[str, list[int | str | tuple]] = {"input": [2, 3, 4]}
    final_shapes = {"input": [2, 3, 4], "output": [], "dim": []}
    final_values = {"output": (4, 3)}
    scalar_info = {
        "dim": IOHyperEdge(value=(-1, -2)),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_size_8():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [2, 3, 4, ("Var1", ...), 1, 5]
    }
    final_shapes = {"input": [2, 3, 4, "(V1, ...)", 1, 5], "output": [], "dim": []}
    final_values = {"output": (2, 3)}
    scalar_info = {
        "dim": IOHyperEdge(value=(0, 1)),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_size_9():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [2, 3, 4, ("Var1", ...), 1, 5]
    }
    final_shapes = {"input": [2, 3, 4, "(V1, ...)", 1, 5], "output": [], "dim": []}
    final_values = {"output": 4}
    scalar_info = {
        "dim": IOHyperEdge(value=2),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_size_10():
    shapes: dict[str, list[int | str | tuple]] = {"input": ["u1", "u2", "u3"]}
    final_shapes = {"input": ["u1", 3, 4], "output": [], "dim": []}
    final_values = {"output": (3, 4)}
    scalar_info = {
        "dim": IOHyperEdge(value=(1, 2)),
        "output": IOHyperEdge(value=(3, 4)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"input"},
        scalar_info,
        final_values,
    )


def test_size_11():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", "u3", ("Var1", ...), "u4", "u5", "u6"]
    }
    final_shapes = {
        "input": ["u1", 10, "u3", "(V1, ...)", "u4", "u5", 7],
        "output": [],
        "dim": [],
    }
    final_values = {"output": (7, 10)}
    scalar_info = {
        "dim": IOHyperEdge(value=(-1, 1)),
        "output": IOHyperEdge(value=(7, 10)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"input"},
        scalar_info,
        final_values,
    )


def test_size_12():
    shapes: dict[str, list[int | str | tuple]] = {"input": [("Var1", ...)]}
    final_shapes = {"input": ["a", 10, "b", "(V1, ...)"], "output": [], "dim": []}
    final_values = {"output": (7, 10)}
    scalar_info = {
        "dim": IOHyperEdge(value=(-3, 1)),
        "output": IOHyperEdge(value=(7, 10)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        False,
        {"input"},
        scalar_info,
        final_values,
    )


def test_size_13():
    shapes: dict[str, list[int | str | tuple]] = {"input": [2, 3, 4]}
    final_shapes = {"input": [2, 3, 4], "output": [], "dim": []}
    final_values = {"output": 4}
    scalar_info = {
        "dim": IOHyperEdge(value=-1),
        "output": IOHyperEdge(type=int | tuple[int, ...], value=TBD),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        size_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


############# FLATTEN #############


def test_flatten_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["a", "b", "c", "x", "e", "f"],
        "input": ["a", "b", "c", "d", "(V1, ...)", "e", "f"],
        "start_dim": [],
        "end_dim": [],
    }
    scalar_info = {"start_dim": IOHyperEdge(value=3), "end_dim": IOHyperEdge(value=-3)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        flatten_constrains,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_flatten_2():
    """Start_dim and end_dim are positive, end_dim is greater than start_dim."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["a", "b", "c", "x", "(V1, ...)"],
        "input": ["a", "b", "c", "d", "e", "f", "(V1, ...)"],
        "start_dim": [],
        "end_dim": [],
    }
    scalar_info = {
        "start_dim": IOHyperEdge(type=int | type(...), value=3),
        "end_dim": IOHyperEdge(type=int | type(...), value=5),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        flatten_constrains,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_flatten_3():
    """Start_dim and end_dim are positive, end_dim is equal to start_dim."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["a", "b", "c", "d", "(V1, ...)"],
        "input": ["a", "b", "c", "d", "(V1, ...)"],
        "start_dim": [],
        "end_dim": [],
    }
    scalar_info = {
        "start_dim": IOHyperEdge(type=int | type(...), value=3),
        "end_dim": IOHyperEdge(type=int | type(...), value=3),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        flatten_constrains,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_flatten_4():
    """Start_dim and end_dim are negative, end_dim is greater than start_dim."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)", "x", "c"],
        "input": ["(V1, ...)", "a", "b", "c"],
        "start_dim": [],
        "end_dim": [],
    }
    scalar_info = {
        "start_dim": IOHyperEdge(int, value=-3),
        "end_dim": IOHyperEdge(int, value=-2),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        flatten_constrains,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_flatten_5():
    """Start_dim and end_dim are negative, end_dim is equal to start_dim."""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)", "a", "b", "c"],
        "input": ["(V1, ...)", "a", "b", "c"],
        "start_dim": [],
        "end_dim": [],
    }
    scalar_info = {
        "start_dim": IOHyperEdge(int, value=-3),
        "end_dim": IOHyperEdge(int, value=-3),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        flatten_constrains,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_flatten_6():
    """Start_dim ellipsis and end dim positive"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["x", "(V2, ...)"],
        "input": ["a", "b", "c", "d", "(V1, ...)"],
        "start_dim": [],
        "end_dim": [],
    }
    scalar_info = {
        "start_dim": IOHyperEdge(int, value=TBD),
        "end_dim": IOHyperEdge(int, value=3),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        flatten_constrains,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_flatten_7():
    """Start_dim ellipsis and end dim negative"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["(V2, ...)", "x", "b", "c"],
        "input": ["(V1, ...)", "a", "b", "c"],
        "start_dim": [],
        "end_dim": [],
    }
    scalar_info = {
        "start_dim": IOHyperEdge(int, value=TBD),
        "end_dim": IOHyperEdge(int, value=-3),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        flatten_constrains,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_flatten_8():
    """Start_dim positive and end dim ellipsis"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["a", "b", "c", "x", "(V2, ...)"],
        "input": ["a", "b", "c", "d", "(V1, ...)"],
        "start_dim": [],
        "end_dim": [],
    }
    scalar_info = {
        "start_dim": IOHyperEdge(int, value=3),
        "end_dim": IOHyperEdge(int, value=TBD),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        flatten_constrains,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_flatten_9():
    """Start_dim positive and end dim ellipsis"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["x", "(V2, ...)"],
        "input": ["(V1, ...)", "a", "b", "c"],
        "start_dim": [],
        "end_dim": [],
    }
    scalar_info = {
        "start_dim": IOHyperEdge(int, value=-3),
        "end_dim": IOHyperEdge(int, value=TBD),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        flatten_constrains,
        False,
        {"input", "output"},
        scalar_info,
    )


############# SWAP_AXES #############


def test_swap_axes_1():
    """Should work with no problem with fully variadic input/output shape and
    1 positive 1 negative axis value. Status is returned as False since swap
    operation can not be performed at this stage.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["x", "y", "z", "t", "(V2, ...)", "k"],
        "input": ["a", "b", "c", "d", "(V1, ...)", "e"],
        "axis1": [],
        "axis2": [],
    }
    scalar_info = {"axis1": IOHyperEdge(value=3), "axis2": IOHyperEdge(value=-5)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        swap_axes_constraints,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_swap_axes_2():
    """Should work with no problem with fully variadic input/output shape and
    1 positive 1 negative axis value. Status is returned as False since swap
    operation can not be performed at this stage.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["a", "b", "x", "y", "(V2, ...)"],
        "input": ["a", "b", "c", "d", "(V1, ...)"],
        "axis1": [],
        "axis2": [],
    }
    scalar_info = {"axis1": IOHyperEdge(value=3), "axis2": IOHyperEdge(value=-2)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        swap_axes_constraints,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_swap_axes_3():
    """Should work with no problem with fully variadic input/output shape and
    2 positive axis values. Status is returned as True since swap
    operation can be performed at this stage (axis values with same signs).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["a", "d", "c", "b", "(V1, ...)"],
        "input": ["a", "b", "c", "d", "(V1, ...)"],
        "axis1": [],
        "axis2": [],
    }
    scalar_info = {"axis1": IOHyperEdge(value=3), "axis2": IOHyperEdge(value=1)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        swap_axes_constraints,
        True,
        {"input", "output"},
        scalar_info,
    )


def test_swap_axes_4():
    """Should work with no problem with fully variadic input/output shape and
    2 negative axis values. Status is returned as True since swap
    operation can be performed at this stage (axis values with same signs).
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)", "c", "b", "a", "d", "e"],
        "input": ["(V1, ...)", "a", "b", "c", "d", "e"],
        "axis1": [],
        "axis2": [],
    }
    scalar_info = {"axis1": IOHyperEdge(value=-3), "axis2": IOHyperEdge(value=-5)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        swap_axes_constraints,
        True,
        {"input", "output"},
        scalar_info,
    )


def test_swap_axes_5():
    """Should work with no problem with variadic output shape,
    uniadic input shape and 2 positive axis values. Status is
    returned as True since swap operation can be performed at
    this stage.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": ["a", "b", "c", "d"],
    }
    final_shapes = {
        "output": ["a", "d", "c", "b"],
        "input": ["a", "b", "c", "d"],
        "axis1": [],
        "axis2": [],
    }
    scalar_info = {"axis1": IOHyperEdge(value=3), "axis2": IOHyperEdge(value=1)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        swap_axes_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_swap_axes_6():
    """Should work with no problem with variadic output shape,
    uniadic input shape and 2 negative axis values. Status is
    returned as True since swap operation can be performed at
    this stage.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("out", ...)],
        "input": ["a", "b", "c", "d"],
    }
    final_shapes = {
        "output": ["a", "d", "c", "b"],
        "input": ["a", "b", "c", "d"],
        "axis1": [],
        "axis2": [],
    }
    scalar_info = {"axis1": IOHyperEdge(value=-3), "axis2": IOHyperEdge(value=-1)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        swap_axes_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_swap_axes_7():
    """Should work with no problem with uniadic output shape,
    variadic input shape and 2 negative axis values. Status is
    returned as True since swap operation can be performed at
    this stage.
    """
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a", "b", "c", "d"],
        "input": [("in", ...)],
    }
    final_shapes = {
        "output": ["a", "b", "c", "d"],
        "input": ["a", "d", "c", "b"],
        "axis1": [],
        "axis2": [],
    }
    scalar_info = {"axis1": IOHyperEdge(value=-3), "axis2": IOHyperEdge(value=-1)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        swap_axes_constraints,
        True,
        {"input"},
        scalar_info,
    )


############# TO_TENSOR #############


def test_to_tensor_1():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {"output": [("V1", ...)]}
    final_shapes = {
        "output": [2, 3],
        "input": [],
    }
    value = [[2, 3, 4], [2, 3, 4]]
    scalar_info = {"input": IOHyperEdge(value=value)}
    final_values = {"input": value}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_tensor_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_to_tensor_2():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {"output": [("V1", ...)]}
    final_shapes = {
        "output": [3, 4, 5, 6, 7, 8],
        "input": [],
    }
    value = np.random.rand(3, 4, 5, 6, 7, 8).tolist()
    scalar_info = {"input": IOHyperEdge(value=value)}
    final_values = {"input": value}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_tensor_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_to_tensor_3():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {"output": ["u1", ("V1", ...), "u2"]}
    final_shapes = {
        "output": [3, 4, 5, 6, 7, 8],
        "input": [],
    }
    value = np.random.rand(3, 4, 5, 6, 7, 8).tolist()
    scalar_info = {"input": IOHyperEdge(value=value)}
    final_values = {"input": value}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_tensor_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_to_tensor_4_error():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {"output": ["u1", ("V1", ...), "u1"]}
    value = np.random.rand(3, 4, 5, 6, 7, 8).tolist()
    scalar_info = {"input": IOHyperEdge(value=value)}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, to_tensor_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == "Possible values mismatch!"


def test_to_tensor_5():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {"output": ["u1", "u1", "u1"]}
    final_shapes = {
        "output": [3, 3, 3],
        "input": [],
    }
    value = np.random.rand(3, 3, 3).tolist()
    scalar_info = {"input": IOHyperEdge(value=value)}
    final_values = {"input": value}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_tensor_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_to_tensor_6():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {"output": [("Var1", ...)]}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input": [],
    }
    value = 3.0
    scalar_info = {"input": IOHyperEdge(value=value)}
    final_values = {"input": value}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_tensor_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_to_tensor_7_error():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {"output": ["u1", "u2"]}
    value = 3.0
    scalar_info = {"input": IOHyperEdge(value=value)}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, to_tensor_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == "Shape dimensions does not match"


def test_to_tensor_8_error():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {"output": ["u1", "u1", "u1"]}
    value = np.random.rand(3, 4, 5).tolist()
    scalar_info = {"input": IOHyperEdge(value=value)}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, to_tensor_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == "Possible values mismatch!"


############# REVERSE #############


def test_reverse_1():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u1", "u1"],
        "input": ["u1", "u1", "u1"],
    }
    final_shapes = {
        "output": ["u1", "u1", "u1"],
        "input": ["u1", "u1", "u1"],
        "axes": [],
    }
    scalar_info = {"axes": IOHyperEdge(value=None)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reverse_constraints, True, set(), scalar_info
    )


def test_reverse_2():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": ["u1", "u2", "u3"],
    }
    final_shapes = {
        "output": ["u3", "u2", "u1"],
        "input": ["u1", "u2", "u3"],
        "axes": [],
    }
    scalar_info = {"axes": IOHyperEdge(value=None)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reverse_constraints, True, {"output"}, scalar_info
    )


def test_reverse_3():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "input": ["u1", "u2", "u3"],
    }
    final_shapes = {
        "output": ["u2", "u1", "u3"],
        "input": ["u1", "u2", "u3"],
        "axes": [],
    }
    scalar_info = {"axes": IOHyperEdge(value=[1, 0, 2])}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reverse_constraints, True, {"output"}, scalar_info
    )


def test_reverse_4():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2", "u3"],
        "input": [("Var1", ...)],
    }
    final_shapes = {
        "output": ["u1", "u2", "u3"],
        "input": ["u2", "u1", "u3"],
        "axes": [],
    }
    scalar_info = {"axes": IOHyperEdge(value=[1, 0, 2])}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, reverse_constraints, True, {"input"}, scalar_info
    )


############# ADDITION #############

# def test_addition_1():
#     """Should work with no problem
#     """
#     shapes: dict[str, list[int | str | tuple]] = {
#         "output": ["u1", "u2", "u3", "u4"],
#         "input1": ["u1", "u2", "u3", 6],
#         "input2": ["u1", "u2", "u3", 4],

#     }
#     shape_map = create_shape_map(shapes)
#     data = shape_map_to_tensor(shape_map)
#     data["indices"] = (-1, -1, -1)
#     in1_map = shape_map["input1"]
#     in2_map = shape_map["input2"]
#     out_map = shape_map["output"]
#     out_pre = out_map.prefix

#     status, updated_symbols = addition_constraints(**data)
#     # assert {out_pre[-1]} == updated_symbols
#     check_updated_symbols({out_pre[-1]}, updated_symbols)
#     assert status == True
#     assert in1_map.get_shapes({}, {}) == ["u1", "u2", "u3", 6]
#     assert in2_map.get_shapes({}, {}) == ["u1", "u2", "u3", 4]
#     assert out_map.get_shapes({}, {}) == ["u1", "u2", "u3", 10]


#     return status, data, addition_constraints


# def test_addition_2():
#     """Should work with no problem
#     """
#     shapes: dict[str, list[int | str | tuple]] = {
#         "output": ["u1", "u2", "u3", "u4"],
#         "input1": ["u1", "u2", 6],
#         "input2": ["u1", "u2", 4],

#     }
#     shape_map = create_shape_map(shapes)
#     data = shape_map_to_tensor(shape_map)
#     data["indices"] = (1, -1, -1)
#     in1_map = shape_map["input1"]
#     in2_map = shape_map["input2"]
#     out_map = shape_map["output"]
#     out_pre = out_map.prefix

#     status, updated_symbols = addition_constraints(**data)
#     # assert updated_symbols == {out_pre[1]}
#     check_updated_symbols({out_pre[1]}, updated_symbols)
#     assert status == True
#     assert in1_map.get_shapes({}, {}) == ["u1", 10, 6]
#     assert in2_map.get_shapes({}, {}) == ["u1", 10, 4]
#     assert out_map.get_shapes({}, {}) == ["u1", 10, "u2", "u3"]

#     return status, data, addition_constraints


# def test_addition_3():
#     """Should work with no problem
#     """
#     shapes: dict[str, list[int | str | tuple]] = {
#         "output": ["u1", "u1", "u1", "u1", "u1", "u1"],
#         "input1": [6],
#         "input2": ["u1", 4, "u1"],

#     }
#     shape_map = create_shape_map(shapes)
#     data = shape_map_to_tensor(shape_map)
#     data["indices"] = (1, -1, 1)
#     in1_map = shape_map["input1"]
#     in2_map = shape_map["input2"]
#     in2_pre = in2_map.prefix
#     out_map = shape_map["output"]
#     out_pre = out_map.prefix

#     status, updated_symbols = addition_constraints(**data)
#     # assert {*out_pre, in2_pre[0], in2_pre[-1]} == updated_symbols
#     check_updated_symbols({*out_pre, in2_pre[0], in2_pre[-1]}, updated_symbols)
#     assert status == True
#     assert in1_map.get_shapes({}, {}) == [6]
#     assert in2_map.get_shapes({}, {}) == [10, 4, 10]
#     assert out_map.get_shapes({}, {}) == [10, 10, 10, 10, 10, 10]

#     return status, data, addition_constraints


# def test_addition_4():
#     """Should work with no problem
#     """
#     shapes: dict[str, list[int | str | tuple]] = {
#         "output": ["u1", "u2", ("Var1", ...), "u3", "u4"],
#         "input1": ["u1", ("Var2", ...), 6],
#         "input2": [4, ("Var3", ...)],

#     }
#     shape_map = create_shape_map(shapes)
#     data = shape_map_to_tensor(shape_map)
#     data["indices"] = (1, -1, 0)
#     in1_map = shape_map["input1"]
#     in1_root = in1_map.root
#     in2_map = shape_map["input2"]
#     in2_root = in2_map.root
#     out_map = shape_map["output"]
#     out_root = out_map.root
#     out_pre = out_map.prefix

#     status, updated_symbols = addition_constraints(**data)
#     # assert {out_pre[1]} == updated_symbols
#     check_updated_symbols({out_pre[1]}, updated_symbols)
#     assert status == True
#     assert in1_map.get_shapes({}, {}) == ["u1", "(V1, ...)", 6]
#     assert in2_map.get_shapes({}, {}) == [4, "(V1, ...)"]
#     assert out_map.get_shapes({}, {}) == ["u1", 10, ("(V1, ...)"), "u2", "u3"]

#     return status, data, addition_constraints


# def test_addition_5():
#     """Should work with no problem
#     """
#     shapes: dict[str, list[int | str | tuple]] = {
#         "output": ["u1", 10, ("Var1", ...), "u3", "u4"],
#         "input1": ["u1", ("Var2", ...), 6],
#         "input2": [4, ("Var3", ...)]
#     }
#     shape_map = create_shape_map(shapes)
#     data = shape_map_to_tensor(shape_map)
#     data["indices"] = (1, -1, 0)
#     in1_map = shape_map["input1"]
#     in2_map = shape_map["input2"]
#     out_map = shape_map["output"]

#     status, updated_symbols = addition_constraints(**data)
#     assert set() == updated_symbols.shape_updates
#     assert status == True
#     assert in1_map.get_shapes({}, {}) == ["u1", "(V1, ...)", 6]
#     assert in2_map.get_shapes({}, {}) == [4, "(V1, ...)"]
#     assert out_map.get_shapes({}, {}) == ["u1", 10, ("(V1, ...)"), "u2", "u3"]

#     return status, data, addition_constraints

# def test_addition_6_error():
#     """Should work with no problem
#     """
#     shapes: dict[str, list[int | str | tuple]] = {
#         "output": ["u1", 13, ("Var1", ...), "u3", "u4"],
#         "input1": ["u1", ("Var2", ...), 6],
#         "input2": [4, ("Var3", ...)],

#     }
#     shape_map = create_shape_map(shapes)
#     data = shape_map_to_tensor(shape_map)
#     data["indices"] = (1, -1, 0)

#     with pytest.raises(ValueError) as err_info:
#         addition_constraints(**data)
#     assert str(err_info.value) == "Dimensions does not match in Tensor slice model!"

############# CONCAT #############


def test_concat_1():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2", "u3", "u4"],
        "axis": [],
        "input1": ["u1", "u2", "u3", 3],
        "input2": ["u1", "u2", "u3", 4],
        "input3": ["u1", "u2", "u3", 5],
    }
    final_shapes = {
        "output": ["u1", "u2", "u3", 12],
        "axis": [],
        "input1": ["u1", "u2", "u3", 3],
        "input2": ["u1", "u2", "u3", 4],
        "input3": ["u1", "u2", "u3", 5],
    }
    scalar_info = {"axis": IOHyperEdge(value=-1)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        concat_constraints,
        True,
        {"output"},
        scalar_info,
        variadic_fn=True,
    )


def test_concat_2():
    """Should work with no problem"""
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2", "u3", "u4"],
        "axis": [],
        "input1": ["u1", "u2", "u3", 3],
        "input2": ["u1", "u2", "u3", 4],
        "input3": ["u1", "u2", "u3", 5],
    }
    final_shapes = {
        "output": ["u1", "u2", "u3", 12],
        "axis": [],
        "input1": ["u1", "u2", "u3", 3],
        "input2": ["u1", "u2", "u3", 4],
        "input3": ["u1", "u2", "u3", 5],
    }
    scalar_info = {"axis": IOHyperEdge(value=-1)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        concat_constraints,
        True,
        {"output"},
        scalar_info,
        variadic_fn=True,
    )


def test_concat_3():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u3"],
        "axis": [],
        "input1": [3],
        "input2": [3],
        "input3": [1],
    }
    final_shapes = {
        "output": [7],
        "axis": [],
        "input1": [3],
        "input2": [3],
        "input3": [1],
    }
    scalar_info = {"axis": IOHyperEdge(value=-1)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        concat_constraints,
        True,
        {"output"},
        scalar_info,
        variadic_fn=True,
    )


def test_concat_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...), 12, "u2"],
        "axis": [],
        "input1": [("Var1", ...), 9, "u2"],
        "input2": [("Var1", ...), 2, "u2"],
        "input3": [("Var1", ...), "u1", "u2"],
    }
    final_shapes = {
        "output": ["(Var1, ...)", 12, "u1"],
        "axis": [],
        "input1": ["(Var1, ...)", 9, "u1"],
        "input2": ["(Var1, ...)", 2, "u1"],
        "input3": ["(Var1, ...)", 1, "u1"],
    }
    scalar_info = {"axis": IOHyperEdge(value=-2)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        concat_constraints,
        True,
        {"input3"},
        scalar_info,
        variadic_fn=True,
    )


def test_concat_5():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "axis": [],
        "input1": [("Var2", ...)],
        "input2": [("Var3", ...)],
        "input3": [("Var4", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)", "x", "b", "c", "d"],
        "axis": [],
        "input1": ["(V1, ...)", "y", "b", "c", "d"],
        "input2": ["(V1, ...)", "z", "b", "c", "d"],
        "input3": ["(V1, ...)", "t", "b", "c", "d"],
    }
    scalar_info = {"axis": IOHyperEdge(value=-4)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        concat_constraints,
        False,
        {"output", "input1", "input2", "input3"},
        scalar_info,
        variadic_fn=True,
    )


def test_concat_6():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "axis": [],
        "input1": [("Var2", ...)],
        "input2": [("Var3", ...)],
        "input3": [("Var4", ...)],
    }
    final_shapes = {
        "output": ["a"],
        "axis": [],
        "input1": ["(Var2, ...)"],
        "input2": ["(Var3, ...)"],
        "input3": ["(Var4, ...)"],
    }
    scalar_info = {"axis": IOHyperEdge(value=None)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        concat_constraints,
        False,
        {"output"},
        scalar_info,
        variadic_fn=True,
    )


def test_concat_7():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1"],
        "axis": [],
        "input1": [2, 3, 4],
        "input2": [6, 7, 8, 9],
        "input3": [2, 1, 1, 2, 3, 4],
    }
    final_shapes = {
        "output": [3096],
        "axis": [],
        "input1": [2, 3, 4],
        "input2": [6, 7, 8, 9],
        "input3": [2, 1, 1, 2, 3, 4],
    }
    scalar_info = {"axis": IOHyperEdge(value=None)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        concat_constraints,
        True,
        {"output"},
        scalar_info,
        variadic_fn=True,
    )


def test_concat_8():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3096],
        "axis": [],
        "input1": [2, 3, 4],
        "input2": [6, 7, 8, "u1"],
        "input3": [2, 1, 1, 2, 3, 4],
    }
    final_shapes = {
        "output": [3096],
        "axis": [],
        "input1": [2, 3, 4],
        "input2": [6, 7, 8, 9],
        "input3": [2, 1, 1, 2, 3, 4],
    }
    scalar_info = {"axis": IOHyperEdge(value=None)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        concat_constraints,
        True,
        {"input2"},
        scalar_info,
        variadic_fn=True,
    )


def test_concat_9():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3096],
        "axis": [],
        "input1": [2, 3, 4],
        "input2": [6, 7, 8, 9],
        "input3": [2, 1, "u1", 2, 3, 4],
    }
    final_shapes = {
        "output": [3096],
        "axis": [],
        "input1": [2, 3, 4],
        "input2": [6, 7, 8, 9],
        "input3": [2, 1, 1, 2, 3, 4],
    }
    scalar_info = {"axis": IOHyperEdge(value=None)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        concat_constraints,
        True,
        {"input3"},
        scalar_info,
        variadic_fn=True,
    )


############# SHAPE #############


def test_shape_1():
    shapes: dict[str, list[int | str | tuple]] = {"input": [3, 4, 5]}
    final_shapes = {"input": [3, 4, 5], "output": []}
    scalar_info = {"output": IOHyperEdge(tuple[int, ...] | type(...), value=TBD)}
    final_values = {"output": (3, 4, 5)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        shape_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_shape_2():
    shapes: dict[str, list[int | str | tuple]] = {"input": [1]}
    final_shapes = {"input": [1], "output": []}
    scalar_info = {"output": IOHyperEdge(tuple[int, ...] | type(...), value=TBD)}
    final_values = {"output": (1,)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        shape_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_shape_3():
    shapes: dict[str, list[int | str | tuple]] = {"input": [("Var1", ...)]}
    final_shapes = {"input": [3, 4, 5, 6, 7], "output": []}
    scalar_info = {"output": IOHyperEdge(value=(3, 4, 5, 6, 7))}
    final_values = {"output": (3, 4, 5, 6, 7)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        shape_constraints,
        True,
        {"input"},
        scalar_info,
        final_values,
    )


def test_shape_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", "u3", ("Var1", ...), "u4", "u5"]
    }
    final_shapes = {"input": [3, 4, 5, 6, 7], "output": []}
    scalar_info = {"output": IOHyperEdge(value=(3, 4, 5, 6, 7))}
    final_values = {"output": (3, 4, 5, 6, 7)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        shape_constraints,
        True,
        {"input"},
        scalar_info,
        final_values,
    )


def test_shape_5():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", "u3", ("Var1", ...), "u4", "u4"]
    }
    final_shapes = {"input": [3, 4, 5, 6, 6], "output": []}
    scalar_info = {"output": IOHyperEdge(value=(3, 4, 5, 6, 6))}
    final_values = {"output": (3, 4, 5, 6, 6)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        shape_constraints,
        True,
        {"input"},
        scalar_info,
        final_values,
    )


def test_to_tuple_forward():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input1": [],
        "input2": [],
        "input3": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=tuple[int, ...] | type(...), value=TBD),
        "input1": IOHyperEdge(value=3),
        "input2": IOHyperEdge(value=4),
        "input3": IOHyperEdge(value=5),
    }
    final_values = {"output": (3, 4, 5), "input1": 3, "input2": 4, "input3": 5}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_tuple_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
        variadic_fn=True,
    )


def test_to_tuple_reverse_1():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input1": [],
        "input2": [],
        "input3": [],
    }
    scalar_info = {
        "output": IOHyperEdge(value=(3, 4, 5)),
        "input1": IOHyperEdge(
            type=int | float | bool | list | tuple | type(...), value=TBD
        ),
        "input2": IOHyperEdge(
            type=int | float | bool | list | tuple | type(...), value=TBD
        ),
        "input3": IOHyperEdge(
            type=int | float | bool | list | tuple | type(...), value=TBD
        ),
    }
    final_values = {"output": (3, 4, 5), "input1": 3, "input2": 4, "input3": 5}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_tuple_constraints,
        True,
        {"input1", "input2", "input3"},
        scalar_info,
        final_values,
        variadic_fn=True,
    )


def test_to_tuple_reverse_2():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input1": [],
        "input2": [],
        "input3": [],
    }
    scalar_info = {
        "output": IOHyperEdge(value=(3, 4, 5)),
        "input1": IOHyperEdge(type=int | float | bool, value=3),
        "input2": IOHyperEdge(type=int | float | bool, value=TBD),
        "input3": IOHyperEdge(type=int | float | bool, value=5),
    }
    final_values = {"output": (3, 4, 5), "input1": 3, "input2": 4, "input3": 5}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_tuple_constraints,
        True,
        {"input2"},
        scalar_info,
        final_values,
        variadic_fn=True,
    )


def test_to_list_forward():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input1": [],
        "input2": [],
        "input3": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=list[int] | type(...), value=TBD),
        "input1": IOHyperEdge(value=3),
        "input2": IOHyperEdge(value=4),
        "input3": IOHyperEdge(value=5),
    }
    final_values = {"output": [3, 4, 5], "input1": 3, "input2": 4, "input3": 5}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_list_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
        variadic_fn=True,
    )


def test_to_list_reverse_1():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input1": [],
        "input2": [],
        "input3": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=list[int] | type(...), value=[3, 4, 5]),
        "input1": IOHyperEdge(
            type=int | float | bool | list | tuple | type(...), value=TBD
        ),
        "input2": IOHyperEdge(
            type=int | float | bool | list | tuple | type(...), value=TBD
        ),
        "input3": IOHyperEdge(
            type=int | float | bool | list | tuple | type(...), value=TBD
        ),
    }
    final_values = {"output": [3, 4, 5], "input1": 3, "input2": 4, "input3": 5}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_list_constraints,
        True,
        {"input1", "input2", "input3"},
        scalar_info,
        final_values,
        variadic_fn=True,
    )


def test_to_list_reverse_2():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input1": [],
        "input2": [],
        "input3": [],
    }
    scalar_info = {
        "output": IOHyperEdge(value=[3, 4, 5]),
        "input1": IOHyperEdge(value=3),
        "input2": IOHyperEdge(type=int | float | bool, value=TBD),
        "input3": IOHyperEdge(value=5),
    }
    final_values = {"output": [3, 4, 5], "input1": 3, "input2": 4, "input3": 5}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        to_list_constraints,
        True,
        {"input2"},
        scalar_info,
        final_values,
        variadic_fn=True,
    )


def test_where_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "cond": [("Var2", ...)],
        "input1": ["u1", "u2", "u3", 3],
        "input2": ["u1", "u2", "u3", 3],
    }
    final_shapes = {
        "output": ["(V1, ...)", "a", "b", "c", 3],
        "cond": ["(V2, ...)"],
        "input1": ["u1", "u2", "u3", 3],
        "input2": ["u1", "u2", "u3", 3],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, where_constrains, False, {"output"}
    )


def test_where_2():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "cond": [("Var2", ...)],
        "input1": [("Var3", ...)],
        "input2": ["u1", "u2", "u3", 3],
    }
    final_shapes = {
        "output": ["(V1, ...)", "a", "b", "c", 3],
        "cond": ["(V2, ...)"],
        "input1": ["(V3, ...)"],
        "input2": ["u1", "u2", "u3", 3],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, where_constrains, False, {"output"}
    )


def test_where_3():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "cond": [("Var2", ...)],
        "input1": ["u1", "u2", "u3", 3],
        "input2": [("Var3", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)", "a", "b", "c", 3],
        "cond": ["(V2, ...)"],
        "input1": ["u1", "u2", "u3", 3],
        "input2": ["(V3, ...)"],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, where_constrains, False, {"output"}
    )


def test_where_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var1", ...)],
        "cond": ["u1", "u2", "u3", 3],
        "input1": [("Var2", ...)],
        "input2": [("Var3", ...)],
    }
    final_shapes = {
        "output": ["(V1, ...)", "a", "b", "c", 3],
        "cond": ["u1", "u2", "u3", 3],
        "input1": ["(V2, ...)"],
        "input2": ["(V3, ...)"],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, where_constrains, False, {"output"}
    )


def test_eye_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2"],
    }
    final_shapes = {"output": [1, 1], "N": [], "M": []}
    scalar_info = {"N": IOHyperEdge(int, value=1), "M": IOHyperEdge(int, value=1)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, eye_constraints, True, {"output"}, scalar_info
    )


def test_eye_2():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2"],
    }
    final_shapes = {"output": [1, 4], "N": [], "M": []}
    scalar_info = {"N": IOHyperEdge(int, value=1), "M": IOHyperEdge(int, value=4)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, eye_constraints, True, {"output"}, scalar_info
    )


def test_eye_3():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2"],
    }
    final_shapes = {"output": [1, "u2"], "N": [], "M": []}
    scalar_info = {"N": IOHyperEdge(int, value=1), "M": IOHyperEdge(int)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, eye_constraints, False, {"output"}, scalar_info
    )


def test_eye_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2"],
    }
    final_shapes = {"output": ["u1", 2], "N": [], "M": []}
    scalar_info = {"N": IOHyperEdge(int), "M": IOHyperEdge(int, value=2)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, eye_constraints, False, {"output"}, scalar_info
    )


def test_eye_5():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["u1", "u2"],
    }
    final_shapes = {"output": ["u1", "u2"], "N": [], "M": []}
    scalar_info = {"N": IOHyperEdge(int), "M": IOHyperEdge(int)}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, eye_constraints, False, set(), scalar_info
    )


# def test_tensor_to_list_forward_1():
#     shapes: dict[str, list[int | str | tuple]] = {"input": [2, 3]}
#     final_shapes = {"input": [2, 3], "output": []}
#     value = np.ones((2, 3))
#     scalar_info = {
#         "output": IOHyperEdge(type=list[list[float]] | type(...)),
#     }
#     final_values = {"output": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], "input": value}
#     assert_constraint_results(
#         shapes,
#         {},
#         final_shapes,
#         {},
#         tensor_to_list_constraints,
#         True,
#         {"output"},
#         scalar_info,
#         final_values,
#         initial_values={"input": value},
#     )


# def test_tensor_to_list_forward_2():
#     shapes: dict[str, list[int | str | tuple]] = {"input": [2, 3]}
#     final_shapes = {"input": [2, 3], "output": []}
#     value = np.ones((2, 3), dtype=int)
#     scalar_info = {
#         "output": IOHyperEdge(type=list[list[int]] | type(...)),
#     }
#     final_values = {"output": [[1, 1, 1], [1, 1, 1]], "input": value}
#     assert_constraint_results(
#         shapes,
#         {},
#         final_shapes,
#         {},
#         tensor_to_list_constraints,
#         True,
#         {"output"},
#         scalar_info,
#         final_values,
#         initial_values={"input": value},
#     )


def test_tensor_to_list_backward_1():
    shapes: dict[str, list[int | str | tuple]] = {"input": [("Var1", ...)]}
    final_shapes = {"input": [2, 3], "output": []}
    scalar_info = {
        "output": IOHyperEdge(value=[[1, 1, 1], [1, 1, 1]]),
    }
    final_values = {"output": [[1, 1, 1], [1, 1, 1]]}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        tensor_to_list_constraints,
        True,
        {"input"},
        scalar_info,
        final_values,
    )


def test_tensor_to_list_backward_2():
    shapes: dict[str, list[int | str | tuple]] = {"input": [("Var1", ...)]}
    scalar_info = {
        "output": IOHyperEdge(value=[[1, 1, 1], [1, 1]]),
    }
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, tensor_to_list_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == "Inconsistent dimensions found in the list."


def test_item_constraints_1():
    shapes: dict[str, list[int | str | tuple]] = {"input": [("Var1", ...)]}
    final_shapes = {"input": ["(Var1, ...)"], "output": []}
    scalar_info = {"output": IOHyperEdge(type=int | float | bool | type(...))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, item_constraints, False, set(), scalar_info
    )


def test_item_constraints_2():
    shapes: dict[str, list[int | str | tuple]] = {"input": []}
    final_shapes: dict[str, list[int | str | tuple]] = {"input": [], "output": []}
    scalar_info = {"output": IOHyperEdge(type=int | float | bool | type(...))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, item_constraints, True, set(), scalar_info
    )


def test_item_constraints_3():
    shapes: dict[str, list[int | str | tuple]] = {"input": [1, 1, 1]}
    final_shapes = {"input": [1, 1, 1], "output": []}
    scalar_info = {"output": IOHyperEdge(type=int | float | bool | type(...))}
    assert_constraint_results(
        shapes, {}, final_shapes, {}, item_constraints, True, set(), scalar_info
    )


def test_item_constraints_4():
    shapes: dict[str, list[int | str | tuple]] = {"input": [1, 2, 1]}
    scalar_info = {"output": IOHyperEdge(type=int | float | bool | type(...))}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes, {}, {}, {}, item_constraints, False, set(), scalar_info
        )
    assert str(err_info.value) == (
        "Only tensors with 1 elements can be converted to scalar, "
        "got input shape as [1, 2, 1]"
    )


def test_scalar_item_1():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input": [],
        "index": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=int | float | bool | type(...)),
        "input": IOHyperEdge(type=list[int], value=[1, 2, 3]),
        "index": IOHyperEdge(type=int, value=2),
    }
    final_values = {"output": 3, "input": [1, 2, 3], "index": 2}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_scalar_item_2():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input": [],
        "index": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=int | float | bool | type(...)),
        "input": IOHyperEdge(type=list[int] | type(...), value=TBD),
        "index": IOHyperEdge(type=int, value=2),
    }
    final_values = {"output": TBD, "input": TBD, "index": 2}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        set(),
        scalar_info,
        final_values,
    )


def test_scalar_item_3():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input": [],
        "index": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=int | float | bool | type(...), value=2),
        "input": IOHyperEdge(type=list[int] | type(...), value=[1, 2, 3]),
        "index": IOHyperEdge(type=int | type(...), value=TBD),
    }
    final_values = {"output": 2, "input": [1, 2, 3], "index": 1}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"index"},
        scalar_info,
        final_values,
    )


def test_polynomial_features_1():
    shapes: dict[str, list[int | str | tuple]] = {"input": [4, 2], "output": [4, "u1"]}
    final_shapes = {"input": [4, 2], "output": [4, 5], "degree": []}
    scalar_info = {"degree": IOHyperEdge(type=int, value=2)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        polynomial_features_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_polynomial_features_2():
    shapes: dict[str, list[int | str | tuple]] = {"input": [4, 2], "output": [4, "u1"]}
    final_shapes = {"input": [4, 2], "output": [4, 9], "degree": []}
    scalar_info = {"degree": IOHyperEdge(type=int, value=3)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        polynomial_features_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_polynomial_features_3():
    shapes: dict[str, list[int | str | tuple]] = {"input": [4, "u1"], "output": [4, 9]}
    final_shapes = {"input": [4, 2], "output": [4, 9], "degree": []}
    scalar_info = {"degree": IOHyperEdge(type=int, value=3)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        polynomial_features_constraints,
        True,
        {"input"},
        scalar_info,
    )


def test_polynomial_features_4():
    shapes: dict[str, list[int | str | tuple]] = {"input": [4, "u1"], "output": [4, 8]}
    scalar_info = {"degree": IOHyperEdge(type=int, value=3)}
    with pytest.raises(ValueError) as err_info:
        assert_constraint_results(
            shapes,
            {},
            {},
            {},
            polynomial_features_constraints,
            False,
            set(),
            scalar_info,
        )
    assert (
        str(err_info.value)
        == "Something went wrong while calculating Polynomial Features shapes!"
    )
    # TODO: Update the error message with a more informative one!


def test_indexer_constraints_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_shapes = {"input": ["a", "(V1, ...)"], "output": ["(V1, ...)"], "index": []}
    scalar_info = {"index": IOHyperEdge(value=1)}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"input"},
        scalar_info,
    )


def test_indexer_constraints_2():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "u3", "(V1, ...)"],
        "output": ["(V1, ...)"],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(1, 2, 3))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"input"},
        scalar_info,
    )


def test_indexer_constraints_3():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", ("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "u3", "(V1, ...)"],
        "output": ["(V1, ...)"],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(1, 2, 3))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"input"},
        scalar_info,
    )


def test_indexer_constraints_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "(V1, ...)"],
        "output": [1, 1, "(V1, ...)"],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(None, None, 3))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"input", "output"},
        scalar_info,
    )


def test_indexer_constraints_5():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [10, ("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_shapes = {"input": [10, "(V1, ...)"], "output": [3, "(V1, ...)"], "index": []}
    scalar_info = {"index": IOHyperEdge(value=slice(2, 5, None))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_6():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", "u3"],
        "output": [("Var1", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "u3"],
        "output": ["u4", "u5", "u3"],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(slice(2, 5, None), slice(2, 5, None)))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_7():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
        "output": [("Var1", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
        "output": ["u4", "u5", "u6"],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(1, 2, 3, ..., 1, 0))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_8():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
        "output": [("Var1", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
        "output": [1, 1, 1, 1, "u4", "u5", "u6", 1, 1],
        "index": [],
    }
    scalar_info = {
        "index": IOHyperEdge(
            value=(1, None, None, 2, None, 3, None, ..., None, 1, 0, None)
        )
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_9():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
        "output": [("Var1", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
        "output": ["u9", 1, 1, 1, "u5", "u6", 1, "u10", 1],
        "index": [],
    }
    scalar_info = {
        "index": IOHyperEdge(
            value=(
                1,
                slice(2, None, None),
                None,
                2,
                None,
                3,
                None,
                ...,
                None,
                slice(2, None, None),
                0,
                None,
            )
        )
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_10():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [8, 9, 10, 11, 12, 13, 14, 15],
        "output": [("Var1", ...)],
    }
    final_shapes = {
        "input": [8, 9, 10, 11, 12, 13, 14, 15],
        "output": [7, 1, 1, 1, 12, 13, 1, 12, 1],
        "index": [],
    }
    scalar_info = {
        "index": IOHyperEdge(
            value=(
                1,
                slice(2, None, None),
                None,
                2,
                None,
                3,
                None,
                ...,
                None,
                slice(2, None, None),
                0,
                None,
            )
        )
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_11():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", "u3", ("Var1", ...), "u4", "u5", "u6"],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "u3", "(V1, ...)", "u4", "u5", "u6"],
        "output": [1, "(V1, ...)"],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(1, 2, None, 3, ..., 2, 3, 4))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_12():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...), "u1", "u2"],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["(V1, ...)", "u1", "u2"],
        "output": ["u3", 1, "u4", "(V2, ...)"],
        "index": [],
    }
    scalar_info = {
        "index": IOHyperEdge(
            value=(slice(None, None, None), None, slice(None, None, None))
        )
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_13():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...), "u1", "u2"],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["(V1, ...)", "u1", "u2"],
        "output": ["u3", "u4", "(V2, ...)"],
        "index": [],
    }
    scalar_info = {
        "index": IOHyperEdge(value=(slice(2, None, None), slice(3, None, None)))
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_14():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", ("Var1", ...), "u2"],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "(V1, ...)", "u3"],
        "output": [1, 1, "u4", "(V2, ...)"],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(3, 4, None, None, slice(2, None, None)))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"output", "input"},
        scalar_info,
    )


def test_indexer_constraints_15():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", ("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "(V1, ...)"],
        "output": ["(V2, ...)", "u3", 1, "u4"],
        "index": [],
    }
    scalar_info = {
        "index": IOHyperEdge(
            value=(..., slice(None, None, None), None, slice(2, None, None))
        )
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_16():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...), "u1", "u2"],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["(V1, ...)", "u1", "u2"],
        "output": ["(V1, ...)", "u1", "u2", 1],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(..., None))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_17():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...), "u1", "u2"],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["(V1, ...)", "u1", "u2"],
        "output": ["(V1, ...)", "u1", 1, "u3", 1],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(..., None, slice(2, None, None), None))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_18():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...), "u1", "u2"],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["(V1, ...)", "u1", "u2"],
        "output": ["(V1, ...)", "u1", 1, 1],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(..., None, 2, None))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_19():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", ("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "(V1, ...)", "u2"],
        "output": ["(V2, ...)", 1, 1],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(..., None, 2, None, 3))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_indexer_constraints_20():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", ("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "(V1, ...)", "u2"],
        "output": ["(V2, ...)", 1, 1, 1, "u3", "u4"],
        "index": [],
    }
    scalar_info = {
        "index": IOHyperEdge(
            value=(..., None, None, None, slice(2, None, None), slice(2, None, None))
        )
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_indexer_constraints_21():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", ("Var1", ...), 7],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "(V1, ...)", "u2", 7],
        "output": ["(V2, ...)", 1, 1, 1, "u3", "u4", 5],
        "index": [],
    }
    scalar_info = {
        "index": IOHyperEdge(
            value=(
                ...,
                None,
                None,
                None,
                slice(2, None, None),
                slice(2, None, None),
                slice(2, None, None),
            )
        )
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        False,
        {"input", "output"},
        scalar_info,
    )


def test_indexer_constraints_22():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", ("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "u2", "(V1, ...)"],
        "output": [1, 1, "u1", "u2", "(V1, ...)"],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(None, None))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_23():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", ("Var1", ...), "u2"],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": ["u1", "(V1, ...)", "u2"],
        "output": [1, "u1", "(V1, ...)", "u2", 1],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(None, ..., None))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_24():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [3, 4, 5, 6, 7],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": [3, 4, 5, 6, 7],
        "output": [1, 3, 4, 1, 3, 2],
        "index": [],
    }
    scalar_info = {
        "index": IOHyperEdge(
            value=(None, ..., None, 1, slice(2, 5, None), slice(2, 4, None))
        )
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_indexer_constraints_25():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [10, 1, 2],
        "output": ["u1", 1, 2],
    }
    final_shapes = {
        "input": [10, 1, 2],
        "output": [5, 1, 2],
        "index": [],
    }
    scalar_info = {"index": IOHyperEdge(value=(slice(5, None, None)))}
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        indexer_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_split_constraints_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [6, 4, 5],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": [6, 4, 5],
        "output": [3, 2, 4, 5],
        "split_size": [],
        "axis": [],
    }
    scalar_info = {
        "split_size": IOHyperEdge(value=3),
        "axis": IOHyperEdge(value=0),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        split_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_split_constraints_2():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [6, 4, 5],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": [6, 4, 5],
        "output": [2, 6, 2, 5],
        "split_size": [],
        "axis": [],
    }
    scalar_info = {
        "split_size": IOHyperEdge(value=2),
        "axis": IOHyperEdge(value=1),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        split_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_split_constraints_3():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [6, 4, 5],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": [6, 4, 5],
        "output": [1, 6, 4, 5],
        "split_size": [],
        "axis": [],
    }
    scalar_info = {
        "split_size": IOHyperEdge(value=1),
        "axis": IOHyperEdge(value=1),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        split_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_split_constraints_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [6, 4, 8],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": [6, 4, 8],
        "output": [2, 6, 4, 4],
        "split_size": [],
        "axis": [],
    }
    scalar_info = {
        "split_size": IOHyperEdge(value=2),
        "axis": IOHyperEdge(value=-1),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        split_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_split_constraints_5():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [6, 4, ("Var1", ...), 8],
        "output": [("Var2", ...)],
    }
    final_shapes = {
        "input": [6, 4, "(V1, ...)", 8],
        "output": [2, 6, 4, "(V1, ...)", 4],
        "split_size": [],
        "axis": [],
    }
    scalar_info = {
        "split_size": IOHyperEdge(value=2),
        "axis": IOHyperEdge(value=-1),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        split_constraints,
        True,
        {"output"},
        scalar_info,
    )


def test_scalar_item_type_constraints_1():
    final_types = {"output": list[int], "input": list[list[int]], "index": int}
    scalar_info = {
        "index": IOHyperEdge(value=3),
        "input": IOHyperEdge(type=list[list[int]]),
        "output": IOHyperEdge(type=list),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        True,
        {"output"},
        scalar_info,
    )


def test_scalar_item_type_constraints_2():
    final_types = {
        "output": list[float],
        "input": list[list[int | float]],
        "index": int,
    }
    scalar_info = {
        "index": IOHyperEdge(value=3),
        "input": IOHyperEdge(type=list[list[int | float]]),
        "output": IOHyperEdge(type=list[float]),
    }
    assert_constraint_results(
        {}, {}, final_types, {}, indexer_type_constraint, True, set(), scalar_info
    )


def test_scalar_item_type_constraints_3():
    final_types = {
        "index": int,
        "input": list[list] | tuple[list, ...],
        "output": list[float],
    }
    scalar_info = {
        "index": IOHyperEdge(value=3),
        "input": IOHyperEdge(type=list | tuple),
        "output": IOHyperEdge(type=list[float]),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        True,
        {"input"},
        scalar_info,
    )


def test_scalar_item_type_constraints_3_1():
    final_types = {
        "index": int,
        "input": list[list] | tuple[list[float], ...],
        "output": list[float],
    }
    scalar_info = {
        "index": IOHyperEdge(value=3),
        "input": IOHyperEdge(type=list | tuple[list[float], ...]),
        "output": IOHyperEdge(type=list[float]),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        True,
        {"input"},
        scalar_info,
    )


def test_scalar_item_type_constraints_4():
    final_types = {
        "index": int,
        "input": list[tuple[list[int | str], ...]],
        "output": tuple[list[int | str], ...],
    }
    scalar_info = {
        "index": IOHyperEdge(value=3),
        "input": IOHyperEdge(type=list[tuple[list[int | str], ...]]),
        "output": IOHyperEdge(type=tuple[list[int | str | bool], ...]),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        False,
        {"output"},
        scalar_info,
    )


def test_scalar_item_type_constraints_5():
    final_types = {
        "index": int,
        "input": tuple[int, float, str, int, float],
        "output": int | float,
    }
    scalar_info = {
        "index": IOHyperEdge(type=int, value=TBD),
        "input": IOHyperEdge(type=tuple[int, float, str, int, float]),
        "output": IOHyperEdge(type=int | bool | float),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        False,
        {"output"},
        scalar_info,
    )


def test_scalar_item_type_constraints_6():
    final_types = {"index": int, "input": list[int], "output": int}
    scalar_info = {
        "index": IOHyperEdge(
            type=int,
            value=TBD,
        ),
        "input": IOHyperEdge(type=list[int] | list[float]),
        "output": IOHyperEdge(type=int),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        True,
        {"input"},
        scalar_info,
    )


def test_scalar_item_type_constraints_7():
    final_types = {
        "index": int,
        "input": tuple[int, float, bool, float, int],
        "output": bool,
    }
    scalar_info = {
        "index": IOHyperEdge(value=2, type=int),
        "input": IOHyperEdge(type=tuple[int, float, bool, float, int]),
        "output": IOHyperEdge(type=int | float | bool),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        True,
        {"output"},
        scalar_info,
    )


def test_scalar_item_type_constraints_8():
    index_type = int
    input_type: type[tuple] = tuple[int, float, float, int]
    output_type = int | bool

    final_index_type = int
    final_input_type = tuple[int, float, float, int]
    final_output_type = int

    final_types = {
        "index": final_index_type,
        "input": final_input_type,
        "output": final_output_type,
    }
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=index_type),
        "input": IOHyperEdge(type=input_type),
        "output": IOHyperEdge(type=output_type),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        True,
        {"output"},
        scalar_info,
    )


def test_scalar_item_type_constraints_9():
    index_type = int
    input_type: type[tuple] = tuple[int, int, int, int]
    output_type = int | float | bool

    final_index_type = int
    final_input_type = tuple[int, int, int, int]
    final_output_type = int

    final_types = {
        "index": final_index_type,
        "input": final_input_type,
        "output": final_output_type,
    }
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=index_type),
        "input": IOHyperEdge(type=input_type),
        "output": IOHyperEdge(type=output_type),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        True,
        {"output"},
        scalar_info,
    )


def test_scalar_item_type_constraints_10():
    index_type = int
    input_type = list
    output_type = int

    final_index_type = int
    final_input_type = list
    final_output_type = int

    final_types = {
        "index": final_index_type,
        "input": final_input_type,
        "output": final_output_type,
    }
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=index_type),
        "input": IOHyperEdge(type=input_type),
        "output": IOHyperEdge(type=output_type),
    }
    assert_constraint_results(
        {}, {}, final_types, {}, indexer_type_constraint, True, set(), scalar_info
    )


def test_scalar_item_type_constraints_11():
    index_type = int
    input_type = tuple
    output_type = int

    final_index_type = int
    final_input_type = tuple
    final_output_type = int

    final_types = {
        "index": final_index_type,
        "input": final_input_type,
        "output": final_output_type,
    }
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=index_type),
        "input": IOHyperEdge(type=input_type),
        "output": IOHyperEdge(type=output_type),
    }
    assert_constraint_results(
        {}, {}, final_types, {}, indexer_type_constraint, True, set(), scalar_info
    )


def test_scalar_item_type_constraints_12():
    index_type = int
    input_type = list[list[int]] | list[int]
    output_type = list[int]

    final_index_type = int
    final_input_type = list[list[int]]
    final_output_type = list[int]

    final_types = {
        "index": final_index_type,
        "input": final_input_type,
        "output": final_output_type,
    }
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=index_type),
        "input": IOHyperEdge(type=input_type),
        "output": IOHyperEdge(type=output_type),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        True,
        {"input"},
        scalar_info,
    )


def test_scalar_item_type_constraints_13():
    index_type = int
    input_type = list[list[int]] | list[int] | list[float]
    output_type = int | float

    final_index_type = int
    final_input_type = list[int] | list[float]
    final_output_type = int | float

    final_types = {
        "index": final_index_type,
        "input": final_input_type,
        "output": final_output_type,
    }
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=index_type),
        "input": IOHyperEdge(type=input_type),
        "output": IOHyperEdge(type=output_type),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        False,
        {"input"},
        scalar_info,
    )


def test_scalar_item_type_constraints_14():
    final_types = {"index": int, "input": list[int], "output": int}
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=int),
        "input": IOHyperEdge(type=list[int]),
        "output": IOHyperEdge(type=int),
    }
    assert_constraint_results(
        {}, {}, final_types, {}, indexer_type_constraint, True, set(), scalar_info
    )


def test_scalar_item_type_constraints_15():
    final_types = {"index": int, "input": list[float], "output": float}
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=int),
        "input": IOHyperEdge(type=list[int] | list[float]),
        "output": IOHyperEdge(type=float),
    }
    assert_constraint_results(
        {}, {}, final_types, {}, indexer_type_constraint, True, set(), scalar_info
    )


def test_scalar_item_type_constraints_16():
    final_types = {
        "index": int,
        "input": list[int] | list[float],
        "output": int | float,
    }
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=int),
        "input": IOHyperEdge(type=list[int] | list[float]),
        "output": IOHyperEdge(type=int | float),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        False,
        {"input"},
        scalar_info,
    )


def test_scalar_item_type_constraints_17():
    final_types = {
        "index": int,
        "input": list[int] | list[float],
        "output": int | float,
    }
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=int),
        "input": IOHyperEdge(type=list[int] | list[float]),
        "output": IOHyperEdge(type=int | float),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        False,
        {"input"},
        scalar_info,
    )


def test_scalar_item_type_constraints_18():
    final_types = {
        "index": int,
        "input": list[int] | list[float],
        "output": int | float,
    }
    scalar_info = {
        "index": IOHyperEdge(value=TBD, type=int),
        "input": IOHyperEdge(type=list[int] | list[float]),
        "output": IOHyperEdge(type=int | float),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        indexer_type_constraint,
        False,
        {"input"},
        scalar_info,
    )


def test_scalar_slice_type_constraints_1():
    final_types = {
        "start": NoneType,
        "stop": NoneType,
        "step": NoneType,
        "input": tuple[int, ...],
        "output": tuple[int, ...],
    }
    scalar_info = {
        "start": IOHyperEdge(value=None, type=int | NoneType),
        "stop": IOHyperEdge(value=None, type=int | NoneType),
        "step": IOHyperEdge(value=None, type=int | NoneType),
        "input": IOHyperEdge(type=tuple[int, ...]),
        "output": IOHyperEdge(type=tuple),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        scalar_slice_type_constraint,
        False,
        {"output"},
        scalar_info,
    )


def test_scalar_slice_type_constraints_2():
    final_types = {
        "start": NoneType,
        "stop": NoneType,
        "step": NoneType,
        "input": tuple[int, int, int, int],
        "output": tuple[int, int, int, int],
    }
    scalar_info = {
        "start": IOHyperEdge(value=None, type=int | NoneType),
        "stop": IOHyperEdge(value=None, type=int | NoneType),
        "step": IOHyperEdge(value=None, type=int | NoneType),
        "input": IOHyperEdge(type=tuple[int, int, int, int]),
        "output": IOHyperEdge(type=tuple | list | int),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        scalar_slice_type_constraint,
        True,
        {"output"},
        scalar_info,
    )


def test_scalar_slice_type_constraints_3():
    final_types = {
        "start": NoneType,
        "stop": NoneType,
        "step": NoneType,
        "input": list[list[list[int]]],
        "output": list[list[list[int]]],
    }
    scalar_info = {
        "start": IOHyperEdge(value=None, type=int | NoneType),
        "stop": IOHyperEdge(value=None, type=int | NoneType),
        "step": IOHyperEdge(value=None, type=int | NoneType),
        "input": IOHyperEdge(type=list[list[list[int]]]),
        "output": IOHyperEdge(type=list | tuple),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        scalar_slice_type_constraint,
        True,
        {"output"},
        scalar_info,
    )


def test_scalar_slice_type_constraints_4():
    final_types = {
        "start": NoneType,
        "stop": NoneType,
        "step": NoneType,
        "input": tuple[int, int, int],
        "output": tuple[int, int, int],
    }
    scalar_info = {
        "start": IOHyperEdge(value=None, type=int | NoneType),
        "stop": IOHyperEdge(value=None, type=int | NoneType),
        "step": IOHyperEdge(value=None, type=int | NoneType),
        "input": IOHyperEdge(type=tuple),
        "output": IOHyperEdge(type=tuple[int, int, int]),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        scalar_slice_type_constraint,
        True,
        {"input"},
        scalar_info,
    )


def test_scalar_slice_type_constraints_5():
    final_types = {
        "start": int | NoneType,
        "stop": int | NoneType,
        "step": int | NoneType,
        "input": tuple,
        "output": tuple[int, int, int],
    }
    scalar_info = {
        "start": IOHyperEdge(value=TBD, type=int | NoneType),
        "stop": IOHyperEdge(value=TBD, type=int | NoneType),
        "step": IOHyperEdge(value=TBD, type=int | NoneType),
        "input": IOHyperEdge(type=tuple),
        "output": IOHyperEdge(type=tuple[int, int, int]),
    }
    assert_constraint_results(
        {}, {}, final_types, {}, scalar_slice_type_constraint, True, set(), scalar_info
    )


def test_scalar_slice_type_constraints_6():
    final_types = {
        "start": int | NoneType,
        "stop": int | NoneType,
        "step": int | NoneType,
        "input": tuple,
        "output": tuple[int, int, int],
    }
    scalar_info = {
        "start": IOHyperEdge(value=TBD, type=int | NoneType),
        "stop": IOHyperEdge(value=TBD, type=int | NoneType),
        "step": IOHyperEdge(value=TBD, type=int | NoneType),
        "input": IOHyperEdge(type=tuple | list | int),
        "output": IOHyperEdge(type=tuple[int, int, int]),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        scalar_slice_type_constraint,
        True,
        {"input"},
        scalar_info,
    )


def test_scalar_slice_type_constraints_7():
    final_types = {
        "start": int,
        "stop": int,
        "step": NoneType,
        "input": tuple[int, float, list, int, int],
        "output": tuple[int, float],
    }
    scalar_info = {
        "start": IOHyperEdge(value=0),
        "stop": IOHyperEdge(value=2),
        "step": IOHyperEdge(value=None, type=int | NoneType),
        "input": IOHyperEdge(type=tuple[int, float, list, int, int]),
        "output": IOHyperEdge(type=tuple),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        scalar_slice_type_constraint,
        True,
        {"output"},
        scalar_info,
    )


def test_scalar_slice_type_constraints_8():
    final_types = {
        "start": NoneType,
        "stop": NoneType,
        "step": NoneType,
        "input": tuple[int, int, int],
        "output": tuple[int, int, int],
    }
    scalar_info = {
        "start": IOHyperEdge(value=None, type=int | NoneType),
        "stop": IOHyperEdge(value=None, type=int | NoneType),
        "step": IOHyperEdge(value=None, type=int | NoneType),
        "input": IOHyperEdge(type=tuple | list | bool | str),
        "output": IOHyperEdge(type=tuple[int, int, int]),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        scalar_slice_type_constraint,
        True,
        {"input"},
        scalar_info,
    )


def test_scalar_slice_type_constraints_9():
    final_types = {
        "start": NoneType,
        "stop": NoneType,
        "step": NoneType,
        "input": list[list[int | float]],
        "output": list[list[int | float]],
    }
    scalar_info = {
        "start": IOHyperEdge(value=None, type=int | NoneType),
        "stop": IOHyperEdge(value=None, type=int | NoneType),
        "step": IOHyperEdge(value=None, type=int | NoneType),
        "input": IOHyperEdge(type=list | tuple),
        "output": IOHyperEdge(type=list[list[int | float]]),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        scalar_slice_type_constraint,
        False,
        {"input"},
        scalar_info,
    )


def test_scalar_slice_type_constraints_10():
    final_types = {
        "start": NoneType,
        "stop": NoneType,
        "step": NoneType,
        "input": list[list[int] | list[float]],
        "output": list[list[int] | list[float]],
    }
    scalar_info = {
        "start": IOHyperEdge(value=None, type=int | NoneType),
        "stop": IOHyperEdge(value=None, type=int | NoneType),
        "step": IOHyperEdge(value=None, type=int | NoneType),
        "input": IOHyperEdge(type=list[list[int] | list[float]]),
        "output": IOHyperEdge(type=list[list[int | float]]),
    }
    assert_constraint_results(
        {},
        {},
        final_types,
        {},
        scalar_slice_type_constraint,
        False,
        {"output"},
        scalar_info,
    )


def test_tensor_to_list_type_constraints_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", ("Var1", ...), "u2"],
    }
    scalar_info = {
        "output": IOHyperEdge(type=TensorToListType, value=TBD),
    }
    final_types = {
        "output": generate_nested_list_type(int | float | bool, min_depth=2),
        "input": int | float | bool,
    }
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        tensor_to_list_type_constraint,
        False,
        {"output"},
        scalar_info,
    )


def test_tensor_to_list_type_constraints_2():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", "u2"],
    }
    scalar_info = {
        "output": IOHyperEdge(type=TensorToListType, value=TBD),
    }
    final_types = {
        "output": generate_nested_list_type(
            int | float | bool, min_depth=3, max_depth=3
        ),
        "input": int | float | bool,
    }
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        tensor_to_list_type_constraint,
        True,
        {"output"},
        scalar_info,
    )


def test_tensor_to_list_type_constraints_3():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1", "u2", ("Var1", ...), "u3"],
    }
    scalar_info = {
        "output": IOHyperEdge(type=generate_nested_list_type(int, 4, 4), value=TBD),
    }
    final_types = {"output": list[list[list[list[int]]]], "input": int}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        tensor_to_list_type_constraint,
        True,
        {"input"},
        scalar_info,
    )


def test_tensor_to_list_type_constraints_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": ["u1"],
    }
    scalar_info = {
        "output": IOHyperEdge(type=list[int | float], value=TBD),
    }
    final_types = {"output": list[int | float], "input": int | float}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        tensor_to_list_type_constraint,
        True,
        {"input"},
        scalar_info,
    )


def test_reduce_type_constraint_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...)],
        "output": [("Var2", ...)],
    }
    final_types = {"input": int | float | bool, "output": int | float}
    assert_constraint_results(
        shapes, {}, final_types, {}, reduce_type_constraint, False, {"output"}
    )


def test_reduce_type_constraint_2():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...)],
        "output": [("Var2", ...)],
    }
    initial_types: dict[str, type | UnionType] = {
        "input": int,
        "output": int | float | bool,
    }
    final_types = {"input": int, "output": int}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        reduce_type_constraint,
        True,
        {"output"},
        initial_types=initial_types,
    )


def test_reduce_type_constraint_3():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...)],
        "output": [("Var2", ...)],
    }
    initial_types = {"input": int | float, "output": int | float | bool}
    final_types = {"input": int | float, "output": int | float}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        reduce_type_constraint,
        False,
        {"output"},
        initial_types=initial_types,
    )


def test_reduce_type_constraint_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...)],
        "output": [("Var2", ...)],
    }
    initial_types: dict[str, type | UnionType] = {
        "input": int | float | bool,
        "output": float,
    }
    final_types = {"input": float, "output": float}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        reduce_type_constraint,
        True,
        {"input"},
        initial_types=initial_types,
    )


def test_reduce_type_constraint_5():
    shapes: dict[str, list[int | str | tuple]] = {
        "input": [("Var1", ...)],
        "output": [("Var2", ...)],
    }
    initial_types: dict[str, type | UnionType] = {
        "input": bool,
        "output": int | float | bool,
    }
    final_types = {"input": bool, "output": int}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        reduce_type_constraint,
        True,
        {"output"},
        initial_types=initial_types,
    )


def test_general_tensor_type_constraint_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var2", ...)],
        "input": [("Var1", ...)],
    }
    initial_types: dict[str, type | UnionType] = {
        "input": bool,
        "output": int | float | bool,
    }
    final_types = {"input": bool, "output": bool}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        general_tensor_type_constraint,
        True,
        {"output"},
        initial_types=initial_types,
        variadic_fn=True,
    )


def test_general_tensor_type_constraint_2():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var2", ...)],
        "input": [("Var1", ...)],
    }
    initial_types = {"input": bool, "output": bool}
    final_types = {"input": bool, "output": bool}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        general_tensor_type_constraint,
        True,
        set(),
        initial_types=initial_types,
        variadic_fn=True,
    )


def test_general_tensor_type_constraint_3():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var2", ...)],
        "input1": [("Var1", ...)],
        "input2": [("Var1", ...)],
        "input3": [("Var1", ...)],
        "input4": [("Var1", ...)],
    }
    initial_types: dict[str, type | UnionType] = {
        "input1": int | float | bool,
        "input2": int | float | bool,
        "input3": int | float | bool,
        "input4": int | float | bool,
        "output": bool,
    }
    final_types = {
        "input1": bool,
        "input2": bool,
        "input3": bool,
        "input4": bool,
        "output": bool,
    }
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        general_tensor_type_constraint,
        True,
        {"input1", "input2", "input3", "input4"},
        initial_types=initial_types,
        variadic_fn=True,
    )


def test_general_tensor_type_constraint_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var2", ...)],
        "input1": [("Var1", ...)],
        "input2": [("Var1", ...)],
        "input3": [("Var1", ...)],
        "input4": [("Var1", ...)],
    }
    initial_types = {
        "input1": int | float | bool,
        "input2": int | float | bool,
        "input3": int | float | bool,
        "input4": int | float | bool,
        "output": int | bool,
    }
    final_types = {
        "input1": int | bool,
        "input2": int | bool,
        "input3": int | bool,
        "input4": int | bool,
        "output": int | bool,
    }
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        general_tensor_type_constraint,
        False,
        {"input1", "input2", "input3", "input4"},
        initial_types=initial_types,
        variadic_fn=True,
    )


def test_general_tensor_type_constraint_5():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var2", ...)],
        "input1": [("Var1", ...)],
        "input2": [("Var1", ...)],
        "input3": [("Var1", ...)],
        "input4": [("Var1", ...)],
    }
    initial_types: dict[str, type | UnionType] = {
        "input1": bool,
        "input2": bool,
        "input3": bool,
        "input4": int | float | bool,
        "output": int,
    }
    final_types = {
        "input1": bool,
        "input2": bool,
        "input3": bool,
        "input4": int,
        "output": int,
    }
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        general_tensor_type_constraint,
        True,
        {"input4"},
        initial_types=initial_types,
        variadic_fn=True,
    )


def test_general_tensor_type_constraint_6():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var2", ...)],
        "input1": [("Var1", ...)],
        "input2": [("Var1", ...)],
        "input3": [("Var1", ...)],
        "input4": [("Var1", ...)],
    }
    initial_types: dict[str, type | UnionType] = {
        "input1": int,
        "input2": bool,
        "input3": bool | int,
        "input4": int | float | bool,
        "output": float,
    }
    final_types = {
        "input1": int,
        "input2": bool,
        "input3": bool | int,
        "input4": float,
        "output": float,
    }
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        general_tensor_type_constraint,
        True,
        {"input4"},
        initial_types=initial_types,
        variadic_fn=True,
    )


def test_general_tensor_type_constraint_8():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [("Var2", ...)],
        "input1": [("Var1", ...)],
        "input2": [("Var1", ...)],
        "input3": [("Var1", ...)],
        "input4": [("Var1", ...)],
    }
    initial_types: dict[str, type | UnionType] = {
        "input1": float,
        "input2": int | bool | float,
        "input3": int | bool | float,
        "input4": int | bool | float,
        "output": int | bool | float,
    }
    final_types = {
        "input1": float,
        "input2": int | bool | float,
        "input3": int | bool | float,
        "input4": int | bool | float,
        "output": float,
    }
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        general_tensor_type_constraint,
        True,
        {"output"},
        initial_types=initial_types,
        variadic_fn=True,
    )


def test_general_tensor_type_constraint_9():
    shapes = {
        "output": [("Var2", ...)],
        "input1": [("Var1", ...)],
        "input2": [("Var1", ...)],
    }
    initial_types = {
        "input1": int | bool,
        "input2": float | int,
        "output": int | bool | float,
    }
    final_types = {"input1": int | bool, "input2": float | int, "output": float | int}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        general_tensor_type_constraint,
        False,
        {"output"},
        initial_types=initial_types,
        variadic_fn=True,
    )


def test_general_tensor_type_constraint_10():
    shapes = {
        "output": [("Var2", ...)],
        "input1": [("Var1", ...)],
        "input2": [("Var1", ...)],
    }
    initial_types = {
        "input1": int | bool,
        "input2": int | bool,
        "output": int | bool | float,
    }
    final_types = {"input1": int | bool, "input2": int | bool, "output": int | bool}
    assert_constraint_results(
        shapes,
        {},
        final_types,
        {},
        general_tensor_type_constraint,
        False,
        {"output"},
        initial_types=initial_types,
        variadic_fn=True,
    )


def test_bcast_with_possibles_1():
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    assignments = {
        "x": {1, 3},
        "y": {1, 5},
    }

    final_shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    final_assignments = {"z": {1, 5, 3}, "y": {1, 5}, "x": {1, 3}}

    assert_constraint_results(
        shapes, assignments, final_shapes, final_assignments, bcast, False, {"output"}
    )


def test_bcast_with_possibles_2():
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    assignments = {"z": {3}}

    final_shapes = {"output": [3], "left": ["y"], "right": ["x"]}

    final_assignments = {"x": {3, 1}, "y": {3, 1}}

    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        final_assignments,
        bcast,
        False,
        {"left", "right"},
    )


def test_bcast_with_possibles_3():
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    assignments = {"z": {3}, "x": {1}, "y": {3}}

    final_shapes = {"output": [3], "left": [3], "right": [1]}

    final_assignments: AssignmentType = {}
    assert_constraint_results(
        shapes, assignments, final_shapes, final_assignments, bcast, True, {"output"}
    )


def test_bcast_with_possibles_4():
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    assignments = {"z": {8, 9, 10, 11}}

    final_shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    final_assignments = {
        "z": {8, 9, 10, 11},
        "x": {8, 9, 10, 11, 1},
        "y": {8, 9, 10, 11, 1},
    }

    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        final_assignments,
        bcast,
        False,
        {"left", "right"},
    )


def test_bcast_with_possibles_5():
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    assignments = {"z": {8, 9, 10, 11}, "y": {3, 4, 5, 6, 7, 8, 9}}

    final_shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    final_assignments = {"z": {8, 9}, "y": {8, 9}, "x": {1, 8, 9}}

    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        final_assignments,
        bcast,
        False,
        {"output", "left", "right"},
    )


def test_bcast_with_possibles_6():
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    assignments = {"z": {8, 9, 10, 11}, "y": {3, 4, 5, 6, 7, 8, 9}, "x": {8, 9, 10, 11}}

    final_shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    final_assignments = {"z": {8, 9}, "y": {8, 9}, "x": {8, 9}}

    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        final_assignments,
        bcast,
        False,
        {"output", "left", "right"},
    )


def test_bcast_with_possibles_7():
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    assignments = {"y": {1, 3}, "x": {1, 5}}

    final_shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    final_assignments = {"z": {1, 3, 5}, "y": {1, 3}, "x": {1, 5}}

    assert_constraint_results(
        shapes, assignments, final_shapes, final_assignments, bcast, False, {"output"}
    )


def test_bcast_with_possibles_8():
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    assignments = {"z": {1, 3, 4, 5, 6, 7}, "y": {6, 7, 8}, "x": {1, 10, 11}}

    final_shapes = {"output": ["z"], "left": ["z"], "right": [1]}

    final_assignments = {
        "z": {6, 7},
    }

    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        final_assignments,
        bcast,
        True,
        {"output", "right"},
    )


def test_bcast_with_possibles_8_error():
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}

    assignments = {"z": {1, 3, 4, 5, 6, 7}, "y": {6, 7, 8}, "x": {1, 10, 11}}

    final_shapes = {"output": ["z"], "left": ["z"], "right": [1]}

    final_assignments = {
        "z": {6, 7, 9},
    }
    with pytest.raises(AssertionError):
        assert_constraint_results(
            shapes,
            assignments,
            final_shapes,
            final_assignments,
            bcast,
            True,
            {"output", "right"},
        )


@pytest.mark.skip("Fails in idempotency checks")
def test_bcast_with_possibles_9():
    shapes = {"output": ["a", "b"], "left": ["c", "d"], "right": ["x", "y"]}

    assignments = {
        "a": {1, 3, 4},
        "c": {1, 7, 8},
        "x": {1, 3, 4},
        "b": {2, 3, 4},
        "d": {3, 4, 5},
        "y": {4, 5, 6},
    }

    final_shapes = {"output": ["a", 4], "left": [1, 4], "right": ["a", 4]}

    final_assignments = {
        "a": {1, 3, 4},
    }

    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        final_assignments,
        bcast,
        True,
        {"output", "left", "right"},
    )


def test_bcast_with_var_possibles_1():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a"],
        "left": [("V1", ...)],
        "right": ["b"],
    }

    assignments: dict[
        tuple[str, EllipsisType] | str, set[int] | list[tuple[str, ...]]
    ] = {
        "a": {7, 8},
        ("V1", ...): [(), ("c",)],
        "b": {1, 2, 3},
        "c": {7, 9},
    }

    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [7],
        "left": [7],
        "right": [1],
    }

    final_assignments: AssignmentType = {}

    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        final_assignments,
        bcast,
        True,
        {"output", "left", "right"},
    )


@pytest.mark.skip("Bcast possible inference bug")
def test_bcast_with_var_possibles_2():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": ["a"],
        "left": [("V1", ...)],
        "right": ["b"],
    }

    assignments: dict[
        tuple[str, EllipsisType] | str, set[int] | list[tuple[str, ...]]
    ] = {
        "a": {7, 8},
        ("V1", ...): [(), ("c",)],
        "b": {1, 2, 3, 7},
        "c": {1, 2, 3, 7, 9},
    }

    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [7],
        "left": [("V1", ...)],
        "right": ["b"],
    }

    final_assignments: dict[
        tuple[str, EllipsisType] | str, set[int] | list[tuple[str, ...]]
    ] = {
        ("V1", ...): [(), ("c",)],
        "b": {1, 7},
        "c": {1, 7},
    }

    assert_constraint_results(
        shapes,
        assignments,
        final_shapes,
        final_assignments,
        bcast,
        False,
        {"left", "right"},
    )


@pytest.mark.skip("Bcast possible inference bug")
def test_bcast_with_var_possibles_3():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3],
        "left": [("V1", ...)],
        "right": [3],
    }

    assignments: AssignmentType = {}

    final_shapes = {"output": [3], "left": [("V1", ...)], "right": [3]}

    final_assignments: dict[
        tuple[str, EllipsisType] | str, set[int] | list[tuple[str, ...]]
    ] = {
        ("V1", ...): [(), ("a",)],
        "a": {1, 3},
    }

    assert_constraint_results(
        shapes, assignments, final_shapes, final_assignments, bcast, False, {"left"}
    )


@pytest.mark.skip("Bcast possible inference bug")
def test_bcast_with_var_possibles_4():
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [3],
        "left": [("V1", ...)],
        "right": [3],
    }

    assignments: AssignmentType = {}

    final_shapes = {"output": [3], "left": [("V1", ...)], "right": [3]}

    final_assignments: dict[
        tuple[str, EllipsisType] | str, set[int] | list[tuple[str, ...]]
    ] = {
        ("V1", ...): [(), ("a",)],
        "a": {1, 3},
    }

    assert_constraint_results(
        shapes, assignments, final_shapes, final_assignments, bcast, False, {"left"}
    )


def test_slice_given_input():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "start": [],
        "stop": [],
        "step": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=slice),
        "start": IOHyperEdge(type=int | None, value=1),
        "stop": IOHyperEdge(type=int | None, value=3),
        "step": IOHyperEdge(type=int | None, value=5),
    }
    final_values = {
        "output": slice(1, 3, 5),
        "start": 1,
        "stop": 3,
        "step": 5,
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        slice_constraints,
        True,
        {"output"},
        scalar_info,
        final_values,
    )


def test_slice_given_missing_input():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "start": [],
        "stop": [],
        "step": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=slice, value=TBD),
        "start": IOHyperEdge(type=int | None, value=1),
        "stop": IOHyperEdge(type=int | None, value=3),
        "step": IOHyperEdge(type=int | None, value=TBD),
    }
    final_values = {
        "output": TBD,
        "start": 1,
        "stop": 3,
        "step": TBD,
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        slice_constraints,
        False,
        set(),
        scalar_info,
        final_values,
    )


def test_slice_given_output():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "start": [],
        "stop": [],
        "step": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=slice, value=slice(1, 3, 5)),
        "start": IOHyperEdge(type=int | None, value=1),
        "stop": IOHyperEdge(type=int | None, value=3),
        "step": IOHyperEdge(type=int | None, value=TBD),
    }
    final_values = {
        "output": slice(1, 3, 5),
        "start": 1,
        "stop": 3,
        "step": 5,
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        slice_constraints,
        True,
        {"step"},
        scalar_info,
        final_values,
    )


def test_slice_given_output_missing_all_inputs():
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "start": [],
        "stop": [],
        "step": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=slice, value=slice(1, 3, 5)),
        "start": IOHyperEdge(type=int | None, value=TBD),
        "stop": IOHyperEdge(type=int | None, value=TBD),
        "step": IOHyperEdge(type=int | None, value=TBD),
    }
    final_values = {
        "output": slice(1, 3, 5),
        "start": 1,
        "stop": 3,
        "step": 5,
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        slice_constraints,
        True,
        {"start", "stop", "step"},
        scalar_info,
        final_values,
    )


def test_general_forward_add_scalar():
    add_constraint = partial(general_forward_constraint, callable=lambda x, y: x + y)
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "left": [],
        "right": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=int | float | bool, value=TBD),
        "left": IOHyperEdge(value=2.0),
        "right": IOHyperEdge(value=3.0),
    }
    final_values = {
        "output": 5.0,
        "left": 2.0,
        "right": 3.0,
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        add_constraint,
        True,
        {"output"},
        scalar_info,
        final_values,
        variadic_fn=True,
    )


def test_general_forward_add_tensor():
    add_constraint = partial(general_forward_constraint, callable=lambda x, y: x + y)
    shapes = {"output": ["z"], "left": ["y"], "right": ["x"]}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": ["z"],
        "left": ["y"],
        "right": ["x"],
    }
    assert_constraint_results(
        shapes, {}, final_shapes, {}, add_constraint, True, set(), variadic_fn=True
    )


def test_general_forward_add_mixed():
    add_constraint = partial(general_forward_constraint, callable=lambda x, y: x + y)
    shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "left": ["c"],
        "right": [],
    }
    final_shapes = {"output": [], "left": ["c"], "right": []}
    scalar_info = {
        "right": IOHyperEdge(value=(1.0)),
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        add_constraint,
        True,
        set(),
        scalar_info,
        variadic_fn=True,
    )


def test_general_forward_add_siso():
    siso_constraint = partial(general_forward_constraint, callable=lambda x: 3 * x)
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=int | float | bool, value=TBD),
        "input": IOHyperEdge(value=2.0),
    }
    final_values = {
        "output": 6.0,
        "input": 2.0,
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        siso_constraint,
        True,
        {"output"},
        scalar_info,
        final_values,
        variadic_fn=True,
    )


def test_general_forward_add_3_inputs_status_true():
    three_input_constraint = partial(
        general_forward_constraint, callable=lambda x, y, z: 2 * x + 3 * y + z
    )
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input1": [],
        "input2": [],
        "input3": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=int | float | bool, value=TBD),
        "input1": IOHyperEdge(value=3.0),
        "input2": IOHyperEdge(value=4.0),
        "input3": IOHyperEdge(value=5.0),
    }
    final_values = {
        "output": 23.0,
        "input1": 3.0,
        "input2": 4.0,
        "input3": 5.0,
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        three_input_constraint,
        True,
        {"output"},
        scalar_info,
        final_values,
        variadic_fn=True,
    )


def test_general_forward_add_3_inputs_status_false():
    three_input_constraint = partial(
        general_forward_constraint, callable=lambda x, y, z: 2 * x + 3 * y + z
    )
    shapes: dict[str, list[int | str | tuple]] = {}
    final_shapes: dict[str, list[int | str | tuple]] = {
        "output": [],
        "input1": [],
        "input2": [],
        "input3": [],
    }
    scalar_info = {
        "output": IOHyperEdge(type=int | float | bool, value=TBD),
        "input1": IOHyperEdge(value=3.0),
        "input2": IOHyperEdge(type=int | float | bool, value=TBD),
        "input3": IOHyperEdge(value=5.0),
    }
    final_values = {
        "output": TBD,
        "input1": 3.0,
        "input2": TBD,
        "input3": 5.0,
    }
    assert_constraint_results(
        shapes,
        {},
        final_shapes,
        {},
        three_input_constraint,
        False,
        set(),
        scalar_info,
        final_values,
        variadic_fn=True,
    )
