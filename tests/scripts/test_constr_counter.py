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


import pytest

from mithril.framework.common import (
    NOT_GIVEN,
    TBD,
    BaseKey,
    IOHyperEdge,
    ShapeRepr,
    Tensor,
    Uniadic,
    Updates,
)
from mithril.framework.constraints import bcast, general_tensor_type_constraint
from mithril.models import (
    Add,
    Buffer,
    Connection,
    ConnectionType,
    ExtendInfo,
    Indexer,
    IOKey,
    Model,
    Operator,
    Relu,
    Slice,
    Transpose,
)
from mithril.models.primitives import PrimitiveModel


def dummy_constraint(output: IOHyperEdge, input: IOHyperEdge):
    # Dummy test constraint that is written for test purposes
    # it basically increment shapes by 1
    # updated_symbols = set()
    updates = Updates()
    status = False
    output_repr = output._temp_shape if output.is_tensor else output.value
    input_repr = input._temp_shape if input.is_tensor else input.value
    assert isinstance(output_repr, ShapeRepr)
    assert isinstance(input_repr, ShapeRepr)
    if bool(input_repr.root) ^ bool(output_repr.root):
        var_repr, non_var_repr = (
            (input_repr, output_repr) if input_repr.root else (output_repr, input_repr)
        )
        add_val = 1 if output_repr.root else -1
        uniadics = []
        values = [uni.value for uni in non_var_repr.prefix]
        uniadics = [
            Uniadic(val + add_val) if val is not None else Uniadic() for val in values
        ]
        updates |= var_repr.update_uniadics(var_repr.prefix, uniadics)
        updates |= var_repr.update_uniadics(var_repr.reverse, uniadics)
        updates |= var_repr.remove_variadic(uniadics)
        status = None not in values
    else:
        status = True
        for in_uni, out_uni in zip(input_repr.prefix, output_repr.prefix, strict=False):
            in_val, out_val = in_uni.value, out_uni.value
            if in_val is None and out_val is not None:
                in_uni.set_value(out_val - 1)
                updates.add(in_uni)
            elif in_val is not None and out_val is None:
                out_uni.set_value(in_val + 1)
                updates.add(out_uni)
            elif in_val is None and out_val is None:
                status = False
        if output_repr.root is not None:
            status = False
            for in_uni, out_uni in zip(
                input_repr.reverse, output_repr.reverse, strict=False
            ):
                in_val, out_val = in_uni.value, out_uni.value
                if in_val is None and out_val is not None:
                    in_uni.set_value(out_val - 1)
                    updates.add(in_uni)
                elif in_val is not None and out_val is None:
                    out_uni.set_value(in_val + 1)
                    updates.add(out_uni)
    return status, updates


class Model1(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=[("Var1", ...)], type=Tensor),
            output=BaseKey(
                shape=[("Var2", ...)],
                type=Tensor,
            ),
        )
        self._add_constraint(fn=dummy_constraint, keys=["output", "input"])


class Model2(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=[("Var1", ...)], type=Tensor),
            output=BaseKey(shape=[("Var2", ...)], type=Tensor),
        )
        self._add_constraint(fn=dummy_constraint, keys=["output", "input"])
        self._add_constraint(
            fn=general_tensor_type_constraint, keys=["output", "input"]
        )


class Model3(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=[("Var1", ...)], type=Tensor[int | bool]),
            output=BaseKey(shape=[("Var2", ...)], type=Tensor[int | bool]),
        )
        self._add_constraint(fn=dummy_constraint, keys=["output", "input"])


class MyAdd2(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(self, left, right, output) -> None:
        super().__init__(
            formula_key="add",
            output=BaseKey(shape=output, type=Tensor),
            left=BaseKey(shape=left, type=Tensor),
            right=BaseKey(shape=right, type=Tensor),
        )
        self._add_constraint(fn=bcast, keys=[Operator.output_key, "left", "right"])

    def __call__(  # type: ignore
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"left": left, "right": right, "output": output}
        return ExtendInfo(self, kwargs)


def assert_constr_counts(ref_dict: dict[Connection, list[int]]):
    for connection, result in ref_dict.items():
        call_list = [
            constr.call_counter for constr in connection.metadata.all_constraints
        ]
        assert result == sorted(call_list)


def make_reference_dict(
    ref_dict: dict[Connection, list[int]],
) -> dict[Connection, list[int]]:
    """This is a helper function which does nothing to its argument.
    We designed constraint counter tests in such a way where values
    of ref_dict are sorted numbers of calls to all constraints in corresponding
    connection. "assert_constr_counts" function also sorts those numbers of calls
    and assert with the expected values.

    For example if a connection.metadata has 2 constraints in it (i.e. for type
    and shape) we have a list with 2 elements in ref_dict. But note that it is not
    obvious which number is for which constraint. If we all know that type constraint
    is called once and shape for 2, we expecte result to be [1, 2]. If the opposite
    was the case (i.e. type is called 2 times while shape was once), then expected
    result would be again [1, 2] since we sort the list wrt. number of calls.

    Although this is not the most precise testing practise, it is quite powerful since
    we control changes in all extend or set_shape calls.
    """
    return ref_dict


# def assert_constr_counts(model_dict):
#     for m, ref_counts in model_dict.items():
#         for ref_count, constr in zip(ref_counts, m.constraints):
#             assert constr.counter == ref_count


def test_shape_constraint_counter_1():
    model = Model()
    model += (add := Add())
    # edge_type_constr = 1
    ref_dict = make_reference_dict(
        {
            add.left: [0, 0, 0, 0, 1],
            add.right: [0, 0, 0, 0, 1],
            add.output: [0, 0, 0, 0, 1],
        }
    )
    assert_constr_counts(ref_dict)
    add.set_shapes(left=[1, 2], right=[1, 2])
    # edge_type_constr solved, bcast solved, only general_type_constr
    ref_dict = make_reference_dict({add.left: [1], add.right: [1], add.output: [1]})

    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_2():
    model = Model()
    model |= (add1 := Add())
    model |= (add2 := Add())(left=add1.output)

    ref_dict = make_reference_dict(
        {
            add1.left: [0, 0, 0, 0, 1],
            add1.right: [0, 0, 0, 0, 1],
            add2.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            add2.right: [0, 0, 0, 0, 2],
            add2.output: [0, 0, 0, 0, 2],
        }
    )
    assert_constr_counts(ref_dict)

    add1.set_shapes(left=[1, 2, 9], right=[1, 2, 1])

    ref_dict = make_reference_dict(
        {
            add1.left: [1],
            add1.right: [1],
            add1.output: [0, 0, 0, 0, 1, 3],
            add2.left: [0, 0, 0, 0, 1, 3],
            add2.right: [0, 0, 0, 0, 3],
            add2.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)

    add2.set_shapes(left=["a", "b", "c"], right=["a", "b", "c"])

    ref_dict = make_reference_dict(
        {
            add1.left: [1],
            add1.right: [1],
            add1.output: [1, 1],
            add2.left: [1, 1],
            add2.right: [1],
            add2.output: [1],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_3():
    model = Model()
    model |= (add1 := Add())
    model |= (add2 := Add())(left=add1.output)
    model |= (add3 := Add())(left=add2.output)

    ref_dict = make_reference_dict(
        {
            add1.left: [0, 0, 0, 0, 1],
            add1.right: [0, 0, 0, 0, 1],
            add1.output: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            add2.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            add2.right: [0, 0, 0, 0, 2],
            add2.output: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            add3.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            add3.right: [0, 0, 0, 0, 2],
            add3.output: [0, 0, 0, 0, 2],
        }
    )
    assert_constr_counts(ref_dict)

    add1.set_shapes(left=[1, 2, 9], right=[1, 2, 1])
    ref_dict = make_reference_dict(
        {
            add1.left: [1],
            add1.right: [1],
            add1.output: [0, 0, 0, 0, 1, 3],
            add2.left: [0, 0, 0, 0, 1, 3],
            add2.right: [0, 0, 0, 0, 3],
            add2.output: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add3.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add3.right: [0, 0, 0, 0, 3],
            add3.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)

    add2.set_shapes(left=["a", "b", "c"], right=["a", "b", "c"])
    ref_dict = make_reference_dict(
        {
            add1.left: [1],
            add1.right: [1],
            add1.output: [1, 1],
            add2.left: [1, 1],
            add2.right: [1],
            add2.output: [0, 0, 0, 0, 1, 3],
            add3.left: [0, 0, 0, 0, 1, 3],
            add3.right: [0, 0, 0, 0, 3],
            add3.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_4():
    model = Model()
    model |= (add1 := Add())
    model |= (add2 := Add())(left=add1.output)
    model |= (add3 := Add())(left=add2.output)
    model |= (add4 := Add())(left=add3.output)

    model = Model()
    model |= Buffer()([IOKey("in2"), IOKey("in2")])

    ref_dict = make_reference_dict(
        {
            add1.left: [0, 0, 0, 0, 1],
            add1.right: [0, 0, 0, 0, 1],
            add2.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            add2.right: [0, 0, 0, 0, 2],
            add3.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            add3.right: [0, 0, 0, 0, 2],
            add4.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            add4.right: [0, 0, 0, 0, 2],
            add4.output: [0, 0, 0, 0, 2],
        }
    )
    assert_constr_counts(ref_dict)

    add1.set_shapes(left=[1, 2, 9], right=[1, 2, 1])
    ref_dict = make_reference_dict(
        {
            add1.left: [1],
            add1.right: [1],
            add2.left: [0, 0, 0, 0, 1, 3],
            add2.right: [0, 0, 0, 0, 3],
            add3.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add3.right: [0, 0, 0, 0, 3],
            add4.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add4.right: [0, 0, 0, 0, 3],
            add4.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)

    add2.set_shapes(left=["a", "b", "c"], right=["a", "b", "c"])
    ref_dict = make_reference_dict(
        {
            add1.left: [1],
            add1.right: [1],
            add2.left: [1, 1],
            add2.right: [1],
            add3.left: [0, 0, 0, 0, 1, 3],
            add3.right: [0, 0, 0, 0, 3],
            add4.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add4.right: [0, 0, 0, 0, 3],
            add4.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_5():
    model = Model()
    model |= (add1 := Add())
    model |= (add2 := Add())(left=add1.output)
    model |= (add3 := Add())(left=add2.output)
    model |= (add4 := Add())(left=add3.output)
    model |= (add5 := Add())(left=add4.output)
    model |= (add6 := Add())(left=add5.output)
    ref_dict = make_reference_dict(
        {
            add1.left: [0, 0, 0, 0, 1],
            add1.right: [0, 0, 0, 0, 1],
            add2.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            add2.right: [0, 0, 0, 0, 2],
            add3.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            add3.right: [0, 0, 0, 0, 2],
            add4.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            add4.right: [0, 0, 0, 0, 2],
            add5.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            add5.right: [0, 0, 0, 0, 2],
            add6.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            add6.right: [0, 0, 0, 0, 2],
            add6.output: [0, 0, 0, 0, 2],
        }
    )
    assert_constr_counts(ref_dict)

    add1.set_shapes(left = [1, 2, 9], right = [1, 2, 1])
    ref_dict = make_reference_dict(
        {
            add1.left: [1],
            add1.right: [1],
            add2.left: [0, 0, 0, 0, 1, 3],
            add2.right: [0, 0, 0, 0, 3],
            add3.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add3.right: [0, 0, 0, 0, 3],
            add4.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add4.right: [0, 0, 0, 0, 3],
            add5.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add5.right: [0, 0, 0, 0, 3],
            add6.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add6.right: [0, 0, 0, 0, 3],
            add6.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)

    add2.set_shapes(left = ["a", "b", "c"], right = ["a", "b", "c"])
    ref_dict = make_reference_dict(
        {
            add1.left: [1],
            add1.right: [1],
            add2.left: [1, 1],
            add1.right: [1],
            add3.left: [0, 0, 0, 0, 1, 3],
            add3.right: [0, 0, 0, 0, 3],
            add4.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add4.right: [0, 0, 0, 0, 3],
            add5.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add5.right: [0, 0, 0, 0, 3],
            add6.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            add6.right: [0, 0, 0, 0, 3],
            add6.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_6():
    model = Model()
    model += (model_1 := Transpose())
    model += (model_2 := Transpose())
    model += (model_3 := Transpose())
    model += Transpose()
    model += (model_5 := Transpose())
    ref_dict = make_reference_dict(
        {
            model_1.input: [1, 1],
            model_1.axes: [1],
            model_2.input: [1, 1, 1, 2],
            model_2.axes: [2],
            model_3.input: [1, 1, 2, 2],
            model_3.axes: [2],
            model_3.input: [1, 1, 2, 2],
            model_3.axes: [2],
            model_5.input: [1, 1, 2, 2],
            model_5.axes: [2],
            model_5.output: [1, 2],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(input=[2, 3, 4])
    ref_dict = make_reference_dict(
        {
            model_1.input: [1],
            model_1.axes: [],
            model_2.input: [1, 1],
            model_2.axes: [],
            model_3.input: [1, 1],
            model_3.axes: [],
            model_3.input: [1, 1],
            model_3.axes: [],
            model_5.input: [1, 1],
            model_5.axes: [],
            model_5.output: [1],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_7():
    model = Model()
    model += (model_1 := Transpose())
    model += (model_2 := Transpose())
    model += (model_3 := Transpose())
    model += Transpose()
    model += (model_5 := Transpose())
    ref_dict = make_reference_dict(
        {
            model_1.input: [1, 1],
            model_1.axes: [1],
            model_2.input: [1, 1, 1, 2],
            model_2.axes: [2],
            model_3.input: [1, 1, 2, 2],
            model_3.axes: [2],
            model_3.input: [1, 1, 2, 2],
            model_3.axes: [2],
            model_5.input: [1, 1, 2, 2],
            model_5.axes: [2],
            model_5.output: [1, 2],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(input=["u1", "u2", "u3", "u4"])
    ref_dict = make_reference_dict(
        {
            model_1.input: [1],
            model_1.axes: [],
            model_2.input: [1, 1],
            model_2.axes: [],
            model_3.input: [1, 1],
            model_3.axes: [],
            model_3.input: [1, 1],
            model_3.axes: [],
            model_5.input: [1, 1],
            model_5.axes: [],
            model_5.output: [1],
        }
    )
    assert_constr_counts(ref_dict)
    model_3.set_shapes(input=[1, "u2", "u3", "u4"])
    ref_dict = make_reference_dict(
        {
            model_1.input: [1],
            model_1.axes: [],
            model_2.input: [1, 1],
            model_2.axes: [],
            model_3.input: [1, 1],
            model_3.axes: [],
            model_3.input: [1, 1],
            model_3.axes: [],
            model_5.input: [1, 1],
            model_5.axes: [],
            model_5.output: [1],
        }
    )
    assert_constr_counts(ref_dict)

    model_3.set_shapes(output=[4, 3, 2, 1])
    ref_dict = make_reference_dict(
        {
            model_1.input: [1],
            model_1.axes: [],
            model_2.input: [1, 1],
            model_2.axes: [],
            model_3.input: [1, 1],
            model_3.axes: [],
            model_3.input: [1, 1],
            model_3.axes: [],
            model_5.input: [1, 1],
            model_5.axes: [],
            model_5.output: [1],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_8():
    model = Model()
    model += (model_1 := Model1())
    model += (model_2 := Model1())
    model += (model_3 := Model1())
    model += (model_4 := Model1())
    model += (model_5 := Model1())
    ref_dict = make_reference_dict(
        {
            model_1.input: [1],
            model_2.input: [1, 2],
            model_3.input: [2, 2],
            model_3.input: [2, 2],
            model_5.input: [2, 2],
            model_5.output: [2],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(input=["u1", "u2", "u3", "u4"])
    ref_dict = make_reference_dict(
        {
            model_1.input: [2],
            model_2.input: [2, 3],
            model_3.input: [3, 3],
            model_3.input: [3, 3],
            model_5.input: [3, 3],
            model_5.output: [3],
        }
    )
    assert_constr_counts(ref_dict)
    model_3.set_shapes(input=[7, "u2", "u3", "u4"])
    ref_dict = make_reference_dict(
        {
            model_1.input: [3],
            model_2.input: [3, 4],
            model_3.input: [4, 4],
            model_4.input: [4, 4],
            model_5.input: [4, 4],
            model_5.output: [4],
        }
    )
    assert_constr_counts(ref_dict)

    model_2.set_shapes(input=[6, 5, 7, 8])
    ref_dict = make_reference_dict(
        {
            model_1.input: [],
            model_2.input: [],
            model_3.input: [],
            model_5.input: [],
            model_5.input: [],
            model_5.output: [],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_9():
    model = Model()
    model += (model_1 := Model1())
    model += (model_2 := Model1())
    model += (model_3 := Model1())
    model += Model1()
    model += (model_5 := Model1())
    ref_dict = make_reference_dict(
        {
            model_1.input: [1],
            model_2.input: [1, 2],
            model_3.input: [2, 2],
            model_3.input: [2, 2],
            model_5.input: [2, 2],
            model_5.output: [2],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(input=["u1"])
    ref_dict = make_reference_dict(
        {
            model_1.input: [2],
            model_2.input: [2, 3],
            model_3.input: [3, 3],
            model_3.input: [3, 3],
            model_5.input: [3, 3],
            model_5.output: [3],
        }
    )
    assert_constr_counts(ref_dict)

    model_2.set_shapes(input=[1])
    ref_dict = make_reference_dict(
        {
            model_1.input: [],
            model_2.input: [],
            model_3.input: [],
            model_5.input: [],
            model_5.input: [],
            model_5.output: [],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_10():
    model = Model()
    model += (model_1 := Model1())
    model += (model_2 := Model1())
    model += (model_3 := Model1())
    model += Model1()
    model += (model_5 := Model1())
    ref_dict = make_reference_dict(
        {
            model_1.input: [1],
            model_2.input: [1, 2],
            model_3.input: [2, 2],
            model_3.input: [2, 2],
            model_5.input: [2, 2],
            model_5.output: [2],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(input=["u1"])
    ref_dict = make_reference_dict(
        {
            model_1.input: [2],
            model_2.input: [2, 3],
            model_3.input: [3, 3],
            model_3.input: [3, 3],
            model_5.input: [3, 3],
            model_5.output: [3],
        }
    )
    assert_constr_counts(ref_dict)

    model_3.set_shapes(output=[3])
    ref_dict = make_reference_dict(
        {
            model_1.input: [],
            model_2.input: [],
            model_3.input: [],
            model_5.input: [],
            model_5.input: [],
            model_5.output: [],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_11():
    model = Model()
    model += (model_1 := Model1())
    model += (model_2 := Model1())
    model += (model_3 := Model1())
    model += (model_4 := Model1())
    model += (model_5 := Model1())
    ref_dict = make_reference_dict(
        {
            model_1.input: [1],
            model_2.input: [1, 2],
            model_3.input: [2, 2],
            model_3.input: [2, 2],
            model_5.input: [2, 2],
            model_5.output: [2],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(input=["u1", "u2"])
    ref_dict = make_reference_dict(
        {
            model_1.input: [2],
            model_2.input: [2, 3],
            model_3.input: [3, 3],
            model_3.input: [3, 3],
            model_5.input: [3, 3],
            model_5.output: [3],
        }
    )
    assert_constr_counts(ref_dict)
    model_5.set_shapes(input=[1, "u2"])
    ref_dict = make_reference_dict(
        {
            model_1.input: [3],
            model_2.input: [3, 4],
            model_4.input: [4, 4],
            model_4.input: [4, 4],
            model_5.input: [4, 4],
            model_5.output: [4],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_12():
    model = Model()
    model |= (model_1 := Add())(left="input1", right="input2")
    model |= (model_2 := Add())(left="input1", right=model_1.output)
    ref_dict = make_reference_dict(
        {
            model_1.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            model_1.right: [0, 0, 0, 0, 1],
            model_2.right: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            model_2.output: [0, 0, 0, 0, 2],
        }
    )
    assert_constr_counts(ref_dict)

    model += (model_3 := Transpose())
    ref_dict = make_reference_dict(
        {
            model_1.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 3],
            model_1.right: [0, 0, 0, 0, 1],
            model_2.right: [0, 0, 0, 0, 0, 0, 0, 0, 1, 3],
            model_3.input: [0, 0, 0, 0, 1, 2, 3],
            model_3.output: [1, 2],
        }
    )
    assert_constr_counts(ref_dict)

    model += (model_4 := Transpose())
    ref_dict = make_reference_dict(
        {
            model_1.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 3],
            model_1.right: [0, 0, 0, 0, 1],
            model_2.right: [0, 0, 0, 0, 0, 0, 0, 0, 1, 3],
            model_3.input: [0, 0, 0, 0, 1, 2, 3],
            model_4.input: [1, 1, 2, 2],
            model_4.output: [1, 2],
        }
    )
    assert_constr_counts(ref_dict)

    model += (model_5 := Transpose())
    ref_dict = make_reference_dict(
        {
            model_1.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 3],
            model_1.right: [0, 0, 0, 0, 1],
            model_2.right: [0, 0, 0, 0, 0, 0, 0, 0, 1, 3],
            model_3.input: [0, 0, 0, 0, 1, 2, 3],
            model_4.input: [1, 1, 2, 2],
            model_5.input: [1, 1, 2, 2],
            model_5.output: [1, 2],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(left=[4, 5], right=[4, 5])
    ref_dict = make_reference_dict(
        {
            model_1.left: [1, 1],
            model_1.right: [1],
            model_2.right: [1, 1],
            model_3.input: [1, 1],
            model_4.input: [1, 1],
            model_5.input: [1, 1],
            model_5.output: [1],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_13():
    model = Model()
    slice_model = Slice(start=0, stop=2, step=None)
    model_1 = Indexer(index=TBD)
    model_2 = Add()
    model_3 = Add()
    model_4 = Add()
    model |= slice_model
    model |= model_1(index=slice_model.output)
    model |= model_2(left=model_1.output)
    model |= model_3(left=model_2.output)
    model |= model_4(left=model_3.output)
    ref_dict = make_reference_dict(
        {
            slice_model.start: [],
            slice_model.stop: [],
            slice_model.step: [],
            model_1.input: [0, 0, 2],
            model_1.index: [0, 0],
            model_2.left: [0, 0, 0, 0, 0, 0, 1, 2],
            model_2.right: [0, 0, 0, 0, 1],
            model_3.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            model_3.right: [0, 0, 0, 0, 2],
            model_4.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            model_4.right: [0, 0, 0, 0, 2],
            model_4.output: [0, 0, 0, 0, 2],
        }
    )
    assert_constr_counts(ref_dict)

    model_2.set_shapes(right=[1])
    ref_dict = make_reference_dict(
        {
            slice_model.start: [],
            slice_model.stop: [],
            slice_model.step: [],
            model_1.input: [0, 0, 2],
            model_1.index: [0, 0],
            model_2.left: [0, 0, 0, 0, 0, 0, 2, 2],
            model_2.right: [0, 0, 0, 0, 2],
            model_3.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 3],
            model_3.right: [0, 0, 0, 0, 3],
            model_4.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            model_4.right: [0, 0, 0, 0, 3],
            model_4.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_14():
    model = Model()

    model_1 = Add()
    model_2 = Add()
    model_3 = Add()
    model_4 = Add()

    model |= model_1
    model |= model_2(left=model_1.output)
    model |= model_3(left=model_2.output)
    model |= model_4(left=model_3.output)
    ref_dict = make_reference_dict(
        {
            model_1.left: [0, 0, 0, 0, 1],
            model_1.right: [0, 0, 0, 0, 1],
            model_2.left: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            model_2.right: [0, 0, 0, 0, 2],
            model_3.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            model_3.right: [0, 0, 0, 0, 2],
            model_4.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
            model_4.right: [0, 0, 0, 0, 2],
            model_4.output: [0, 0, 0, 0, 2],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(right=[4, 4])
    ref_dict = make_reference_dict(
        {
            model_1.left: [0, 0, 0, 0, 2],
            model_1.right: [0, 0, 0, 0, 2],
            model_2.left: [0, 0, 0, 0, 0, 0, 0, 0, 2, 3],
            model_2.right: [0, 0, 0, 0, 3],
            model_3.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            model_3.right: [0, 0, 0, 0, 3],
            model_4.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            model_4.right: [0, 0, 0, 0, 3],
            model_4.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(left=[4, 4])
    ref_dict = make_reference_dict(
        {
            model_1.left: [1],
            model_1.right: [1],
            model_2.left: [0, 0, 0, 0, 1, 3],
            model_2.right: [0, 0, 0, 0, 3],
            model_3.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            model_3.right: [0, 0, 0, 0, 3],
            model_4.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            model_4.right: [0, 0, 0, 0, 3],
            model_4.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)

    model_2.set_shapes(right=[4, 4, 1, 1])
    ref_dict = make_reference_dict(
        {
            model_1.left: [1],
            model_1.right: [1],
            model_2.left: [1, 1],
            model_2.right: [1],
            model_3.left: [0, 0, 0, 0, 1, 3],
            model_3.right: [0, 0, 0, 0, 3],
            model_4.left: [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
            model_4.right: [0, 0, 0, 0, 3],
            model_4.output: [0, 0, 0, 0, 3],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_15():
    model = Model()

    slice_1 = Slice()
    slice_2 = Slice()
    slice_3 = Slice()
    slice_4 = Slice()

    item_model_1 = Indexer()
    item_model_1.set_types(input=Tensor)
    item_model_2 = Indexer()
    item_model_3 = Indexer()
    item_model_4 = Indexer()

    model_1 = Model()
    model_1 |= slice_1(start="start", stop="stop", step="step")
    model_1 |= item_model_1(input="input", index=slice_1.output, output=IOKey("output"))

    model_2 = Model()
    model_2 |= slice_2(start="start", stop="stop", step="step")
    model_2 |= item_model_2(input="input", index=slice_2.output, output=IOKey("output"))

    model_3 = Model()
    model_3 |= slice_3(start="start", stop="stop", step="step")
    model_3 |= item_model_3(input="input", index=slice_3.output, output=IOKey("output"))

    model_4 = Model()
    model_4 |= slice_4(start="start", stop="stop", step="step")
    model_4 |= item_model_4(input="input", index=slice_4.output, output=IOKey("output"))

    model |= model_1(start=1, stop=None, step=None)
    model += model_2(start=1, stop=None, step=None)
    model += model_3(start=1, stop=None, step=None)
    model += model_4(start=1, stop=None, step=None)
    ref_dict = make_reference_dict(
        {
            model_1.input: [2],  # type: ignore
            model_1.start: [],  # type: ignore
            model_1.stop: [],  # type: ignore
            model_1.step: [],  # type: ignore
            model_2.input: [1, 2],  # type: ignore
            model_2.start: [],  # type: ignore
            model_2.stop: [],  # type: ignore
            model_2.step: [],  # type: ignore
            model_3.input: [1, 1],  # type: ignore
            model_3.start: [],  # type: ignore
            model_3.stop: [],  # type: ignore
            model_3.step: [],  # type: ignore
            model_4.input: [1, 1],  # type: ignore
            model_4.start: [],  # type: ignore
            model_4.stop: [],  # type: ignore
            model_4.step: [],  # type: ignore
            model_4.output: [1],  # type: ignore
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(input=[9])
    ref_dict = make_reference_dict(
        {
            model_1.input: [],  # type: ignore
            model_1.start: [],  # type: ignore
            model_1.stop: [],  # type: ignore
            model_1.step: [],  # type: ignore
            model_2.input: [],  # type: ignore
            model_2.start: [],  # type: ignore
            model_2.stop: [],  # type: ignore
            model_2.step: [],  # type: ignore
            model_3.input: [],  # type: ignore
            model_3.start: [],  # type: ignore
            model_3.stop: [],  # type: ignore
            model_3.step: [],  # type: ignore
            model_4.input: [],  # type: ignore
            model_4.start: [],  # type: ignore
            model_4.stop: [],  # type: ignore
            model_4.step: [],  # type: ignore
            model_4.output: [],  # type: ignore
        }
    )
    assert_constr_counts(ref_dict)

    model_2.set_shapes(input=[8])
    assert_constr_counts(ref_dict)

    model_3.set_shapes(input=[7])
    assert_constr_counts(ref_dict)

    model_4.set_shapes(input=[6])
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_16():
    model = Model()

    model_1 = Add()
    model_1.set_types(left=Tensor, right=Tensor)
    model_2 = Add()
    model_2.set_types(left=Tensor, right=Tensor)

    model |= model_1
    model |= model_2(left=model_1.output)
    ref_dict = make_reference_dict(
        {
            model_1.left: [0, 1, 1],
            model_1.right: [0, 1, 1],
            model_2.left: [0, 0, 1, 1, 1, 2],
            model_2.right: [0, 1, 2],
            model_2.output: [0, 1, 2],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(right=[1] * 100)
    ref_dict = make_reference_dict(
        {
            model_1.left: [0, 1, 2],
            model_1.right: [0, 1, 2],
            model_2.left: [0, 0, 1, 1, 2, 3],
            model_2.right: [0, 1, 3],
            model_2.output: [0, 1, 3],
        }
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(left=[4] * 100)
    ref_dict = make_reference_dict(
        {
            model_1.left: [1],
            model_1.right: [1],
            model_2.left: [0, 1, 1, 4],
            model_2.right: [0, 1, 4],
            model_2.output: [0, 1, 4],
        }
    )
    assert_constr_counts(ref_dict)


def test_shape_constraint_counter_17():
    model = Model()
    model_1 = Model1()
    model_2 = Model1()

    model += model_1
    model += model_2
    ref_dict = make_reference_dict(
        {model_1.input: [1], model_2.input: [1, 2], model_2.output: [2]}
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(output=[f"u{i}" for i in range(3)])
    ref_dict = make_reference_dict(
        {model_1.input: [2], model_2.input: [2, 3], model_2.output: [3]}
    )

    model_1.set_shapes(input=[5] * 2 + ["u1"])
    ref_dict = make_reference_dict(
        {model_1.input: [3], model_2.input: [3, 4], model_2.output: [4]}
    )
    assert_constr_counts(ref_dict)


def test_init_shape_constraints():
    add_model_1 = MyAdd2(left=[1, 4], right=[4, 1], output=[("Var1", ...)])
    ref_dict = make_reference_dict(
        {add_model_1.left: [], add_model_1.right: [], add_model_1.output: []}
    )
    assert_constr_counts(ref_dict)


def test_init_shape_constraints_2():
    model = Model()
    buff = Buffer()
    relu = Relu()
    add_model_1 = MyAdd2(left=[1, 4], right=[4, 1], output=[("Var1", ...)])
    model |= buff(input="my_input", output="output")
    ref_dict = make_reference_dict(
        {
            add_model_1.left: [],
            add_model_1.right: [],
            add_model_1.output: [],
            buff.input: [1],
            buff.output: [1],
        }
    )
    assert_constr_counts(ref_dict)

    model |= relu(input="output", output="output1")
    ref_dict = make_reference_dict(
        {
            add_model_1.left: [],
            add_model_1.right: [],
            add_model_1.output: [],
            buff.input: [2],
            relu.input: [1, 2],
            relu.output: [1],
        }
    )
    assert_constr_counts(ref_dict)

    model |= add_model_1(output="my_input", left="left", right="right")
    ref_dict = make_reference_dict(
        {
            add_model_1.left: [],
            add_model_1.right: [],
            # add_model_1.output: [1],
            buff.input: [2],
            relu.input: [1, 2],
            relu.output: [1],
        }
    )
    assert_constr_counts(ref_dict)


def test_type_constraint_counter_1():
    model = Model()
    model_1 = Model3()
    model_2 = Model2()

    model += model_1
    model += model_2
    ref_dict = make_reference_dict(
        {model_1.input: [1], model_2.input: [1, 2, 2], model_2.output: [2, 2]}
    )
    assert_constr_counts(ref_dict)

    model_1.set_shapes(output=[f"u{i}" for i in range(3)])
    ref_dict = make_reference_dict(
        {model_1.input: [2], model_2.input: [2, 2, 3], model_2.output: [2, 3]}
    )

    model_1.set_shapes(input=[5] * 2 + ["u1"])
    ref_dict = make_reference_dict(
        {model_1.input: [3], model_2.input: [2, 3, 4], model_2.output: [2, 4]}
    )
    assert_constr_counts(ref_dict)


@pytest.mark.skip(
    reason="Will be available after the implementation "
    "of nested post_processes. bcast_error_check "
    "is not working as expected."
)
def test_error_check_counter_1():
    """Checks if bcast_error_check works robust."""
    model = Model()
    add1 = Add()
    add1.set_types(left=Tensor, right=Tensor)
    add2 = Add()
    add2.set_types(left=Tensor, right=Tensor)
    model += add1
    model += add2

    ref_dict = make_reference_dict(
        {
            add1.left: [1, 1],
            add1.right: [1, 1],
            add2.left: [1, 1, 1, 2],
            add2.right: [1, 2],
            add2.output: [1, 2],
        }
    )
    assert_constr_counts(ref_dict)
    add1.set_shapes(left=[1, 2, 9], right=[1, 2, "u1"])
    # Note that actually bcast eliminated from add1 constraints but
    # bcast_error_check comes to the play after bcast eliminated.
    # So [1, 1] is still the correct counts.
    ref_dict = make_reference_dict(
        {
            add1.left: [1, 1],
            add1.right: [1, 1],
            add2.left: [1, 1, 1, 3],
            add2.right: [1, 3],
            add2.output: [1, 3],
        }
    )
    assert_constr_counts(ref_dict)

    # Note add2 bcast and error_check solved but add1 still has error_check.
    add2.set_shapes(left=["a", "b", "c"], right=["a", "b", "c"])

    ref_dict = make_reference_dict(
        {
            add1.left: [1, 1],
            add1.right: [1, 1],
            add2.left: [1, 1, 1],
            add2.right: [1],
            add2.output: [1],
        }
    )
    assert_constr_counts(ref_dict)


@pytest.mark.skip(
    reason="Will be available after the implementation "
    "of nested post_processes. bcast_error_check "
    "is not working as expected."
)
def test_error_check_counter_2():
    """Checks if bcast_error_check works robust."""
    model = Model()
    model += (add1 := Add())
    model += (add2 := Add())

    ref_dict = make_reference_dict(
        {
            add1.left: [1, 1],
            add1.right: [1, 1],
            add2.left: [1, 1, 1, 2],
            add2.right: [1, 2],
            add2.output: [1, 2],
        }
    )
    assert_constr_counts(ref_dict)
    add1.set_shapes(left=[1, 2, 9], right=[1, 2, "u1"])
    # Note that actually bcast eliminated from add1 constraints but
    # bcast_error_check comes to the play after bcast eliminated.
    # So [1, 1] is still the correct counts.
    ref_dict = make_reference_dict(
        {
            add1.left: [1, 1],
            add1.right: [1, 1],
            add2.left: [1, 1, 1, 3],
            add2.right: [1, 3],
            add2.output: [1, 3],
        }
    )
    assert_constr_counts(ref_dict)

    with pytest.raises(ValueError) as err_info:
        add1.set_shapes(right=[1, 2, 5])

    assert str(err_info.value) == "Possible values mismatch!"


@pytest.mark.skip(
    reason="Why were they called once before updates???. \
                  Now they are called once more after first set_shapes call."
)
def test_error_check_counter_3():
    """Checks if bcast_error_check works robust."""
    model = Model()
    model += (add1 := Add())
    model += (add2 := Add())

    ref_dict = make_reference_dict(
        {
            add1.left: [1, 1],
            add1.right: [1, 1],
            add2.left: [1, 1, 1, 2],
            add2.right: [1, 2],
            add2.output: [1, 2],
        }
    )
    assert_constr_counts(ref_dict)
    add1.set_shapes(left=[1, 2, 9], right=[1, 2, "u1"])
    # Note that bcast can not be eliminated.
    # TODO: After broadcast updated, results will be updated.
    # ref_dict = make_reference_dict({
    #     add1.left: [1, 1],
    #     add1.right: [1, 1],
    #     add2.left: [1, 1, 1, 3],
    #     add2.right: [1, 3],
    #     add2.output: [1, 3]
    # })
    ref_dict = make_reference_dict(
        {
            add1.left: [1, 1],
            add1.right: [1, 1],
            add2.left: [1, 1, 1, 2],
            add2.right: [1, 2],
            add2.output: [1, 2],
        }
    )
    assert_constr_counts(ref_dict)

    with pytest.raises(ValueError) as err_info:
        add1.set_shapes(right=[1, 2, 5])

    assert str(err_info.value) == (
        "Shape mismatch for broadcast. Dimensionalities for the corresponding "
        "shape index are left: 9, right: 5, output: 9"
    )
