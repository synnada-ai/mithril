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
    TBD,
    Constraint,
    IOHyperEdge,
    MyTensor,
    ShapeNode,
    ShapeRepr,
    UpdateType,
    Variadic,
)
from mithril.framework.constraints import reduce_constraints, reduce_type_constraint

############## Type Setting Tests ##############


def test_init_with_tensor_default_type():
    edge = IOHyperEdge(MyTensor)
    assert (
        edge.edge_type is MyTensor
        and isinstance(edge._value, MyTensor)
        and edge.value_type == int | float | bool
        and edge.value is TBD
    )
    assert isinstance(edge.shape, ShapeNode)
    assert edge in edge._value.referees and edge in edge._value.shape.referees


def test_init_with_tensor_int_or_float_type():
    edge = IOHyperEdge(MyTensor[int | float])
    assert (
        edge.edge_type is MyTensor
        and isinstance(edge._value, MyTensor)
        and edge.value_type == int | float
        and edge.value is TBD
    )
    assert isinstance(edge.shape, ShapeNode)
    assert edge in edge._value.referees and edge in edge._value.shape.referees


def test_set_tensor_type():
    edge = IOHyperEdge()
    assert edge.edge_type is TBD and edge._value is TBD and edge.value is TBD
    assert edge.shape is None
    edge.set_type(MyTensor)
    assert (
        edge.edge_type is MyTensor
        and isinstance(edge._value, MyTensor)
        and edge.value is TBD
    )
    assert edge.value_type == int | float | bool
    assert isinstance(edge.shape, ShapeNode)
    assert edge in edge._value.referees and edge in edge._value.shape.referees


def test_set_generic_tensor_type():
    edge = IOHyperEdge()
    assert edge.edge_type is TBD and edge._value is TBD and edge.value is TBD
    assert edge.shape is None
    edge.set_type(MyTensor[int | float])
    assert (
        edge.edge_type is MyTensor
        and isinstance(edge._value, MyTensor)
        and edge.value is TBD
    )
    assert edge.value_type == int | float
    assert isinstance(edge.shape, ShapeNode)
    assert edge in edge._value.referees and edge in edge._value.shape.referees


def test_set_scalar_type():
    edge = IOHyperEdge()
    assert edge.edge_type is TBD and edge._value is TBD and edge.value is TBD
    assert edge.shape is None
    edge.set_type(int | float)
    assert edge.edge_type == int | float and edge._value is TBD and edge.value is TBD
    assert edge.shape is None


def test_set_scalar_edge_type_to_tensor_type():
    edge = IOHyperEdge(type=int | float)
    with pytest.raises(TypeError) as err_info:
        edge.set_type(MyTensor)
    assert str(err_info.value) == "Can not set Tensor type to a Scalar edge."


def test_set_tensor_edge_type_to_scalar_type():
    edge = IOHyperEdge(type=MyTensor)
    with pytest.raises(TypeError) as err_info:
        edge.set_type(int | float)
    assert str(err_info.value) == "Can not set Scalar type to a Tensor edge."


############## Value Setting Tests ##############


def test_init_with_tensor_value():
    shape_node = ShapeRepr(root=Variadic()).node
    tensor = MyTensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge(value=tensor)
    assert (
        edge.edge_type is MyTensor
        and isinstance(edge._value, MyTensor)
        and edge.value_type is float
        and edge.value == [[2.0]]
    )
    assert (
        isinstance(edge.shape, ShapeNode)
        and edge.shape.referees == {edge}
        and tensor.referees == {edge}
    )
    assert edge.shape.get_shapes() == [1, 1]


def test_set_non_typed_edge_with_tensor_value():
    shape_node = ShapeRepr(root=Variadic()).node
    tensor = MyTensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge()
    edge.set_value(tensor)
    assert (
        edge.edge_type is MyTensor
        and isinstance(edge._value, MyTensor)
        and edge.value_type is float
        and edge.value == [[2.0]]
    )
    assert (
        isinstance(edge.shape, ShapeNode)
        and edge.shape.referees == {edge}
        and tensor.referees == set()
        # Tensor is not referred by any edge since edge's own
        # tensor is created and matched with the given tensor.
    )
    assert edge.shape.get_shapes() == [1, 1]


def test_set_tensor_edge_with_tensor_value():
    shape_node = ShapeRepr(root=Variadic()).node
    tensor = MyTensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge(type=MyTensor)
    assert isinstance(edge._value, MyTensor)
    edge_tensor = edge._value
    edge.set_value(tensor)
    assert (
        edge.edge_type is MyTensor
        and isinstance(edge._value, MyTensor)
        and edge.value_type is float
        and edge.value == [[2.0]]
    )
    assert (
        isinstance(edge.shape, ShapeNode)
        and edge.shape.referees == {edge}
        and tensor.referees == set()
        and edge_tensor.referees == {edge}
    )
    assert edge.shape.get_shapes() == [1, 1]


def test_set_scalar_edge_with_scalar_value():
    edge = IOHyperEdge(type=int | float | bool)
    assert edge.shape is None
    edge.set_value(True)
    assert (
        edge.edge_type is bool and isinstance(edge._value, bool) and edge.value is True
    )
    assert edge.shape is None


def test_set_scalar_edge_with_tensor_value():
    shape_node = ShapeRepr(root=Variadic()).node
    tensor = MyTensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge(type=int | float)
    with pytest.raises(ValueError) as err_info:
        edge.set_value(tensor)
    assert str(err_info.value) == "Can not set Tensor value to a Scalar edge."


def test_set_tensor_edge_with_scalar_value():
    edge = IOHyperEdge(type=MyTensor)
    with pytest.raises(ValueError) as err_info:
        edge.set_value(3)
    assert str(err_info.value) == "Can not set Scalar value to a Tensor edge."


def test_set_tensor_edge_with_different_tensor_value():
    shape_node = ShapeRepr(root=Variadic()).node
    tensor = MyTensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge(value=tensor)
    with pytest.raises(ValueError) as err_info:
        edge.set_value(MyTensor([[3.0]], shape=ShapeRepr(root=Variadic()).node))
    assert (
        str(err_info.value)
        == "Value is set before as [[2.0]]. A value can not be reset."
    )


def test_set_tensor_edge_with_different_type_tensor_value():
    edge = IOHyperEdge(type=MyTensor[int | bool])
    with pytest.raises(TypeError) as err_info:
        shape_node = ShapeRepr(root=Variadic()).node
        tensor = MyTensor([[2.0]], shape=shape_node)
        edge.set_value(tensor)
    assert (
        str(err_info.value)
        == "Acceptable types are int | bool, but <class 'float'> type "
        "value is provided!"
    )


def test_set_scalar_edge_with_different_type_scalar_value():
    edge = IOHyperEdge(type=int | bool)
    with pytest.raises(TypeError) as err_info:
        edge.set_value([1, 2])
    assert (
        str(err_info.value) == "Acceptable types are int | bool, but list[int] type "
        "value is provided!"
    )


############## HyperEdge Matching Tests ##############


def test_match_tensor_edge_with_tensor_edge_with_common_types():
    constr1 = Constraint(fn=reduce_constraints, type=UpdateType.SHAPE)
    edge1 = IOHyperEdge(type=MyTensor[int | float])
    edge1.add_constraint(constr1)

    constr2 = Constraint(fn=reduce_type_constraint, type=UpdateType.TYPE)
    edge2 = IOHyperEdge(type=MyTensor[float | bool])
    edge2.add_constraint(constr2)
    node2 = edge2.shape

    updates = edge1.match(edge2)
    assert isinstance(edge1._value, MyTensor) and edge1.value_type is float
    assert edge1._value.referees == {edge1} and edge2._value.referees == {edge1}
    assert edge1.shape is edge2.shape
    assert edge1.shape.referees == {edge1}
    assert (
        edge1.all_constraints == {constr1, constr2} and edge2.all_constraints == set()
    )
    assert updates.constraints[UpdateType.SHAPE] == set() and updates.constraints[
        UpdateType.TYPE
    ] == {constr2}
    assert updates.value_updates == set()
    assert updates.shape_updates == set()
    assert updates.node_updates == {node2}  # NOTE: The shape node is useless actually.


def test_match_tensor_edge_with_tensor_edge_with_no_common_types():
    edge1 = IOHyperEdge(type=MyTensor[int | float])
    edge2 = IOHyperEdge(type=MyTensor[bool])

    with pytest.raises(TypeError) as err_info:
        edge1.match(edge2)
    assert (
        str(err_info.value)
        == "Acceptable types are int | float, but <class 'bool'> type "
        "value is provided!"
    )


def test_match_tensor_edge_with_scalar_edge():
    edge1 = IOHyperEdge(type=MyTensor[int | float])
    edge2 = IOHyperEdge(type=float | bool)

    with pytest.raises(TypeError) as err_info:
        edge1.match(edge2)
    assert str(err_info.value) == "Can not set Scalar type to a Tensor edge."


def test_match_scalar_edge_with_tensor_edge():
    edge1 = IOHyperEdge(type=MyTensor[int | float])
    edge2 = IOHyperEdge(type=float | bool)

    with pytest.raises(TypeError) as err_info:
        edge2.match(edge1)
    assert str(err_info.value) == "Can not set Tensor type to a Scalar edge."


def test_match_untyped_edge_with_tensor_edge():
    edge1 = IOHyperEdge()  # Untyped edge

    constr = Constraint(fn=reduce_type_constraint, type=UpdateType.TYPE)
    edge2 = IOHyperEdge(type=MyTensor[float | bool])
    edge2.add_constraint(constr)
    node2 = edge2.shape

    updates = edge1.match(edge2)
    assert isinstance(edge1._value, MyTensor) and edge1.value_type == float | bool
    assert edge1._value.referees == {edge1} and edge2._value.referees == {edge1}
    assert edge1.shape is edge2.shape
    assert edge1.shape.referees == {edge1}
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints[UpdateType.SHAPE] == set() and updates.constraints[
        UpdateType.TYPE
    ] == {constr}
    assert updates.value_updates == set()
    assert updates.shape_updates == set()
    assert updates.node_updates == {node2}  # NOTE: The shape node is useless actually.


def test_match_untyped_edge_with_scalar_edge():
    edge1 = IOHyperEdge()  # Untyped edge

    constr = Constraint(fn=reduce_type_constraint, type=UpdateType.TYPE)
    edge2 = IOHyperEdge(type=float | bool)
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type == float | bool
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert (
        updates.constraints[UpdateType.SHAPE] == set()
        and updates.constraints[UpdateType.TYPE] == set()
    )
    assert updates.value_updates == set()
    assert updates.shape_updates == set()


def test_match_scalar_edge_with_untyped_edge():
    edge1 = IOHyperEdge()  # Untyped edge

    constr = Constraint(fn=reduce_type_constraint, type=UpdateType.TYPE)
    edge2 = IOHyperEdge(type=float | bool)
    edge2.add_constraint(constr)

    updates = edge2.match(edge1)
    assert edge1.edge_type == float | bool
    assert edge1.all_constraints == set() and edge2.all_constraints == {constr}
    assert (
        updates.constraints[UpdateType.SHAPE] == set()
        and updates.constraints[UpdateType.TYPE] == set()
    )
    assert updates.value_updates == set()
    assert updates.shape_updates == set()
