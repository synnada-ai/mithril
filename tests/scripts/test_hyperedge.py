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
    ScalarValueType,
    ShapeNode,
    ShapeRepr,
    Tensor,
    ToBeDetermined,
    UpdateType,
    Variadic,
)
from mithril.framework.constraints import reduce_constraints, reduce_type_constraint

############## Type Setting and Initialization Tests ##############


def test_init_with_tensor_default_type():
    edge = IOHyperEdge(Tensor[int | float | bool])
    assert (
        edge.is_tensor
        and isinstance(edge._value, Tensor)
        and edge.value_type == int | float | bool
        and edge.value is TBD
    )
    assert isinstance(edge.shape, ShapeNode)
    assert edge in edge._value.referees
    assert edge._value.shape.referees == {edge._value}


def test_init_with_tensor_int_or_float_type():
    edge = IOHyperEdge(Tensor[int | float])
    assert (
        edge.is_tensor
        and isinstance(edge._value, Tensor)
        and edge.value_type == int | float
        and edge.value is TBD
    )
    assert isinstance(edge.shape, ShapeNode)
    assert edge in edge._value.referees
    assert edge._value.shape.referees == {edge._value}


def test_init_with_tensor_type_tensor_value():
    edge = IOHyperEdge(Tensor[int | float], value=Tensor([[2.0]]))
    assert (
        edge.is_tensor
        and isinstance(edge._value, Tensor)
        and edge.value_type is float
        and edge.value == [[2.0]]
    )
    assert isinstance(edge.shape, ShapeNode)
    assert edge in edge._value.referees
    assert edge._value.shape.referees == {edge._value}


def test_init_with_scalar_type_scalar_value():
    edge = IOHyperEdge(int | float, value=2.0)
    assert (
        edge.edge_type is float
        and isinstance(edge._value, float)
        and edge.value_type is float
        and edge.value == 2.0
    )
    assert edge.shape is None


def test_init_with_wrong_tensor_type_tensor_value():
    with pytest.raises(TypeError) as err_info:
        IOHyperEdge(Tensor[int | bool], value=Tensor([[2.0]]))
    assert (
        str(err_info.value)
        == "Acceptable types are mithril.framework.common.Tensor[int | bool], "
        "but mithril.framework.common.Tensor[float] type is provided!"
    )


def test_init_with_wrong_scalar_type_scalar_value():
    with pytest.raises(TypeError) as err_info:
        IOHyperEdge(int | bool, value=2.0)
    assert (
        str(err_info.value)
        == "Acceptable types are bool | int, but <class 'float'> type "
        "is provided!"
    )


def test_init_with_scalar_type_tensor_value():
    with pytest.raises(TypeError) as err_info:
        IOHyperEdge(int | bool, value=Tensor(2.0))
    assert (
        str(err_info.value) == "Acceptable types are bool | int, but "
        "mithril.framework.common.Tensor[float] type is provided!"
    )


def test_init_with_tensor_type_scalar_value():
    with pytest.raises(TypeError) as err_info:
        IOHyperEdge(Tensor[int | bool], value=2.0)
    assert (
        str(err_info.value)
        == "Acceptable types are mithril.framework.common.Tensor[int | bool], "
        "but <class 'float'> type is provided!"
    )


def test_set_tensor_type():
    edge = IOHyperEdge()
    assert edge.edge_type is ToBeDetermined and edge._value is TBD and edge.value is TBD
    assert edge.shape is None
    edge.set_type(Tensor[int | float | bool])
    assert edge.is_tensor and isinstance(edge._value, Tensor) and edge.value is TBD
    assert edge.value_type == int | float | bool
    assert isinstance(edge.shape, ShapeNode)
    assert edge in edge._value.referees
    assert edge._value.shape.referees == {edge._value}


def test_set_generic_tensor_type():
    edge = IOHyperEdge()
    assert edge.edge_type is ToBeDetermined and edge._value is TBD and edge.value is TBD
    assert edge.shape is None
    edge.set_type(Tensor[int | float])
    assert edge.is_tensor and isinstance(edge._value, Tensor) and edge.value is TBD
    assert edge.value_type == int | float
    assert isinstance(edge.shape, ShapeNode)
    assert edge in edge._value.referees
    assert edge._value.shape.referees == {edge._value}


def test_set_scalar_type():
    edge = IOHyperEdge()
    assert edge.edge_type is ToBeDetermined and edge._value is TBD and edge.value is TBD
    assert edge.shape is None
    edge.set_type(int | float)
    assert edge.edge_type == int | float and edge._value is TBD and edge.value is TBD
    assert edge.shape is None


def test_set_scalar_edge_type_to_tensor_type():
    edge = IOHyperEdge(type=int | float)
    with pytest.raises(TypeError) as err_info:
        edge.set_type(Tensor[int | float | bool])
    assert (
        str(err_info.value) == "Acceptable types are float | int, but "
        "mithril.framework.common.Tensor[int | float | bool] type is provided!"
    )


def test_set_tensor_edge_type_to_scalar_type():
    edge = IOHyperEdge(type=Tensor[int | float | bool])
    with pytest.raises(TypeError) as err_info:
        edge.set_type(int | float)
    assert (
        str(err_info.value)
        == "Acceptable types are mithril.framework.common.Tensor[int | float | bool], "
        "but float | int type is provided!"
    )


############## Value Setting Tests ##############


def test_init_with_tensor_value():
    shape_node = ShapeRepr(root=Variadic()).node
    tensor: Tensor[float] = Tensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge(value=tensor)
    assert (
        edge.is_tensor
        and isinstance(edge._value, Tensor)
        and edge.value_type is float
        and edge.value == [[2.0]]
    )
    assert (
        isinstance(edge.shape, ShapeNode)
        and edge.shape.referees == {tensor}
        and tensor.referees == {edge}
    )
    assert edge.shape.get_shapes() == [1, 1]


def test_set_non_typed_edge_with_tensor_value():
    shape_node = ShapeRepr(root=Variadic()).node
    tensor: Tensor[float] = Tensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge()
    edge.set_value(tensor)
    assert (
        edge.is_tensor
        and isinstance(edge._value, Tensor)
        and edge.value_type is float
        and edge.value == [[2.0]]
    )
    assert (
        isinstance(edge.shape, ShapeNode)
        and edge.shape.referees == {tensor}
        and tensor.referees == {edge}
        # Tensor is not referred by any edge since edge's own
        # tensor is created and matched with the given tensor.
    )
    assert edge.shape.get_shapes() == [1, 1]


def test_set_tensor_edge_with_tensor_value():
    shape_node = ShapeRepr(root=Variadic()).node
    tensor: Tensor[float] = Tensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge(type=Tensor[int | float | bool])
    assert isinstance(edge._value, Tensor)
    edge_tensor = edge._value
    edge.set_value(tensor)
    assert (
        edge.is_tensor
        and isinstance(edge._value, Tensor)
        and edge.value_type is float
        and edge.value == [[2.0]]
    )
    assert (
        isinstance(edge.shape, ShapeNode)
        and edge.shape.referees == {edge_tensor}
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
    tensor: Tensor[float] = Tensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge(type=int | float)
    with pytest.raises(TypeError) as err_info:
        edge.set_value(tensor)
    assert (
        str(err_info.value) == "Acceptable types are float | int, "
        "but mithril.framework.common.Tensor[float] type is provided!"
    )


def test_set_tensor_edge_with_scalar_value():
    edge = IOHyperEdge(type=Tensor[int | float | bool])
    with pytest.raises(TypeError) as err_info:
        edge.set_value(3)
    assert (
        str(err_info.value)
        == "Acceptable types are mithril.framework.common.Tensor[int | float | bool], "
        "but <class 'int'> type is provided!"
    )


def test_set_tensor_edge_with_different_tensor_value():
    shape_node = ShapeRepr(root=Variadic()).node
    tensor: Tensor[float] = Tensor([[2.0]], shape=shape_node)
    edge = IOHyperEdge(value=tensor)
    with pytest.raises(ValueError) as err_info:
        edge.set_value(Tensor([[3.0]], shape=ShapeRepr(root=Variadic()).node))
    assert (
        str(err_info.value)
        == "Value is set before as [[2.0]]. A value can not be reset."
    )


def test_set_tensor_edge_with_different_type_tensor_value():
    edge = IOHyperEdge(type=Tensor[int | bool])
    with pytest.raises(TypeError) as err_info:
        shape_node = ShapeRepr(root=Variadic()).node
        tensor: Tensor[float] = Tensor([[2.0]], shape=shape_node)
        edge.set_value(tensor)
    assert (
        str(err_info.value)
        == "Acceptable types are mithril.framework.common.Tensor[int | bool], "
        "but mithril.framework.common.Tensor[float] type is provided!"
    )


def test_set_scalar_edge_with_different_type_scalar_value():
    edge = IOHyperEdge(type=int | bool)
    with pytest.raises(TypeError) as err_info:
        edge.set_value([1, 2])
    assert (
        str(err_info.value) == "Acceptable types are bool | int, but list[int] type "
        "is provided!"
    )


############## HyperEdge Matching Tests ##############


def test_match_tensor_edge_with_tensor_edge_having_common_types():
    constr1 = Constraint(fn=reduce_constraints, types=[UpdateType.SHAPE])
    edge1 = IOHyperEdge(type=Tensor[int | float])
    edge1.add_constraint(constr1)

    constr2 = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=Tensor[float | bool])
    edge2.add_constraint(constr2)
    node2 = edge2.shape

    updates = edge1.match(edge2)
    assert edge1.shape is not None
    assert edge2.shape is not None
    assert (
        isinstance(edge1._value, Tensor)
        and edge1._value.referees == {edge1}
        and edge1.edge_type == Tensor[float]
        and edge1.value_type is float
    )
    assert isinstance(edge2._value, Tensor) and edge2._value.referees == {edge1}
    assert edge1.shape is edge2.shape
    assert edge1._value is edge2._value
    assert edge1.shape.referees == {edge1._value}
    assert (
        edge1.all_constraints == {constr1, constr2} and edge2.all_constraints == set()
    )
    assert updates.constraints == {constr2}
    assert updates.shape_updates == set()
    assert updates.node_updates == {node2}  # NOTE: The shape node is useless actually.


def test_match_tensor_edge_with_tensor_edge_with_no_common_types():
    edge1 = IOHyperEdge(type=Tensor[int | float])
    edge2 = IOHyperEdge(type=Tensor[bool])

    with pytest.raises(TypeError) as err_info:
        edge1.match(edge2)
    assert (
        str(err_info.value)
        == "Acceptable types are mithril.framework.common.Tensor[int | float], "
        "but mithril.framework.common.Tensor[bool] type is provided!"
    )


def test_match_tensor_edge_with_scalar_edge():
    edge1 = IOHyperEdge(type=Tensor[int | float])
    edge2 = IOHyperEdge(type=float | bool)

    with pytest.raises(TypeError) as err_info:
        edge1.match(edge2)
    assert (
        str(err_info.value)
        == "Acceptable types are mithril.framework.common.Tensor[int | float], "
        "but bool | float type is provided!"
    )


def test_match_scalar_edge_with_tensor_edge():
    edge1 = IOHyperEdge(type=Tensor[int | float])
    edge2 = IOHyperEdge(type=float | bool)

    with pytest.raises(TypeError) as err_info:
        edge2.match(edge1)
    assert (
        str(err_info.value) == "Acceptable types are bool | float, but "
        "mithril.framework.common.Tensor[int | float] type is provided!"
    )


def test_match_untyped_edge_with_tensor_edge():
    edge1 = IOHyperEdge()  # Untyped edge

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=Tensor[float | bool])
    edge2.add_constraint(constr)
    node2 = edge2.shape

    updates = edge1.match(edge2)
    assert edge1.shape is not None
    assert edge2.shape is not None
    assert (
        isinstance(edge1._value, Tensor)
        and edge1._value.referees == {edge1}
        and edge1.value_type == float | bool
    )
    assert isinstance(edge2._value, Tensor) and edge2._value.referees == {edge1}
    assert edge1.shape is edge2.shape
    assert edge1._value is edge2._value
    assert edge1.shape.referees == {edge1._value}
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == set()
    assert updates.shape_updates == set()
    assert updates.node_updates == {node2}  # NOTE: The shape node is useless actually.


def test_match_untyped_edge_with_scalar_edge():
    edge1 = IOHyperEdge()  # Untyped edge

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=float | bool)
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type == float | bool
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == set()
    assert updates.shape_updates == set()


def test_match_scalar_edge_with_untyped_edge():
    edge1 = IOHyperEdge()  # Untyped edge

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=float | bool)
    edge2.add_constraint(constr)

    updates = edge2.match(edge1)
    assert edge1.edge_type == float | bool
    assert edge1.all_constraints == set() and edge2.all_constraints == {constr}
    assert updates.constraints == set()
    assert updates.shape_updates == set()


def test_match_mixed_type_edge_with_tensor_edge():
    edge1 = IOHyperEdge(type=Tensor[int | float] | int | float)  # Mixed type edge.

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=Tensor[float | bool])
    edge2.add_constraint(constr)
    node2 = edge2.shape

    updates = edge1.match(edge2)
    assert edge1.shape is not None
    assert edge2.shape is not None
    assert (
        isinstance(edge1._value, Tensor)
        and edge1._value.referees == {edge1}
        and edge1.value_type is float
    )
    assert edge1.edge_type == Tensor[float]
    assert not edge1.is_polymorphic
    assert isinstance(edge2._value, Tensor) and edge2._value.referees == {edge1}
    assert edge1.shape is edge2.shape
    assert edge1._value is edge2._value
    assert edge1.shape.referees == {edge1._value}
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == {constr}
    assert updates.shape_updates == set()
    assert updates.node_updates == {node2}  # NOTE: The shape node is useless actually.


def test_match_mixed_type_edge_with_scalar_edge():
    edge1 = IOHyperEdge(type=Tensor[int | float] | int | float)  # Mixed type edge.

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=float | bool)
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type is float
    assert not edge1.is_polymorphic
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == {constr}
    assert updates.shape_updates == set()


def test_match_mixed_type_edge_with_mixed_type_edge_1():
    edge1 = IOHyperEdge(type=Tensor[int | float] | int | float)  # Mixed type edge.

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=float | bool | Tensor[float | bool])
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type == float | Tensor[float]
    assert edge1.is_polymorphic
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == {constr}
    assert updates.shape_updates == set()


def test_match_mixed_type_edge_with_mixed_type_edge_2():
    edge1 = IOHyperEdge(type=Tensor[int | float] | int | float)  # Mixed type edge.

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=float | bool | Tensor[bool])
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type is float
    assert not edge1.is_polymorphic
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == {constr}
    assert updates.shape_updates == set()


def test_match_mixed_type_edge_with_mixed_type_edge_3():
    edge1 = IOHyperEdge(type=Tensor[int | float] | int | float)  # Mixed type edge.

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=float | bool | Tensor[int | float | bool])
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type == float | Tensor[int | float]
    assert edge1.is_polymorphic
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == {constr}
    assert updates.shape_updates == set()


def test_list_of_tensor_type_edge_match():
    edge1 = IOHyperEdge(type=list[Tensor[int | float | bool]])

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=list[Tensor[int | float]])
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type == list[Tensor[int | float]]
    assert not edge1.is_polymorphic
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == set()
    assert updates.shape_updates == set()


def test_set_list_of_mixed_type_value():
    edge1 = IOHyperEdge(value=[Tensor(1), 5.0, Tensor(False)])

    assert edge1.edge_type == list[Tensor[int | bool] | float]
    assert not edge1.is_polymorphic


def test_list_of_tensor_type_edge_match_with_list_of_tensor_value_edge():
    value: list[Tensor[int | float]] = [Tensor(1.0), Tensor(2), Tensor(3.0)]
    edge1 = IOHyperEdge(value=value)

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    edge2 = IOHyperEdge(type=list[Tensor[int | float]])
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type == list[Tensor[int | float]]
    assert edge1._value == value
    assert not edge1.is_polymorphic
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == set()
    assert updates.shape_updates == set()
    assert updates.value_updates == set()


def test_list_of_tensor_value_edge_match_with_list_of_tensor_value_edge_1():
    value1: list[Tensor[int | float]] = [Tensor(1.0), Tensor(2), Tensor(3.0)]
    edge1 = IOHyperEdge(value=value1)
    assert edge1._value == value1
    assert isinstance(edge1._value, list) and all(
        isinstance(t, Tensor) for t in edge1._value
    )
    assert edge1._value[0].shape.referees == {value1[0]}
    assert edge1._value[1].shape.referees == {value1[1]}
    assert edge1._value[2].shape.referees == {value1[2]}
    assert value1[0].referees == {edge1}
    assert value1[1].referees == {edge1}
    assert value1[2].referees == {edge1}

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    value2: list[Tensor[int | float]] = [Tensor(), Tensor(), Tensor()]
    edge2 = IOHyperEdge(value=value2)
    assert edge2._value == value2
    assert isinstance(edge2._value, list) and all(
        isinstance(t, Tensor) for t in edge2._value
    )
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type == list[Tensor[int | float]]
    assert all(
        edge1._value[idx].value == tensor.value for idx, tensor in enumerate(value1)
    )
    assert all(tensor.referees == {edge1} for tensor in edge1._value)
    assert all(tensor.shape.referees == {tensor} for tensor in edge1._value)
    assert all(tensor.shape.referees == {tensor} for tensor in edge2._value)
    assert edge1._value == edge2._value == value1
    assert not edge1.is_polymorphic
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == {constr}
    assert updates.shape_updates == set()
    assert updates.value_updates == set()


def test_list_of_tensor_value_edge_match_with_list_of_tensor_value_edge_2():
    t: Tensor[int | float | bool] = Tensor()
    value1: list[Tensor[int | float]] = [Tensor(1.0), Tensor(2), t]
    edge1 = IOHyperEdge(value=value1)
    assert edge1._value == value1
    assert isinstance(edge1._value, list) and all(
        isinstance(t, Tensor) for t in edge1._value
    )
    assert edge1._value[0].shape.referees == {value1[0]}
    assert edge1._value[1].shape.referees == {value1[1]}
    assert edge1._value[2].shape.referees == {value1[2]}
    assert value1[0].referees == {edge1}
    assert value1[1].referees == {edge1}
    assert value1[2].referees == {edge1}

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    value2: list[Tensor[int | float]] = [Tensor(), Tensor(), Tensor(4)]
    edge2 = IOHyperEdge(value=value2)
    assert edge2._value == value2
    assert isinstance(edge2._value, list) and all(
        isinstance(t, Tensor) for t in edge2._value
    )
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type == list[Tensor[int | float]]
    assert all(
        edge1._value[idx].value == tensor.value for idx, tensor in enumerate(value1)
    )
    assert all(tensor.referees == {edge1} for tensor in edge1._value)
    assert all(tensor.shape.referees == {tensor} for tensor in edge1._value)
    assert all(tensor.shape.referees == {tensor} for tensor in edge2._value)
    assert edge1._value == edge2._value == value1
    assert not edge1.is_polymorphic
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == {constr}
    assert updates.shape_updates == {t}
    assert updates.value_updates == {edge1}


def test_list_of_tensor_value_edge_match_with_list_of_tensor_value_edge_reverse():
    value1: list[Tensor[int | float]] = [Tensor(1.0), Tensor(2), Tensor(3.0)]
    edge1 = IOHyperEdge(value=value1)
    assert edge1._value == value1
    assert isinstance(edge1._value, list) and all(
        isinstance(t, Tensor) for t in edge1._value
    )
    assert edge1._value[0].shape.referees == {value1[0]}
    assert edge1._value[1].shape.referees == {value1[1]}
    assert edge1._value[2].shape.referees == {value1[2]}
    assert value1[0].referees == {edge1}
    assert value1[1].referees == {edge1}
    assert value1[2].referees == {edge1}

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    t1: Tensor[int | float | bool] = Tensor()
    t2: Tensor[int | float | bool] = Tensor()
    t3: Tensor[int | float | bool] = Tensor()
    value2: list[Tensor[int | float]] = [t1, t2, t3]
    edge2 = IOHyperEdge(value=value2)
    assert edge2._value == value2
    assert isinstance(edge2._value, list) and all(
        isinstance(t, Tensor) for t in edge2._value
    )
    edge2.add_constraint(constr)

    updates = edge2.match(edge1)
    assert edge1.edge_type == list[Tensor[int | float]]
    assert all(
        edge1._value[idx].value == tensor.value for idx, tensor in enumerate(value1)
    )
    assert all(tensor.referees == {edge2} for tensor in edge1._value)
    assert all(tensor.shape.referees == {tensor} for tensor in edge1._value)
    assert all(tensor.shape.referees == {tensor} for tensor in edge2._value)
    assert edge1._value == edge2._value == value2
    assert not edge1.is_polymorphic
    assert edge1.all_constraints == set() and edge2.all_constraints == {constr}
    assert updates.constraints == {constr}
    assert updates.shape_updates == {t1, t2, t3}
    assert updates.value_updates == {edge2}


def test_tuple_of_tensor_value_edge_match_with_tuple_of_tensor_value_edge():
    value1: tuple[Tensor[float], Tensor[int], Tensor[float]] = (
        Tensor(1.0),
        Tensor(2),
        Tensor(3.0),
    )
    edge1 = IOHyperEdge(value=value1)
    assert edge1._value == value1
    assert isinstance(edge1._value, tuple) and all(
        isinstance(t, Tensor) for t in edge1._value
    )
    assert edge1._value[0].shape.referees == {value1[0]}
    assert edge1._value[1].shape.referees == {value1[1]}
    assert edge1._value[2].shape.referees == {value1[2]}
    assert value1[0].referees == {edge1}
    assert value1[1].referees == {edge1}
    assert value1[2].referees == {edge1}

    constr = Constraint(fn=reduce_type_constraint, types=[UpdateType.TYPE])
    value2: tuple[Tensor[int | float | bool], ...] = (Tensor(), Tensor(), Tensor())
    edge2 = IOHyperEdge(value=value2)
    assert edge2._value == value2
    assert isinstance(edge2._value, tuple) and all(
        isinstance(t, Tensor) for t in edge2._value
    )
    edge2.add_constraint(constr)

    updates = edge1.match(edge2)
    assert edge1.edge_type == tuple[Tensor[float], Tensor[int], Tensor[float]]
    assert all(
        edge1._value[idx].value == tensor.value for idx, tensor in enumerate(value1)
    )
    assert edge1._value == edge2._value == value1
    assert all(tensor.referees == {edge1} for tensor in edge1._value)
    assert all(tensor.shape.referees == {tensor} for tensor in edge1._value)
    assert not edge1.is_polymorphic
    assert edge1.all_constraints == {constr} and edge2.all_constraints == set()
    assert updates.constraints == {constr}
    assert updates.shape_updates == set()
    assert updates.value_updates == set()


ValueType = Tensor[int | float | bool] | ScalarValueType | ToBeDetermined


def test_match_tuple_of_two_scalar_values():
    """
    Tests the following case:

    edge1: (1, TBD) -+
                     |-> expected_result: (1, 2)
    edge2: (TBD, 2) -+
    """
    edge1 = IOHyperEdge(value=(1, TBD))
    edge2 = IOHyperEdge(value=(TBD, 2))
    updates = edge1.match(edge2)

    assert edge1._type == tuple[int, int]
    assert edge1._value == (1, 2)
    assert updates.value_updates == {edge1}


def test_match_recursive_values():
    """
    Tests the following case:

    edge1: [(2, TBD), -+
            (3, 4)]    |
                       |-> expected_result: [(2, 5),
                       |                     (3, 4)]
    edge2: [(TBD, 5), -+
            (3, 4)]
    """
    edge1 = IOHyperEdge(value=[(2, TBD), (3, 4)])
    edge2 = IOHyperEdge(value=[(TBD, 5), (3, 4)])
    updates = edge1.match(edge2)

    assert edge1._type == list[tuple[int, int]]
    assert edge1._value == [(2, 5), (3, 4)]
    assert updates.value_updates == {edge1}


def test_match_recursive_heterogenous_values():
    """
    Tests the following case:

    edge1: [(2, TBD),   -+
           (3, 4, TBD)]  |
                         |-> expected_result: [(2, 5),
                         |                    (3, 4, 2)]
    edge2: [(TBD, 5),   -+
           (3, 4, 2)]
    """
    edge1 = IOHyperEdge(value=[(2, TBD), (3, 4, TBD)])
    edge2 = IOHyperEdge(value=[(TBD, 5), (3, 4, 2)])
    updates = edge1.match(edge2)

    assert edge1._type == list[tuple[int, int] | tuple[int, int, int]]
    assert edge1._value == [(2, 5), (3, 4, 2)]
    assert updates.value_updates == {edge1}


def test_match_values_with_different_types():
    """
    Tests the following case:

    edge1: [1, TBD] -+
                      |-> expected_result: [1, 2]
    edge2: [TBD, 2] -+
    """

    edge1 = IOHyperEdge(value=[1, TBD])
    edge2 = IOHyperEdge(value=[TBD, 2])
    updates = edge1.match(edge2)

    assert edge1._type == list[int]
    assert edge1._value == [1, 2]
    assert updates.value_updates == {edge1}


@pytest.mark.skip(
    reason="Known Bug, see https://github.com/synnada-ai/mithril/issues/333"
)
def test_match_two_tensors():
    """
    Tests the following case:

    edge1: [T1, TBD] -+
                      |-> expected_result: [T1, T2]
    edge2: [TBD, T2] -+

    T1 and T2 are all different Tensor objects.
    """
    t1: Tensor[int | float] = Tensor(type=int | float)
    t2: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[t1, TBD])
    edge2 = IOHyperEdge(value=[TBD, t2])
    updates = edge1.match(edge2)

    assert edge1._type == list[Tensor[int | float]]
    assert edge1._value == [t1, t2]

    assert t1.referees == {edge1}
    assert t2.referees == {edge1}
    assert updates.value_updates == {edge1}


def test_match_three_tensors():
    """
    Tests the following case:

    edge1: [T1, TBD, T3]  -+
                           |-> expected_result: [T1, T2, T3]
    edge2: [TBD, T2, TBD] -+

    T1, T2 and T3 are all different Tensor objects.

    """
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=(t1, TBD, t3))
    edge2 = IOHyperEdge(value=(TBD, t2, TBD))

    updates = edge1.match(edge2)
    assert edge1._type == tuple[Tensor[int], Tensor[int], Tensor[int]]
    assert edge1._value == (t1, t2, t3)

    assert t1.referees == {edge1}
    assert t2.referees == {edge1}
    assert t3.referees == {edge1}

    assert edge1.tensors == [t1, t2, t3]

    assert updates.value_updates == {edge1}


def test_match_three_same_tensors():
    """
    Tests the following case:

    edge1: [T1, TBD, T3] -+
                          |-> expected_result: [T1, T1, T3]
    edge2: [T2, T2, TBD] -+

    T1, T2 and T3 are all different Tensor objects.
    """
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=(t1, TBD, t3))
    edge2 = IOHyperEdge(value=(t2, t2, TBD))

    updates = edge1.match(edge2)
    assert edge1._type == tuple[Tensor[int], Tensor[int], Tensor[int]]
    assert edge1._value == (t1, t1, t3)

    assert t1.referees == {edge1}
    assert t2.referees == set()
    assert t3.referees == {edge1}

    assert set(edge1.tensors) == {t1, t3}

    assert updates.value_updates == {edge1}


def test_match_three_same_tensors_backwards():
    """
    Tests the following case:

    edge1: [T3, TBD, T1] -+
                          |-> expected result: [T3, T1, T1]
    edge2: [TBD, T2, T2] -+

    T1, T2 and T3 are all different Tensor objects.
    """
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=(t3, TBD, t1))
    edge2 = IOHyperEdge(value=(TBD, t2, t2))

    updates = edge1.match(edge2)
    assert edge1._type == tuple[Tensor[int], Tensor[int], Tensor[int]]
    assert edge1._value == (t3, t1, t1)

    assert t1.referees == {edge1}
    assert t2.referees == set()
    assert t3.referees == {edge1}

    assert set(edge1.tensors) == {t1, t3}

    assert updates.value_updates == {edge1}


def test_match_two_tensors_in_two_containers():
    """
    Tests the following case:

    edge1: [T1, T2] -+
                     |-> expected result: [T1, T1]
    edge2: [T2, T1] -+

    T1 and T2 are all different Tensor objects.
    """
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[t1, t2])
    edge2 = IOHyperEdge(value=[t2, t1])

    updates = edge1.match(edge2)
    assert edge1._type == list[Tensor[int]]
    assert edge1._value == [t1, t1]

    assert t1.referees == {edge1}
    assert t2.referees == set()

    assert set(edge1.tensors) == {t1}

    assert updates.value_updates == set()


def test_match_four_tensors_in_two_nested_containers():
    """
    Tests the following case:

    edge1:  [[T1, T2], -+
             [T3, T4]]  |
                        |-> expected result: [[T1, T1],
                        |                     [T1, T1]]
    edge2:  [[T2, T3], -+
             [T4, T5]]

    T1, T2, T3, T4 and T5 are all different Tensor objects.
    """
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)
    t4: Tensor[int] = Tensor(type=int)
    t5: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[[t1, t2], [t3, t4]])
    edge2 = IOHyperEdge(value=[[t2, t3], [t4, t5]])

    updates = edge1.match(edge2)
    assert edge1._type == list[list[Tensor[int]]]
    assert edge1._value == [[t1, t1], [t1, t1]]

    assert t1.referees == {edge1}
    assert t2.referees == set()
    assert t3.referees == set()
    assert t4.referees == set()
    assert t5.referees == set()

    assert set(edge1.tensors) == {t1}

    assert updates.value_updates == set()


def test_match_four_tensors_in_two_nested_containers_with_tbd():
    """
    Tests the following case:

    edge1:  [[T1, T2], -+
             [T3, T4]]  |
                        |-> expected result: [T1, T1]
                        |                    [T1, T4]
    edge2:  [[T2, T3], -+
             TBD]

    T1, T2, T3 and T4 are all different Tensor objects.
    """
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)
    t4: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[[t1, t2], [t3, t4]])
    edge2 = IOHyperEdge(value=[[t2, t3], TBD])

    updates = edge1.match(edge2)
    assert edge1._type == list[list[Tensor[int]]]
    assert edge1._value == [[t1, t1], [t1, t4]]

    assert t1.referees == {edge1}
    assert t2.referees == set()
    assert t3.referees == set()
    assert t4.referees == {edge1}

    assert set(edge1.tensors) == {t1, t4}

    assert updates.value_updates == {edge1}


def test_list_of_tensors_with_three_hyperedges():
    """
    Tests the following case:



    edge1:  [[T1, TBD], -+
             [T2, T3]]   |
                         |-> expected: [[T1, TBD], -+
                         |              [T1, T3]]   |
    edge2:  [[T2, TBD], -+                          |-> expected: [[T1, T3],
             TBD]                                   |              [T1, T3]]
                                edge3: [[TBD, T3], -+
                                         TBD]


    T1, T2, T3 and T4 are all different Tensor objects.
    """
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[[t1, TBD], [t2, t3]])
    edge2 = IOHyperEdge(value=[[t2, TBD], TBD])

    updates = edge1.match(edge2)
    assert edge1._type == list[list[Tensor[int] | ToBeDetermined]]
    assert edge1._value == [[t1, TBD], [t1, t3]]

    assert t1.referees == {edge1}
    assert t2.referees == set()
    assert t3.referees == {edge1}

    assert set(edge1.tensors) == {t1, t3}

    assert updates.value_updates == {edge1}

    edge3 = IOHyperEdge(value=[[TBD, t3], TBD])

    updates |= edge1.match(edge3)
    assert edge1._type == list[list[Tensor[int]]]
    assert edge1._value == [[t1, t3], [t1, t3]]

    assert t1.referees == {edge1}
    assert t2.referees == set()
    assert t3.referees == {edge1}

    assert updates.value_updates == {edge1}


def test_list_and_tuple_of_tensors_error():
    """
    Tests the following case:



    edge1:  [[T1, TBD], -+
             [T2, T3]]   |
                         |-> expected: ValueError
                         |
    edge2:  [[T2, TBD], -+
             (TBD, TBD)]


    T1, T2, T3 are all different Tensor objects.

    ValueError is expected from this test as
    determined list and tuple are tried to be matched.
    """
    t1: Tensor[int] = Tensor(type=int | float)
    t2: Tensor[int] = Tensor(type=int | float)
    t3: Tensor[int] = Tensor(type=int | float)

    edge1 = IOHyperEdge(value=[[t1, TBD], [t2, t3]])
    edge2 = IOHyperEdge(value=[[t2, TBD], (TBD, TBD)])
    with pytest.raises(ValueError) as err_info:
        edge1.match(edge2)
    assert str(err_info.value) == (
        "Given value is not compatible with the current value\n"
        "    Current value: [Tensor[int | float], Tensor[int | float]]\n"
        "    Given value: (TBD, TBD)"
    )


def test_list_of_two_values_and_list_of_three_values_error():
    """
    Tests the following case:



    edge1:  [[T1, TBD], -+
             [T2, T3]]   |
                         |-> expected: ValueError
                         |
    edge2:  [[T2, TBD], -+
             TBD,
             TBD]


    T1, T2, T3 and T4 are all different Tensor objects.

    ValueError is expected from this test as a list with
    lenght 3 is tried to be matched with a list with lenght 2
    """
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[[t1, TBD], [t2, t3]])
    edge2 = IOHyperEdge(value=[[t2, TBD], TBD, TBD])
    with pytest.raises(ValueError) as err_info:
        edge1.match(edge2)
    assert str(err_info.value) == (
        "Given value is not compatible with the current value\n"
        "    Current value: [[Tensor[int], TBD], [Tensor[int], Tensor[int]]]\n"
        "    Given value: [[Tensor[int], TBD], TBD, TBD]"
    )


def test_list_of_two_values_and_list_of_one_value_error():
    """
    Tests the following case:



    edge1:  [[T1, TBD], -+
             [T2, T3]]   |
                         |-> expected: ValueError
                         |
    edge2:  [[T2, TBD], -+
             [TBD]]


    T1, T2, T3 and T4 are all different Tensor objects.

    ValueError is expected from this test as a list with
    lenght 1 is tried to be matched with a list with lenght 2
    """
    t1: Tensor[bool] = Tensor(type=bool)
    t2: Tensor[bool] = Tensor(type=bool)
    t3: Tensor[bool] = Tensor(type=bool)

    edge1 = IOHyperEdge(value=[[t1, TBD], [t2, t3]])
    edge2 = IOHyperEdge(value=[[t2, TBD], [TBD]])
    with pytest.raises(ValueError) as err_info:
        edge1.match(edge2)
    assert str(err_info.value) == (
        "Given value is not compatible with the current value\n"
        "    Current value: [Tensor[bool], Tensor[bool]]\n"
        "    Given value: [TBD]"
    )


def test_edge_set_values_three_times():
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge()

    edge1.set_value(value=[[t1, TBD], [t2, t3]])
    edge1.set_value(value=[[t2, TBD], TBD])

    assert edge1._type == list[list[Tensor[int] | ToBeDetermined]]
    assert edge1._value == [[t1, TBD], [t1, t3]]

    assert t1.referees == {edge1}
    assert t2.referees == set()
    assert t3.referees == {edge1}

    assert set(edge1.tensors) == {t1, t3}
    edge1.set_value(value=[[TBD, t3], TBD])

    assert edge1._type == list[list[Tensor[int]]]
    assert edge1._value == [[t1, t3], [t1, t3]]

    assert t1.referees == {edge1}
    assert t2.referees == set()
    assert t3.referees == {edge1}


def test_three_edge_reference():
    """
    Tests the following case:

    edge1:  [[T1, T2], -+
             [T3, T4]]  |
                        |-> expected result: [[T1, T1],
                        |                     [T1, T1]]
    edge2:  [[T2, T3], -+
             [T4, T5]]

    edge3:  [[TBD, T3],   --->  [[TBD, T1],
             [T4, TBD]]          [T1, TBD]]

    T1, T2, T3, T4 and T5 are all different Tensor objects.
    """

    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)
    t4: Tensor[int] = Tensor(type=int)
    t5: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[[t1, t2], [t3, t4]])
    edge2 = IOHyperEdge(value=[[t2, t3], [t4, t5]])
    edge3 = IOHyperEdge(value=[[TBD, t3], [t4, TBD]])

    edge1.match(edge2)
    assert edge3._value == [[TBD, t1], [t1, TBD]]
    assert t1.referees == {edge3, edge1}
    assert t2.referees == set()
    assert t3.referees == set()
    assert t4.referees == set()
    assert t5.referees == set()
