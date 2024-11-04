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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, reduce
from itertools import combinations, cycle, product, zip_longest
from types import EllipsisType, GenericAlias, UnionType
from typing import Any, Literal, TypeVar

from ..backends.backend import Backend
from ..core import (
    Constant,
    DataType,
    Dtype,
    # data_types,
    # is_tensor_type,
    GenericDataType,
    constant_type_table,
    epsilon_table,
)
from ..utils.utils import OrderedSet, PaddingType, find_dominant_type
from .utils import (
    NestedListType,
    align_shapes,
    find_intersection_type,
    find_type,
    sort_type,
)

__all__ = [
    "_get_shapes",
    "NOT_GIVEN",
    "TBD",
    "IOKey",
    "KeyType",
    "ConnectionType",
    "IOHyperEdge",
    "Connection",
    "ConnectionData",
    "Connect",
    "Connections",
    "ShapeNode",
    "ShapeRepr",
    "Constraint",
    "create_shape_map",
    "TensorType",
    "Tensor",
    "Scalar",
    "ShapesType",
    "_ShapesType",
    "_get_summary_shapes",
    "_get_summary_types",
    "ConstraintSolver",
    "NOT_AVAILABLE",
    "NotAvailable",
]


class NullConnection:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


class ToBeDetermined:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


class NotAvailable:
    _instance = None
    key: str

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


NOT_GIVEN = NullConnection()
TBD = ToBeDetermined()
NOT_AVAILABLE = NotAvailable()


class UpdateType(Enum):
    SHAPE = 1
    TYPE = 2


class KeyType(Enum):
    INPUT = 1
    OUTPUT = 2
    INTERNAL = 3


type FixedValueType = (
    None
    | int
    | tuple[int, ...]
    | list[int]
    | dict
    | slice
    | Constant
    | tuple[int | None, ...]
    | dict
)
type DeferredValueType = (
    float | tuple[float, ...] | list[float] | EllipsisType | ToBeDetermined
)
type TypeValueType = Dtype
type ValueType = FixedValueType | DeferredValueType | TypeValueType
type ScalarType = (
    type[int]
    | type[float]
    | type[bool]
    | type[tuple]
    | type[list]
    | type[dict]
    | type[Constant]
    | type[slice]
    | type[PaddingType]
    | type[EllipsisType]
    | type[ToBeDetermined]
    | NestedListType
    | type[None]
    | UnionType
    | GenericAlias
)
ShapeTemplateType = Sequence[int | str | tuple[str, EllipsisType] | None]
ReduceType = int | tuple[int, ...] | None | ToBeDetermined
MainValueType = (
    int
    | float
    | tuple
    | list
    | dict
    | None
    | EllipsisType
    | PaddingType
    | Constant
    | slice
    | ToBeDetermined
    | Dtype
)
TensorValueType = (
    int | float | tuple | list | None | Constant | ToBeDetermined | NestedListType
)

LossKey = "loss"
FinalCost = "final_cost"


ItemType = TypeVar("ItemType")


def update_equivalence_table(
    item1: ItemType, item2: ItemType, lookup_table: dict[ItemType, set[ItemType]]
):
    item_set1 = lookup_table.get(item1)
    item_set2 = lookup_table.get(item2)
    if item_set1 is None and item_set2 is None:
        items = {item1, item2}
        lookup_table[item1] = lookup_table[item2] = items
    elif item_set1 is None and item_set2 is not None:
        item_set2.add(item1)
        lookup_table[item1] = item_set2
    elif item_set1 is not None and item_set2 is None:
        item_set1.add(item2)
        lookup_table[item2] = item_set1
    elif item_set1 is not None and item_set2 is not None:
        item_set1 |= item_set2
        for _item in item_set2:
            lookup_table[_item] = item_set1


@dataclass
class ConstraintSolver:
    symbol_store: dict[int, Uniadic] = field(
        default_factory=lambda: {}
    )  # contains valued Uniadics' UniadicRecords
    # TODO: empty_node will not be None when it is created
    # with weak_ref.
    empty_node: ShapeNode | None = field(default_factory=lambda: ShapeRepr().node)
    constraint_map: dict[Constraint, list[ConnectionData]] = field(
        default_factory=lambda: {}
    )

    def __call__(self, updates: Updates):
        self.update_shapes(updates)
        solved_constrs: set[Constraint] = set()
        for constr_type in UpdateType:
            self._solver_loop(constr_type, updates, solved_constrs)

    def _solver_loop(
        self,
        constraint_type: UpdateType,
        updates: Updates,
        solved_constraints: set[Constraint],
    ):
        constraints = updates.constraints[constraint_type]
        while constraints:
            constr = constraints.pop()

            if constr not in solved_constraints and constr in self.constraint_map:
                inputs = self.constraint_map[constr]
                status, newly_added_symbols = constr(
                    [conn.metadata.data for conn in inputs]
                )
                if constraint_type is UpdateType.SHAPE:
                    self.update_shapes(newly_added_symbols)
                updates |= newly_added_symbols
                new_constraints = newly_added_symbols.constraints[constraint_type]

                # If a constraint is solved, get its post_constraints and add to
                # constraints set.
                if status:
                    solved_constraints.add(constr)
                    self.constraint_map.pop(constr)
                    # Remove constraint from inputs.
                    for input in inputs:
                        input.metadata.data.remove_constraint(constr)

                    post_constraints = constr.create_post_constraints()
                    for post_constr in post_constraints:
                        self.constraint_map[post_constr] = inputs

                        # Add post_constraints to inputs.
                        for input in inputs:
                            input.metadata.data.add_constraint(post_constr)

                    constraints |= post_constraints

                constraints |= new_constraints
                constraints.discard(constr)

    @staticmethod
    def _combine_nodes(updates: Updates):
        # Check if any node could be reduced after variadic updates add into
        # node_updates field.
        while updates.node_updates:
            node = updates.node_updates.pop()
            updates |= node.combine()

    def _reduce_uniadic_referees(self, updates: Updates):
        while updates.uniadic_updates:
            uni = updates.uniadic_updates.pop()
            uni_val = uni.value
            rec = uni.metadata
            for var in list(rec.vars_dict):
                # TODO: Use also pos_val!
                lengths = rec.vars_dict.get(var)
                if lengths is not None:
                    updates |= var.update_possibilities(lengths)
            # Update symbol store.
            if uni_val is not None:
                if valued_uni := self.symbol_store.get(uni_val):
                    # directly take from symbolstore if an uniadic with a same value
                    # already exists
                    valued_uni.match(uni)
                    uni = valued_uni
                    # TODO: Decide if we need to add match updates to updates
                    # (seems like no need)!
                else:
                    self.symbol_store[uni_val] = uni
            rec = uni.metadata
            # Reduce referees of rec to 1.
            # NOTE: Update all vars in these uniadics (UniadicRecord) accordingly.
            for repr, indices in rec.reprs_dict.items():
                for idx in indices:
                    if repr[idx].metadata != rec:
                        raise KeyError("Uniadic record index mismatch.")
                    repr[idx] = uni
            rec.referees = {uni}

    @staticmethod
    def _find_intersection_reprs(repr1):
        intersection_reprs: set[ShapeRepr] = set()
        # Collect visited repr's symbols.
        symbols = repr1.prefix + repr1.suffix

        if repr1.root is not None:
            intersection_reprs = set(repr1.root.reprs)

        elif len(symbols) >= 1:
            intersection_reprs = set(symbols.pop().metadata.reprs_dict.keys())

        # Iterate over reprs symbols.
        for symbol in symbols:
            # Find intersection of all symbols' reprs.
            if intersection_reprs:
                intersection_reprs &= symbol.metadata.reprs_dict.keys()

        return intersection_reprs

    @staticmethod
    def _add_sublists(repr1, intersection_reprs, deletion_nodes):
        updates = Updates()
        for repr2 in intersection_reprs:
            if (repr1.node != repr2.node) and (repr1 in repr2):
                if repr2 in repr1:
                    # Find duplicated nodes and add them to deletion_nodes.
                    update_equivalence_table(repr1.node, repr2.node, deletion_nodes)
                else:
                    updates |= subset_match(repr1, repr2)

        return updates

    def clear(self):
        self.symbol_store = {}
        self.constraint_map = {}
        self.empty_node = None

    @staticmethod
    def _delete_nodes(deletion_nodes: dict[ShapeNode, set[ShapeNode]]) -> Updates:
        updates = Updates()
        # Delete duplicated nodes
        while deletion_nodes:
            # Select one remaining node from the set.
            remaining = next(iter(deletion_nodes))
            deletion_set = deletion_nodes.pop(remaining)

            for deleted in deletion_set:
                if deleted != remaining:
                    updates |= ConstraintSolver._delete_node(remaining, deleted)
                    # clear deletion_nodes
                    deletion_nodes.pop(deleted)
        return updates

    @staticmethod
    def _delete_node(remaining: ShapeNode, deleted: ShapeNode) -> Updates:
        # Merge remaining node with the node that will be deleted.
        updates = remaining.merge(deleted)
        # Iterate over deleted nodes referees to remove deleted node.
        for ref in deleted.referees:
            ref.shape = remaining
            remaining.referees.add(ref)

        deleted.referees = set()
        deleted.reprs = []
        return updates

    def update_shapes(self, updates: Updates):
        deletion_nodes: dict[ShapeNode, set[ShapeNode]] = {}
        # Note that update can be tuple also. First element of update
        # is always Tensor | Scalar. So this is the case, get first element
        # as update.

        # Reduce updated nodes' reprs if possible.
        self._combine_nodes(updates)
        # Reduce updated UniadicRecords' referees field.
        self._reduce_uniadic_referees(updates)

        all_reprs = {
            repr for update in updates.shape_updates for repr in update.shape.reprs
        }

        # Visit all updated tensors' nodes' reprs.
        assert self.empty_node is not None
        for repr in all_reprs:
            if repr.root is None and repr.prefix == [] and self.empty_node != repr.node:
                # Unify all empty ShapeReprs use same Node
                self._delete_node(self.empty_node, repr.node)

            intersection_reprs = self._find_intersection_reprs(repr)
            # Iterate over intersections.
            updates |= self._add_sublists(repr, intersection_reprs, deletion_nodes)

        updates |= self._delete_nodes(deletion_nodes)

        # Reduce updated nodes' reprs if possible.
        self._combine_nodes(updates)
        # Reduce updated UniadicRecords' referees field.
        self._reduce_uniadic_referees(updates)

    def match(self, other: ConstraintSolver) -> Updates:
        # This method updates symbol store values.
        # TODO: Do we need to clear other
        updates = Updates()
        for val, uni in other.symbol_store.items():
            if (current_uni := self.symbol_store.get(val)) is not None:
                updates |= current_uni.match(uni)
                other.symbol_store[val] = uni
        self.symbol_store |= other.symbol_store
        self.constraint_map |= other.constraint_map
        return updates


@dataclass
class Updates:
    shape_updates: set[Tensor] = field(default_factory=lambda: set())
    value_updates: set[Tensor | Scalar] = field(default_factory=lambda: set())
    uniadic_updates: set[Uniadic] = field(default_factory=lambda: set())
    node_updates: set[ShapeNode] = field(default_factory=lambda: set())
    constraints: dict[UpdateType, set[Constraint]] = field(
        default_factory=lambda: {UpdateType.SHAPE: set(), UpdateType.TYPE: set()}
    )

    def add(
        self,
        symbol: BaseData | Uniadic | Variadic,
        update_type: UpdateType = UpdateType.SHAPE,
    ) -> None:
        # TODO: Use match case here
        if update_type == UpdateType.SHAPE:
            if isinstance(symbol, Uniadic):
                self._add_uniadic(symbol)
            elif isinstance(symbol, Variadic):
                self._add_variadic(symbol)
            else:
                assert isinstance(symbol, Scalar)
                self._add_scalar(symbol)

            # TODO: Fill here after type_updates added to class
        elif update_type == UpdateType.TYPE:
            assert isinstance(symbol, Tensor | Scalar)
            self._add_type_update(symbol)

    def _add_scalar(self, symbol: Scalar):
        self.value_updates.add(symbol)
        self.constraints[UpdateType.SHAPE] |= symbol.shape_constraints

    def _add_uniadic(self, symbol: Uniadic):
        self.uniadic_updates.add(symbol)
        for repr in symbol.metadata.reprs_dict:
            for tensor in repr.node.referees:
                self.shape_updates.add(tensor)
                self.constraints[UpdateType.SHAPE] |= tensor.shape_constraints

    def _add_variadic(self, symbol: Variadic):
        # self.symbol_updates.add(symbol)
        for repr in symbol.reprs:
            self.node_updates.add(repr.node)
            for tensor in repr.node.referees:
                self.shape_updates.add(tensor)
                self.constraints[UpdateType.SHAPE] |= tensor.shape_constraints

    def _add_type_update(self, symbol: Tensor | Scalar):
        self.constraints[UpdateType.TYPE] |= symbol.type_constraints

    def __ior__(self, other: Updates) -> Updates:
        self.constraints[UpdateType.SHAPE] |= other.constraints[UpdateType.SHAPE]
        self.constraints[UpdateType.TYPE] |= other.constraints[UpdateType.TYPE]
        self.shape_updates |= other.shape_updates
        self.uniadic_updates |= other.uniadic_updates
        self.node_updates |= other.node_updates
        self.value_updates |= other.value_updates
        return self


def _get_shapes(
    data_dict: dict[str, Tensor | Scalar],
    uniadic_keys=None,
    varadic_keys=None,
    symbolic: bool = True,
    verbose: bool = False,
    key_mappings: dict | None = None,
) -> _ShapesType:
    if key_mappings is None:
        key_mappings = {}
    if uniadic_keys is None:
        uniadic_keys = {}
    if varadic_keys is None:
        varadic_keys = {}
    shapes: dict = {}
    for key, data in data_dict.items():
        key_name = key_mappings.get(key, key)
        if isinstance(data, Tensor):
            shapes[key_name] = data.shape.get_shapes(
                uniadic_keys, varadic_keys, symbolic, verbose
            )
        else:
            shapes[key_name] = None
    return shapes


class BaseData:
    def __init__(self, type: type[TensorValueType] | ScalarType) -> None:
        self._type = type
        # self.shape: ShapeNode | None = None
        self._logical_data = True  # Every data starts as logical data.
        self.shape_constraints: set[Constraint] = set()
        self.type_constraints: set[Constraint] = set()
        self._temp_shape: ShapeRepr | None = None  # set random repr

    @property
    def is_non_diff(self) -> bool:
        raise NotImplementedError("No 'is_non_diff' property implemented.")

    @property
    def is_valued(self) -> bool:
        raise NotImplementedError("No 'is_valued' property implemented.")

    @property
    def all_constraints(self) -> set[Constraint]:
        return self.shape_constraints | self.type_constraints

    def _convert_value(self, backend: Backend) -> Any:
        raise NotImplementedError("No '_convert_value' method implemented.")

    def finalize_match(self, other: Tensor | Scalar):
        if (typ_1 := type(other)) != (typ_2 := type(self)):
            raise TypeError(
                f"Replacement can be done for only same types. Got {typ_1} and {typ_2}"
            )

        # After modifications propagate other constraints into self.
        self.shape_constraints |= other.shape_constraints
        self.type_constraints |= other.type_constraints
        other.shape_constraints = set()
        other.type_constraints = set()

    Val_Type = TypeVar("Val_Type", MainValueType, DataType)  # type: ignore

    def set_value(self, value: Val_Type):
        raise NotImplementedError("No 'set_value' method implemented.")

    def set_type(self, type: type[TensorValueType] | ScalarType | UnionType) -> Updates:
        updates = Updates()
        if not self._types_equal(type):
            updates.add(self, UpdateType.TYPE)
            new_type = find_intersection_type(type, self._type)

            if not new_type:
                raise TypeError(
                    f"Acceptable types are {self._type}, but {type} type value "
                    "is provided!"
                )
            self._type = new_type
        return updates

    def _types_equal(self, other_type: type[TensorValueType] | ScalarType) -> bool:
        self_type = (
            self._type.base_type
            if isinstance(self._type, NestedListType)
            else self._type
        )
        other_type = (
            other_type.base_type
            if isinstance(other_type, NestedListType)
            else other_type
        )
        return self_type == other_type

    def _set_as_physical(self):
        if self._logical_data:
            self._logical_data = False

    def add_constraint(self, constraint: Constraint):
        if constraint.type == UpdateType.SHAPE:
            self.shape_constraints.add(constraint)
        elif constraint.type == UpdateType.TYPE:
            self.type_constraints.add(constraint)

    def remove_constraint(self, constraint: Constraint):
        # TODO: check why pop raises!
        if constraint.type == UpdateType.SHAPE:
            self.shape_constraints.discard(constraint)
        elif constraint.type == UpdateType.TYPE:
            self.type_constraints.discard(constraint)

    def match_shapes(self, other: BaseData):
        return Updates()

    def match(self, other: Tensor | Scalar) -> Updates:
        updates = Updates()
        if self != other:
            updates = Updates()
            updates |= self.set_type(other._type)
            updates |= other.set_type(self._type)
            if isinstance(other, Tensor):
                updates |= self.match_shapes(other)

            if self.is_valued ^ other.is_valued:
                valued, non_valued = (self, other) if self.is_valued else (other, self)
                assert isinstance(valued, Tensor | Scalar)
                updates |= non_valued.set_value(valued.value)
                if non_valued == other:
                    if isinstance(other, Tensor):
                        updates.shape_updates.discard(other)
                    updates.value_updates.discard(other)

            self.finalize_match(other)
        return updates


class Tensor(BaseData, GenericDataType[DataType]):
    _type: type[float] | type[int] | type[bool] | UnionType
    temp_value: int | float | tuple | list | None | Constant | NestedListType
    value: DataType | ToBeDetermined | None

    def __init__(
        self,
        shape: ShapeNode,
        possible_types: type[float] | type[int] | type[bool] | UnionType,
        value: TensorValueType,
        interval: list[float | int] | None,
    ) -> None:
        super().__init__(type=possible_types)
        self.temp_value = None
        self.value = None
        # Update type if any value is given.
        if not isinstance(value, ToBeDetermined | None):
            self.set_type(find_dominant_type(value))
            self.temp_value = value
        else:
            self.value = value
        self.shape: ShapeNode = shape
        # Update tensor field of ShapeNode
        self.shape.referees.add(self)
        self.interval = interval

    @property
    def is_non_diff(self) -> bool:
        return self.temp_value is not None or self.value is not None

    @property
    def is_valued(self) -> bool:
        return self.value is not None and self.value is not TBD

    def _set_as_physical(self):
        super()._set_as_physical()

    def _convert_value(self, backend: Backend) -> DataType | ToBeDetermined | None:
        if isinstance(self.temp_value, Constant):
            self.value = backend.array(
                epsilon_table[backend.precision][self.temp_value]
            )
        elif self.temp_value is not None:
            self.value = backend.array(self.temp_value)
        return self.value

    def make_physical(self, backend: Backend, memo: dict[int, Tensor | Scalar]):
        physical_tensor = deepcopy(self, memo)
        # Update data as physical data.
        physical_tensor._set_as_physical()
        # Update value of physical data taking backend into account.
        physical_tensor._convert_value(backend)
        return physical_tensor

    def __deepcopy__(self, memo: dict[int, Tensor | Scalar]):
        # Check if the object is already in the memo dictionary.
        if id(self) in memo:
            return memo[id(self)]
        # Create a new instance of the class without deep copying.
        new_instance = self.__class__.__new__(self.__class__)
        # Add the new instance to the memo dictionary before populating it.
        memo[id(self)] = new_instance
        # Do not deepcopy "value" attribute.
        for k, v in self.__dict__.items():
            if k == "value":
                setattr(new_instance, k, v)
            else:
                setattr(new_instance, k, deepcopy(v, memo))
        return new_instance

    def match_shapes(self, other: Tensor):  # type: ignore[override]
        updates = Updates()
        if other.shape != self.shape:
            updates |= self.shape.merge(other.shape)
            self.shape.referees |= other.shape.referees
            prev_node = other.shape
            for ref in other.shape.referees:
                ref.shape = self.shape
            prev_node.reprs = []
            prev_node.referees = set()
        self.shape.referees.discard(other)
        return updates

        # TODO: Should we change all occurances of other object in other's shape
        # to self object? If we should, we also need to transfer "interval" attribute
        # which requires handling of interval arithmetic in logical level also.

    def set_value(self, value: DataType | TensorValueType) -> Updates:  # type: ignore[override]
        if self._logical_data:
            assert isinstance(value, TensorValueType)
            return self._set_logical_value(value)
        else:
            assert self.is_tensor_type(value)
            return self._set_physical_value(value)

    def _set_logical_value(self, value: TensorValueType) -> Updates:
        if isinstance(value, ToBeDetermined | None):
            if self.temp_value is not None:
                raise ValueError(
                    f"Value is set before as {self.temp_value}. Can not be reset."
                )
            if self.value is TBD and value is None:
                raise ValueError(
                    "Already set as non-differentiable. Can not be reverted \
                    to a differentiable state."
                )
            self.value = value
        else:
            if self.temp_value is not None and self.temp_value != value:
                raise ValueError(
                    f"Value is set before as {self.temp_value}. Can not be reset."
                )
            self.temp_value = value
        return Updates()

    def _set_physical_value(self, value: DataType) -> Updates:
        if self.is_tensor_type(self.value):
            raise ValueError(f"Value is set before as {self.value}. Can not be reset.")
        updates = Updates()
        if value is not None:
            val_type: type[bool] | type[int] | type[float]
            data_dtype = str(value.dtype)
            # Check value type is OK, and update type accordinly.
            if "bool" in data_dtype:
                val_type = bool
            elif "int" in data_dtype:
                val_type = int
            elif "float" in data_dtype:
                val_type = float
            else:
                raise TypeError(
                    f"Given type ({data_dtype}) is not supported. Only float, int or "
                    "bool types are accepted."
                )
            self.set_type(val_type)
            self.value = value
            # Set corresponding shape values.
            if value is not TBD:
                shape = list(value.shape)
                updates |= self.shape.set_values(shape)
                updates.value_updates.add(self)
        return updates


class Scalar(BaseData):
    _type: ScalarType

    def __init__(
        self,
        possible_types: ScalarType | UnionType | None = None,
        value: MainValueType = TBD,
    ) -> None:
        if possible_types is None:
            if value is TBD:
                raise Exception("No possible types or value is given!")
            else:
                possible_types = self.find_type(value)

        super().__init__(type=possible_types)
        # Update type if any value is given.
        if not isinstance(value, ToBeDetermined | str):
            self.set_type(self.find_type(value))
        self.value = value

    @property
    def is_non_diff(self) -> bool:
        return True

    @property
    def is_valued(self) -> bool:
        return self.value is not TBD

    def find_type(self, value: MainValueType | str) -> ScalarType:
        if isinstance(value, Constant):
            return constant_type_table[value]
        else:
            return find_type(value)

    def _convert_value(self, backend):
        self.value = backend.cast(self.value)
        return self.value

    def make_physical(self, backend: Backend, memo: dict[int, Tensor | Scalar]):
        new_scalar = deepcopy(self, memo)
        if id(self) not in memo:
            # Update data as physical data.
            new_scalar._set_as_physical()

        if isinstance(self.value, Constant):
            new_scalar.value = epsilon_table[backend.precision][self.value]

        return new_scalar

    def set_value(self, value: MainValueType) -> Updates:
        # Check value type!

        updates = Updates()
        prev_value = self.value

        if self.value is not TBD and self.value != value and value is not TBD:
            raise ValueError(
                f"Value is set before as {self.value}. A scalar value can not "
                "be reset."
            )

        if value is not TBD:
            updates |= self.set_type(self.find_type(value))
            self.value = value

        #  Call constraints if any change occured.
        if self.value != prev_value:
            updates.add(self)
        return updates


class TensorType:
    def __init__(
        self,
        shape_template: ShapeTemplateType,
        possible_types: type | UnionType = float | int | bool,
        value: int | float | bool | Constant | ToBeDetermined | None = None,
        interval: list[float | int] | None = None,
    ) -> None:
        self._type = possible_types
        self.shape_template = shape_template
        self.value = value
        if interval is None:
            interval = []
        self.interval = interval

    def construct(self, shape_node: ShapeNode) -> Tensor:
        return Tensor(
            shape=shape_node,
            possible_types=self._type,
            value=self.value,
            interval=self.interval,
        )


@dataclass
class IOHyperEdge:
    data: Tensor | Scalar
    key_origin: str | None = None

    @property
    def shape(self) -> ShapeNode | None:
        if isinstance(self.data, Tensor):
            return self.data.shape
        # TODO: Consider raising an error for Scalar type.
        return None

    def __hash__(self) -> int:
        return hash(id(self))


class TemplateBase:
    def __getitem__(
        self,
        key: slice
        | int
        | EllipsisType
        | tuple[slice | int | None | EllipsisType, ...]
        | IOKey
        | None,
    ):
        if key is ...:
            key = slice(None)
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            return ExtendTemplate(connections=[self, start, stop, step], model="slice")
        elif isinstance(key, int | tuple):
            return ExtendTemplate(connections=[self, key], model="item")
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def __add__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="add")

    def __radd__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="add")

    def __sub__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="sub")

    def __rsub__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="sub")

    def __mul__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="mul")

    def __rmul__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="mul")

    def __truediv__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="div")

    def __rtruediv__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="div")

    def __floordiv__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="fdiv")

    def __rfloordiv__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="fdiv")

    def __pow__(self, other: TemplateConnectionType):
        return ExtendTemplate(
            connections=[self, other], model="pow", defaults={"robust", "threshold"}
        )

    def __rpow__(self, other: TemplateConnectionType):
        return ExtendTemplate(
            connections=[other, self], model="pow", defaults={"robust", "threshold"}
        )

    def __matmul__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="matmul")

    def __gt__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="gt")

    def __rgt__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="gt")

    def __ge__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="ge")

    def __rge__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="ge")

    def __lt__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="lt")

    def __rlt__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="lt")

    def __le__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="le")

    def __rle__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="le")

    def __eq__(self, other: object):
        if isinstance(other, int | float | bool | list | Connection | IOKey | tuple):
            return ExtendTemplate(connections=[self, other], model="eq")
        else:
            raise ValueError("Unsupported type for equality operation.")

    def __req__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="eq")

    def __ne__(self, other: object):
        if isinstance(other, int | float | bool | list | Connection | IOKey | tuple):
            return ExtendTemplate(connections=[self, other], model="ne")
        else:
            raise ValueError("Unsupported type for equality operation.")

    def __rne__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="ne")

    def __and__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="and")

    def __rand__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="and")

    def __or__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="or")

    def __ror__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="or")

    def __xor__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="xor")

    def __rxor__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="xor")

    def __lshift__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="lshift")

    def __rlshift__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="lshift")

    def __rshift__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[self, other], model="rshift")

    def __rrshift__(self, other: TemplateConnectionType):
        return ExtendTemplate(connections=[other, self], model="rshift")

    def __invert__(self):
        return ExtendTemplate(connections=[self], model="not")

    def __neg__(self):
        return ExtendTemplate(connections=[self], model="minus")

    def abs(self):
        return ExtendTemplate(connections=[self], model="abs")

    def len(self):
        return ExtendTemplate(connections=[self], model="len")

    def shape(self):
        return ExtendTemplate(connections=[self], model="shape")

    def reshape(self, shape: tuple[int, ...] | TemplateBase):
        return ExtendTemplate(connections=[self, shape], model="reshape")

    def size(self, dim: int | tuple[int, ...] | TemplateBase | None = None):
        return ExtendTemplate(connections=[self, dim], model="size")

    def tensor(self):
        return ExtendTemplate(connections=[self], model="tensor")

    def mean(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ):
        return ExtendTemplate(connections=[self, axis, keepdim], model="mean")

    def sum(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ):
        return ExtendTemplate(connections=[self, axis, keepdim], model="sum")

    def max(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ):
        return ExtendTemplate(connections=[self, axis, keepdim], model="max")

    def min(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ):
        return ExtendTemplate(connections=[self, axis, keepdim], model="min")

    def prod(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ):
        return ExtendTemplate(connections=[self, axis, keepdim], model="prod")

    def var(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
        correction: float | None = 0.0,
    ):
        return ExtendTemplate(
            connections=[self, axis, keepdim, correction], model="var"
        )

    def sqrt(self):
        return ExtendTemplate(
            connections=[self], model="sqrt", defaults={"robust", "cutoff"}
        )

    def transpose(self, axes: tuple[int, ...] | TemplateBase | None = None):
        return ExtendTemplate(connections=[self, axes], model="transpose")


class ExtendTemplate(TemplateBase):
    output_connection: ConnectionData | None

    def __init__(
        self,
        connections: list[TemplateConnectionType],
        model: str,
        defaults: set[str] | None = None,
    ) -> None:
        for connection in connections:
            if isinstance(connection, str):
                raise ValueError(
                    "In extend template operations, 'str' is not a valid type."
                )

        self.connections = connections
        self.model = model

        if defaults is None:
            defaults = set()
        self.defaults = defaults
        self.output_connection = None


class IOKey(TemplateBase):
    def __init__(
        self,
        name: str | None = None,
        value: MainValueType | NullConnection = NOT_GIVEN,
        shape: ShapeTemplateType | None = None,
        type: UnionType | type | None = None,
        expose: bool = True,
    ) -> None:
        super().__init__()
        self._name = name
        self._value = value
        self._shape = shape
        self._type = type
        self._expose = expose

        # TODO: Shape should not be [] also!
        if self._value != NOT_GIVEN and self._shape is not None and self._shape != []:
            raise ValueError(
                f"Scalar values are shapeless, shape should be None or []. "
                f"Got {self._shape}."
            )

        if self._value != NOT_GIVEN and self._type is not None:
            value_type = find_type(self._value)
            if value_type != self._type:
                raise TypeError(
                    f"type of the given value and given type does not match. Given "
                    f"type is {self._type} while type of value is {value_type}"
                )

    def __hash__(self) -> int:
        return hash(id(self))


class Connection(TemplateBase):
    def __init__(self, key, metadata, is_key_autogenerated: bool):
        self.data = ConnectionData(key, metadata, is_key_autogenerated, self)

    @property
    def key(self):
        return self.data.key

    @property
    def metadata(self):
        return self.data.metadata

    def __hash__(self) -> int:
        return hash(id(self))


ShapesType = (
    Mapping[str | Connection, ShapeTemplateType]
    | Mapping[str, ShapeTemplateType]
    | Mapping[Connection, ShapeTemplateType]
)
_ShapesType = Mapping[str, ShapeTemplateType]


@dataclass
class ConnectionData:
    # TODO: This class updated as mutable. Update docstrings accordingly!
    """Immutable dataclass object which holds model instance, key
    and I/O info. It is immutable because once a model's input-output
    names are defined, changing them is not allowed for proper DAG
    connections.
    """

    key: str
    metadata: IOHyperEdge
    # TODO: remove is_key_autogenerated field
    is_key_autogenerated: bool
    conn: Connection

    def __hash__(self) -> int:
        return hash(id(self))

    def __and__(self, other) -> Connect:
        return Connect(self.conn) & other

    def __eq__(self, other) -> bool:
        return id(self) == id(other)


TemplateConnectionType = (
    TemplateBase
    | int
    | float
    | list[int | float]
    | tuple[slice | int | None | EllipsisType, ...]
    | None
)


class Connect:
    def __init__(self, *connections: Connection | str, key: IOKey | None = None):
        self.connections: OrderedSet[ConnectionData | str] = OrderedSet()
        self.key = key
        for item in connections:
            conn: ConnectionData | str
            if isinstance(item, Connection):
                conn = item.data
            elif isinstance(item, str):
                conn = item
            else:
                raise KeyError("Requires Connection object or string!")

            self.connections.add(conn)


ConnectionType = (
    str
    | ConnectionData
    | Connect
    | MainValueType
    | ExtendTemplate
    | NullConnection
    | IOKey
    | Connection
    | NotAvailable
    | NestedListType
)


class Connections:
    """This class maintains all the connections and their operations in model.
    _input_dict, _output_dict and _internal_dict stores added / updated connections
    and metadata_dict contains all metadata of the connections in _input_dict,
    _output_dict and _internal_dict. Metadata dict is updated in case of metadata
    merger.
    """

    def __init__(self) -> None:
        self._connection_dict: dict[KeyType, dict[str, ConnectionData]] = {
            key_type: {} for key_type in KeyType
        }

        self.metadata_dict: dict[IOHyperEdge, set[ConnectionData]] = {}
        self.connections_dict: dict[IOHyperEdge, set[Connections]] = {}

    @property
    def input_keys(self):
        return self._connection_dict[KeyType.INPUT].keys()

    @property
    def input_connections(self):
        return self._connection_dict[KeyType.INPUT].values()

    @property
    def output_keys(self):
        return self._connection_dict[KeyType.OUTPUT].keys()

    @property
    def output_connections(self):
        return self._connection_dict[KeyType.OUTPUT].values()

    @property
    def internal_keys(self):
        return self._connection_dict[KeyType.INTERNAL].keys()

    @property
    def internal_connections(self):
        return self._connection_dict[KeyType.INTERNAL].values()

    @property
    def all(self) -> dict[str, ConnectionData]:
        return (
            self._connection_dict[KeyType.INTERNAL]
            | self._connection_dict[KeyType.INPUT]
            | self._connection_dict[KeyType.OUTPUT]
        )

    @property
    def io_keys(self):
        return (
            self._connection_dict[KeyType.INPUT] | self._connection_dict[KeyType.OUTPUT]
        ).keys()

    def add(
        self,
        connection: ConnectionData,
    ):
        metadata = connection.metadata
        self.metadata_dict.setdefault(metadata, set()).add(connection)
        self.connections_dict.setdefault(metadata, set()).add(self)

    def set_connection_type(
        self, connection: ConnectionData, con_type: KeyType
    ) -> None:
        if con_type == KeyType.OUTPUT and connection.is_key_autogenerated:
            raise KeyError("Connection without a name cannot be set as output")
        self._set_connection_type(connection, con_type)

    def _set_connection_type(
        self, connection: ConnectionData, con_type: KeyType
    ) -> None:
        key = connection.key
        for _type in KeyType:
            if _type == con_type:
                self._connection_dict[_type][key] = connection
            else:
                self._connection_dict[_type].pop(key, None)

    def remove_connection(self, connection: ConnectionData) -> None:
        for _type in KeyType:
            self._connection_dict[_type].pop(connection.key, None)

    def get_data(self, key: str) -> Scalar | Tensor:
        # if (metadata := self._get_metadata(key)) is not None:
        #     return metadata.data
        # raise KeyError(f"Key {key} is not found in connections.")
        return self._get_metadata(key).data

    def get_non_diff_keys(self):
        return {key for key, conn in self.all.items() if conn.metadata.data.is_non_diff}

    def is_key_non_diff(self, key: str) -> bool:
        return self.get_data(key).is_non_diff

    def get_connection(self, key: str) -> ConnectionData | None:
        internals = self._connection_dict[KeyType.INTERNAL]
        inputs = self._connection_dict[KeyType.INPUT]
        outputs = self._connection_dict[KeyType.OUTPUT]
        return internals.get(key, inputs.get(key, outputs.get(key)))

    def get_con_by_metadata(self, key: IOHyperEdge) -> ConnectionData | None:
        conns = self.metadata_dict.get(key)
        if conns is not None:
            return next(iter(conns))
        return conns

    def get_cons_by_metadata(self, key: IOHyperEdge):
        return self.metadata_dict.get(key)

    def _get_metadata(self, key: str) -> IOHyperEdge:
        if (con := self.get_connection(key)) is not None:
            return con.metadata
        raise KeyError(f"Key {key} is not found in connections.")

    def get_key_origin(self, key: str) -> str | None:
        return self._get_metadata(key).key_origin

    def get_shape_node(self, key: str) -> ShapeNode:
        data = self._get_metadata(key).data
        assert isinstance(data, Tensor)
        return data.shape

    def set_value(self, con: ConnectionData, value: MainValueType):
        self.get_data(con.key).set_value(value)

    def extract_metadata(self, key: str | Connection) -> IOHyperEdge:
        if isinstance(key, Connection):
            # Extract the key from the Connection object.
            metadata = key.metadata
        else:
            metadata = self._get_metadata(key)
        return metadata


class Uniadic:
    def __init__(self, value: int | set[int] | None = None) -> None:
        # TODO: we could accept *value as input to initialize Uniadic.
        self.metadata = UniadicRecord()
        self.metadata.update_possible_values(value)
        self.metadata.referees.add(self)

    @property
    def value(self):
        return self.metadata.value

    @property
    def possible_values(self):
        return self.metadata.possible_values

    @property
    def reprs(self):
        return self.metadata.reprs

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Uniadic) -> bool:  # type: ignore
        return id(self.metadata) == id(other.metadata)

    def set_value(self, value):  # Do we need set_value
        prev_value = self.metadata.possible_values
        new_value = self.metadata.update_possible_values(value)
        return prev_value != new_value

    def update_possible_values(self, values: int | set[int] | None) -> Updates:
        updates = Updates()
        prev_values = self.metadata.possible_values
        new_values = self.metadata.update_possible_values(values)
        if prev_values != new_values:
            updates.add(self)
        return updates

    def match(self, other: Uniadic) -> Updates:
        updates = Updates()
        if self.metadata != other.metadata:
            if self.value == other.value and (
                len(self.metadata.reprs_dict) > 0 and len(other.metadata.reprs_dict) > 0
            ):
                updates.add(self)
            else:
                main_pos_val = copy(self.possible_values)
                updates |= self.update_possible_values(other.possible_values)
                updates |= other.update_possible_values(main_pos_val)
            self.metadata.match(other.metadata)
        return updates

    def __and__(self, other: Uniadic) -> set[int] | None:
        match (self.possible_values, other.possible_values):
            case (None, _ as pos) | (_ as pos, None):
                return pos
            case _:
                return self.possible_values & other.possible_values

    # def __and__(self, other: Uniadic) -> set[int] | None:
    #     match (self.possible_values, other.possible_values):
    #         case (None, _ as pos) | (_ as pos, None):
    #             return pos
    #         case (int() as _number, set() as pos) | (set() as pos, int() as _number):
    #             return pos & {_number}
    #         case _:
    #             return self.possible_values & other.possible_values


@dataclass
class UniadicRecord:
    possible_values: set[int] | None = None
    referees: set[Uniadic] = field(default_factory=lambda: set())
    reprs_dict: dict[ShapeRepr, set[int]] = field(default_factory=lambda: {})
    vars_dict: dict[Variadic, set[int]] = field(default_factory=lambda: {})

    @property
    def reprs(self):
        return self.reprs_dict.keys()

    @property
    def value(self):
        if self.possible_values is not None and len(self.possible_values) == 1:
            return next(iter(self.possible_values))
        else:
            return None

    def update_possible_values(self, values: int | set[int] | None) -> set[int] | None:
        # TODO: Check if all elements of set are int!
        if isinstance(values, int):
            values = {values}
        if values == set():
            raise ValueError("Possible value set could not be empty!")
        elif values is None:
            return self.possible_values
        elif self.possible_values is None:
            self.possible_values = values
        elif values is not None and len(intersect := self.possible_values & values) > 0:
            self.possible_values = intersect
        elif values is not None and self.possible_values is not None:
            raise ValueError("Possible values mismatch!")
        return self.possible_values

    def match(self, other: UniadicRecord):
        if id(self) != id(other):
            self.update_possible_values(other.possible_values)
            # TODO: Is it required to check other.referees!
            # if not other.referees:
            #     raise ValueError("Encountered a removed UniadicRecord!")
            for uniadic in other.referees:
                # Update all referees' metadata
                uniadic.metadata = self
                self.referees.add(uniadic)
            for repr, indices in other.reprs_dict.items():
                self.reprs_dict.setdefault(repr, set()).update(indices)
            for var, pos_set in other.vars_dict.items():
                self.vars_dict.setdefault(var, set()).update(pos_set)
                var.uni_metadata_set.discard(other)
                var.uni_metadata_set.add(self)
            other.reprs_dict = {}
            other.referees = set()
            return self.value

    def __hash__(self) -> int:
        return hash(id(self))


def intersect_values(
    values1: set[int] | None, values2: set[int] | None
) -> set[int] | None:
    if values1 is None and values2 is None:
        return None
    elif values1 is None:
        return values2
    elif values2 is None:
        return values1
    else:
        return values1 & values2


@dataclass
class Equivalences:
    uniadics: set[Uniadic] = field(default_factory=lambda: set())
    values: set[int] | None = None

    def __contains__(self, other: Equivalences) -> bool:
        for uni in other.uniadics:
            # TODO: list makes this comparison N2 complexity.
            # Note that set looks for hash values but we have
            # __eq__ method for Uniadic class.
            if uni not in list(self.uniadics):
                return False
        if other.values is not None:
            if self.values is None:
                return True
            return self.values.issubset(other.values)
        return True

    def add_uniadic(self, uni: Uniadic) -> tuple[bool, bool]:
        self.uniadics.add(uni)
        return self.intersect_values(uni.possible_values)

    def intersect_values(self, values: set[int] | None) -> tuple[bool, bool]:
        # Returns tuple[is_applicable, is_updated]
        prev_val = None if self.values is None else set(self.values)
        self.values = intersect_values(self.values, values)
        return self.values != set(), prev_val != self.values


@dataclass
class PossibleValues:
    # Expresses Variadic's single possible value in terms of Possible Uniadics.
    uniadics: tuple[Uniadic, ...] = ()
    # Expresses required condition of given Possible Uniadics in Disjunctive Normal Form
    dnf_list: list[DNF] = field(default_factory=lambda: [])

    def __post_init__(self):
        # TODO: Check applicability
        self._is_applicable: bool = True
        self.dnf_lookup_table: dict[Uniadic, Equivalences] = {}
        self.update_dnf()

    @property
    def is_applicable(self) -> bool:
        return self._is_applicable

    def check_is_subset(self, other: PossibleValues) -> bool:
        # If any Uniadics differ, return False
        for other_uni, self_uni in zip(other.uniadics, self.uniadics, strict=False):
            if other_uni != self_uni:
                look_up = self.dnf_lookup_table.get(self_uni)
                if look_up is None or other_uni not in look_up.uniadics:
                    return False

        # if self.uniadics != other.uniadics:
        #     return False

        for uni, equ in other.dnf_lookup_table.items():
            if (self_equ := self.dnf_lookup_table.get(uni)) is None:
                for _uni, val in self.dnf_lookup_table.items():
                    if uni.metadata == _uni.metadata:
                        self_equ = val
            if self_equ is None or equ not in self_equ:
                return False
        return True

    def merge(self, other: PossibleValues) -> tuple[bool, bool]:
        # Add other's dnf_list and uniadics equality to dnf_list
        if self == other or self.check_is_subset(other):
            return True, False
        # TODO: Check if other has same info but different object
        self.dnf_list += other.dnf_list + [
            DNF([AND({u1: u2})])
            for u1, u2 in zip(self.uniadics, other.uniadics, strict=True)
        ]

        # Update dnf
        return self.update_dnf(), True

    def update_dnf(self) -> bool:
        while True:
            is_updated = False
            for dnf in self.dnf_list:
                is_applicable, _is_updated, new_dnfs = dnf.update(self.dnf_lookup_table)
                self.dnf_list += new_dnfs
                is_updated |= _is_updated
                if not is_applicable:
                    # TODO: DELETE PossibleValues from all Uniadics in this
                    # PossibleValues.
                    self._is_applicable = False
                    return False
            if not is_updated:
                break
        return True

    def get_all_uniadics(self) -> set[Uniadic]:
        # TODO: add all ANDs Uniadics into lookup table, then remove this method and
        # directly call self.lookup_table.keys()
        uniadics = set()
        for dnf in self.dnf_list:
            for item in dnf.item_list:
                for key, value in item.uni_table.items():
                    uniadics.add(key)
                    if isinstance(value, Uniadic):
                        uniadics.add(value)
        return uniadics


@dataclass
class AND:
    uni_table: dict[Uniadic, Uniadic | int]

    def check(self, lookup_table: dict[Uniadic, Equivalences]) -> bool:
        result = True
        local_lookup: dict[Uniadic, set[int] | None] = {}
        for key, value in self.uni_table.items():
            if key in lookup_table:
                local_lookup |= {
                    _key: lookup_table[key].values
                    for _key in lookup_table[key].uniadics
                }
            else:
                local_lookup[key] = key.possible_values

            if isinstance(value, Uniadic):
                if value in lookup_table:
                    local_lookup |= {
                        _key: lookup_table[value].values
                        for _key in lookup_table[value].uniadics
                    }
                else:
                    local_lookup[value] = value.possible_values
                eq_val = local_lookup[value]
            else:
                eq_val = {value}
            result &= intersect_values(local_lookup[key], eq_val) != set()
            # TODO: break if false?
        return result

    def is_equal(self, other: AND) -> bool:
        if len(self.uni_table) != len(other.uni_table):
            return False
        for uni1, val1 in self.uni_table.items():
            for uni2, val2 in self.uni_table.items():
                if uni1 == uni2 and val1 == val2:
                    break
            else:
                return False
        return True


@dataclass
class DNF:
    item_list: list[AND]

    def update(
        self, lookup_table: dict[Uniadic, Equivalences]
    ) -> tuple[bool, bool, list[DNF]]:
        current_len = len(self.item_list)
        # Returns tuple[is_applicable, is_updated]
        self.item_list = [item for item in self.item_list if item.check(lookup_table)]
        if len(self.item_list) == 0:
            return current_len != 0, current_len != len(self.item_list), []
        elif len(self.item_list) == 1:
            # Update lookup table
            for key, value in self.item_list[0].uni_table.items():
                if key not in lookup_table:
                    lookup_table[key] = Equivalences()
                is_applicable, is_updated = lookup_table[key].add_uniadic(key)
                if isinstance(value, Uniadic):
                    if value in lookup_table:
                        _is_applicable, _is_updated = lookup_table[
                            key
                        ].intersect_values(lookup_table[value].values)
                        is_applicable &= _is_applicable
                        is_updated |= _is_updated
                        for uni in lookup_table[value].uniadics:
                            lookup_table[uni] = lookup_table[key]
                            lookup_table[key].uniadics.add(uni)
                    else:
                        lookup_table[value] = lookup_table[key]
                    _is_applicable, _is_updated = lookup_table[key].add_uniadic(value)
                else:
                    _is_applicable, _is_updated = lookup_table[key].intersect_values(
                        {value}
                    )
                is_applicable &= _is_applicable
                is_updated |= _is_updated
                if not is_applicable:
                    return is_applicable, is_updated, []
            return is_applicable, is_updated, []

        # Extract common terms in ANDs
        uniadics: dict[Uniadic, Uniadic | int] = {}
        for idx, item in enumerate(self.item_list):
            if idx == 0:
                uniadics |= item.uni_table
            elif intersect := (uniadics.keys() & item.uni_table.keys()):
                uniadics = {
                    uni: uniadics[uni]
                    for uni in intersect
                    if uniadics[uni] == item.uni_table[uni]
                }
            else:
                uniadics = {}
                break

        new_dnfs = []
        for uni, value in uniadics.items():
            new_dnfs.append(DNF([AND({uni: value})]))
            # get rid of this uni in all ANDs
            for _item in self.item_list:
                _item.uni_table.pop(uni, None)

        return True, new_dnfs != [], new_dnfs


@dataclass
class Variadic:
    reprs: set[ShapeRepr] = field(default_factory=lambda: set())
    # Key indicates the length of possible values length is unique.
    possibles: dict[int, PossibleValues] | None = None
    uni_metadata_set: set[UniadicRecord] = field(default_factory=lambda: set())

    def _match(
        self,
        new_root: Variadic | None,
        new_prefix: list[Uniadic],
        new_suffix: list[Uniadic],
    ) -> Updates:
        for repr in self.reprs:
            if repr.root != new_root:
                # Update reprs_dict indices for new suffix and new_prefix.
                for idx, uniadic in enumerate(new_prefix):
                    uniadic.metadata.reprs_dict.setdefault(repr, set()).add(
                        len(repr.prefix) + idx
                    )
                for idx, uniadic in enumerate(new_suffix[::-1]):
                    uniadic.metadata.reprs_dict.setdefault(repr, set()).add(
                        -(len(repr.suffix) + idx + 1)
                    )
                # Update repr's root, prefix and suffix accordingly.
                repr.root = new_root
                if new_root is not None:
                    new_root.reprs.add(repr)
                    repr.prefix += new_prefix
                    repr.suffix = new_suffix + repr.suffix
                else:
                    repr.prefix += new_prefix + repr.suffix
                    repr.suffix = []
        updates = Updates()
        updates.add(self)
        # Add uniadics to only uniadic_updates
        updates.uniadic_updates |= {
            uni for uni in new_prefix + new_suffix if uni.value is not None
        }
        # TODO: Remove self.uni_metadata_set access uniadics using DNF as below:
        # if self.possibles is not None:
        # for pos in self.possibles.values():
        #     for uni in pos.get_all_uniadics():
        #         uni.metadata.vars_dict.pop(self, None)
        #     for uni in pos.uniadics:
        #         uni.metadata.vars_dict.pop(self, None)
        for metadata in self.uni_metadata_set:
            metadata.vars_dict.pop(self, None)
        return updates

    def match(
        self,
        new_root: Variadic | None = None,
        new_prefix: list[Uniadic] | None = None,
        new_suffix: list[Uniadic] | None = None,
    ) -> Updates:
        if new_prefix is None:
            new_prefix = []
        if new_suffix is None:
            new_suffix = []
        if new_root is not None:
            updates = self._match(new_root, new_prefix, new_suffix)
            if self.possibles is not None:
                updates |= self._match_possibles(new_root, new_prefix, new_suffix)
        else:
            if new_suffix != []:
                raise ValueError(
                    "Suffix could only be non-empty if another root is given!"
                )
            updates = self.update_possible_values(PossibleValues(tuple(new_prefix)))
        self.reprs = set()
        return updates

    def _match_possibles(
        self, root: Variadic, prefix: list[Uniadic], suffix: list[Uniadic]
    ) -> Updates:
        assert self.possibles is not None
        updates = Updates()
        # Clip self.possibles with new_prefix and new_suffix
        # Add clipped uniadics to new equivalences.
        possibles = []
        for _len, pos in self.possibles.items():
            if _len < len(prefix) + len(suffix):
                continue
            # Insert equivalences after clipping
            prefix_eq = [
                DNF([AND({u1: u2})])
                for u1, u2 in zip(prefix, pos.uniadics, strict=False)
            ]
            suffix_eq = [
                DNF([AND({u1: u2})])
                for u1, u2 in zip(suffix[::-1], pos.uniadics[::-1], strict=False)
            ]
            # Clip possible values according to given prefix and suffix
            pos = PossibleValues(
                pos.uniadics[len(prefix) : len(pos.uniadics) - len(suffix)],
                pos.dnf_list + prefix_eq + suffix_eq,
            )
            # if pos.is_applicable:
            possibles.append(pos)

        updates |= root.update_possible_values(*possibles)
        return updates

    def update_possible_values(self, *possibles: PossibleValues) -> Updates:
        updates = Updates()
        # This method accepts all possible values for the Variadic.
        if len(possibles) == 0:
            raise ValueError("Variadic possible values could not be empty!")

        possibles_dict: dict[int, PossibleValues] = {
            len(pos.uniadics): pos for pos in possibles
        }
        for length, pos in possibles_dict.items():
            for uni in pos.get_all_uniadics():
                uni.metadata.vars_dict.setdefault(self, set()).add(length)
                self.uni_metadata_set.add(uni.metadata)

        # Initially set possibles
        if self.possibles is None:
            self.possibles = possibles_dict
            updates.add(self)
        else:
            _possibles = {}
            for _len in self.possibles.keys() & possibles_dict.keys():
                status, any_change = self.possibles[_len].merge(possibles_dict[_len])
                if status:
                    _possibles[_len] = self.possibles[_len]
                if any_change:
                    updates.add(self)

            if len(_possibles) != len(self.possibles):
                updates.add(self)

            self.possibles = _possibles

        # TODO: DNFs updated two times!
        updates |= self.update_possibilities()
        return updates

    def extract_uniadics(self) -> Updates:
        # Extract uniadic if it exists in all possible values.
        updates = Updates()
        if self.possibles is None or self.min_len == 0:
            return updates

        # Initially check from prefix.
        extracted_prefix: list[Uniadic] = []
        for idx, uni in enumerate(self.possibles[self.min_len].uniadics):
            extracted_uni_len = len(extracted_prefix)
            for pos in self.possibles.values():
                unis = pos.uniadics
                u_idx = unis[idx]
                if u_idx.metadata != uni.metadata or (
                    u_idx.value is not None and u_idx.value != uni.value
                ):
                    break
            else:
                extracted_prefix.append(uni)

            if extracted_uni_len == len(extracted_prefix):  # No uniadic extracted
                break

        # After checking from prefix then check from suffix.
        # TODO: Functionalize this part
        extracted_suffix: list[Uniadic] = []
        for idx, uni in enumerate(self.possibles[self.min_len].uniadics[::-1]):
            rev_idx = -idx - 1
            if self.min_len - len(extracted_prefix) - len(extracted_suffix) <= 0:
                break
            extracted_len = len(extracted_suffix)
            for pos in self.possibles.values():
                unis = pos.uniadics
                _uni = unis[rev_idx]
                if (
                    len(unis) <= idx
                    or _uni.metadata != uni.metadata
                    or (_uni.value is not None and _uni.value != uni.value)
                ):
                    break
            else:
                extracted_suffix.append(uni)

            if extracted_len == len(extracted_suffix):  # No uniadic extracted
                break
        extracted_suffix = extracted_suffix[::-1]

        if extracted_prefix != [] or extracted_suffix != []:
            new_root = Variadic()
            updates |= self.match(new_root, extracted_prefix, extracted_suffix)

        return updates

    def update_possibilities(self, lengths: set[int] | None = None) -> Updates:
        updates = Updates()
        if self.possibles is None:
            return updates

        if lengths is None:
            lengths = set(self.possibles.keys())

        # Iterate over all possibilities
        single_dnfs: list[DNF] | None = None
        for _len in set(self.possibles.keys()):
            # Check validity of PossibleValues considering
            # its cnf & equivalences, also update cnf.
            if _len in lengths and not self.possibles[_len].update_dnf():
                updates.add(self)
                pos = self.possibles.pop(_len)
                # Remove Variadic from corresponding Uniadics vars_dict.
                for uni in pos.get_all_uniadics():
                    vars_dict = uni.metadata.vars_dict
                    # Another Uniadic may have same UniadicRecord with this Uniadic
                    # and already removed required _len or self. Check if self
                    # exists in vars_dict first.
                    if self in vars_dict:
                        vars_dict[self].discard(_len)
                        # If no index left for this Variadic,
                        # remove it from vars_dict.
                        if vars_dict[self] == []:
                            vars_dict.pop(self)

            elif single_dnfs is None:
                # Initially fill single_dnfs with first length.
                single_dnfs = []
                for dnf in self.possibles[_len].dnf_list:
                    if len(dnf.item_list) == 1:
                        single_dnfs.append(dnf)
            elif single_dnfs != []:
                # Intersect with existing single dnfs. If single_dnfs == [] this
                # means intersection is empty, no need for further intersection.
                _single_dnfs = []
                for dnf in self.possibles[_len].dnf_list:
                    if len(dnf.item_list) == 1:
                        and1 = dnf.item_list[0]
                        for s_dnf in single_dnfs:
                            if and1.is_equal(s_dnf.item_list[0]):
                                _single_dnfs.append(dnf)
                                break
                single_dnfs = _single_dnfs

        if single_dnfs is not None:
            for dnf in single_dnfs:
                # TODO: clear applied DNFs from possible values
                for uni, val in dnf.item_list[0].uni_table.items():
                    # TODO: clear matched uniadics' vars_dict
                    if isinstance(val, Uniadic):
                        updates |= uni.match(val)
                    else:
                        if uni.set_value(val):
                            updates.add(uni)

        # If there exists single possibility, apply post processes.
        if len(self.possibles) == 1:
            # Match all equivalences in this possibility
            possibles_vals: PossibleValues = next(iter(self.possibles.values()))
            for dnf in possibles_vals.dnf_list:
                if len(dnf.item_list) != 1:
                    break
            else:
                # Apply DNF conditions
                for dnf in possibles_vals.dnf_list:
                    for key, value in dnf.item_list[0].uni_table.items():
                        if isinstance(value, Uniadic):
                            updates |= key.match(value)
                        else:
                            if key.set_value(value):
                                updates.add(key)

                # If there exists single dnf possibilities,
                # then update Uniadics with values in dnf.
                self.possibles = None

                updates |= self._match(
                    new_root=None,
                    new_prefix=list(possibles_vals.uniadics),
                    new_suffix=[],
                )

                # Remove Variadic from corresponding Uniadics vars_dict.
                for uni in possibles_vals.get_all_uniadics():
                    # NOTE: Uniadics which share same UniadicRecord can call
                    # pop method multiple times. So we add None into pop method.
                    uni.metadata.vars_dict.pop(self, None)
                return updates

        elif self.possibles == {}:
            # raise ValueError("Possible values of Variadic could not be empty!")
            raise ValueError("Incompatible possible values for Variadic!")

        updates |= self.extract_uniadics()
        return updates

    @property  # TODO: If not necessary, remove it
    def min_len(self):
        assert self.possibles is not None
        return min(self.possibles.keys())

    @property  # TODO: If not necessary, remove it
    def max_len(self):
        assert self.possibles is not None
        return max(self.possibles.keys())

    def __hash__(self) -> int:
        return hash(id(self))


def subset_match(sub_repr: ShapeRepr, main_repr: ShapeRepr):
    updates = Updates()
    for nodes_repr in list(sub_repr.node.reprs):
        if not nodes_repr.is_equal(sub_repr):
            prefix = (
                main_repr.prefix[: len(main_repr.prefix) - len(sub_repr.prefix)]
                + nodes_repr.prefix
            )
            suffix = nodes_repr.suffix + main_repr.suffix[len(sub_repr.suffix) :]
            for repr in list(main_repr.node.reprs):
                if not (
                    (len(prefix) > len(repr.prefix) and len(suffix) < len(repr.suffix))
                    or (
                        (len(prefix) < len(repr.prefix))
                        and (len(suffix) > len(repr.suffix))
                    )
                ):
                    # Match corresponding parts and break the loop
                    updates |= repr.inner_match(prefix, nodes_repr.root, suffix)
                    break
            else:
                main_repr.node.add_repr(ShapeRepr(prefix, nodes_repr.root, suffix))
    return updates


def are_unis_identical(unis1: list[Uniadic], unis2: list[Uniadic]):
    for uni1, uni2 in zip(unis1, unis2, strict=False):
        # if (
        #     uni1.possible_values is None
        #     or uni2.possible_values is None
        #     or uni1.possible_values & uni2.possible_values == set()
        # ):
        if (
            uni1.possible_values is not None
            and uni2.possible_values is not None
            and uni1.possible_values & uni2.possible_values == set()
        ):
            return False
    return True


def handle_numerical_incompatibility(
    main_root: Variadic,
    other_root: Variadic,
    remaining_prefix: list[Uniadic],
    remaining_suffix: list[Uniadic],
) -> tuple[Updates, bool]:
    updates = Updates()
    update_status = False

    # In the case of one of the remaining affix has longer length than
    # other one, clip the long one to the length of small one
    if len(remaining_prefix) > len(remaining_suffix):
        # clipped suffix will remain same in this case
        clipped_prefix = remaining_prefix[
            len(remaining_prefix) - len(remaining_suffix) :
        ]
        clipped_suffix = remaining_suffix

    elif len(remaining_prefix) < len(remaining_suffix):
        # clipped prefix will remain same in this case
        clipped_suffix = remaining_suffix[: len(remaining_prefix)]
        clipped_prefix = remaining_prefix
    else:
        # both will remain same as they have equal lenghts
        clipped_prefix = remaining_prefix
        clipped_suffix = remaining_suffix

    # find clipped amonut for each prefix and suffix
    clipped_prefix_amount = len(remaining_prefix) - len(clipped_prefix)
    clipped_suffix_amount = len(remaining_suffix) - len(clipped_suffix)

    for idx in range(len(clipped_prefix)):
        # Check all combinations of clipped_prefix and clipped_suffix,
        # Find the first combination that can be identical and break the loop
        if are_unis_identical(
            clipped_suffix[: len(clipped_prefix) - idx], clipped_prefix[idx:]
        ):
            if idx > 0:
                # meaning that first few combinations of prefix and suffix cannot be
                # the same. However, latter combinations of prefix and suffix can
                # overlap and different variadics cannot be removed

                # Example case:
                # [1, 2, V1] => [1, 2, V3, 3]
                # [V2, 2, 3]    [1, V4, 2, 3]

                update_status = True
                updates |= main_root.match(
                    Variadic(),
                    [],
                    remaining_suffix[
                        len(remaining_suffix) - idx - clipped_suffix_amount :
                    ],
                )
                updates |= other_root.match(
                    Variadic(), remaining_prefix[: idx + clipped_prefix_amount], []
                )
            break
    else:
        # meaning that any combinations of prefix and suffix uniadics cannot be the
        # same, This means their roots will be the same.

        # Example case:
        # [1, V1] => [1, V3, 2]
        # [V2, 2]

        update_status = True
        _root = Variadic()
        updates |= main_root.match(_root, [], remaining_suffix)
        updates |= other_root.match(_root, remaining_prefix, [])

    return updates, update_status


class ShapeNode:
    __slots__ = "reprs", "referees"

    def __init__(self) -> None:
        self.reprs: list[ShapeRepr] = []
        self.referees: set[Tensor] = set()

    def add_repr(self, repr: ShapeRepr):
        self.reprs.append(repr)
        repr.node = self

    def merge(self, other: ShapeNode) -> Updates:
        updates = Updates()
        resolved_reprs: set[ShapeRepr] = set()
        remaining_reprs = []
        add_constraint = False

        if self != other:
            for repr2 in other.reprs:
                for repr1 in self.reprs:
                    # Match all reprs of other with self.reprs.
                    updates |= repr1._match(repr2)
                    if (
                        len(repr1.prefix) == len(repr2.prefix)
                        and len(repr1.suffix) == len(repr2.suffix)
                        and repr1.root == repr2.root
                    ):
                        resolved_reprs.add(repr2)
                        break
                else:
                    # If other.reprs could not match with any of self.reprs
                    # add it to remaining_reprs and set repr2's root to updates.
                    remaining_reprs.append(repr2)
                    add_constraint = True

            for repr in remaining_reprs:
                self.add_repr(repr)

            if add_constraint:
                for tensor in self.referees:
                    updates.constraints[UpdateType.SHAPE] |= tensor.shape_constraints

            for repr in resolved_reprs:
                # remove_repr_from_symbols(repr)
                repr.clear()

        return updates

    def combine(self):
        updates = Updates()
        same_reprs = set()
        # Iterate over all repr pairs and remove matching reprs.
        for repr, other_repr in combinations(self.reprs, 2):
            if repr not in same_reprs and other_repr not in same_reprs:
                updates |= repr._match(other_repr)
                if (
                    len(repr.prefix) == len(other_repr.prefix)
                    and len(repr.suffix) == len(other_repr.suffix)
                    and repr.root == other_repr.root
                ):
                    same_reprs.add(other_repr)
                    # remove_repr_from_symbols(other_repr)
                    other_repr.clear()
                    # other_repr.node = None

        self.reprs = [repr for repr in self.reprs if repr not in same_reprs]

        return updates

    def set_values(self, values: Sequence[int | None]):
        updates = Updates()
        for repr in self.reprs:
            updates |= repr.set_values(values)
        return updates

    def get_shapes(
        self,
        u_keys: dict[UniadicRecord | Variadic, str] | None = None,
        v_keys: dict[UniadicRecord | Variadic, str] | None = None,
        symbolic=True,
        verbose=False,
    ) -> list[int | str | None] | list[list[int | str | None]]:
        if u_keys is None:
            u_keys = {}
        if v_keys is None:
            v_keys = {}

        if verbose and len(self.reprs) > 1:
            return [repr.get_shapes(u_keys, v_keys, symbolic) for repr in self.reprs]
        else:
            repr = self.get_most_informative_repr()
            return repr.get_shapes(u_keys, v_keys, symbolic)

    def get_most_informative_repr(self) -> ShapeRepr:
        """
        Loop through all Shape representations and get the most informative one.
        Rule 1: The most informative repr is the one with the most uniadics in
            prefix and suffix.
        Rule 2: The most informative repr is the one with the most valued uniadics.
        Rule 3: The most informative repr is the one with the its uniadics and
            variadics is used most.
        Rule 4: The most informative repr is the one with the most prefix uniadics.
        """
        most_informative_repr = None
        for repr in self.reprs:
            # Rule 1
            if most_informative_repr is None or len(repr) > len(most_informative_repr):
                most_informative_repr = repr
            elif len(repr) == len(most_informative_repr):
                # Count valued uniadics in current repr
                valued_uniadics = sum(1 for uni in repr.prefix if uni.value is not None)
                valued_uniadics += sum(
                    1 for uni in repr.suffix if uni.value is not None
                )

                # Count valued uniadics in most informative repr
                best_valued_uniadics = sum(
                    1 for uni in most_informative_repr.prefix if uni.value is not None
                )
                best_valued_uniadics += sum(
                    1 for uni in most_informative_repr.suffix if uni.value is not None
                )

                # Count number of used reprs
                n_reprs = set()
                for uni in repr.prefix + repr.suffix:
                    n_reprs |= uni.reprs

                if repr.root is not None:
                    n_reprs |= repr.root.reprs

                best_n_reprs = set()
                for uni in most_informative_repr.prefix + most_informative_repr.suffix:
                    best_n_reprs |= uni.reprs

                if most_informative_repr.root is not None:
                    best_n_reprs |= most_informative_repr.root.reprs

                # Rule 2
                if (
                    valued_uniadics > best_valued_uniadics
                    or len(n_reprs) > len(best_n_reprs)
                    or (
                        valued_uniadics == best_valued_uniadics
                        and len(repr.prefix) > len(most_informative_repr.prefix)
                        and len(n_reprs) == len(best_n_reprs)
                    )
                ):
                    most_informative_repr = repr
        assert most_informative_repr is not None
        return most_informative_repr


type ShapeType = Uniadic | Variadic
type ConstrainResultType = tuple[bool, Updates]


class ShapeRepr:
    __slots__ = "prefix", "root", "suffix", "node"

    def __init__(
        self,
        prefix: list[Uniadic] | None = None,
        root: Variadic | None = None,
        suffix: list[Uniadic] | None = None,
    ) -> None:
        if prefix is None:
            prefix = []
        self.prefix = prefix

        if suffix is None:
            suffix = []
        self.suffix = suffix

        self.root = root
        self.node: ShapeNode = ShapeNode()
        self.node.add_repr(self)

        self.set_symbol_order()
        if isinstance(self.root, Variadic):
            self.root.reprs.add(self)

    @property
    def reverse(self) -> list[Uniadic]:
        return self.suffix[::-1] if self.root is not None else self.prefix[::-1]

    def set_symbol_order(self):
        for idx, uni in enumerate(self.prefix):
            uni.metadata.reprs_dict.setdefault(self, set()).add(idx)
        if self.root is None:
            for idx, uni in enumerate(self.suffix):
                uni.metadata.reprs_dict.setdefault(self, set()).add(
                    len(self.prefix) + idx
                )
        else:
            for idx, uni in enumerate(self.suffix[::-1]):
                uni.metadata.reprs_dict.setdefault(self, set()).add(-(idx + 1))

    @staticmethod
    def _is_subset(main_list: list[Uniadic], sub_list: list[Uniadic]) -> bool:
        return all(uni == main_list[idx] for idx, uni in enumerate(sub_list))

    @staticmethod
    def _is_subset_rootless(main_list: list[Uniadic], sub_list: list[Uniadic]) -> bool:
        # NOTE: This check is not very efficient and currently not used, try not to use
        # if not necessary!
        if not sub_list:
            return True

        sub_list_len = len(sub_list)
        for i in range(len(main_list) - sub_list_len + 1):
            if main_list[i : i + sub_list_len] == sub_list:
                return True
        return False

    def is_equal(self, other: ShapeRepr):
        if self.root != other.root:
            return False
        else:
            if self.root is not None:
                if len(self.prefix) != len(other.prefix) and len(self.suffix) != len(
                    other.suffix
                ):
                    return False
                else:
                    return self._is_subset(
                        self.prefix[::-1], other.prefix[::-1]
                    ) and self._is_subset(self.suffix, other.suffix)
            else:
                if len(self.prefix) != len(other.prefix):
                    return False
                else:
                    return self._is_subset(self.prefix, other.prefix)

    def __contains__(self, key: ShapeRepr) -> bool:
        if self.root == key.root:
            if self.root is None:
                return self._is_subset_rootless(self.prefix, key.prefix)
            elif len(self.prefix) >= len(key.prefix) and len(self.suffix) >= len(
                key.suffix
            ):
                return self._is_subset(
                    self.prefix[::-1], key.prefix[::-1]
                ) and self._is_subset(self.suffix, key.suffix)
        elif self.root is not None and key.root is None:
            return self._is_subset_rootless(
                self.prefix, key.prefix
            ) or self._is_subset_rootless(self.suffix, key.prefix)
        return False

    def __getitem__(self, position):
        # TODO: Currently position could only be int, but we should support slicing
        # operations too (e.g. repr[:2]) if it is possible (if index of Variadic
        # field allows the operation).
        if not isinstance(position, int):
            raise ValueError("Requires int index!")
        if position < 0 and self.root is not None:
            return self.suffix[position]
        else:
            return self.prefix[position]

    def __setitem__(self, position, new_item):
        if position < 0 and self.root is not None:
            self.suffix[position] = new_item
        else:
            self.prefix[position] = new_item

    def remove_variadic(self, exact_list: list[Uniadic]) -> Updates:
        if (root := self.root) is None:
            raise ValueError("Requires Variadic Shape Representation.")
        else:
            exact_len = len(exact_list)
            if exact_len < len(self):
                raise ValueError(
                    f"Requires minimum of {len(self)} dimensionality, got {exact_len}."
                )
            var_list = exact_list[len(self.prefix) : exact_len - len(self.suffix)]
            return root.match(new_prefix=var_list, new_root=None, new_suffix=[])

    def __len__(self) -> int:
        return len(self.prefix) + len(self.suffix)

    @staticmethod
    def _update_uniadics(
        outer_list: list[Uniadic], inner_list: list[Uniadic]
    ) -> Updates:
        updates = Updates()
        for outer, inner in zip(outer_list, inner_list, strict=False):
            if outer.metadata != inner.metadata:
                updates |= outer.match(inner)
        return updates

    def get_shapes(
        self,
        u_keys: dict[UniadicRecord | Variadic, str] | None = None,
        v_keys: dict[UniadicRecord | Variadic, str] | None = None,
        symbolic=True,
    ) -> list[int | str | None]:
        if u_keys is None:
            u_keys = {}
        if v_keys is None:
            v_keys = {}
        prefix_list = self._get_uniadic_shapes(self.prefix, u_keys, symbolic=symbolic)
        var_list = []
        if self.root is not None:
            if symbolic:
                var_list = [
                    v_keys.setdefault(
                        self.root, ("(V" + str(len(v_keys) + 1) + ", ...)")
                    )
                ]
            else:
                var_list = [v_keys.setdefault(self.root, "...")]
        suffix_list = self._get_uniadic_shapes(self.suffix, u_keys, symbolic=symbolic)
        return prefix_list + var_list + suffix_list

    @staticmethod
    def _get_uniadic_shapes(uniadic_list, cache, symbolic=True):
        final_list = []
        for uniadic in uniadic_list:
            if (value := uniadic.value) is None and symbolic:
                value = cache.setdefault(uniadic.metadata, "u" + str(len(cache) + 1))
            final_list.append(value)
        return final_list

    @staticmethod
    def get_remainings(
        outer_list: list[Uniadic], inner_list: list[Uniadic]
    ) -> list[Uniadic]:
        if (out_len := len(outer_list)) < (in_len := len(inner_list)):
            remaining = inner_list[out_len:]
        else:
            remaining = outer_list[in_len:]
        return remaining

    def inner_match(
        self,
        prefix: list[Uniadic] | None = None,
        root: Variadic | None = None,
        suffix: list[Uniadic] | None = None,
    ) -> Updates:
        if prefix is None:
            prefix = []
        if suffix is None:
            suffix = []
        updates = Updates()
        other_len = len(prefix) + len(suffix)
        if root is None and self.root is None and other_len != len(self):
            raise ValueError(
                "Determined shape representations should have same length."
            )
        # Match all parallel uniadics.
        updates |= self._update_uniadics(self.prefix, prefix)
        updates |= self._update_uniadics(
            self.reverse, suffix[::-1] if root is not None else prefix[::-1]
        )
        if bool(root) ^ bool(self.root):
            # If only one root is not None: remove single Variadic
            if root is None:
                updates |= self.remove_variadic(prefix)
            else:
                exact_len = len(self.prefix)
                if exact_len < len(prefix) + len(suffix):
                    raise ValueError(
                        f"Requires minimum of {len(prefix) + len(suffix)} "
                        f"dimensionality, got {exact_len}."
                    )
                var_list = self.prefix[len(prefix) : len(self.prefix) - len(suffix)]
                updates |= root.match(new_prefix=var_list, new_root=None, new_suffix=[])
                # updates |= other.remove_variadic(self.prefix)
        elif root is not None and self.root is not None:
            if id(self.root) == id(root) and len(self) != other_len:
                raise ValueError("Shape mismatch!")
            elif self.root != root:
                # If only two different roots exists
                # Find all leftover prefixes and suffixes
                remaining_prefix = self.get_remainings(self.prefix, prefix)
                remaining_suffix = self.get_remainings(self.reverse, suffix[::-1])[::-1]
                # Find which root will be updated.
                if len(self) >= other_len:
                    removed_root = root
                    new_root = self.root
                else:
                    removed_root = self.root
                    new_root = root

                is_prefix_longer = len(self.prefix) > len(prefix)
                is_prefix_shorter = len(self.prefix) < len(prefix)
                is_suffix_longer = len(self.suffix) > len(suffix)
                is_suffix_shorter = len(self.suffix) < len(suffix)

                if (
                    is_prefix_longer
                    and is_suffix_shorter
                    or is_prefix_shorter
                    and is_suffix_longer
                ):
                    # This case is the unmatchable repr case
                    # Update remaining_suffix and prefix accordingly.
                    # This will also equalize lengths of reprs.
                    if is_prefix_longer and is_suffix_shorter:
                        main_root = self.root
                        other_root = root
                    else:
                        main_root = root
                        other_root = self.root

                    _updates, is_updated = handle_numerical_incompatibility(
                        main_root, other_root, remaining_prefix, remaining_suffix
                    )
                    updates |= _updates

                    if not is_updated and len(self) != other_len:
                        if len(remaining_prefix) < len(remaining_suffix):
                            remaining_suffix = remaining_suffix[len(remaining_prefix) :]
                            remaining_prefix = []
                        else:
                            remaining_prefix = remaining_prefix[
                                : len(remaining_prefix) - len(remaining_suffix)
                            ]
                            remaining_suffix = []
                        updates |= removed_root.match(
                            Variadic(), remaining_prefix, remaining_suffix
                        )

                else:
                    updates |= removed_root.match(
                        new_root, remaining_prefix, remaining_suffix
                    )
        return updates

    def _match(self, other: ShapeRepr) -> Updates:
        return self.inner_match(other.prefix, other.root, other.suffix)

    def set_values(self, values: Sequence[int | None]) -> Updates:
        updates = Updates()
        if self.root is not None:
            uniadics = [Uniadic(value) for value in values]
            updates |= self._update_uniadics(self.prefix, uniadics)
            updates |= self._update_uniadics(self.reverse, uniadics[::-1])
            updates |= self.remove_variadic(uniadics)
            # updates -= set(uniadics)
        else:
            if len(values) != len(self.prefix):
                raise ValueError(
                    "Shape representation's dimension mismatched with given "
                    "list of values."
                )
            for uni, value in zip(self.prefix, values, strict=False):
                if uni.set_value(value):
                    updates.add(uni)
        return updates

    def clear(self):
        # Clear given repr's symbols' reprs field from itself.
        for symbol in self.prefix + self.suffix:
            # symbol.reprs.discard(repr)
            symbol.metadata.reprs_dict.pop(self, None)
        if self.root is not None:
            self.root.reprs.discard(self)

        self.root = None
        self.prefix = []
        self.suffix = []
        # self.node.reprs.remove(self)


# # Below functions are used in various constraints.
# prod_fn = lambda a, b: (a if isinstance(a, int) else a.value) * (b if
# isinstance(b, int) else b.value)
# is_repr_known = lambda repr: repr.root is None and repr.prefix and
# all([uni.value is not None for uni in repr.prefix])
@dataclass
class Constraint:
    fn: Callable
    type: UpdateType = UpdateType.SHAPE
    call_counter: int = 0
    post_processes: set[Callable] = field(default_factory=lambda: set())

    def __call__(self, keys: list[Scalar | Tensor]):
        status = False
        updates = Updates()
        if self.type == UpdateType.SHAPE:
            tensor_keys = [key for key in keys if isinstance(key, Tensor)]
            for reprs in product(*[key.shape.reprs for key in tensor_keys]):
                for idx, repr in enumerate(reprs):
                    tensor_keys[idx]._temp_shape = repr
                status, newly_added_symbols = self.fn(*keys)
                updates |= newly_added_symbols
        elif self.type == UpdateType.TYPE:
            status, newly_added_symbols = self.fn(*keys)
            updates |= newly_added_symbols

        self.call_counter += 1
        return status, updates

    def add_post_process(self, fn: Callable):
        self.post_processes.add(fn)

    def create_post_constraints(self):
        constraints = set()
        for fn in self.post_processes:
            constraints.add(Constraint(fn, self.type))
        return constraints

    def __hash__(self) -> int:
        return hash(id(self))


def add_lens(x, y, const):
    return x + next(const) + y


type RowColumnType = list[str | list[str]] | list[str] | list[list[str]]


class Table:
    """
    Class for creating summary tables
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.headers: list[RowColumnType] = []
        self.cells: list[RowColumnType] = []
        self.cell_str = ""
        self.header_str = ""

    def add_header(self, header: list[str]):
        self.headers.append(header)

    def add_row(self, row: RowColumnType):
        self.cells.append(row)

    def add_column(self, column: RowColumnType):
        for idx, row in enumerate(column[: len(self.headers)]):
            self.headers[idx].append(row)  # type: ignore

        for idx_, row in enumerate(column[idx + 1 :]):
            self.cells[idx_].append(row)  # type: ignore

    def _calculate_table_specs(self):
        """Calculates table specifications for constructing the table. Calculates cell
        widths for each column and calculates cell heights for each column
        """

        # concatenate all headers and cells into a one list
        all_elems = self.headers + self.cells

        #  Initialize cell_heights and cell_widths lists, these
        # lists will hold minimum height and width that every cell
        # can have respectively.
        cell_heights = []
        cell_widths = []

        for row in all_elems:
            row_heights = []
            row_widths = []
            for cell in row:
                if isinstance(cell, list):
                    # if cell is a list, length of the list will give minimum height
                    # of the cell. And maximum length of of each string in the list
                    # will give minimum width of the cell
                    row_widths.append(len(max(cell, key=len)))
                    row_heights.append(len(cell))
                else:
                    # if cell is a string, minimum height of that cell will be equal
                    # to 1. And minimum width of the cell will be equal to length of
                    # that string
                    row_widths.append(len(cell))
                    row_heights.append(1)
            cell_heights.append(row_heights)
            cell_widths.append(row_widths)

        # width of each column should be equal to maximum of widths in that column
        self.each_row_width = [
            max(cell_widths[k][i] for k in range(len(cell_widths)))
            for i in range(len(cell_heights[0]))
        ]

        # similarly, height of each row should be equal to maximum of heights in
        # each row
        self.each_col_height = [max(cell_height) for cell_height in cell_heights]

        # TODO: change names self.each_row_width -> self.each_col_width and
        # self.each_col_height -> self.each_row_height

    def _adjust_table(self):
        """Adjusts the table and manipulates the cells and headers based on already
        calculated each_col_height and each_row_width"""
        # partialize the self.fill_spaces function as it will be used multiple times
        fill_fn_left = partial(self.fill_spaces, align="left")
        # seperate row heights of headers and row heights of cells
        header_col_height, cell_col_height = (
            self.each_col_height[: len(self.headers)],
            self.each_col_height[len(self.headers) :],
        )

        # make adjustments in headers according to each_col_width and each_row_height
        # (set row width of that cell to corresponding max_row_width and set column
        # height of that cell to corresponding max_col_height)
        for i, row in enumerate(self.headers):
            self.headers[i] = list(
                map(
                    partial(fill_fn_left, max_height=header_col_height[i]),
                    row,
                    self.each_row_width,
                )
            )
        # also make adjustments in cells according to each_col_width and each_row_weight
        for i, row in enumerate(self.cells):
            self.cells[i] = list(
                map(
                    partial(fill_fn_left, max_height=cell_col_height[i]),
                    row,
                    self.each_row_width,
                )
            )

    def _compile(
        self,
        row_sep: str | list[str] = "   |   ",
        col_sep: str = "-",
        table_sep: str = "=",
        empty_row_sep: bool = False,
    ) -> None:
        # TODO: Swap the names of row_sep and col_sep
        """compiles the table and constructs pretty printable cell strings and
            header strings

        Args:
            row_sep (str, optional): seperator of columns. Defaults to "   |   ".
            col_sep (str, optional): seperator of rows. Defaults to "-".
            table_sep (str, optional): seperator of headers and cells. Defaults to "=".
        """
        # Initialize the strings
        header_str = ""
        cell_str = ""
        row_sep = (
            [row_sep] * len(self.headers[0])
            if (self.headers and isinstance(row_sep, str))
            else row_sep
        )

        # calculate table specs
        self._calculate_table_specs()
        # adjust the table accordingly
        self._adjust_table()
        # calculate total table width
        table_width = reduce(
            partial(add_lens, const=(len(row) for row in row_sep)), self.each_row_width
        )
        table_constructor_fn = partial(add_lens, const=cycle(row_sep))
        table_constructor_fn_w_spaces = partial(
            add_lens, const=cycle(len(row) * " " for row in row_sep)
        )
        end = "\n"
        header_list = []
        cell_list = []

        # Construct the header if it exists
        header_list.append(self.name.center(table_width) + end)
        if self.headers:
            header_list.append(col_sep * table_width + end)
            for row in self.headers:
                for idx in range(len(row[0])):
                    header_list.append(
                        reduce(table_constructor_fn, [cell[idx] for cell in row]) + end
                    )
        header_list.append(table_sep * table_width)
        header_str = header_str.join(header_list)

        # Construct the cells
        for _row in self.cells:
            for idx in range(len(_row[0])):
                if empty_row_sep:
                    if idx == 0:
                        cell_list.append(
                            reduce(table_constructor_fn, [cell[idx] for cell in _row])
                            + end
                        )
                    else:
                        cell_list.append(
                            reduce(
                                table_constructor_fn_w_spaces,
                                [cell[idx] for cell in _row],
                            )
                            + end
                        )
                cell_list.append(
                    reduce(table_constructor_fn, [cell[idx] for cell in _row]) + end
                )
            cell_list.append(col_sep * table_width + end)
        cell_str = cell_str.join(cell_list) + end
        self.header_str = header_str
        self.cell_str = cell_str

    def display(self):
        """Prints the table"""
        print(self.header_str)
        print(self.cell_str)

    @staticmethod
    def construct_subtable_row(
        stub: str,
        arg_max_lengths: list[int],
        adjustments: list[str],
        *args: list[list[str]],
    ):
        # Constructs subtable with given args
        subtable_list: list[str] = []
        elems: tuple[list[str], ...]
        for elems in zip(*args, strict=False):
            elem: tuple[str, ...]
            for idx, elem in enumerate(zip_longest(*elems, fillvalue="")):
                stub_str = stub + " : " if not subtable_list else " " * (len(stub) + 3)
                split = " : " if idx == 0 else "   "
                row_str = stub_str + Table.construct_subtable_str(
                    elem, arg_max_lengths, adjustments, split
                )
                subtable_list.append(row_str)
        return subtable_list

    @staticmethod
    def construct_subtable_str(
        args: tuple[str, ...] | list[str],
        arg_max_lengths: list[int],
        adjustments: list[str],
        split: str = " : ",
    ) -> str:
        row_str = ""
        for single_elem, max_len, adjustment in zip(
            args, arg_max_lengths, adjustments, strict=False
        ):
            if adjustment == "left":
                row_str += single_elem.ljust(max_len) + split
            elif adjustment == "right":
                row_str += single_elem.rjust(max_len) + split
        row_str = row_str[:-3]
        return row_str

    @staticmethod
    def fill_spaces(
        value: list | str,
        max_width: int = 0,
        max_height: int = 0,
        align: Literal["left", "right", "center"] = "left",
    ) -> list[str]:
        """Adjust given list of string based on maximum length of strigs in that list

        Examples:

        >>> list_1 = ["abcd", "def, "g"]
        >>> aligned_list = table.fill_spaces(list_1)
        >>> list_1
        ["abcd", "def ", "g   "]

        Args:
            value (list | str): list of strings or string to be aligned
            max_width (int, optional): Maximum width of the list. If specified,
                length of every string of the list at least will be equal to max_width.
                Defaults to 0.
            max_height (int, optional): maximum height of a string. If specified,
                length of the list at least will be equal to max_height. Defaults
                to 0.
            align (str, optional): alignment mode. Defaults to "left".

        Returns:
            list[str]: aligned list
        """
        if isinstance(value, str):
            value = [value]
        elif not value:
            value = ["None"]
        value += [""] * max(max_height - len(value), 0)
        max_width = max(len(max(value, key=len)), max_width)
        match align:
            case "left":
                return [row.ljust(max_width) for row in value]
            case "right":
                return [row.rjust(max_width) for row in value]
            case "center":
                return [row.center(max_width) for row in value]

    @staticmethod
    def adjust_list(strings: list[str], max_len: int = 0) -> list[str]:
        """Adjusts and reconstucts a list based on specified maximum length.
        takes all strings in the list, put it to new string seperated
        with comma, if length of the string exceeds the max_len, append this string to a
        new list and continue to proceed

        Examples:
        >>> list1 = ["abcdef", "ghf", "1234", "ab", "c"]
        >>> list2 = Table.adjust_list(list1, 7)
        >>> list2
        ["abcdef, ",
        "ghf, 1234, ",
        "ab, c"]

        Args:
            strings (list[str]): list of strings
            max_len (int, optional): maximum length of a string in a list. Defaults
                to 0.

        Returns:
            list[str]: list of re-oredered strings
        """
        if not strings:
            #  Fill empty strings with None
            new_list = ["None"]
        else:
            line_len = 0
            new_str = ""
            new_list = []
            for string in strings:
                new_str += string + ", "
                line_len += len(string) + 2
                if line_len > max_len:
                    new_list.append(new_str)
                    line_len = 0
                    new_str = ""
            new_list.append(new_str)
            new_list[-1] = new_list[-1][:-2]
        return new_list

    @staticmethod
    def dict_to_table(
        in_dict: Mapping[str, list | dict],
        seperator: str = " : ",
        left_align: Literal["left", "right", "center"] = "left",
        right_align: Literal["left", "right", "center"] = "left",
        left_length: int = 0,
        right_length: int = 0,
        len_space: int = 0,
        space_fill: str = "-",
        r_len=0,
    ) -> list[str]:
        """takes a dicts and creates a list of strings from that dict by filling empty
        spaces and making necessary alignments This function will work recursivey of
        values of the input dict is also dicts

        Examples:

        >>> dict1 = {
            "key_1": ["val_1", "val_2", "val_3"],
            "key_2": ["val_21", "val_22", "val_23"],
        }
        >>> table = Table.dict_to_table(dict1)
        >>> table
        ["key_1 : val_1 ",
         "        val_2 ",
         "        val_3 ",
         "key_2 : val_21",
         "        val_22",
         "        val_23"]

        Args:
            in_dict (dict[str, list  |  dict]): input dict
            seperator (_type_, optional): seperator between key and value pairs.
                Defaults to " : ".
            left_align (str, optional): alignment mode of keys. Defaults to "left".
            right_align (str, optional): alignment mode of values. Defaults to "left".
            left_length (int, optional): initial length of keys, if it is bigger than
                maximum length of keys, adjustments will be made according to this
                number. Defaults to 0.
            right_length (int, optional): initial length of values, if it is bigger
                than maximum length of values, adjustments will be made according to
                this number. Defaults to 0.
            len_space (int, optional): vertical space between each key value list of
                tables. Defaults to 0.
            space_fill (str, optional): if len_space is specified, fill the spaces with
                that value. Defaults to "-".

        Returns:
            list[str]: list of strings with same length
        """

        table = []
        left_list: list[str] = []
        right_list: list[str] = []

        # run the funcion recursively if value is dict
        new_dict = {
            key: Table.dict_to_table(value) if isinstance(value, dict) else value
            for key, value in in_dict.items()
        }
        sep_indexes = [0]
        for _key, value in new_dict.items():
            if r_len != 0:
                value = Table.adjust_list(value, r_len)
            if isinstance(_key, tuple):
                key = list(_key)
                k_len = len(key)
            else:
                key = [_key]
                k_len = 1
            v_len = len(value)
            max_len = max(k_len, v_len)
            right_list.extend(value + ["" for _ in range(len_space + max_len - v_len)])
            left_list.extend(key + ["" for _ in range(len_space + max_len - k_len)])
            sep_indexes.append(sep_indexes[-1] + max_len + len_space)
        right_list = Table.fill_spaces(
            right_list, align=right_align, max_width=right_length
        )
        left_list = Table.fill_spaces(
            left_list, align=left_align, max_width=left_length
        )
        for idx, (left, right) in enumerate(zip(left_list, right_list, strict=False)):
            sep = (" " * len(seperator)) if idx not in sep_indexes else seperator
            row = left + sep + right
            if row.isspace():
                row = row.replace(" ", space_fill)
            table.append(row)
        return table


UsedKeysType = dict[str | int, ShapeType] | dict[int, ShapeType] | dict[str, ShapeType]


def create_shape_map(
    shape_template: _ShapesType,
    solver: ConstraintSolver,
) -> dict[str, ShapeRepr]:
    used_keys: UsedKeysType = {}
    return {
        key: create_shape_repr(shp_list, solver, used_keys)
        for key, shp_list in shape_template.items()
    }


def create_shape_repr(
    shp_list: ShapeTemplateType,
    solver: ConstraintSolver,
    used_keys: UsedKeysType | None = None,
) -> ShapeRepr:
    if used_keys is None:
        _used_keys: UsedKeysType = {}
    else:
        _used_keys = used_keys
    if shp_list == []:
        assert solver.empty_node is not None
        return next(iter(solver.empty_node.reprs))

    shp_prefix: list[Uniadic] = []
    shp_root: Variadic | None = None
    shp_suffix: list[Uniadic] = []

    for symbol in shp_list:
        symbol_obj: Uniadic | Variadic | None
        key_symbol: str | int
        if isinstance(symbol, tuple):
            if len(symbol) != 2 or symbol[1] != ...:
                raise KeyError("Requires valid variadic format!")
            # Get symbol of variadic
            symbol_obj = _used_keys.get(key_symbol := symbol[0], Variadic())  # type: ignore
            assert isinstance(symbol_obj, Variadic), "Must be Variadic!"
            shp_root = symbol_obj
        else:
            key_symbol = symbol  # type: ignore
            if symbol not in _used_keys:
                match symbol:
                    case int() if not isinstance(symbol, bool):
                        if (_uni := solver.symbol_store.get(symbol)) is not None:
                            symbol_obj = _uni
                        else:
                            symbol_obj = solver.symbol_store[symbol] = Uniadic(symbol)
                    case str() | None:
                        symbol_obj = Uniadic()
                    case _:
                        raise TypeError(
                            f"Given type ({type(symbol)}) is not supported. Only int, "
                            "str, or None types are accepted."
                        )
            else:
                assert isinstance(symbol, str | int)
                symbol_obj = _used_keys[symbol]  # type: ignore

            assert isinstance(symbol_obj, Uniadic), "Must be Uniadic!"

            if shp_root is None:
                shp_prefix.append(symbol_obj)
            else:
                shp_suffix.append(symbol_obj)

        if symbol is not None:
            _used_keys[key_symbol] = symbol_obj  # type: ignore

    return ShapeRepr(prefix=shp_prefix, root=shp_root, suffix=shp_suffix)


def get_summary(
    conns: dict,
    name: str,
    shape: dict | None = None,
    types: dict | None = None,
    params: dict | None = None,
) -> Table:
    """Constructs the summary table based on connections and shapes

    Args:
        conns (dict): connection dict
        name (str): given name of the table
        shape (dict | None, optional): Shape information of all keys. Defaults to None.

    Returns:
        Table: Table object that holds all summary information
    """

    stub_input = "Inputs"
    stub_output = "Outputs"
    stub_len = len(max(stub_input, stub_output, key=len))
    stub_input = stub_input.ljust(stub_len)
    stub_output = stub_output.ljust(stub_len)

    keys_name = "Keys"
    shapes_name = "Shapes"
    conn_name = "Connections"
    type_name = "Types"
    params_name = "Parameters"
    sub_columns = [keys_name, shapes_name, type_name, conn_name, params_name]
    adjustments = ["left", "right", "left", "left", "right"]

    removed_cols = []
    if shape is not None:
        align_shapes(
            [shape_dict for shape_tuple in shape.values() for shape_dict in shape_tuple]
        )
    else:
        # if shape is not given, add it to the removed cols, shape sub column will be
        # removed after
        shape = {}
        removed_cols.append(1)

    if params is None:
        # if params is not given, add it to the removed cols, params subcolumn will be
        # removed after
        params = {}
        removed_cols.append(4)

    if types is None:
        # if types is not given, add it to the removed cols, types subcolumn will be
        # removed after
        types = {}
        removed_cols.append(2)

    all_keys = {keys_name}
    all_shapes = {shapes_name}
    all_types = {type_name}
    all_connections = {conn_name}
    all_params = {params_name}

    for model_name in conns:
        # collect all keys and conections into a their corresponding sets (also shape
        # and params if possible) these sets will be used in finding their
        # corresponding column lengths
        io_conn_tuple = conns.get(model_name, ({}, {}))
        io_shape_tuple = shape.get(model_name, ({}, {}))
        io_param_tuple = params.get(model_name, ({}, {}))
        io_type_tuple = types.get(model_name, ({}, {}))

        input_conns, output_conns = io_conn_tuple
        input_shape, output_shape = io_shape_tuple
        input_params, output_params = io_param_tuple
        input_types, output_types = io_type_tuple

        all_keys.update(input_conns.keys())
        all_keys.update(output_conns.keys())

        all_connections.update(*input_conns.values())
        all_connections.update(*output_conns.values())

        all_shapes.update(*input_shape.values())
        all_shapes.update(*output_shape.values())

        all_params.update(input_params.values())
        all_params.update(output_params.values())

        all_types.update(input_types.values())
        all_types.update(output_types.values())

    # Find max lengths of each sets
    max_key_len = len(max(*all_keys, key=len))
    max_conn_len = len(max(*all_connections, key=len))
    max_shape_len = len(max(*all_shapes, key=len))
    max_params_len = len(max(*all_params, key=len))
    max_types_len = len(max(*all_types, key=len))

    sub_column_lengths = [
        max_key_len,
        max_shape_len,
        max_types_len,
        max_conn_len,
        max_params_len,
    ]

    # remove not given columns
    sub_column_lengths = [
        col for idx, col in enumerate(sub_column_lengths) if idx not in removed_cols
    ]
    adjustments = [
        col for idx, col in enumerate(adjustments) if idx not in removed_cols
    ]
    sub_columns = [
        col for idx, col in enumerate(sub_columns) if idx not in removed_cols
    ]

    table = Table(name=name)

    for model_name in conns:
        # iterate in all models, construct spanner columns
        input_shape, output_shape = shape.get(model_name, ({}, {}))
        input_conn, output_conn = conns.get(model_name, ({}, {}))
        input_params, output_params = params.get(model_name, ({}, {}))
        input_types, output_types = types.get(model_name, ({}, {}))

        in_keys = [[key] for key in input_conn]
        out_keys = [[key] for key in output_conn]

        input_shape = input_shape.values()
        output_shape = output_shape.values()

        in_types = [[key] for key in input_types.values()]
        out_types = [[key] for key in output_types.values()]

        input_conn = [value if value else ["--"] for value in input_conn.values()]
        output_conn = [value if value else ["--"] for value in output_conn.values()]

        in_params = [[param] for param in input_params.values()]
        out_params = [[param] for param in output_params.values()]

        input_args = [in_keys, input_shape, in_types, input_conn, in_params]
        output_args = [out_keys, output_shape, out_types, output_conn, out_params]

        # remove not given columns
        input_args = [
            col for idx, col in enumerate(input_args) if idx not in removed_cols
        ]
        output_args = [
            col for idx, col in enumerate(output_args) if idx not in removed_cols
        ]

        # construct Inputs and outputs stub
        input_table = table.construct_subtable_row(
            stub_input, sub_column_lengths, adjustments, *input_args
        )
        sep = ["-" * len(input_table[0])]
        output_table = table.construct_subtable_row(
            stub_output, sub_column_lengths, adjustments, *output_args
        )

        cell = input_table + sep + output_table
        table.add_row([model_name, cell])

    subheader_adjustments = ["left", "left", "left", "left", "left"]
    subheader_adjustments = [
        col for idx, col in enumerate(subheader_adjustments) if idx not in removed_cols
    ]

    sub_row_header = table.construct_subtable_str(
        sub_columns, sub_column_lengths, subheader_adjustments
    )
    sub_row_length = len(sub_row_header)
    header_stub_space = " " * (stub_len + 3)
    table.add_header(
        ["Model Name", header_stub_space + "Model Keys".center(sub_row_length)]
    )
    table.add_header(["", header_stub_space + "-" * sub_row_length])
    table.add_header(["", header_stub_space + sub_row_header])
    return table


def _get_summary_shapes(model_shapes: dict, conn_info: dict):
    shape_info = {}
    for model_name in conn_info:
        shape = model_shapes[model_name]
        input_conns, output_conns = conn_info[model_name]
        input_shapes = {key: shape[key] for key in input_conns}
        output_shapes = {key: shape[key] for key in output_conns}
        shape_info[model_name] = (input_shapes, output_shapes)
    return shape_info


def _get_summary_types(name_mappings: dict, data_memo=None):
    if data_memo is None:
        data_memo = {}

    type_info: dict[str, tuple[dict, dict]] = {}

    for model, model_name in name_mappings.items():
        in_dict, out_dict = type_info.setdefault(model_name, ({}, {}))
        for key in model.conns.all:
            key_mappings = model._generate_keys(include_outputs=True)
            in_key = key_mappings.get(key, key)
            data = model.conns.get_data(key)
            pm_data = data_memo.get(id(data), data)
            data_type = pm_data._type
            if not hasattr(data_type, "__args__"):
                str_type = data_type.__name__
            else:
                sorted_type = sort_type(pm_data._type)
                str_type = str(sorted_type)
            if key in model._input_keys:
                in_dict[in_key] = str_type
            else:
                out_dict[in_key] = str_type
    return type_info


def is_type_adjustment_required(data: dict[str, Tensor | Scalar], inputs: list[str]):
    if len(inputs) <= 2:
        return False
    inputs = inputs[:2]
    left = data[inputs[0]]
    right = data[inputs[1]]
    if not isinstance(left, Tensor) or not isinstance(right, Tensor):
        return False

    rule1 = issubclass(float, left._type) and issubclass(int, right._type)
    rule2 = issubclass(float, right._type) and issubclass(int, left._type)

    return rule1 | rule2
