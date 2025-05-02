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

from collections.abc import Callable, Iterator, KeysView, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, reduce
from itertools import chain, combinations, cycle, product, zip_longest
from types import EllipsisType, GenericAlias, UnionType
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

from ..common import PaddingType, get_specific_types_from_value
from ..types import (
    Constant,
    DataType,
    Dtype,
    constant_shp_table,
    constant_type_table,
)
from ..utils.type_utils import is_union_type
from .utils import (
    align_shapes,
    sort_type,
)

__all__ = [
    "get_shapes",
    "NOT_GIVEN",
    "TBD",
    "KeyType",
    "IOHyperEdge",
    "Tensor",
    "ShapeNode",
    "ShapeRepr",
    "Constraint",
    "create_shape_map",
    "ShapeResultType",
    "get_summary_shapes",
    "get_summary_types",
    "ConstraintSolver",
    "Constant",
]


class SingletonObject:
    _instance = None
    """
    A base class for custom objects that ensures a singleton pattern.
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
    It ensures that only one instance of any subclass is created (singleton pattern).
    Usage:
        class MySingletonObject(SingletonObject):
            pass
        obj1 = MySingletonObject()
        obj2 = MySingletonObject()
        assert obj1 is obj2  # True, both are the same instance
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


class NullConnection(SingletonObject):
    """
    A singleton class representing a null connection indicating
    that no connection is specified.
    """

    pass


class ToBeDetermined(SingletonObject):
    """
    A singleton class representing a null data indicating
    that no data is provided.
    """

    def __repr__(self) -> str:
        return "TBD"

    pass


NOT_GIVEN = NullConnection()
TBD = ToBeDetermined()


class UpdateType(Enum):
    SHAPE = 1
    TYPE = 2
    VALUE = 3


class KeyType(Enum):
    INPUT = 1
    OUTPUT = 2
    LATENT_INPUT = 3
    INTERNAL = 4
    LATENT_OUTPUT = 5


@dataclass
class StateKey:
    in_key: str
    out_key: str
    is_exposed: bool
    initial_value: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD


type FixedValueType = (
    None
    | int
    | tuple[int, ...]
    | list[int]
    | dict[Any, Any]
    | slice
    | Constant
    | tuple[int | None, ...]
    | str
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
    | type[Sequence[Any]]
    | type[dict[Any, Any]]
    | type[Mapping[Any, Any]]
    | type[Constant]
    | type[slice]
    | type[PaddingType]
    | type[EllipsisType]
    | type[ToBeDetermined]
    | type[str]
    | type[None]
    | type[Dtype]
    | UnionType
    | GenericAlias
)
# TODO: remove Constant from ScalarValueType
ScalarValueType = (
    int
    | float
    | bool
    | Sequence[Any]
    | dict[Any, Any]
    | Constant
    | slice
    | PaddingType
    | EllipsisType
    | Dtype
    | ToBeDetermined
    | str
    | None
)
ShapeTemplateType = Sequence[int | str | tuple[str, EllipsisType] | None]
ReduceType = int | tuple[int, ...] | None | ToBeDetermined
MainValueType = (
    int
    | float
    | Sequence[Any]
    | dict[Any, Any]
    | bool
    | None
    | EllipsisType
    | PaddingType
    | Constant
    | slice
    | Dtype
    | ToBeDetermined
)
# Mainvalue type for isintance check
MainValueInstance = (
    int
    | float
    | Sequence  # type: ignore
    | dict  # type: ignore
    | bool
    | None
    | EllipsisType
    | PaddingType
    | Constant
    | slice
    | Dtype
)

TypeVarTensorType = TypeVar(
    "TypeVarTensorType",
    bound=int | float | bool,
    covariant=True,
)
# Availale types for Tensor type ("_type" attribute of Tensor class).
_TensorTypes = type[int] | type[float] | type[bool] | UnionType
ValType = int | float | bool
VariableSequenceType = (
    Sequence[TypeVarTensorType]
    | Sequence[Sequence[TypeVarTensorType]]
    | Sequence[Sequence[Sequence[TypeVarTensorType]]]
    | Sequence[Sequence[Sequence[Sequence[TypeVarTensorType]]]]
    | Sequence[Sequence[Sequence[Sequence[Sequence[TypeVarTensorType]]]]]
)
SequenceValType = VariableSequenceType[ValType]

VariableListType = (
    list[TypeVarTensorType]
    | list[list[TypeVarTensorType]]
    | list[list[list[TypeVarTensorType]]]
    | list[list[list[list[TypeVarTensorType]]]]
    | list[list[list[list[list[TypeVarTensorType]]]]]
)
ListValType = VariableListType[ValType]

# Nested Sequence type values for Tensor class.
_TensorValueType = ValType | SequenceValType
# Logical value types for Tensor class (i.e. "value" attribute of
# Tensor class).
TensorValueType = _TensorValueType | Constant

# TODO: This kind of type definitions will be updated as recursive
# definitions when mypy supports recursive types.
TensorToListType = ValType | ListValType

MaxNestedListDepth = 5

ParamsEvalType = dict[str, DataType]
DataEvalType = Mapping[str, DataType | ScalarValueType | str]


class EvaluateType(Protocol, Generic[DataType]):
    def __call__(
        self,
        params: ParamsEvalType[DataType] | None,
        data: DataEvalType[DataType] | None,
        cache: DataEvalType[DataType] | None = None,
    ) -> DataEvalType[DataType]: ...


class EvaluateAllType(Protocol, Generic[DataType]):
    def __call__(
        self,
        params: ParamsEvalType[DataType] | None,
        data: DataEvalType[DataType] | None,
        output_gradients: ParamsEvalType[DataType] | None,
    ) -> tuple[DataEvalType[DataType], ParamsEvalType[DataType]]: ...


class AssignedConstraintType(TypedDict):
    fn: str
    keys: list[str]


LossKey = "loss"
FinalCost = "final_cost"

ItemType = TypeVar("ItemType")


def update_equivalence_table(
    item1: ItemType, item2: ItemType, lookup_table: dict[ItemType, set[ItemType]]
) -> None:
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
    constraint_map: dict[Constraint, list[IOHyperEdge]] = field(
        default_factory=lambda: {}
    )

    def __call__(self, updates: Updates) -> None:
        self.update_shapes(updates)
        # Here we are updating Updates object because we are
        # using it in DataStore's `update_cached_data`.
        updates |= self.solver_loop(updates.constraints)

    def solver_loop(self, constraints: set[Constraint]) -> Updates:
        updates = Updates()
        while constraints:
            constr = constraints.pop()
            if (not constr.parents) and (constr in self.constraint_map):
                hyper_edges = self.constraint_map[constr]
                status, newly_added_symbols = constr(hyper_edges)
                if UpdateType.SHAPE in constr.types:
                    self.update_shapes(newly_added_symbols)
                updates |= newly_added_symbols
                new_constraints = newly_added_symbols.constraints

                if status:
                    # Remove all occurences of constraint.
                    self.constraint_map.pop(constr)
                    for hyper_edge in hyper_edges:
                        hyper_edge.remove_constraint(constr)

                constraints |= new_constraints
                constraints.discard(constr)
        return updates

    @staticmethod
    def _combine_nodes(updates: Updates) -> None:
        # Check if any node could be reduced after variadic updates add into
        # node_updates field.
        while updates.node_updates:
            node = updates.node_updates.pop()
            updates |= node.combine()

    def _reduce_uniadic_referees(self, updates: Updates) -> None:
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
    def _find_intersection_reprs(repr1: ShapeRepr) -> set[ShapeRepr]:
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
    def _add_sublists(
        repr1: ShapeRepr,
        intersection_reprs: set[ShapeRepr],
        deletion_nodes: dict[ShapeNode, set[ShapeNode]],
    ) -> Updates:
        updates = Updates()
        for repr2 in intersection_reprs:
            if (repr1.node != repr2.node) and (repr1 in repr2):
                if repr2 in repr1:
                    # Find duplicated nodes and add them to deletion_nodes.
                    update_equivalence_table(repr1.node, repr2.node, deletion_nodes)
                else:
                    updates |= subset_match(repr1, repr2)

        return updates

    def clear(self) -> None:
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
        for tensor in deleted.referees:
            tensor.shape = remaining
            remaining.referees.add(tensor)

        deleted.referees = set()
        deleted.reprs = []
        return updates

    def update_shapes(self, updates: Updates) -> None:
        deletion_nodes: dict[ShapeNode, set[ShapeNode]] = {}
        # Reduce updated nodes' reprs if possible.
        self._combine_nodes(updates)
        # Reduce updated UniadicRecords' referees field.
        self._reduce_uniadic_referees(updates)

        all_reprs = {
            repr for tensor in updates.shape_updates for repr in tensor.shape.reprs
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

    def update_constraint_map(self, new: IOHyperEdge, old: IOHyperEdge) -> None:
        # Replaces old hyperedge with the new one in constraint_map.
        for constr in old.all_constraints:
            old_edges = self.constraint_map[constr]
            new_edges = [new if key is old else key for key in old_edges]
            self.constraint_map[constr] = new_edges

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
    shape_updates: set[Tensor[int | float | bool]] = field(
        default_factory=lambda: set()
    )
    value_updates: set[IOHyperEdge] = field(default_factory=lambda: set())
    uniadic_updates: set[Uniadic] = field(default_factory=lambda: set())
    node_updates: set[ShapeNode] = field(default_factory=lambda: set())
    constraints: set[Constraint] = field(default_factory=lambda: set())

    def add(
        self,
        symbol: IOHyperEdge | Uniadic | Variadic,
        update_type: UpdateType = UpdateType.SHAPE,
    ) -> None:
        match symbol:
            case Uniadic():
                self._add_uniadic(symbol)
            case Variadic():
                self._add_variadic(symbol)
            case IOHyperEdge():
                self.constraints |= symbol.constraints[update_type]

    def _add_uniadic(self, symbol: Uniadic) -> None:
        self.uniadic_updates.add(symbol)
        all_tensors = {
            tensor for repr in symbol.metadata.reprs for tensor in repr.node.referees
        }
        self.shape_updates.update(all_tensors)
        self.constraints |= set().union(
            *(
                edge.constraints[UpdateType.SHAPE]
                for tensor in all_tensors
                for edge in tensor.referees
            )
        )

    def _add_variadic(self, symbol: Variadic) -> None:
        for repr in symbol.reprs:
            self.node_updates.add(repr.node)
            for tensor in repr.node.referees:
                self.shape_updates.add(tensor)
                for edge in tensor.referees:
                    self.constraints |= edge.constraints[UpdateType.SHAPE]

    def __ior__(self, other: Updates) -> Updates:
        self.constraints |= other.constraints
        self.shape_updates |= other.shape_updates
        self.uniadic_updates |= other.uniadic_updates
        self.node_updates |= other.node_updates
        self.value_updates |= other.value_updates
        return self


def get_shapes(
    data_dict: dict[str, IOHyperEdge],
    uniadic_keys: dict[UniadicRecord, str] | None = None,
    varadic_keys: dict[Variadic, str] | None = None,
    symbolic: bool = True,
    verbose: bool = False,
    key_mappings: dict[str, str] | None = None,
) -> dict[str, ShapeTemplateType | list[ShapeTemplateType] | None]:
    if key_mappings is None:
        key_mappings = {}
    if uniadic_keys is None:
        uniadic_keys = {}
    if varadic_keys is None:
        varadic_keys = {}
    shapes: dict[str, ShapeTemplateType | list[ShapeTemplateType] | None] = {}
    for key, data in data_dict.items():
        key_name = key_mappings.get(key, key)
        if data.is_tensor:
            assert data.shape is not None
            shapes[key_name] = data.shape.get_shapes(
                uniadic_keys, varadic_keys, symbolic, verbose
            )
        else:
            shapes[key_name] = None
    return shapes


# Is it required to include TensorValueType, it is naturally covered
# by ScalarValueType.
AllValueType = TensorValueType | ScalarValueType | ToBeDetermined


@overload
def find_type(value: Constant) -> type[int] | type[float] | type[bool]: ...
@overload
def find_type(
    value: Tensor[int | float | bool],
) -> type[Tensor[int | float | bool]]: ...
@overload
def find_type(value: ScalarValueType) -> ScalarType: ...


def find_type(
    value: Tensor[int | float | bool] | ScalarValueType | range,
) -> type[Tensor[int | float | bool]] | ScalarType | list[int]:
    typ: type
    if isinstance(value, Tensor):
        typ = Tensor[value.type]  # type: ignore
    elif isinstance(value, Constant):
        typ = constant_type_table[value]
    elif isinstance(value, tuple | list):
        element_types: list[Any] = [find_type(elem) for elem in value]
        if isinstance(value, tuple):
            typ = tuple[*element_types]  # type: ignore
        else:
            result: UnionType | type = reduce(lambda x, y: x | y, element_types)
            typ = list[squash_tensor_types(result)]  # type: ignore
    else:
        typ = type(value)
    return typ


def check_uniformity(sublist: SequenceValType) -> None:
    """Check if all sublists have the same length."""
    lengths = {len(item) if isinstance(item, Sequence) else -1 for item in sublist}
    if len(lengths) > 1:
        raise ValueError("Inconsistent dimensions found in the list.")


def process_value(
    value: TensorValueType,
) -> tuple[list[int] | None, TensorValueType, type[int] | type[float] | type[bool]]:
    # If value is not a sequence, directly return empty shape, value and
    # its type directly.
    if not isinstance(value, tuple | list | range):
        return (
            [] if not isinstance(value, Constant) else constant_shp_table[value],
            value,
            type(value) if not isinstance(value, Constant) else find_type(value),
        )  # type: ignore
    elif isinstance(value, range):
        # Convert range types into list.
        value = list(value)
    else:
        # Check for incompatible dimensions.
        check_uniformity(value)

    # Initialize result as an empty sequence of same type as value.
    result: list[Any] | tuple[Any, ...] = list() if isinstance(value, list) else tuple()

    dominant_type: type[bool] | type[int] | type[float] = bool

    for item in value:
        # Recursively determine the shape, value and type of sub items.
        sub_shape, sub_val, sub_type = process_value(item)
        assert not isinstance(sub_val, Constant)

        if isinstance(result, list):
            result.append(sub_val)
        else:
            result += (sub_val,)

        if sub_type is float:
            dominant_type = float
        elif sub_type is int and dominant_type is bool:
            dominant_type = int
    if sub_shape is not None:
        sub_shape = [len(value)] + sub_shape
    return sub_shape, result, dominant_type


def find_intersection_type(
    type_1: type
    | UnionType
    | GenericAlias
    | type[Tensor[int | float | bool]]
    | type[Any],
    type_2: type
    | UnionType
    | GenericAlias
    | type[Tensor[int | float | bool]]
    | type[Any],
) -> type | UnionType | GenericAlias | type[Tensor[int | float | bool]] | None:
    # If both types are same, return the recursively squashed result.
    if type_1 == type_2:
        return squash_tensor_types_recursively(type_1)
    # ToBeDetermined and Any type can be coerced to all types. Since no recursion
    # will be performed here, squashing should be done recursively.
    if type_1 in (ToBeDetermined, Any):
        return squash_tensor_types_recursively(type_2)
    if type_2 in (ToBeDetermined, Any):
        return squash_tensor_types_recursively(type_1)

    # First squash tensor types in a single tensor type if required.
    type_1 = squash_tensor_types(type_1)
    type_2 = squash_tensor_types(type_2)

    # Find direct intersections.
    subtypes_1 = set(get_args(type_1)) if is_union_type(type_1) else {type_1}
    subtypes_2 = set(get_args(type_2)) if is_union_type(type_2) else {type_2}
    intersect = subtypes_1 & subtypes_2

    # Handle coercion of Any (typing.Any) type to all other types.
    if Any in subtypes_1:
        intersect.update(subtypes_2)
        subtypes_1.remove(Any)
    if Any in subtypes_2:
        intersect.update(subtypes_1)
        subtypes_2.remove(Any)

    # if one of the subtypes have list or tuple without an origin (without square
    # brackets, ex: tuple), look for other set if it contains corresponding type
    # with origin (ex: tuple[int, int]) if the set contains it, add that type with
    # origin (since it contains more information)

    for s_types in (subtypes_1, subtypes_2):
        other_set = subtypes_2 if s_types == subtypes_1 else subtypes_1
        for orig_type in (list, tuple, range):
            if orig_type in s_types:
                for typ in other_set:
                    if isinstance(typ, GenericAlias):
                        if typ.__origin__ == orig_type:
                            intersect.add(typ)
                        elif typ.__origin__ == Sequence:
                            if orig_type is range:
                                if find_intersection_type(int, typ.__args__[0]):
                                    intersect.add(range)
                            else:
                                intersect.add(
                                    orig_type[reduce(lambda x, y: x | y, typ.__args__)]  # type: ignore
                                )

    # Take tuple types from remaining sets and find intesection types
    # of all consistent pairs of cartesian product.
    for typ_1 in subtypes_1.difference(intersect):
        # if not isinstance(typ_1, GenericAlias):
        if get_origin(typ_1) is not None:
            args_1 = typ_1.__args__
            for typ_2 in subtypes_2.difference(intersect):
                # if not isinstance(typ_2, GenericAlias):
                if get_origin(typ_2) is not None:
                    args_2 = typ_2.__args__
                    if typ_1.__origin__ == typ_2.__origin__:
                        if len(args_1) == 0 or len(args_2) == 0:
                            # if one of the lengths of the args_1 and args_2 are zero,
                            # this means one of the types with origin are empty list or
                            # tuple, in that case, take the empty one (tuple[()], or
                            # list[()]) as intersection type.
                            common: Any = typ_1.__origin__[()]

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
                                intersect.add(tuple[*common])

                        elif typ_1.__origin__ in (list, Tensor):
                            if len(args_2) > 1 or len(args_1) > 1:
                                raise TypeError(
                                    f"args of type {typ_1.__origin__} cannot take "
                                    "more than 1 element"
                                )
                            else:
                                common = find_intersection_type(args_1[0], args_2[0])
                            if common:
                                intersect.add(
                                    list[common]
                                    if typ_1.__origin__ is list
                                    else Tensor[common]
                                )
                        # TODO: Below code is duplicate of above code, refactor it.
                        elif typ_1.__origin__ is Sequence:
                            if len(args_2) > 1 or len(args_1) > 1:
                                raise TypeError(
                                    "args of type Sequence cannot take "
                                    "more than 1 element"
                                )
                            else:
                                common = find_intersection_type(args_1[0], args_2[0])
                            if common:
                                intersect.add(Sequence[common])

                    elif Sequence in (typ_1.__origin__, typ_2.__origin__):
                        if typ_1.__origin__ == Sequence:
                            coerced_type = typ_1
                            other_type = typ_2
                        else:
                            coerced_type = typ_2
                            other_type = typ_1

                        other_origin = other_type.__origin__
                        if other_origin is not Tensor:
                            # Sequence type can only be replaced with list or tuple.
                            assert isinstance(other_origin, type(list) | type(tuple))

                            # Replace Sequence with other origin type and resend them
                            # to find_intersection_type.
                            inner_args = reduce(
                                lambda x, y: x | y, coerced_type.__args__
                            )
                            updated_type = (
                                other_origin[inner_args]
                                if other_type.__origin__ is list
                                else other_origin[inner_args, ...]
                            )
                            common = find_intersection_type(updated_type, other_type)
                            if common:
                                intersect.add(common)

    if intersect:
        result = reduce(lambda x, y: x | y, intersect)
        return result
    return None


def enhanced_isinstance[T](obj: Any, type_def: type[T]) -> TypeGuard[T]:
    """isinstance that supports generic type checking.

    Since Python does not support generic type checking feature, naive
    implementation of enhanced_isnstance is introduced

    Examples:

    >>> isinstance((2, 3, 4), tuple[int, int, int]) # python raises error
    >>> enhanced_isinstance((2, 3, 4), tuple[int, int, int]) # returns True

    NOTE: This functionality will be removed when Python brings support
    generic type checking feature.

    """
    return find_intersection_type(find_type(obj), type_def) is not None


def is_tensor_type(
    typ: type | UnionType | GenericAlias | type[Tensor[int | float | bool]] | None,
) -> TypeGuard[type[Tensor[int | float | bool]]]:
    return get_origin(typ) is Tensor or typ is Tensor


def squash_tensor_types(typ: Any) -> Any:
    """
    Reduces multiple Tensor types declarations into a single Tensor type with
    all subtypes.

    This function takes a type that may include multiple Tensor type declarations and
    reduces them into a single Tensor type with all subtypes. It also expands the
    generic "Tensor" type to "Tensor[int | float | bool]".

    Args:
        typ (Any): The type to be squashed. It can be a single type, a union of types,
        or a generic alias.

    Returns:
        Any: The squashed type with reduced Tensor type declarations.

    Example:
        >>> squash_tensor_types(Tensor[int] | Tensor[float])
        Tensor[int | float]
        >>> squash_tensor_types(Tensor[int | float] | bool)
        Tensor[int | float] | bool
        >>> squash_tensor_types(Tensor)
        Tensor[int | float | bool]
    """
    if typ is Tensor:
        typ = Tensor[int | float | bool]
    if is_union_type(typ):
        updated_args = set()
        regular_args = set()
        for arg in get_args(typ):
            if is_tensor_type(arg):
                if arg is Tensor:
                    # Expand generic "Tensor" type to "Tensor[int | float | bool]".
                    arg = Tensor[int | float | bool]
                updated_args.add(arg)
            else:
                # Non-tensor types are added to regular_args.
                regular_args.add(arg)

        if len(updated_args) == 1:
            typ = next(iter(updated_args))
        elif len(updated_args) > 1:
            # If any updated_args are found, create a new Tensor type with all subtypes.
            # Note that if one updated_args exists, there is no need to create it again.
            inner_types = {inner for arg in updated_args for inner in get_args(arg)}
            typ = Tensor[reduce(lambda x, y: x | y, inner_types)]  # type: ignore
        # If there are regular types, add them to the final type.
        if regular_args:
            regular_type = reduce(lambda x, y: x | y, regular_args)
            typ = typ | regular_type if updated_args else regular_type

    return typ


def squash_tensor_types_recursively(typ: Any) -> Any:
    """
    Recursively squashes tensor types within a given type annotation.

    This function takes a type annotation and recursively processes it to squash
    tensor types. It handles Union and generic types by recursively squashing
    their arguments.

    Args:
        typ (Any): The type annotation to process.

    Returns:
        Any: The processed type annotation with tensor types squashed.

    Example:
        >>> squash_tensor_types_recursively(list[Tensor[int] | Tensor[float]])
        list[Tensor[int | float]]
        >>> squash_tensor_types_recursively(Tensor[int | float] | bool)
        Tensor[int | float] | bool
        >>> squash_tensor_types_recursively(
            tuple[Tensor, ...] | Tensor[int | float] | Tensor[float]
        )
        tuple[Tensor[int | float | bool], ...] | Tensor[int | float]

    Notes:
        - If the type is a Union, it will extract tensor types and squash the
          remaining types.
        - If the type is a generic type (excluding Tensor), it will recursively
          squash its arguments.
        - Tuple types with ellipsis are handled specifically to maintain the ellipsis
          in the resulting type.
    """
    # First squash current level tensor types.
    typ = squash_tensor_types(typ)
    if (origin := get_origin(typ)) in (Union, UnionType):
        args = list(get_args(typ))
        # Exclude Tensor type arg since it is processed at the beginning.
        tensor_arg = next((arg for arg in args if get_origin(arg) is Tensor), None)
        if tensor_arg:
            args.remove(tensor_arg)
        squashed_regulars = [squash_tensor_types_recursively(arg) for arg in args]
        if squashed_regulars:
            regular_types = reduce(lambda x, y: x | y, squashed_regulars)
            typ = regular_types | tensor_arg if tensor_arg else regular_types
    elif origin not in (None, Tensor):
        # Simple generic types are squashed by recursively squashing their args.
        # NOTE: Tuple type can contain ellipsis in its args.
        squashed_args = [
            squash_tensor_types_recursively(arg) for arg in get_args(typ) if arg != ...
        ]
        typ = (
            origin[*squashed_args, ...]
            if ... in get_args(typ)
            else origin[*squashed_args]
        )
    return typ


def replace_tensors(
    tensor_map: dict[Tensor[int | float | bool], Tensor[int | float | bool]]
    | None = None,
    value: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined | None = None,
) -> Tensor[int | float | bool] | ScalarValueType:
    """
    Replace occurrences of `current_tensor` with `new_tensor` in the given `value`.

    This function recursively traverses the `value` and replaces any instance of
    `current_tensor` with `new_tensor`. The `value` can be a tensor, a scalar,
    a list, a tuple, or a dictionary containing these types.

    Args:
        tensor_map (dict[Tensor[int | float | bool], Tensor[int | float | bool]] |None):
            dict that holds the mapping of current tensor to new tensor.
        value (Tensor[int | float | bool] | ScalarValueType | ToBeDetermined | None):
            The value in which to replace the tensor. It can be a tensor, a scalar,
            a list, a tuple, a dictionary, or None. Defaults to None.

    Returns:
        Tensor[int | float | bool] | ScalarValueType: The modified value with
        `current_tensor` replaced by `new_tensor`. The return type matches the
        type of the input `value`.
    """
    if tensor_map is None:
        tensor_map = {}
    if isinstance(value, Tensor) and (new_tensor := tensor_map.get(value)):
        new_tensor.is_used = value.is_used
        value = new_tensor
    elif isinstance(value, list | tuple):
        new_value = [replace_tensors(tensor_map, item) for item in value]
        value = tuple(new_value) if isinstance(value, tuple) else new_value
    elif isinstance(value, dict):
        value = {key: replace_tensors(tensor_map, val) for key, val in value.items()}
    return value


def any_differentiable(value: Any) -> bool:
    """
    Check if any element within the given value is a differentiable Tensor.

    This function traverses the input value, which can be a Tensor, list,
    tuple, or dictionary, and checks if any Tensor within the structure
    is differentiable.

    Args:
        value (Any): The input value to check. It can be a Tensor, list,
        tuple, or dictionary.

    Returns:
        bool: True if any Tensor within the input value is differentiable,
        False otherwise.

    Example:
        >>> any_differentiable(
            [
                Tensor(differentiable=False),
                Tensor(differentiable=False),
                (Tensor(differentiable=True), 3.0)
            ]
        )
        True
        >>> any_differentiable((Tensor(differentiable=False), True))
        False
    """
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Tensor):
            # Check if the current Tensor is differentiable.
            if current.differentiable:
                return True
        elif isinstance(current, list | tuple):
            stack.extend(current)
        elif isinstance(current, dict):
            stack.extend(current.values())
    return False


def set_differentiabilities(value: Any, diff_info: Any, updates: Updates) -> None:
    if isinstance(value, Tensor):
        # Set the differentiability of the value Tensor.
        if not isinstance(diff_info, bool):
            raise TypeError("Differentiability must be a boolean value.")
        value.differentiable = diff_info
        if diff_info and value.type is not float:
            updates |= value.set_type(float)

    elif diff_info is True:
        raise ValueError("Non-tensor values cannot be differentiable.")

    elif isinstance(value, list | tuple):
        assert isinstance(diff_info, list | tuple)
        for val, info in zip(value, diff_info, strict=True):
            set_differentiabilities(val, info, updates)

    elif isinstance(value, dict):
        assert isinstance(diff_info, dict)
        for val, info in zip(value.values(), diff_info, strict=True):
            set_differentiabilities(val, info, updates)


def convert_to_numeric_value(
    value: Tensor[int | float | bool] | ScalarValueType,
) -> ScalarValueType:
    if isinstance(value, Tensor):
        return value.value
    elif isinstance(value, list | tuple):
        result = [convert_to_numeric_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(result)
        return result
    elif isinstance(value, dict):
        return {key: convert_to_numeric_value(val) for key, val in value.items()}
    return value


def check_used_tensors(value: Any) -> None:
    """
    Checks if any tensors within the given value are marked as used.

    This function iterates through all tensors extracted from the provided value
    and raises a ValueError if any of them are marked as used. Used tensors cannot
    be reused as a value for another connection.

    Args:
        value (Any): The input value that may contain tensors to be checked.

    Raises:
        ValueError: If any tensor within the value is marked as used.
    """
    for tensor in get_specific_types_from_value(value, Tensor):
        if tensor.is_used:
            raise ValueError(
                "Used tensors can not be used as a value for another connection."
            )


class Tensor(Generic[TypeVarTensorType]):
    def __init__(
        self,
        value: TensorValueType | ToBeDetermined = TBD,
        type: _TensorTypes = int | float | bool,
        shape: ShapeNode | None = None,
        differentiable: bool | None = None,
    ):
        if shape is None:
            # If shape is not provided, create a new shape with a Variadic root.
            shape = ShapeRepr(root=Variadic()).node
        shape.referees.add(self)
        self.shape: ShapeNode = shape
        self._temp_shape: ShapeRepr | None = None  # set random repr
        self.type: _TensorTypes = type
        self.referees: set[IOHyperEdge] = set()
        self.initial_valued: bool = False
        self.differentiable = differentiable
        if differentiable:
            if find_intersection_type(type, float) is None:
                raise TypeError(f"{type} type tensors can be differentiable.")
            self.set_type(float)
        # Initialize value as TBD and then set if any value is provided.
        self.value: TensorValueType | ToBeDetermined = TBD
        if not isinstance(value, ToBeDetermined):
            self.set_value(value)
        self.is_used = False

    def set_type(self, typ: _TensorTypes) -> Updates:
        updates = Updates()
        if self.type != (new_type := find_intersection_type(typ, self.type)):
            if not new_type:
                raise TypeError(
                    f"Acceptable types are {sort_type(self.type)}, but "
                    f"{sort_type(typ)} type is provided!"
                )
            # TODO: Update below assertion!
            assert not (is_tensor_type(new_type) or isinstance(new_type, GenericAlias))
            self.type = new_type
            # Add all referee edges into the updates.
            for edge in self.referees:
                # Note that we have to update the type of each referee edge
                # since a tensor instance in corresponding value is updated.
                updates |= edge.update_type()
        return updates

    def set_value(self, value: TensorValueType) -> Updates:
        if self.value is not TBD and self.value != value:
            raise ValueError(
                f"Value is set before as {self.value}. A value can not be reset."
            )

        updates = Updates()
        # Set value.
        if self.value is TBD:
            # Infer shape, final_value and type from the value.
            shape, val, typ = process_value(value)
            # Set type.
            updates |= self.set_type(typ)
            # Set shape.
            if shape is not None:
                updates |= self.shape.set_values(shape)
            # Add all referee edges into the updates.
            for edge in self.referees:
                updates.add(edge, update_type=UpdateType.VALUE)
                updates.add(edge, update_type=UpdateType.SHAPE)
                updates.value_updates.add(edge)
            self.value = val
        if self.initial_valued is False:
            self.differentiable = False
        return updates

    def match(self, other: Tensor[int | float | bool]) -> Updates:
        updates = Updates()
        if self is not other:
            self.initial_valued |= other.initial_valued
            updates |= self.set_type(other.type)
            updates |= other.set_type(self.type)
            if self.value is not TBD or other.value is not TBD:
                valued, non_valued = (
                    (other, self) if other.value is not TBD else (self, other)
                )
                assert not isinstance(valued.value, ToBeDetermined)
                updates |= non_valued.set_value(valued.value)
                if self.initial_valued is False:
                    self.differentiable = False
                    other.differentiable = False
            elif self.differentiable is None:
                self.differentiable = other.differentiable
                # Differentiable tensors can only be float type.
                if self.differentiable:
                    updates |= self.set_type(float)
            elif (
                other.differentiable is not None
                and self.differentiable != other.differentiable
            ):
                raise ValueError("Differentiability mismatch!")
            # Match shapes.
            updates |= self.match_shapes(other.shape)
            updates.shape_updates.discard(other)
            # Transfer all referees of other to self and update all
            # Tensors in all edges of other with self.
            self.referees |= other.referees
            # Remove other from referees of shape node.
            self.shape.referees.discard(other)

            for edge in other.referees:
                edge.replace_tensor(other, self)
            other.referees = set()
        return updates

    def match_shapes(self, node: ShapeNode) -> Updates:
        updates = Updates()
        if node is not self.shape:
            updates |= self.shape.merge(node)
            self.shape.referees |= node.referees
            prev_node = node
            for ref in node.referees:
                ref.shape = self.shape
            prev_node.reprs = []
            prev_node.referees = set()
        return updates

    def __repr__(self) -> str:
        if isinstance(self.type, UnionType):
            _type = " | ".join(type_def.__name__ for type_def in get_args(self.type))
        else:
            assert isinstance(self.type, type)
            _type = self.type.__name__

        return f"Tensor[{_type}]"


class IOHyperEdge:
    _type: type[Tensor[int | float | bool]] | ScalarType
    _value: Tensor[int | float | bool] | ScalarValueType

    def __init__(
        self,
        type: type[Tensor[int | float | bool]] | ScalarType = ToBeDetermined,
        value: Tensor[int | float | bool] | ScalarValueType = TBD,
        key_origin: str | None = None,
        interval: list[float | int] | None = None,
    ) -> None:
        self.key_origin = key_origin
        self.constraints: dict[UpdateType, set[Constraint]] = {
            type: set() for type in UpdateType
        }
        self.interval: list[float | int] | None = interval
        # Initially set type and value as not determined yet.
        self._type = ToBeDetermined
        self._value = TBD
        self._initial_valued: bool = False
        # Set given type.
        self.set_type(type)
        # If any value is provided, set it.
        if value is not TBD:
            self.set_value(value)

    @property
    def _temp_shape(self) -> ShapeRepr | None:
        if isinstance(self._value, Tensor):
            return self._value._temp_shape
        return None

    @property
    def differentiable(self) -> bool | None:
        if isinstance(self._value, Tensor):
            return self._value.differentiable
        elif self.is_scalar:
            # Scalars are always non-differentiable.
            return False
        # Differentiability of polymorphic edges are defined
        # as None. Depending on its instant type updates, it can
        # become True or False (e.g Tensor or int type).
        return None

    @property
    def tensors(self) -> list[Tensor[int | float | bool]]:
        return get_specific_types_from_value(self._value, Tensor)

    @property
    def is_polymorphic(self) -> bool:
        # Returns if the edge is of polymorphic type or not.
        if self._type is ToBeDetermined:
            return True
        # Look for possible tensor and scalar types.
        return not (self.is_tensor or self.is_scalar)

    @property
    def is_tensor(self) -> bool:
        return get_origin(self._type) is Tensor

    @property
    def is_scalar(self) -> bool:
        has_tensor = bool(
            find_intersection_type(Tensor[int | float | bool], self._type)
        )
        is_tbd = self._type is ToBeDetermined
        return not (is_tbd or has_tensor)

    @property
    def is_valued(self) -> bool:
        return not self.initial_valued and is_valued(self._value)

    @property
    def initial_valued(self) -> bool:
        if isinstance(self._value, Tensor):
            return self._value.initial_valued
        return self._initial_valued

    @property
    def all_constraints(self) -> set[Constraint]:
        result: set[Constraint] = set().union(*self.constraints.values())
        return result

    @property
    def value(self) -> _TensorValueType | ScalarValueType | ToBeDetermined:
        return convert_to_numeric_value(self._value)

    @property
    def shape(self) -> ShapeNode | None:
        if isinstance(self._value, Tensor):
            return self._value.shape
        return None

    @property
    def value_type(self) -> _TensorTypes | ScalarType:
        if isinstance(self._value, Tensor):
            return self._value.type
        else:
            return self._type  # type: ignore

    @property
    def edge_type(self) -> type[Tensor[int | float | bool]] | ScalarType:
        return self._type

    @staticmethod
    def _discard_edge_from_tensors(
        edge: IOHyperEdge,
        value: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined,
    ) -> None:
        if isinstance(value, Tensor):
            # Add self to referees of value and shape.
            value.referees.discard(edge)
        elif isinstance(value, tuple | list):
            # If value is a Sequence, update all its values.
            for val in value:
                IOHyperEdge._discard_edge_from_tensors(edge, val)
        elif isinstance(value, dict):
            # If value is a dict, update all its values.
            for val in value.values():
                IOHyperEdge._discard_edge_from_tensors(edge, val)

    def _create_and_set_tensor_value(
        self, typ: type[Tensor[int | float | bool]]
    ) -> Updates:
        updates = Updates()
        # Create a new tensor and add self to its referees
        # and shape referees.
        tensor_typ = get_args(typ)[0]
        tensor: Tensor[int | float | bool] = Tensor(type=tensor_typ)
        tensor.referees.add(self)
        tensor.shape.referees.add(tensor)
        # Set type of the edge to Tensor.
        self._type = typ
        updates.add(self, UpdateType.TYPE)
        updates.add(self, UpdateType.VALUE)
        updates.add(self, UpdateType.SHAPE)
        self._value = tensor
        tensor.is_used = True
        return updates

    def replace_tensor(
        self,
        current_tensor: Tensor[int | float | bool],
        new_tensor: Tensor[int | float | bool],
    ) -> None:
        self._value = replace_tensors({current_tensor: new_tensor}, self._value)

    def set_type(
        self,
        typ: type[Tensor[int | float | bool]] | ScalarType,
        create_tensor: bool = True,
    ) -> Updates:
        updates = Updates()
        if self._type != typ:
            new_type = find_intersection_type(self._type, typ)
            # If new_type is not different from the current type, return updates.
            if self._type == new_type:
                return updates
            # None new_type means incompatible types are provided,
            # raise TypeError.
            if new_type is None:
                raise TypeError(
                    f"Acceptable types are {sort_type(self._type)}, but "
                    f"{sort_type(typ)} type is provided!"
                )
            elif is_tensor_type(new_type):
                # new_type is strictly a tensor type.
                if not isinstance(self._value, Tensor):
                    if create_tensor:
                        # This is the case when the base type is not determined yet,
                        # meaning it can be of any type. So, if it is requested
                        # to set type to Tensor, we need to create a new Tensor
                        # and set it as the value of the edge.
                        updates |= self._create_and_set_tensor_value(new_type)
                else:
                    # Set type of Tensor object using available_types
                    updates |= self._value.set_type(get_args(new_type)[0])
            # Add self as type update, set new type and update differentiability.
            updates.add(self, UpdateType.TYPE)
            self._type = new_type
        return updates

    def _match_values(
        self,
        value: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined,
        self_value: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined,
        updates: Updates,
        updated_tensors: dict[Tensor[int | float | bool], Tensor[int | float | bool]]
        | None = None,
    ) -> Tensor[int | float | bool] | ScalarValueType:
        if updated_tensors is None:
            updated_tensors = {}

        match (value, self_value):
            case (val1, val2) if val1 == val2:
                return val1

            case (Tensor(), Tensor()):
                # check if Tensors are already updated in previous iterations.
                # If yes, replace them with updated ones.
                value_1 = updated_tensors.get(self_value, self_value)
                value_2 = updated_tensors.get(value, value)
                if value_1 is not value_2:
                    updates |= value_1.match(value_2)
                    updated_tensors[value_2] = value_1
                value_2.is_used = True
                return value_1

            case (list(), list()) if len(value) == len(self_value):
                return [
                    self._match_values(val, self_val, updates, updated_tensors)
                    for val, self_val in zip(value, self_value, strict=True)
                ]

            case (tuple(), tuple()) if len(value) == len(self_value):
                return tuple(
                    self._match_values(val, self_val, updates, updated_tensors)
                    for val, self_val in zip(value, self_value, strict=True)
                )

            case (dict(), dict()) if self_value.keys() == value.keys():
                return {
                    key: self._match_values(
                        value[key], self_value[key], updates, updated_tensors
                    )
                    for key in value
                }

            case (val, ToBeDetermined()) | (ToBeDetermined(), val):
                if self_value is TBD:
                    # add value to updates to inform constraints later on
                    updates.add(self, UpdateType.VALUE)
                    updates.value_updates.add(self)

                # if val is a tensor (or a container that includes Tensor)
                # update Tensor's IOHyperEdge referees and also return
                # updated tensors
                return self.update_tensor_values(val, updates, updated_tensors)

            case _:
                raise ValueError(
                    f"Given value is not compatible with the current value\n"
                    f"    Current value: {self_value}\n"
                    f"    Given value: {value}"
                )

    def update_tensor_values(
        self,
        value: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined,
        updates: Updates,
        updated_tensors: dict[Tensor[int | float | bool], Tensor[int | float | bool]]
        | None = None,
    ) -> ScalarValueType | Tensor[int | float | bool]:
        if updated_tensors is None:
            updated_tensors = {}
        # Updates all required fields of all tensor values in value and
        # update Updates object accordingly
        if isinstance(value, Tensor):
            # Add self to referees of value and shape.
            value = updated_tensors.get(value, value)
            value.referees.add(self)
            value.is_used = True
            # TODO: When two edges set to the same tensor value using
            # different Tensor objects, we need to merge their nodes into
            # a single node. In order to track this, we need to add all
            # uniadic symbols of all reprs to the updates.
            for repr in value.shape.reprs:
                for symbol in repr.prefix + repr.suffix:
                    updates.add(symbol)
            return value

        elif isinstance(value, tuple | list):
            _value = [
                self.update_tensor_values(val, updates, updated_tensors)
                for val in value
            ]
            return tuple(value) if isinstance(value, tuple) else _value

        elif isinstance(value, dict):
            return {
                key: self.update_tensor_values(val, updates, updated_tensors)
                for key, val in value.items()
            }
        else:
            return value

    def set_value(
        self,
        value: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined,
        initial: bool = False,
    ) -> Updates:
        updates = Updates()
        self._initial_valued = initial
        if isinstance(self._value, Tensor):
            self._value.initial_valued = initial
        if isinstance(value, Tensor):
            value.initial_valued = initial
        # Note that two tensor objects having same value are not equal.
        # Tensor values always have to be matched with the existing one
        # or set as the new value.
        if isinstance(value, Constant):
            if not isinstance(self._value, Tensor):
                self.set_type(Tensor[int | float | bool])
            self._value.set_value(value)  # type: ignore
        elif not (isinstance(value, ToBeDetermined) or self._value == value):
            # Set new type without automatic tensor value creation.
            # Note that this first type setting actually validates or
            # rejects value setting based on the compatibility of two types.
            # If they are incompatible, raises an error and value setting is
            # not performed. Else, type is updated, value is set and then
            # type is re-updated based on the final value.
            updated_tensors: dict[
                Tensor[int | float | bool], Tensor[int | float | bool]
            ] = {}
            updates |= self.set_type(find_type(value), create_tensor=False)
            new_value = self._match_values(value, self._value, updates, updated_tensors)
            self._value = replace_tensors(updated_tensors, new_value)
            # Update new type without automatic tensor value creation.
            updates |= self.set_type(find_type(self._value), create_tensor=False)
        if self.is_tensor and self.is_valued:
            self.set_differentiability(False)
        return updates

    def match(self, other: IOHyperEdge) -> Updates:
        # TODO: Get global Updates object for global consistency.
        updates = Updates()
        if self is not other:
            self._initial_valued |= other.initial_valued
            # TODO: If any valued edge, set_value only since it sets types as well.
            updates |= self.set_type(other._type)
            updates |= other.set_type(self._type)
            if self._value != other._value:
                updates.value_updates.add(self)
            if self._value is not TBD or other._value is not TBD:
                valued, non_valued = (
                    (other, self) if other._value is not TBD else (self, other)
                )
                updates |= non_valued.set_value(valued._value, self._initial_valued)
                updates.value_updates.discard(other)
                IOHyperEdge._discard_edge_from_tensors(other, self._value)

        # After modifications done, propagate other constraints into self.
        for type in UpdateType:
            self.constraints[type] |= other.constraints[type]
            other.constraints[type] = set()

        return updates

    def set_differentiability(self, differentiable: Any) -> Updates:
        updates = Updates()
        if differentiable is True:
            # Differentiable edges can only be Tensor[float] type.
            updates |= self.set_type(Tensor[float])

        set_differentiabilities(self._value, differentiable, updates)
        return updates

    def add_constraint(self, constraint: Constraint) -> None:
        for type in constraint.types:
            self.constraints[type].add(constraint)

    def remove_constraint(self, constraint: Constraint) -> None:
        for type in constraint.types:
            self.constraints[type].discard(constraint)

    def update_type(self) -> Updates:
        if self._value is not TBD:
            return self.set_type(find_type(self._value))
        return Updates()


class Uniadic:
    def __init__(self, value: int | set[int] | None = None) -> None:
        # TODO: we could accept *value as input to initialize Uniadic.
        self.metadata = UniadicRecord()
        self.metadata.update_possible_values(value)
        self.metadata.referees.add(self)

    @property
    def value(self) -> int | None:
        return self.metadata.value

    @property
    def possible_values(self) -> set[int] | None:
        return self.metadata.possible_values

    @property
    def reprs(self) -> KeysView[ShapeRepr]:
        return self.metadata.reprs

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Uniadic) -> bool:  # type: ignore
        return id(self.metadata) == id(other.metadata)

    def set_value(self, value: int | set[int] | None) -> bool:  # Do we need set_value
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
                return self.possible_values & other.possible_values  # type: ignore

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
    def reprs(self) -> KeysView[ShapeRepr]:
        return self.reprs_dict.keys()

    @property
    def value(self) -> int | None:
        if self.possible_values is not None and len(self.possible_values) == 1:
            return next(iter(self.possible_values))
        else:
            return None

    def __deepcopy__(self, memo: dict[int, Any]) -> UniadicRecord:
        if id(self) in memo:
            return memo[id(self)]
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance
        # First copy referee Uniadics.
        deepcopy(self.referees, memo)
        for k, v in self.__dict__.items():
            setattr(new_instance, k, deepcopy(v, memo))
        return new_instance

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
        elif len(intersect := self.possible_values & values) > 0:
            self.possible_values = intersect
        else:
            raise ValueError("Possible values mismatch!")
        return self.possible_values

    def match(self, other: UniadicRecord) -> int | None:
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
        return None

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

    def __post_init__(self) -> None:
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

    # def merge(self, other: PossibleValues) -> tuple[bool, bool]:
    #     # Add other's dnf_list and uniadics equality to dnf_list
    #     if self == other or self.check_is_subset(other):
    #         return True, False
    #     # TODO: Check if other has same info but different object
    #     self.dnf_list += other.dnf_list + [
    #         DNF([AND({u1: u2})])
    #         for u1, u2 in zip(self.uniadics, other.uniadics, strict=True)
    #     ]

    #     # Update dnf
    #     return self.update_dnf(), True

    def merge(self, other: PossibleValues) -> tuple[bool, Updates]:
        # Add other's dnf_list and uniadics equality to dnf_list
        updates = Updates()
        if self == other or self.check_is_subset(other):
            return True, updates
        # TODO: Check if other has same info but different object
        for u1, u2 in zip(self.uniadics, other.uniadics, strict=True):
            if u1.metadata != u2.metadata:
                updates.add(u2)
        self.dnf_list += other.dnf_list + [
            DNF([AND({u1: u2})])
            for u1, u2 in zip(self.uniadics, other.uniadics, strict=True)
        ]

        # Update dnf
        return self.update_dnf(), updates

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
        uniadics: set[Uniadic] = set()
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

    def __deepcopy__(self, memo: dict[int, Any]) -> Variadic:
        if id(self) in memo:
            return memo[id(self)]
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance
        # First copy shape reprs.
        deepcopy(self.reprs, memo)
        for k, v in self.__dict__.items():
            setattr(new_instance, k, deepcopy(v, memo))
        return new_instance

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
        updates = self._match(new_root, new_prefix, new_suffix)
        if new_root is not None:
            if self.possibles is not None:
                updates |= self._match_possibles(new_root, new_prefix, new_suffix)
        else:
            if new_suffix != []:
                raise ValueError(
                    "Suffix could only be non-empty if another root is given!"
                )
            updates |= self.update_possible_values(PossibleValues(tuple(new_prefix)))
        self.reprs = set()
        return updates

    def _match_possibles(
        self, root: Variadic, prefix: list[Uniadic], suffix: list[Uniadic]
    ) -> Updates:
        assert self.possibles is not None
        updates = Updates()
        # Clip self.possibles with new_prefix and new_suffix
        # Add clipped uniadics to new equivalences.
        possibles: list[PossibleValues] = []
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
                status, _updates = self.possibles[_len].merge(possibles_dict[_len])
                if status:
                    _possibles[_len] = self.possibles[_len]
                updates |= _updates

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
    def min_len(self) -> int:
        assert self.possibles is not None
        return min(self.possibles.keys())

    @property  # TODO: If not necessary, remove it
    def max_len(self) -> int:
        assert self.possibles is not None
        return max(self.possibles.keys())

    def __hash__(self) -> int:
        return hash(id(self))


def subset_match(sub_repr: ShapeRepr, main_repr: ShapeRepr) -> Updates:
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


def are_unis_identical(unis1: list[Uniadic], unis2: list[Uniadic]) -> bool:
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
        self.referees: set[Tensor[int | float | bool]] = set()

    def __deepcopy__(self, memo: dict[int, Any]) -> ShapeNode:
        if id(self) in memo:
            return memo[id(self)]
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance
        # First copy shape reprs.
        deepcopy(self.reprs, memo)
        for k in self.__slots__:
            setattr(new_instance, k, deepcopy(getattr(self, k), memo))
        return new_instance

    def add_repr(self, repr: ShapeRepr) -> None:
        self.reprs.append(repr)
        repr.node = self

    def merge(self, other: ShapeNode) -> Updates:
        updates = Updates()
        resolved_reprs: set[ShapeRepr] = set()
        remaining_reprs: list[ShapeRepr] = []
        add_constraint = False

        if self != other:
            for repr2 in other.reprs:
                for repr1 in self.reprs:
                    # Match all reprs of other with self.reprs.
                    updates |= repr1.match(repr2)
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
                    for edge in tensor.referees:
                        updates.constraints |= edge.constraints[UpdateType.SHAPE]

            for repr in resolved_reprs:
                # remove_repr_from_symbols(repr)
                repr.clear()

        return updates

    def combine(self) -> Updates:
        updates = Updates()
        same_reprs: set[ShapeRepr] = set()
        # Iterate over all repr pairs and remove matching reprs.
        for repr, other_repr in combinations(self.reprs, 2):
            if repr not in same_reprs and other_repr not in same_reprs:
                updates |= repr.match(other_repr)
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

    def set_values(self, values: Sequence[int | None]) -> Updates:
        updates = Updates()
        for repr in self.reprs:
            updates |= repr.set_values(values)
        return updates

    def get_shapes(
        self,
        u_keys: dict[UniadicRecord, str] | None = None,
        v_keys: dict[Variadic, str] | None = None,
        symbolic: bool = True,
        verbose: bool = False,
    ) -> ShapeTemplateType | list[ShapeTemplateType]:
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
                n_reprs: set[ShapeRepr] = set()
                for uni in repr.prefix + repr.suffix:
                    n_reprs |= uni.reprs

                if repr.root is not None:
                    n_reprs |= repr.root.reprs

                best_n_reprs: set[ShapeRepr] = set()
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
type ConstraintFunctionType = Callable[..., ConstrainResultType]


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

    def __deepcopy__(self, memo: dict[int, Any]) -> ShapeRepr:
        if id(self) in memo:
            return memo[id(self)]
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance
        # First copy shape node.
        deepcopy(self.node, memo)
        for k in self.__slots__:
            setattr(new_instance, k, deepcopy(getattr(self, k), memo))
        return new_instance

    @property
    def reverse(self) -> list[Uniadic]:
        return self.suffix[::-1] if self.root is not None else self.prefix[::-1]

    def set_symbol_order(self) -> None:
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

    def is_equal(self, other: ShapeRepr) -> bool:
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

    def __getitem__(self, position: int) -> Uniadic:
        # TODO: Currently position could only be int, but we should support slicing
        # operations too (e.g. repr[:2]) if it is possible (if index of Variadic
        # field allows the operation).
        if position < 0 and self.root is not None:
            return self.suffix[position]
        else:
            return self.prefix[position]

    def __setitem__(self, position: int, new_item: Uniadic) -> None:
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
    def update_uniadics(
        outer_list: list[Uniadic], inner_list: list[Uniadic]
    ) -> Updates:
        updates = Updates()
        for outer, inner in zip(outer_list, inner_list, strict=False):
            if outer.metadata != inner.metadata:
                updates |= outer.match(inner)
        return updates

    def get_shapes(
        self,
        u_keys: dict[UniadicRecord, str] | None = None,
        v_keys: dict[Variadic, str] | None = None,
        symbolic: bool = True,
    ) -> ShapeTemplateType:
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
    def _get_uniadic_shapes(
        uniadic_list: list[Uniadic],
        cache: dict[UniadicRecord, str],
        symbolic: bool = True,
    ) -> list[int | str | None]:
        final_list: list[int | str | None] = []
        for uniadic in uniadic_list:
            if (value := uniadic.value) is None and symbolic:
                _value = cache.setdefault(uniadic.metadata, "u" + str(len(cache) + 1))
                final_list.append(_value)
            else:
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
        updates |= self.update_uniadics(self.prefix, prefix)
        updates |= self.update_uniadics(
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

    def match(self, other: ShapeRepr) -> Updates:
        return self.inner_match(other.prefix, other.root, other.suffix)

    def set_values(self, values: Sequence[int | None]) -> Updates:
        updates = Updates()
        if self.root is not None:
            uniadics = [Uniadic(value) for value in values]
            updates |= self.update_uniadics(self.prefix, uniadics)
            updates |= self.update_uniadics(self.reverse, uniadics[::-1])
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

    def clear(self) -> None:
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
    fn: ConstraintFunctionType
    types: list[UpdateType] = field(default_factory=lambda: [UpdateType.SHAPE])
    call_counter: int = 0
    parents: set[Constraint] = field(default_factory=lambda: set())
    children: set[Constraint] = field(default_factory=lambda: set())

    def __call__(self, keys: list[IOHyperEdge]) -> ConstrainResultType:
        status = False
        updates = Updates()
        if UpdateType.SHAPE in self.types:
            # tensor_keys = [key for key in keys if key.is_tensor]
            tensor_keys: list[Tensor[int | float | bool]] = list(
                chain(*[key.tensors for key in keys])
            )
            for reprs in product(*[key.shape.reprs for key in tensor_keys]):
                for idx, repr in enumerate(reprs):
                    tensor_keys[idx]._temp_shape = repr
                status, newly_added_symbols = self.fn(*keys)
                updates |= newly_added_symbols
                # Clear temp_shape.
                for idx in range(len(reprs)):
                    tensor_keys[idx]._temp_shape = None
        else:
            status, newly_added_symbols = self.fn(*keys)
            updates |= newly_added_symbols
        if status:
            updates.constraints |= self.children
            self.clear()
        self.call_counter += 1
        return status, updates

    def add_dependencies(self, *args: Constraint) -> None:
        self.parents.update(args)
        for constr in args:
            constr.children.add(self)

    def clear(self) -> None:
        for constr in self.children:
            constr.parents.remove(self)

        for constr in self.parents:
            constr.children.remove(self)

        self.parents = set()
        self.children = set()

    def __hash__(self) -> int:
        return hash(id(self))


@overload
def add_lengths(x: int, y: int, const: Iterator[int]) -> int: ...


@overload
def add_lengths(x: str, y: str, const: Iterator[str]) -> str: ...


def add_lengths(x: str | int, y: str | int, const: Iterator[str | int]) -> str | int:
    return x + next(const) + y  # type: ignore


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

    def add_header(self, header: list[str]) -> None:
        self.headers.append(header)

    def add_row(self, row: RowColumnType) -> None:
        self.cells.append(row)

    def add_column(self, column: RowColumnType) -> None:
        for idx, row in enumerate(column[: len(self.headers)]):
            self.headers[idx].append(row)  # type: ignore

        for idx_, row in enumerate(column[len(self.headers) :]):
            self.cells[idx_].append(row)  # type: ignore

    def _calculate_table_specs(self) -> None:
        """Calculates table specifications for constructing the table. Calculates cell
        widths for each column and calculates cell heights for each column
        """

        # concatenate all headers and cells into a one list
        all_elems = self.headers + self.cells

        #  Initialize cell_heights and cell_widths lists, these
        # lists will hold minimum height and width that every cell
        # can have respectively.
        cell_heights: list[list[int]] = []
        cell_widths: list[list[int]] = []

        for row in all_elems:
            row_heights: list[int] = []
            row_widths: list[int] = []
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

    def _adjust_table(self) -> None:
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

    def compile(
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
            partial(add_lengths, const=(len(row) for row in row_sep)),
            self.each_row_width,
        )
        table_constructor_fn: Callable[[str, str], str] = partial(
            add_lengths, const=cycle(row_sep)
        )  # type: ignore
        table_constructor_fn_w_spaces: Callable[[str, str], str] = partial(  # type: ignore
            add_lengths, const=cycle(len(row) * " " for row in row_sep)
        )
        end = "\n"
        header_list: list[str] = []
        cell_list: list[str] = []

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

    def display(self) -> None:
        """Prints the table"""
        print(self.header_str)
        print(self.cell_str)

    @staticmethod
    def construct_subtable_row(
        stub: str,
        arg_max_lengths: list[int],
        adjustments: list[str],
        *args: list[list[str]],
    ) -> list[str]:
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
        value: list[str] | str,
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
            new_list: list[str] = ["None"]
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
        in_dict: Mapping[str, list[str] | dict[str, Any]],
        seperator: str = " : ",
        left_align: Literal["left", "right", "center"] = "left",
        right_align: Literal["left", "right", "center"] = "left",
        left_length: int = 0,
        right_length: int = 0,
        len_space: int = 0,
        space_fill: str = "-",
        r_len: int = 0,
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

        table: list[str] = []
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
    shape_template: Mapping[str, ShapeTemplateType],
    solver: ConstraintSolver,
) -> dict[str, ShapeRepr]:
    used_keys: UsedKeysType = {}
    return {
        key: create_shape_repr(shp_list, solver, used_keys)
        for key, shp_list in shape_template.items()
    }


def create_shape_repr(
    shp_list: ShapeTemplateType,
    solver: ConstraintSolver | None = None,
    used_keys: UsedKeysType | None = None,
) -> ShapeRepr:
    if used_keys is None:
        _used_keys: UsedKeysType = {}
    else:
        _used_keys = used_keys
    if solver is not None and shp_list == []:
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
                        if solver is None:
                            symbol_obj = Uniadic(symbol)
                        elif (_uni := solver.symbol_store.get(symbol)) is not None:
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
    conns: dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]],
    name: str,
    shape: dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]] | None = None,
    types: dict[str, tuple[dict[str, str], dict[str, str]]] | None = None,
    params: dict[str, tuple[dict[str, str], dict[str, str]]] | None = None,
) -> Table:
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

    removed_cols: list[int] = []
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

        input_shapes = list(input_shape.values())
        output_shapes = list(output_shape.values())

        in_types = [[key] for key in input_types.values()]
        out_types = [[key] for key in output_types.values()]

        model_input_conn = [value if value else ["--"] for value in input_conn.values()]
        model_output_conn = [
            value if value else ["--"] for value in output_conn.values()
        ]

        in_params = [[param] for param in input_params.values()]
        out_params = [[param] for param in output_params.values()]

        input_args = [in_keys, input_shapes, in_types, model_input_conn, in_params]
        output_args = [
            out_keys,
            output_shapes,
            out_types,
            model_output_conn,
            out_params,
        ]

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
        table.add_row([[model_name], cell])

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


ShapeResultType = Mapping[str, ShapeTemplateType | list[ShapeTemplateType] | None]


def get_summary_shapes(
    model_shapes: dict[str, ShapeResultType],
    conn_info: dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]],
) -> dict[str, tuple[ShapeResultType, ShapeResultType]]:
    shape_info: dict[str, tuple[ShapeResultType, ShapeResultType]] = {}
    for model_name in conn_info:
        shape = model_shapes[model_name]
        input_conns, output_conns = conn_info[model_name]
        input_shapes = {key: shape[key] for key in input_conns}
        output_shapes = {key: shape[key] for key in output_conns}
        shape_info[model_name] = (input_shapes, output_shapes)
    return shape_info


# TODO: Name mappings in there should more specialized as Any is not a good choice
# name mappings should be dict[BaseModel, str]
# data_memo should be dict[int, IOHyperEdge]
# however, if this happens, there will be circular import problem
# So carrying this function to another module may be a better idea
# (maybe this function could be a method of BaseModel?)
def get_summary_types(
    name_mappings: dict[Any, Any], data_memo: dict[Any, Any] | None = None
) -> dict[str, tuple[dict[str, str], dict[str, str]]]:
    if data_memo is None:
        data_memo = {}

    type_info: dict[str, tuple[dict[str, str], dict[str, str]]] = {}

    for model, model_name in name_mappings.items():
        in_dict, out_dict = type_info.setdefault(model_name, ({}, {}))
        for key in model.conns.all:
            key_mappings = model.generate_keys(include_outputs=True)
            in_key = key_mappings.get(key, key)
            data = model.conns.get_data(key)
            pm_data = data_memo.get(id(data), data)
            data_type = pm_data.value_type
            if not hasattr(data_type, "__args__"):
                str_type = data_type.__name__
            else:
                sorted_type = sort_type(pm_data.value_type)
                str_type = str(sorted_type)
            if key in model.input_keys:
                in_dict[in_key] = str_type
            else:
                out_dict[in_key] = str_type
    return type_info


def is_type_adjustment_required(
    data: dict[str, IOHyperEdge], inputs: list[str]
) -> bool:
    if len(inputs) <= 2:
        return False
    inputs = inputs[:2]
    left = data[inputs[0]]
    right = data[inputs[1]]
    if not (isinstance(left._value, Tensor) and isinstance(right._value, Tensor)):
        return False

    rule1 = issubclass(float, left._value.type) and issubclass(int, right._value.type)
    rule2 = issubclass(float, right._value.type) and issubclass(int, left._value.type)

    return rule1 | rule2


def is_index_type(
    index: Any,
) -> TypeGuard[tuple[int | slice | EllipsisType | Tensor[int] | None, ...]]:
    return isinstance(index, tuple) and all(
        isinstance(val, int | slice | EllipsisType | None)
        or find_intersection_type(
            find_type(val),
            Tensor[int] | VariableSequenceType[int],  # type: ignore
        )
        is not None
        for val in index
    )


def is_valued(
    value: ScalarValueType | Tensor[int | float | bool] | ToBeDetermined,
) -> bool:
    match value:
        case ToBeDetermined():
            return False
        case Tensor():
            return value.value is not TBD
        case dict():
            return all(is_valued(v) for v in value.values())
        case list() | tuple():
            return all(is_valued(item) for item in value)
        case _:
            return True
