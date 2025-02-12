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

from collections.abc import Iterable, Iterator, MutableMapping
from enum import Enum, IntEnum
from itertools import compress
from typing import Any, Generic, TypeVar

from ..backends.backend import Backend
from ..core import Constant, DataType, constant_type_table

__all__ = [
    "convert_specs_to_dict",
    "BiMap",
    "BiMultiMap",
    "pack_data_into_time_slots",
    "unpack_time_slot_data",
    "pack_encoder_target_data_into_time_slots",
    "find_dominant_type",
]

IOType = dict[str, Any]


class PaddingType(IntEnum):
    VALID = 0
    SAME = 1


def convert_specs_to_dict(specs: Any) -> str | dict[str, Any] | list[Any]:
    """This function converts given specs to
    dict which could be saved to JSON.

    Parameters
    ----------
    specs : Any
        Specs is the input to be converted.
        Specs Callable, Enum, dict of specs,
        or list of specs.

    Returns
    -------
    Any
        Converted specs which could be saved to JSON.
    """
    if callable(specs):
        return specs.__name__
    elif isinstance(specs, Enum):
        return specs.name
    elif isinstance(specs, dict):
        return {k: convert_specs_to_dict(v) for k, v in specs.items()}
    elif isinstance(specs, list):
        return [convert_specs_to_dict(k) for k in specs]
    else:
        return specs


T = TypeVar("T")


class OrderedSet(Generic[T]):
    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._data: dict[T, None] = dict.fromkeys(iterable or [])

    def add(self, item: T) -> None:
        self._data[item] = None

    def discard(self, item: T) -> None:
        self._data.pop(item, None)

    def remove(self, item: T) -> None:
        if item not in self._data:
            raise KeyError(item)
        self._data.pop(item)

    def pop(self) -> T:
        if not self._data:
            raise KeyError("pop from an empty set")
        return self._data.popitem()[0]

    def clear(self) -> None:
        self._data.clear()

    def update(self, *iterables: Iterable[T]) -> None:
        for iterable in iterables:
            for item in iterable:
                self.add(item)

    def difference_update(self, *iterables: Iterable[T]) -> None:
        for iterable in iterables:
            for item in iterable:
                self.discard(item)

    def intersection_update(self, *iterables: Iterable[T]) -> None:
        common_items = set(self._data).intersection(*iterables)
        self._data = dict.fromkeys(common_items)

    def symmetric_difference_update(self, iterable: Iterable[T]) -> None:
        for item in iterable:
            if item in self._data:
                self.discard(item)
            else:
                self.add(item)

    def union(self, *iterables: Iterable[T]) -> OrderedSet[T]:
        new_set = OrderedSet(self)
        new_set.update(*iterables)
        return new_set

    def difference(self, *iterables: Iterable[T]) -> OrderedSet[T]:
        new_set = OrderedSet(self)
        new_set.difference_update(*iterables)
        return new_set

    def intersection(self, *iterables: Iterable[T]) -> OrderedSet[T]:
        common_items = set(self._data).intersection(*iterables)
        return OrderedSet(common_items)

    def symmetric_difference(self, iterable: Iterable[T]) -> OrderedSet[T]:
        new_set = OrderedSet(self)
        new_set.symmetric_difference_update(iterable)
        return new_set

    def issubset(self, other: OrderedSet[T]) -> bool:
        return set(self._data).issubset(other)

    def issuperset(self, other: OrderedSet[T]) -> bool:
        return set(self._data).issuperset(other)

    def isdisjoint(self, other: OrderedSet[T]) -> bool:
        return set(self._data).isdisjoint(other)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item: T) -> bool:
        return item in self._data

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"OrderedSet({list(self._data)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, set):
            return set(self._data) == other
        if not isinstance(other, OrderedSet):
            return False
        return list(self._data) == list(other._data)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __or__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        return self.union(other)

    def __ior__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        self._data |= other._data
        return self

    def __ror__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        return self.union(other)

    def __and__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        return self.intersection(other)

    def __sub__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        return self.difference(other)

    def __xor__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        return self.symmetric_difference(other)


K = TypeVar("K")
V = TypeVar("V")


class BiMap(MutableMapping[K, V]):
    # Implements a bi-directional map for storing unique keys/values using two
    # dictionaries.
    # TODO: override __reversed__ for BiMap
    inverse: dict[V, K]
    _table: dict[K, V]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.inverse = inverse = {}
        self._table = table = dict(*args, **kwargs)
        for key, value in table.items():
            if value in inverse:
                raise ValueError(f"Value {value} maps to multiple keys")
            inverse[value] = key

    def __getitem__(self, key: K) -> V:
        return self._table[key]

    def __setitem__(self, key: K, value: V) -> None:
        if value in (inverse := self.inverse):
            existing_key = inverse[value]
            if key != existing_key:
                msg = f"Value {value} already exists with key {existing_key}"
                raise ValueError(msg)
        else:
            if key in (table := self._table):
                del inverse[table[key]]
            inverse[value] = key
            table[key] = value

    def __delitem__(self, key: K) -> None:
        del self.inverse[(table := self._table)[key]]
        del table[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)


class BiMultiMap(MutableMapping[K, V]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.inverse: dict[V, list[K]] = {}
        self._table = table = dict(*args, **kwargs)
        for key, value in table.items():
            self.__add_inverse_values(key, value)

    def __add_inverse_values(self, key: K, value: V) -> None:
        if isinstance(value, list):
            for v in value:
                self.inverse.setdefault(v, []).append(key)
        else:
            raise TypeError("Requires list type for values!")

    def remove_inverse_values(self, key: K) -> None:
        for k in self._table[key]:
            self.inverse[k].remove(key)
            # Remove k from inverse map if it has no output connection.
            if self.inverse[k] == []:
                del self.inverse[k]

    def __getitem__(self, key: K) -> V:
        return self._table[key]

    def __setitem__(self, key: K, value: V) -> None:
        if key in self._table:
            self.remove_inverse_values(key)
        self._table[key] = value
        self.__add_inverse_values(key, value)

    def __delitem__(self, key: K) -> None:
        # TODO: test here.
        self.remove_inverse_values(key)
        del self._table[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)


# TODO: annotate this fn's types more precisely.
def pack_data_into_time_slots(
    data: Any,
    backend: Any,
    lengths: Any = None,
    key: tuple[Any, ...] = ("input",),
    index: int = 0,
    return_indices: bool = False,
    sorted_data: bool = False,
) -> tuple[Any, ...]:
    # TODO: Very slow. Try to jit or something else for faster execution.
    def find_slot_lengths(stacked_data: Any, lengths: Any) -> dict[str, DataType]:
        time_slots: dict[str, DataType] = {}
        for i in range(max_length):
            slot_samples = len(lengths)
            time_slots[key[0] + f"{i}"] = stacked_data[:slot_samples, i, :][
                :, None, ...
            ]
            lengths -= 1
            lengths = lengths[lengths != 0]
        return time_slots

    if lengths is None:
        lengths = backend.array([d[index].shape[0] for d in data])
    # If given data not sorted wrt. sequence length, sort it.
    if not sorted_data:
        if return_indices:
            indices, data = zip(
                *sorted(
                    enumerate(data), key=lambda d: d[1][index].shape[0], reverse=True
                ),
                strict=False,
            )
        else:
            data = sorted(data, key=lambda d: d[index].shape[0], reverse=True)
        lengths = backend.sort(lengths, descending=True)
    max_length = lengths[0]
    data_sorted = data

    max_size = lengths[0]
    if backend.backend_type == "torch":
        padded = [
            backend.pad(arr[index], (0, 0, 0, max_size - arr[index].shape[0]))
            for arr in data
        ]
    else:
        padded = [
            backend.pad(arr[index], ((0, max_size - arr[index].shape[0]), (0, 0)))
            for arr in data
        ]
    stacked_data = backend.stack(padded)
    # stacked_data = backend.concatenate(padded)
    time_slots = find_slot_lengths(stacked_data, lengths)
    # print("Second method takes: ", time.time() - start)
    return (
        (time_slots, data_sorted)
        if not return_indices
        else (indices, time_slots, data_sorted)
    )


def unpack_time_slot_data(
    backend: Backend[DataType],
    data: dict[str, DataType],
    max_length: int,
    max_size: int,
    output_dim: int,
    key: str = "decoder_out",
    indices: list[int] | None = None,
) -> DataType:
    # TODO: Can we infer max_size automatically???
    # Pad all time slot to the length of t0.
    concat_data = backend.cat(
        [
            backend.pad(
                data[f"{key}{idx}"],
                [(0, max_size - data[f"{key}{idx}"].shape[0]), (0, 0), (0, 0)],
            )
            for idx in range(max_length)
            if f"{key}{idx}" in data
        ],
        axis=-1,
    ).reshape(max_size, -1, output_dim)
    # Indices were calculated wrt target sequence lengths but we must
    # invert it wrt initial input sequences for correct comparison and calculation.
    if indices is not None:
        reverse_indices = backend.array(
            [idx for idx, _ in sorted(enumerate(indices), key=lambda x: x[1])]
        )
        concat_data = concat_data[reverse_indices]
    return concat_data


def pack_encoder_target_data_into_time_slots(
    backend: Backend[DataType],
    data: list[tuple[DataType, ...]],
    lengths: None | DataType = None,
    key: tuple[str, ...] = ("target",),
) -> dict[str, DataType]:
    # TODO: Very slow. Try to jit or something else for faster execution.
    # In order to use this function properly, data should be sorted
    # wrt input lengths.
    time_slots = {}  # type: ignore
    if lengths is None:
        lengths = backend.array([d[0].shape[0] for d in data])
    unique_lengths = backend.unique(lengths)
    for length in unique_lengths:
        assert isinstance(length, int)
        target_key = key[0] + f"{length - 1}"
        filter = lengths == length
        filtered_data: list[DataType] = list(compress(data, filter))
        time_data = backend.stack([d[1] for d in filtered_data], axis=0)
        time_slots[target_key] = time_data
    return time_slots


def convert_to_tuple[T: Any](
    nested_list: list[T] | tuple[T, ...] | T,
) -> tuple[T, ...] | T:
    if isinstance(nested_list, list):
        return tuple(convert_to_tuple(item) for item in nested_list)  # type: ignore
    else:
        return nested_list


def convert_to_list(
    value: int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | Any,
) -> Any:
    if isinstance(value, tuple):
        return [convert_to_list(element) for element in value]
    else:
        return value


# Other utils
def find_dominant_type(
    lst: Any, raise_error: bool = True
) -> type[int] | type[float] | type[bool]:
    # return dominant type of parameters in the list.
    # dominant type is referenced from numpy and in folloing order: bool -> int -> float
    # if any of the parameters are different from these three types, returns ValueError
    # if raise_error set to True. Otherwise returns this type.

    # Examples:
    # list contains both floats and ints -> return float
    # list contains both ints and bools -> return int
    # list contains only bools -> return bool
    # list contains all three of types -> return float

    if isinstance(lst, list | tuple):
        curr_val: type[bool] | type[int] | type[float] = bool
        for elem in lst:
            val = find_dominant_type(elem, raise_error)
            if val is float:
                curr_val = float
            elif val is int:
                if curr_val is bool:
                    curr_val = int
            elif val is not bool:
                curr_val = val
                break
        return curr_val
    elif isinstance(lst, bool | float | int):
        return type(lst)
    elif isinstance(lst, Constant):
        return constant_type_table[lst]
    elif raise_error:
        raise ValueError(
            f"given input contains {type(lst)} type. Allowed types are: list, tuple, "
            "float, int, bool"
        )
    return type(lst)
