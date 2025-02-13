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

from collections.abc import Callable, Iterator, MutableMapping
from typing import Any, TypeVar

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

    # elif isinstance(lst, Constant):
    #     return constant_type_table[lst]


# Other utils
def find_dominant_type(
    lst: Any,
    raise_error: bool = True,
    special_fn: Callable[[Any], type[bool] | type[int] | type[float] | None]
    | None = None,
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
            val = find_dominant_type(elem, raise_error, special_fn)
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
    elif special_fn and (res := special_fn(lst)):
        return res
    elif raise_error:
        raise ValueError(
            f"given input contains {type(lst)} type. Allowed types are: list, tuple, "
            "float, int, bool"
        )
    return type(lst)
