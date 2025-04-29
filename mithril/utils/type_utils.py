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

from types import GenericAlias, UnionType
from typing import (
    Any,
    TypeGuard,
    Union,
    get_origin,
)


def is_int_tuple_tuple(
    data: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
) -> TypeGuard[tuple[tuple[int, int], tuple[int, int]]]:
    return isinstance(data[0], tuple)


def is_tuple_int(t: Any) -> TypeGuard[tuple[int, ...]]:
    return isinstance(t, tuple) and all(isinstance(i, int) for i in t)


def is_list_int(t: Any) -> TypeGuard[list[int]]:
    return isinstance(t, list) and all(isinstance(i, int) for i in t)


def is_tuple_of_two_ints(obj: Any) -> TypeGuard[tuple[int, int]]:
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and all(isinstance(i, int) for i in obj)
    )


def is_union_type(
    type: Any,
) -> TypeGuard[UnionType]:
    return (get_origin(type)) in (Union, UnionType)


def is_generic_alias_type(
    typ: Any,
) -> TypeGuard[GenericAlias]:
    true_generic_alias = type(typ) is GenericAlias
    not_union = not is_union_type(typ)
    is_origin_single_type = type(get_origin(typ)) is type
    return true_generic_alias or (not_union and is_origin_single_type)
