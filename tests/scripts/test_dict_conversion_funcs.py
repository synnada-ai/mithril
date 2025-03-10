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

from mithril.framework.common import Tensor
from mithril.utils.dict_conversions import _deserialize_type_info, _serialize_type_info


def test_serialize_deserialize_basic():
    type_info = int
    serialized = _serialize_type_info(type_info)
    assert serialized == "int"
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_union():
    type_info = int | float
    serialized = _serialize_type_info(type_info)
    assert serialized == ["int", "float"]
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_generic_tuple_1():
    type_info = tuple[int, ...]
    serialized = _serialize_type_info(type_info)
    assert serialized == {"tuple": ["int", "..."]}
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_generic_tuple_2():
    type_info = tuple[int, float]
    serialized = _serialize_type_info(type_info)  # type: ignore
    assert serialized == {"tuple": ["int", "float"]}
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_generic_list():
    type_info = list[int]
    serialized = _serialize_type_info(type_info)
    assert serialized == {"list": ["int"]}
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_generic_tensor():
    type_info = Tensor[int | float]
    serialized = _serialize_type_info(type_info)
    assert serialized == {"Tensor": ["int", "float"]}
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_generic_union():
    type_info = tuple[int | float, ...]
    serialized = _serialize_type_info(type_info)
    assert serialized == {"tuple": [["int", "float"], "..."]}
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_union_generic():
    type_info = bool | tuple[int | float, ...]
    serialized = _serialize_type_info(type_info)
    assert serialized == ["bool", {"tuple": [["int", "float"], "..."]}]
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_union_generic_tensor():
    type_info = bool | Tensor[int | float]
    serialized = _serialize_type_info(type_info)
    assert serialized == ["bool", {"Tensor": ["int", "float"]}]
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_list_generic_union_generic():
    type_info = list[bool | tuple[Tensor[int | float], ...]]
    serialized = _serialize_type_info(type_info)
    assert serialized == {
        "list": ["bool", {"tuple": [{"Tensor": ["int", "float"]}, "..."]}]
    }
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_tuple_generic_union_generic_3_types():
    type_info = tuple[bool | tuple[Tensor[int | float], ...], int | float, str]
    serialized = _serialize_type_info(type_info)  # type: ignore
    assert serialized == {
        "tuple": [
            ["bool", {"tuple": [{"Tensor": ["int", "float"]}, "..."]}],
            ["int", "float"],
            "str",
        ]
    }
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info


def test_serialize_deserialize_dict_generic_union_generic_3_types():
    type_info = dict[
        str, tuple[bool | tuple[Tensor[int | float], ...], int | float, str]
    ]
    serialized = _serialize_type_info(type_info)  # type: ignore
    assert serialized == {
        "dict": [
            "str",
            {
                "tuple": [
                    ["bool", {"tuple": [{"Tensor": ["int", "float"]}, "..."]}],
                    ["int", "float"],
                    "str",
                ]
            },
        ]
    }
    deserialized = _deserialize_type_info(serialized)
    assert deserialized == type_info
