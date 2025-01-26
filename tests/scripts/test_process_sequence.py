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

from collections.abc import Sequence

import pytest

from mithril.framework.common import Tensor, process_value
from mithril.models import Model, ToTensor


def test_process_value_all_list_int():
    input = [[0, 1, 2], [4, 5, 6]]
    expected_result = [2, 3], [[0, 1, 2], [4, 5, 6]], int
    result = process_value(input)
    assert result == expected_result


def test_process_value_all_list_float():
    input: list[list[float | int]] = [[0, 1, 2.0], [4, 5, 6]]
    expected_result = [2, 3], [[0, 1, 2], [4, 5, 6]], float
    result = process_value(input)
    assert result == expected_result


def test_process_value_range_list_int():
    input = [range(3), [4, 5, 6]]
    expected_result = [2, 3], [[0, 1, 2], [4, 5, 6]], int
    result = process_value(input)
    assert result == expected_result


def test_process_value_list_range_int():
    input = [[4, 5, 6], range(3)]
    expected_result = [2, 3], [[4, 5, 6], [0, 1, 2]], int
    result = process_value(input)
    assert result == expected_result


def test_process_value_list_value_list_range_int():
    input = [[[4, 5, 6], range(3)], [range(3), [4, 5, 6]]]
    expected_result = [2, 2, 3], [[[4, 5, 6], [0, 1, 2]], [[0, 1, 2], [4, 5, 6]]], int
    result = process_value(input)
    assert result == expected_result


def test_process_value_list_value_list_range_int_with_bool():
    input = [[[True, 5, 6], range(3)], [range(3), [4, False, 6]]]
    expected_result = [2, 2, 3], [[[1, 5, 6], [0, 1, 2]], [[0, 1, 2], [4, 0, 6]]], int
    result = process_value(input)
    assert result == expected_result


def test_process_value_list_value_list_range_float():
    input: list[list[list[int | float] | Sequence[int]]] = [
        [[4, 5, 6], range(3)],
        [range(3), [4, 5.0, 6]],
    ]
    expected_result = [2, 2, 3], [[[4, 5, 6], [0, 1, 2]], [[0, 1, 2], [4, 5, 6]]], float
    result = process_value(input)
    assert result == expected_result


def test_process_value_full_range():
    input = range(3)
    expected_result = [3], [0, 1, 2], int
    result = process_value(input)
    assert result == expected_result


def test_process_value_inner_full_range():
    input = [range(3)]
    expected_result = [1, 3], [[0, 1, 2]], int
    result = process_value(input)
    assert result == expected_result


def test_process_value_inconsistent_shape():
    input = [range(3), [1, 2]]
    with pytest.raises(ValueError) as err_info:
        process_value(input)
    assert str(err_info.value) == "Inconsistent dimensions found in the list."


def test_tensor_initialization_1():
    sequence: list[list[int] | Sequence[int]] = [[0, 1, 2], range(3, 6)]
    tensor = Tensor(sequence)
    assert tensor.value == [[0, 1, 2], [3, 4, 5]]
    assert tensor.shape.get_shapes() == [2, 3]
    assert tensor.type is int


def test_tensor_initialization_2():
    sequence: list[list[list[int | float] | Sequence[int]]] = [
        [[0, 1.0, 2], range(3, 6)]
    ]
    tensor = Tensor(sequence)
    assert tensor.value == [[[0, 1.0, 2], [3, 4, 5]]]
    assert tensor.shape.get_shapes() == [1, 2, 3]
    assert tensor.type is float


def test_tensor_initialization_3():
    sequence: list[list[list[int | float] | Sequence[int]]] = [
        [[0, 1.0, 2], [True, False, True]]
    ]
    tensor = Tensor(sequence)
    assert tensor.value == [[[0, 1.0, 2], [True, False, True]]]
    assert tensor.shape.get_shapes() == [1, 2, 3]
    assert tensor.type is float


def test_range_input_to_tensor_model():
    sequence: list[list[list[int | float | bool] | Sequence[int]]] = [
        [[0, 1.0], [2, 3]],
        [range(4, 6), [True, False]],
    ]
    model = Model()
    model += (tt := ToTensor())(input=sequence)
    assert tt.input == [[[0, 1.0], [2, 3]], [range(4, 6), [True, False]]]
    assert (
        tt.input.metadata.value_type
        == list[list[list[int | float] | list[int]] | list[range | list[bool]]]
    )
    assert tt.output == [[[0, 1.0], [2, 3]], [[4, 5], [True, False]]]
    assert tt.output.metadata.value_type is float
