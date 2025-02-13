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

from mithril.framework.common import Tensor, squash_tensor_types


def test_1():
    res = squash_tensor_types(list[Tensor[int] | Tensor[float]])
    assert res == list[Tensor[int | float]]


def test_2():
    res = squash_tensor_types(
        list[list[Tensor[int] | Tensor[int | bool]] | Tensor[float]]
    )
    assert res == list[list[Tensor[int | bool]] | Tensor[float]]


def test_3():
    res = squash_tensor_types(int | list[int])
    assert res == int | list[int]


def test_4():
    res = squash_tensor_types(int | tuple[int, ...])
    assert res == int | tuple[int, ...]


def test_5():
    res = squash_tensor_types(tuple[int, ...])
    assert res == tuple[int, ...]


def test_6():
    res = squash_tensor_types(tuple[int, int])
    assert res == tuple[int, int]


def test_7():
    res = squash_tensor_types(tuple[tuple[int, ...], ...])
    assert res == tuple[tuple[int, ...], ...]


def test_8():
    res = squash_tensor_types(tuple[Tensor, ...])
    assert res == tuple[Tensor[int | float | bool], ...]


def test_9():
    res = squash_tensor_types(list[Tensor | int | float])
    assert res == list[Tensor[int | float | bool] | int | float]


def test_10():
    res = squash_tensor_types(Tensor[int] | Tensor[float])
    assert res == Tensor[int | float]


def test_11():
    res = squash_tensor_types(Tensor[int] | int | Tensor[float])
    assert res == Tensor[int | float] | int

def test_12():
    res = squash_tensor_types(Tensor[float])
    assert res == Tensor[float]
