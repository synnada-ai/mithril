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

from mithril.framework.common import squash_tensor_types, Tensor


def test_1():
    res = squash_tensor_types(list[Tensor[int] | Tensor[float]])
    assert res == list[Tensor[int | float]]


def test_2():
    res = squash_tensor_types(list[list[Tensor[int] | Tensor[int | bool]] | Tensor[float]])
    assert res == list[list[Tensor[int | bool]] | Tensor[float]]


def test_3():
    res = squash_tensor_types(int | list[int])
    assert res == int | list[int]

