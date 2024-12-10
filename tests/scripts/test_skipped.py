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

import mithril as ml
from mithril.framework.common import NestedListType, find_intersection_type
from mithril.models import Mean, Model


@pytest.mark.skip(
    reason="call operation currently add Connect "
    "object with expose=False flag. Open this test when it is fixed."
)
def test_auto_connect_in_call():
    mean = Mean(axis=ml.TBD)
    model = Model()
    model += mean(axis=ml.IOKey(name="my_axis", expose=True))
    model += Mean(axis=3)(axis=mean.axis)

    # This should be equivalent to the above
    # mean = Mean(axis=ml.TBD)
    # model = Model()
    # model += mean(axis = ml.IOKey(name="my_axis", expose=True))
    # model += Mean(axis=ml.TBD)(axis = ml.Connect(mean.axis, key=ml.IOKey(value = 3)))

    assert "my_axis" in model.conns.input_keys


@pytest.mark.skip(
    reason="bug in find_intersection_type function. Open this test when it is fixed."
)
def test_find_intersection_type_nested_list_type():
    type1 = int | float | list | tuple
    type2 = NestedListType(int | float)
    assert find_intersection_type(type1, type2) == NestedListType(int | float)
