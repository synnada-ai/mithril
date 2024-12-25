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

from mithril.models import Add, Linear, Model, MyTensor, ScalarItem


def test_freeze_set_values_primitive():
    model = Add()
    assert model.is_frozen is True

    model._freeze()
    assert model.is_frozen is True

    with pytest.raises(ValueError) as error_info:
        model.set_values({"left": 1.0})
    assert str(error_info.value) == "Model is frozen, can not set the key: left!"


def test_freeze_set_values_extend_defined_logical():
    model = Linear()
    assert model.is_frozen is True

    model._freeze()
    assert model.is_frozen is True

    with pytest.raises(ValueError) as error_info:
        model.set_values({"input": 1.0})
    assert str(error_info.value) == "Model is frozen, can not set the key: input!"

    with pytest.raises(AttributeError) as attr_error_info:
        model += Add()
    assert str(attr_error_info.value) == "Model is frozen and can not be extended!"


def test_freeze_set_values_extend_logical():
    model = Model()
    model += Add()(left="left", right="right")
    assert model.is_frozen is False

    model.set_values({"left": MyTensor(1.0)})
    model._freeze()
    assert model.is_frozen is True

    with pytest.raises(ValueError) as error_info:
        model.set_values({"right": 1.0})
    assert str(error_info.value) == "Model is frozen, can not set the key: right!"

    with pytest.raises(AttributeError) as attr_error_info:
        model += Add()
    assert str(attr_error_info.value) == "Model is frozen and can not be extended!"


def test_freeze_set_values_scalar():
    model = Model()
    model += ScalarItem()(input="input")
    assert model.is_frozen is False

    model._freeze()
    model.set_values({"input": [1.0]})
    assert model.is_frozen is True

    assert model.input.metadata.value == [1.0]  # type: ignore
