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

import numpy as np

import mithril as ml
from mithril.framework.physical import PhysicalModel
from mithril.models import Model, Randn


def testrandom_keys_not_provided():
    example_model = Model()
    example_model |= Randn()(shape=(3, 4, 5, 6), output=ml.IOKey("out1"))
    example_model |= Randn()(shape=(3, 4, 5, 1), output=ml.IOKey("out2"))

    backend = ml.JaxBackend()
    pm = ml.compile(example_model, backend, use_short_namings=False, inference=True)

    assert pm._random_seeds == {"randn_0_key": 0, "randn_1_key": 0}
    res1 = pm.evaluate()["out1"]
    assert pm._random_seeds != {"randn_0_key": 0, "randn_1_key": 0}
    res2 = pm.evaluate()["out1"]

    assert isinstance(res1, backend.DataType)
    assert isinstance(res2, backend.DataType)

    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, res1, res2)


def testrandom_keys_some_of_provided():
    example_model = Model()
    # Static inference will infer this function
    example_model |= Randn(key=42)(shape=(3, 4, 5, 6), output=ml.IOKey("out1"))
    example_model |= Randn()(shape=(3, 4, 5, 1), output=ml.IOKey("out2"))

    backend = ml.JaxBackend()
    pm = ml.compile(example_model, backend, use_short_namings=False, inference=True)

    assert pm._random_seeds == {"randn_1_key": 0}
    res1 = pm.evaluate()["out1"]
    assert pm._random_seeds != {"randn_1_key": 0}
    res2 = pm.evaluate()["out1"]

    assert isinstance(res1, backend.DataType)
    assert isinstance(res2, backend.DataType)

    np.testing.assert_array_equal(res1, res2)


def test_setrandom_keys():
    example_model = Model()
    # Static inference will infer this function
    example_model |= Randn(key=42)(shape=(3, 4, 5, 6), output=ml.IOKey("out1"))
    example_model |= Randn()(shape=(3, 4, 5, 1), output=ml.IOKey("out2"))

    backend = ml.JaxBackend()
    pm = PhysicalModel(
        example_model,
        backend,
        use_short_namings=False,
        discard_keys=set(),
        data_keys=set(),
        constant_keys={},
        trainable_keys=set(),
        inference=True,
        safe_shapes=False,
        safe_names=False,
        shapes={},
    )
    assert pm._random_seeds == {"randn_1_key": 0}
    pm.set_random_seed_values(randn_1_key=42)
    assert pm._random_seeds == {"randn_1_key": 42}
