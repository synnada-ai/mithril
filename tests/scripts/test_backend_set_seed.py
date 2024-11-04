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

import platform
import random

from mithril import Backend, JaxBackend, MlxBackend, NumpyBackend, TorchBackend


def test_backend_randomizations():
    backends: list[Backend] = [JaxBackend(), TorchBackend(), NumpyBackend()]
    if platform.system() == "Darwin":
        backends.append(MlxBackend())
    # For all possible backends set a seed, change it and compare randomization results.
    # Then, again set the seed as the initial seed value and check randomized tensors
    # are equal as the first case. Test for "normal" and "uniform" distributions.
    seed_list = [num for num in range(1000)]  # Pick any shape for testing.
    shape = (3, 4, 5, 6)
    for backend in backends:
        for seed1 in random.sample(seed_list, 50):
            # Ensure that seeds are not same.
            while (seed2 := random.sample(seed_list, 1)[0]) == seed1:
                ...
            for fn in (backend.randn, backend.rand):
                backend.set_seed(seed1)
                random_1 = fn(*shape)
                backend.set_seed(seed2)
                random_2 = fn(*shape)
                assert backend.any(random_1 != random_2)
                backend.set_seed(seed1)
                random_3 = fn(*shape)
                assert backend.all(random_1 == random_3)
