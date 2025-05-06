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

from mithril import IOKey
from mithril.models import Gelu, Linear, Model


def mlp_async(input_dim: int):
    block = Model(name="mlp_async")
    block += Linear(dimension=input_dim * 4, name="fc")(input=IOKey("input"))
    block += Gelu()
    block += Linear(dimension=input_dim, name="proj")(output=IOKey("output"))
    return block


def create_mlp_async(input_dim: int = 1024):
    model = Model(name="async_mlp")
    model += mlp_async(input_dim)(input="input")
    return model
