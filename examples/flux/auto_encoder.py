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
from mithril.models import Add, Convolution2D, GroupNorm, Model, SiLU


def resnet_block(in_channels: int, out_channels: int | None = None) -> Model:
    out_channels = in_channels if out_channels is None else out_channels

    block = Model()
    block += GroupNorm(num_groups=32, eps=1e-6)("input")
    block += SiLU()
    block += Convolution2D(3, out_channels, padding=1)

    block += GroupNorm(num_groups=32, eps=1e-6)
    block += SiLU()
    block += Convolution2D(3, out_channels, padding=1)(output="h")

    # TODO: We need to find better implementation than below. It is a conditional skip
    # connection.
    if in_channels != out_channels:
        block += Convolution2D(1, out_channels)(input="input")
        block += Add()(right="h", output=IOKey("output"))

    else:
        block += Add()("input", "h", output=IOKey("output"))

    return block
