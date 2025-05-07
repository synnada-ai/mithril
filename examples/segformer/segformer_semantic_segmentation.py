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

from mithril.framework import IOKey, Tensor
from mithril.models import Model

from .segformer_decode_head import segformer_decode_head
from .segformer_encoder import segformer_encoder


def segformer(config, *, name: str | None = None):
    pixel_values = IOKey("input", type=Tensor[float])
    last_hidden, encoder_hidden_states = segformer_encoder(config, name="encoder")(
        input=pixel_values
    )
    return Model.create(
        name=name, last_hidden_state=last_hidden, hidden_states=encoder_hidden_states
    )


def segformer_semantic_segmentation(config):
    pixel_values = IOKey("input", type=Tensor[float])
    last_hidden_state, hidden_states = segformer(config, name="segformer")(
        input=pixel_values
    )
    logits = segformer_decode_head(config, name="decode_head")(input=hidden_states)
    return Model.create(name="segformer_semantic_segmentation", logits=logits)
