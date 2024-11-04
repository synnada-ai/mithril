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

from typing import Any

import mithril

__all__ = ["Adam"]


class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2=0.999) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def init(
        self,
        backend: mithril.Backend,
        parameters: dict[str, mithril.DataType],
    ):
        state: dict[str, Any] = {}
        state["m"] = {k: backend.zeros_like(v) for k, v in parameters.items()}
        state["v"] = {k: backend.zeros_like(v) for k, v in parameters.items()}
        state["it"] = 1
        return state

    def update_params(
        self,
        parameters: dict[str, mithril.DataType],
        gradients: dict[str, mithril.DataType],
        state: dict,
    ):
        state["m"] = {
            k: self.beta1 * state["m"][k] + grad * (1 - self.beta1)
            for k, grad in gradients.items()
        }
        state["v"] = {
            k: self.beta2 * state["v"][k] + (grad**2) * (1 - self.beta2)
            for k, grad in gradients.items()
        }

        m_hat = {k: state["m"][k] / (1 - self.beta1 ** state["it"]) for k in state["m"]}
        v_hat = {k: state["v"][k] / (1 - self.beta2 ** state["it"]) for k in state["v"]}

        parameters |= {
            k: v - self.lr * m_hat[k] / ((v_hat[k]) ** 0.5 + 1e-16)
            for k, v in parameters.items()
        }
        state["it"] += 1
        return parameters, state
