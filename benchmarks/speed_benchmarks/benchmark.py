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

from benchmarks.speed_benchmarks.jax_fns import mlp_v_jax
from benchmarks.speed_benchmarks.speed_helper import colorize_str
from benchmarks.speed_benchmarks.torch_fns import conv_v_torch, mlp_v_torch
from mithril.framework.common import Table
from mithril.models import Relu, Sigmoid, Tanh

# MLX is not included due to Ubuntu OS in Github
backends = ["Torch", "Jax"]
precisions = [64, 32, 16]

iterations = 100
table = Table()
table.add_header(
    [
        "Model",
        "Backend",
        "Precision",
        "# of parameters",
        "Time Backend (s)",
        "Time Composite ML (s)",
        "Ratio",
    ]
)


# Set up parameters for large MLP model
activations = [
    Sigmoid,
    Relu,
    Sigmoid,
    Relu,
    Tanh,
    Sigmoid,
    Sigmoid,
    Sigmoid,
    Relu,
    Tanh,
]
dimensions = [2048, 1024, 512, 2048, 1024, 1024, 512, 2048, 1024, 128]
input_shape = (128, 256)

for backend in backends:
    fn = mlp_v_jax if backend == "Jax" else mlp_v_torch
    for precision in precisions:
        if not (precision == 16 and backend == "Torch"):
            num_params, time_backend, time_mithril = fn(
                activations=activations,
                dimensions=dimensions,
                input_shape=input_shape,
                iterations=iterations,
                precision=precision,
            )
            table.add_row(
                [
                    "MLP Large",
                    backend,
                    str(precision),
                    str(num_params),
                    f"{time_backend:.4f}",
                    f"{time_mithril:.4f}",
                    colorize_str(time_mithril / time_backend),
                ]
            )


activations = [Sigmoid, Relu, Tanh]
dimensions = [256, 256, 1]

for backend in backends:
    fn = mlp_v_jax if backend == "Jax" else mlp_v_torch
    for precision in precisions:
        if not (precision == 16 and backend == "Torch"):
            num_params, time_backend, time_mithril = fn(
                activations=activations,
                dimensions=dimensions,
                input_shape=(128, 128),
                iterations=iterations,
                precision=precision,
            )
            table.add_row(
                [
                    "MLP Small",
                    backend,
                    str(precision),
                    str(num_params),
                    f"{time_backend:.4f}",
                    f"{time_mithril:.4f}",
                    colorize_str(time_mithril / time_backend),
                ]
            )

activations = [Sigmoid, Relu, Tanh]
dimensions = [12, 16, 32]
stride = (2, 2)
padding = 1
for precision in [32, 64]:
    num_params, time_backend, time_mithril = conv_v_torch(
        activations=activations,
        dimensions=dimensions,
        input_shape=(4, 4, 128, 128),
        iterations=iterations,
        precision=precision,
        stride=stride,
        padding=padding,
    )
    table.add_row(
        [
            "Conv Small",
            "Torch",
            str(precision),
            str(num_params),
            f"{time_backend:.4f}",
            f"{time_mithril:.4f}",
            colorize_str(time_mithril / time_backend),
        ]
    )


activations = [Sigmoid, Relu, Tanh, Sigmoid]
dimensions = [1024, 1024, 1024, 256]
stride = (2, 2)
padding = 2
for precision in [32, 64]:
    num_params, time_backend, time_mithril = conv_v_torch(
        activations=activations,
        dimensions=dimensions,
        input_shape=(2, 1, 128, 128),
        iterations=iterations,
        precision=precision,
        stride=stride,
        padding=padding,
    )
    table.add_row(
        [
            "Conv Large",
            "Torch",
            str(precision),
            str(num_params),
            f"{time_backend:.4f}",
            f"{time_mithril:.4f}",
            colorize_str(time_mithril / time_backend),
        ]
    )

table.compile()
table.display()
