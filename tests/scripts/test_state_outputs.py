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

import torch

import mithril as ml
from mithril.models import Add, BatchNorm2D, Buffer, Model


def test_running_mean():
    # Manually expose state inputs & outputs and manually
    # update running_output in the evaluation loop.
    model = Model()
    model |= Add()("input", "running_input", output="add_output")
    model |= Buffer()("add_output", output="output")

    main_model = Model()
    main_model |= model(
        input="input", running_input="running_input", output="running_output"
    )
    main_model |= Buffer()("running_output", output="output")
    main_model.expose_keys("running_output", "output")
    # TODO: use_short_namings -> short_names
    backend = ml.TorchBackend()
    pm = ml.compile(
        main_model, backend, use_short_namings=False, inference=True, jit=False
    )
    manual_state = backend.ones((1, 1))
    for _ in range(10):
        data = {"input": backend.ones((1, 1)), "running_input": manual_state}
        manual_outputs = pm.evaluate(data=data)
        assert isinstance(manual_outputs["running_output"], torch.Tensor)
        manual_state = manual_outputs["running_output"]

    # Automatically bind state inputs & outputs.
    model = Model()
    model |= Add()("local_input", "running_input", output="add_output")
    model |= Buffer()("add_output", output="local_output")
    model.bind_state_input("running_input", "add_output", 1)
    model.expose_keys("add_output")

    main_model = Model()
    main_model |= model(local_input="input", local_output="output")
    main_model.expose_keys("output")

    pm = ml.compile(
        main_model,
        backend,
        use_short_namings=False,
        inference=True,
        safe_names=False,
        jit=False,
    )
    state = pm.initial_state_dict
    for _ in range(10):
        data = {"input": backend.ones((1, 1))}
        outputs, state = pm.evaluate(data=data, state=state)

    assert isinstance(outputs["output"], torch.Tensor)
    assert isinstance(manual_outputs["output"], torch.Tensor)
    assert torch.allclose(outputs["output"], manual_outputs["output"])


def generate_data(batch_size, channels, height, width):
    return torch.randn(batch_size, channels, height, width)


# Loop-based comparison test
def test_batchnorm_vs_pytorch_training():
    # Hyperparameters
    momentum = 0.1
    eps = 1e-5

    num_features = 3
    batch_size = 32
    height = 8
    width = 8

    # Initialize BatchNorm layers
    batchnorm_torch = torch.nn.BatchNorm2d(num_features, momentum=momentum, eps=eps)
    batchnorm_torch.train()
    # Mithril BatchNorm2d
    batch_norm_mithril = BatchNorm2D(
        num_features=3, eps=1e-5, use_bias=False, use_scale=False, inference=False
    )
    pm = ml.compile(batch_norm_mithril, ml.TorchBackend(), inference=True, jit=False)
    state = pm.initial_state_dict

    # Training loop
    for _ in range(1000):
        x_torch = generate_data(batch_size, num_features, height, width)

        # Mithril BatchNorm
        output_mithril, state = pm.evaluate(data={"input": x_torch}, state=state)

        # PyTorch BatchNorm
        output_torch = batchnorm_torch(x_torch)
        assert isinstance(output_mithril["output"], torch.Tensor)
        assert isinstance(state["running_mean"], torch.Tensor)
        assert isinstance(state["running_var"], torch.Tensor)
        assert isinstance(batchnorm_torch.running_mean, torch.Tensor)
        assert isinstance(batchnorm_torch.running_var, torch.Tensor)
        # Compare outputs
        assert torch.allclose(
            output_mithril["output"], output_torch, atol=1e-5
        ), "Output mismatch"

        # Compare running statistics
        assert torch.allclose(
            state["running_mean"], batchnorm_torch.running_mean, atol=1e-5
        ), "Running mean mismatch"
        assert torch.allclose(
            state["running_var"], batchnorm_torch.running_var, atol=1e-5
        ), "Running var mismatch"


# Loop-based comparison test
def test_batchnorm_vs_pytorch_inference():
    # Hyperparameters
    momentum = 0.1
    eps = 1e-5

    num_features = 3
    batch_size = 32
    height = 8
    width = 8

    # Initialize BatchNorm layers
    batchnorm_torch = torch.nn.BatchNorm2d(num_features, momentum=momentum, eps=eps)
    batchnorm_torch.eval()
    # Mithril BatchNorm2d
    batch_norm_mithril = BatchNorm2D(
        num_features=3, eps=1e-5, use_bias=False, use_scale=False, inference=True
    )
    backend = ml.TorchBackend()
    pm = ml.compile(batch_norm_mithril, backend, inference=True, jit=False)

    # Training loop
    mean = backend.zeros(num_features)
    var = backend.ones(num_features)
    for _ in range(1000):
        x_torch = generate_data(batch_size, num_features, height, width)

        data = {"input": x_torch, "running_mean": mean, "running_var": var}
        # Mithril BatchNorm
        output_mithril = pm.evaluate(data=data)

        # PyTorch BatchNorm
        output_torch = batchnorm_torch(x_torch)
        assert isinstance(output_mithril["output"], torch.Tensor)
        # Compare outputs
        assert torch.allclose(
            output_mithril["output"], output_torch, atol=1e-5
        ), "Output mismatch"
