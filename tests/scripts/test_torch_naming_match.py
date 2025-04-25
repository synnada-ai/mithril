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
from mithril.models import (
    Convolution1D,
    Convolution2D,
    GroupNorm,
    LayerNorm,
    Linear,
    Model,
    Relu,
)

# In this test we check that the naming of the parameters in the PyTorch model
# matches the naming of the parameters in the Mithril model.


def test_conv1d():
    mithril_model = Model()
    mithril_model += Convolution1D(3, 18, name="conv1d").connect(input="input")

    backend = ml.TorchBackend()
    pm_mithril = ml.compile(
        mithril_model,
        backend=backend,
        shapes={"input": [8, 3, 32]},
        use_short_namings=False,
    )

    class TorchModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.conv1d = torch.nn.Conv1d(3, 18, 3)

    torch_model = TorchModel()

    ml_params = pm_mithril.randomize_params()
    torch_params = torch_model.state_dict()

    for key, value in ml_params.items():
        torch_name = key.replace("_", ".")

        assert torch_name in torch_params
        assert torch_params[torch_name].shape == value.squeeze().shape


def test_conv2d():
    mithril_model = Model()
    mithril_model += Convolution2D(3, 18, name="conv2d").connect(input="input")

    backend = ml.TorchBackend()
    pm_mithril = ml.compile(
        mithril_model,
        backend=backend,
        shapes={"input": [8, 3, 32, 32]},
        use_short_namings=False,
    )

    class TorchModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.conv2d = torch.nn.Conv2d(3, 18, 3)

    torch_model = TorchModel()

    ml_params = pm_mithril.randomize_params()
    torch_params = torch_model.state_dict()

    for key, value in ml_params.items():
        torch_name = key.replace("_", ".")

        assert torch_name in torch_params
        assert torch_params[torch_name].shape == value.squeeze().shape


def test_linear():
    mithril_model = Model()
    mithril_model += Linear(18, name="linear").connect(input="input")

    backend = ml.TorchBackend()
    pm_mithril = ml.compile(
        mithril_model,
        backend=backend,
        shapes={"input": [8, 12, 3]},
        use_short_namings=False,
    )

    class TorchModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.linear = torch.nn.Linear(3, 18)

    torch_model = TorchModel()

    ml_params = pm_mithril.randomize_params()
    torch_params = torch_model.state_dict()

    for key, value in ml_params.items():
        torch_name = key.replace("_", ".")

        assert torch_name in torch_params
        assert torch_params[torch_name].shape == value.squeeze().shape


def test_layernorm():
    mithril_model = Model()
    mithril_model += LayerNorm(name="layernorm").connect(input="input")

    backend = ml.TorchBackend()
    pm_mithril = ml.compile(
        mithril_model,
        backend=backend,
        shapes={"input": [8, 12, 3]},
        use_short_namings=False,
    )

    class TorchModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.layernorm = torch.nn.LayerNorm(3)

    torch_model = TorchModel()

    ml_params = pm_mithril.randomize_params()
    torch_params = torch_model.state_dict()

    for key, value in ml_params.items():
        torch_name = key.replace("_", ".")

        assert torch_name in torch_params
        assert torch_params[torch_name].shape == value.squeeze().shape


def test_groupnorm():
    mithril_model = Model()
    mithril_model += GroupNorm(3, name="groupnorm").connect(input="input")

    backend = ml.TorchBackend()

    pm_mithril = ml.compile(
        mithril_model,
        backend=backend,
        shapes={"input": [8, 3, 12, 8]},
        use_short_namings=False,
    )

    class TorchModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.groupnorm = torch.nn.GroupNorm(3, 3)

    torch_model = TorchModel()

    ml_params = pm_mithril.randomize_params()
    torch_params = torch_model.state_dict()

    for key, value in ml_params.items():
        torch_name = key.replace("_", ".")

        assert torch_name in torch_params
        assert torch_params[torch_name].shape == value.squeeze().shape


def test_multi_layer():
    mithril_model = Model()
    mithril_model += Linear(18, name="linear1").connect(input="input")
    mithril_model += Relu(name="relu").connect()
    mithril_model += Linear(18, name="linear2").connect()

    backend = ml.TorchBackend()
    pm_mithril = ml.compile(
        mithril_model,
        backend=backend,
        shapes={"input": [8, 12, 3]},
        use_short_namings=False,
    )

    class TorchModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.linear1 = torch.nn.Linear(3, 18)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(18, 18)

    torch_model = TorchModel()

    ml_params = pm_mithril.randomize_params()
    torch_params = torch_model.state_dict()

    for key, value in ml_params.items():
        torch_name = key.replace("_", ".")

        assert torch_name in torch_params
        assert torch_params[torch_name].shape == value.squeeze().shape


def test_multi_layer_cascaded():
    submodel = Model(name="submodel")
    submodel += Linear(18, name="linear1").connect(input="input")
    submodel += Relu(name="relu").connect()
    submodel += Linear(18, name="linear2").connect()

    mithril_model = Model()
    mithril_model += submodel.connect(input="input")
    mithril_model += Linear(36, name="linear")
    mithril_model += Linear(36, name="linear2")

    backend = ml.TorchBackend()
    pm_mithril = ml.compile(
        mithril_model,
        backend=backend,
        shapes={"input": [8, 12, 3]},
        use_short_namings=False,
    )

    class SubModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.linear1 = torch.nn.Linear(3, 18)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(18, 18)

    class TorchModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.submodel = SubModel()
            self.linear = torch.nn.Linear(18, 36)
            self.linear2 = torch.nn.Linear(36, 36)

        def forward(self, x):
            x = self.submodel(x)
            x = self.linear(x)
            x = self.linear2(x)

            return x

    torch_model = TorchModel()

    ml_params = pm_mithril.randomize_params()
    torch_params = torch_model.state_dict()

    for key, value in ml_params.items():
        torch_name = key.replace("_", ".")

        assert torch_name in torch_params
        assert torch_params[torch_name].shape == value.squeeze().shape
