import torch
import torch.nn  as nn
import mithril as ml
from examples.clip.model import multi_head_attention
import numpy as np
import platform
from collections.abc import Mapping

import numpy as np
import pytest
import torch
def load_weights(
    param_shapes: Mapping, torch_model: torch.nn.Module, backend: ml.Backend
):
    ml_params = {}
    torch_state_dict = torch_model.state_dict()

    for torch_key in torch_state_dict:
        ml_key = torch_key.replace(".", "_").lower()
        if ml_key not in param_shapes:
            continue

        param_shape = param_shapes[ml_key]

        if torch_state_dict[torch_key].shape != param_shape:
            parameter = torch_state_dict[torch_key].numpy().reshape(param_shape)
        else:
            parameter = torch_state_dict[torch_key].numpy().reshape(param_shape)
        ml_params[ml_key] = backend.array(parameter)

    return ml_params

class TestLayers:
    def test_resnet_block(self, backend_type: type[ml.Backend]):
        m_model = multi_head_attention(768,12,True)
        o_model = nn.MultiHeadAttention(768,12)

        backend = ml.TorchBackend(device="cuda")

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"queries": (1,77,768),"mask":(77,77)},
            data_keys={"queries","mask"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_query = torch.randn(1,77,768)
        torch_mask = torch.randn(77,77)

        expected_result = o_model(torch_query,torch_query,torch_query,need_weights=False,mask = torch_mask)[0]

        qr = backend.array(torch_query.numpy())
        msk = backend.array(torch_mask.numpy())

        res = pm(params, {"queries": qr, "mask": msk})["output"]
        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore