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

import os
import platform
import sys
from collections.abc import Mapping

import clip as cliptorch
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import mithril as ml
from examples.clip.model import (
    clip,
    multi_head_attention,
    residual_attention_block,
    transformer,
    vision_transformer,
)

from .clip_torch import CLIP, ResidualAttentionBlock, Transformer, VisionTransformer

installed_backends = [ml.TorchBackend, ml.JaxBackend]
sys.setrecursionlimit(3500)

if platform.system() == "Darwin":
    installed_backends.append(ml.MlxBackend)


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


@pytest.mark.parametrize("backend_type", installed_backends)
class TestLayers:
    def test_multi_head(self, backend_type):
        m_model = multi_head_attention(768, 12, True)
        o_model = nn.MultiheadAttention(768, 12)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"queries": (77, 1, 768)},
            data_keys={"queries"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_query = torch.randn(77, 1, 768)

        expected_result = o_model(
            torch_query,
            torch_query,
            torch_query,
            need_weights=False,
            attn_mask=build_attention_mask(),
        )[0]

        qr = backend.array(torch_query.numpy())
        outs = pm(params, {"queries": qr})

        res = outs["output"]
        np.testing.assert_allclose(
            np.array(res), expected_result.cpu().detach().numpy(), 1e-5, 1e-5
        )  # type: ignore

    def test_res_block(self, backend_type):
        torch_query = torch.randn(77, 1, 768)
        m_model = residual_attention_block(768, 12, True)
        o_model = ResidualAttentionBlock(768, 12, attn_mask=build_attention_mask())

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": (77, 1, 768)},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        expected_result = o_model(torch_query)

        qr = backend.array(torch_query.numpy())
        outs = pm(params, {"input": qr})
        res = outs["output"]
        np.testing.assert_allclose(
            np.array(res), expected_result.cpu().detach().numpy(), 1e-5, 1e-5
        )  # type: ignore

    def test_transformer(self, backend_type):
        torch_query = torch.randn(77, 1, 768)
        m_model = transformer(768, 12, 12, True)
        o_model = Transformer(768, 12, 12, attn_mask=build_attention_mask())

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": (77, 1, 768)},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        expected_result = o_model(torch_query)

        qr = backend.array(torch_query.numpy())
        outs = pm(params, {"input": qr})
        res = outs["output"]
        np.testing.assert_allclose(
            np.array(res), expected_result.cpu().detach().numpy(), 1e-5, 1e-5
        )  # type: ignore

    def test_vision_transformer(self, backend_type):
        torch_input = torch.randn(1, 3, 224, 224)
        m_model = vision_transformer(224, 14, 1024, 24, 16, 768, True)
        o_model = VisionTransformer(224, 14, 1024, 24, 16, 768)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": (1, 3, 224, 224)},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        expected_result = o_model(torch_input)

        qr = backend.array(torch_input.numpy())
        outs = pm(params, {"input": qr})
        res = outs["output"]
        np.testing.assert_allclose(
            np.array(res), expected_result.cpu().detach().numpy(), 1e-5, 1e-5
        )  # type: ignore

    def test_clip(self, backend_type):
        torch_image = torch.randn(2, 3, 224, 224)
        torch_text = torch.randint(0, 49408, size=(6, 77))
        m_model = clip(
            embed_dim=768,  # Embedding dimension for ViT-L/14
            image_resolution=224,  # Input image resolution
            vision_layers=24,  # Number of transformer layers in the vision model
            vision_width=1024,  # Width of the vision model
            vision_patch_size=14,  # Patch size for the ViT-L/14 model
            context_length=77,  # Maximum length of text input
            vocab_size=49408,  # Size of the text tokenizer's vocabulary
            transformer_width=768,  # Width of the text transformer
            transformer_heads=12,  # Number of attention heads in the transformer
            transformer_layers=12,  # Number of transformer layers for the text model
        )
        o_model = CLIP(768, 224, 24, 1024, 14, 77, 49408, 768, 12, 12)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"image": (2, 3, 224, 224), "text": (6, 77)},
            data_keys={"image", "text"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        _, expected_result = o_model(torch_image, torch_text)

        image = backend.array(torch_image.numpy())
        text = backend.array(torch_text.numpy())
        outs = pm(params, {"image": image, "text": text})
        res = outs["logits_per_text"]
        np.testing.assert_allclose(
            np.array(res), expected_result.cpu().detach().numpy(), 1e-5, 1e-5
        )  # type: ignore


def build_attention_mask():
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(77, 77)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


class TestClipEndToEnd:
    def test_clip(self):
        device = "cpu"
        model, preprocess = cliptorch.load("ViT-L/14", device=device)
        img_path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "clip", "CLIP.png"
        )
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        text = cliptorch.tokenize(["a dog", "a diagram", "a cat"]).to(device)
        logits_per_image, logits_per_text = model(image, text)

        clip_mithril = clip(
            embed_dim=768,  # Embedding dimension for ViT-L/14
            image_resolution=224,  # Input image resolution
            vision_layers=24,  # Number of transformer layers in the vision model
            vision_width=1024,  # Width of the vision model
            vision_patch_size=14,  # Patch size for the ViT-L/14 model
            context_length=77,  # Maximum length of text input
            vocab_size=49408,  # Size of the text tokenizer's vocabulary
            transformer_width=768,  # Width of the text transformer
            transformer_heads=12,  # Number of attention heads in the transformer
            transformer_layers=12,  # Number of transformer layers for the text model
        )

        pm = ml.compile(
            clip_mithril,
            backend=ml.TorchBackend(),
            shapes={"image": (1, 3, 224, 224), "text": (3, 77)},
            data_keys={"image", "text"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, model, ml.TorchBackend())

        output = pm(params, {"image": image, "text": text})

        mithril_logits_per_image = output["logits_per_image"]
        mithril_logits_per_text = output["logits_per_text"]

        torch.testing.assert_close(
            mithril_logits_per_image, logits_per_image, atol=1e-3, rtol=1e-3
        )

        torch.testing.assert_close(
            mithril_logits_per_text, logits_per_text, atol=1e-3, rtol=1e-3
        )
