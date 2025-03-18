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
import transformers
from PIL import Image
from transformers import AutoTokenizer

import mithril as ml
from examples.clip.model_mlx import clip_model, clip_text_model, clip_vision_model


def get_config():
    config = {
        "text_config": {
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "max_position_embeddings": 77,
            "vocab_size": 49408,
            "layer_norm_eps": 1e-5,
        },
        "vision_config": {
            "num_hidden_layers": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_channels": 3,
            "image_size": 224,
            "patch_size": 14,
            "layer_norm_eps": 1e-5,
        },
        "projection_dim": 768,
    }
    return config


HF_PATH = "openai/clip-vit-large-patch14"
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
    def test_clip_vision_model(self, backend_type):
        backend = backend_type()
        config = get_config()

        m_model = clip_vision_model(config["vision_config"], "vit-l/14")

        _, _, o_model = load_hf_models(HF_PATH)
        o_model = o_model.vision_model

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": (2, 3, 224, 224)},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)
        torch_input_ids = torch.rand((2, 3, 224, 224))
        input = backend.array(torch_input_ids.clone().numpy())
        expected_result = o_model(torch_input_ids).pooler_output

        outs = pm.evaluate(params, {"input": input})
        res = outs["output"]

        np.testing.assert_allclose(
            np.array(res), expected_result.cpu().detach().numpy(), 1e-5, 1e-5
        )  # type: ignore

    def test_clip_text_model(self, backend_type):
        backend = backend_type()
        config = get_config()

        m_model = clip_text_model(config["text_config"], "vit-l/14")

        _, _, o_model = load_hf_models(HF_PATH)
        o_model = o_model.text_model

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": (2, 77)},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)
        text = ["a photo of a cat", "a photo of a dog"]
        tokens = cliptorch.tokenize(text)
        torch_input = torch.tensor(tokens)
        input = backend.array(torch_input.clone().numpy())
        expected_result = o_model(torch.tensor(input)).pooler_output

        outs = pm.evaluate(params, {"input": input})

        res = outs["output"]
        np.testing.assert_allclose(
            np.array(res), expected_result.cpu().detach().numpy(), 1e-5, 1e-5
        )  # type: ignore

    def test_clip_model(self, backend_type):
        backend = backend_type()
        config = get_config()

        m_model = clip_model(config, "vit-l/14")

        _, _, o_model = load_hf_models(HF_PATH)

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input_ids": (2, 77), "pixel_values": (1, 3, 224, 224)},
            data_keys={"input_ids", "pixel_values"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)
        text = ["a photo of a lion", "a photo of a dog"]
        tokens = cliptorch.tokenize(text)
        torch_input_ids = torch.tensor(tokens)
        torch_pixel_values = torch.rand(1, 3, 224, 224)
        input_ids = backend.array(torch_input_ids.clone().numpy())
        pixel_values = backend.array(torch_pixel_values.clone().numpy())

        expected_result = o_model(
            input_ids=torch_input_ids, pixel_values=torch_pixel_values
        )

        outs = pm.evaluate(
            params, {"input_ids": input_ids, "pixel_values": pixel_values}
        )

        np.testing.assert_allclose(
            np.array(outs["text_embeds"]),
            expected_result.text_embeds.cpu().detach().numpy(),
            1e-5,
            1e-5,
        )  # type: ignore
        np.testing.assert_allclose(
            np.array(outs["image_embeds"]),
            expected_result.image_embeds.cpu().detach().numpy(),
            1e-5,
            1e-5,
        )  # type: ignore


@pytest.mark.parametrize("backend_type", installed_backends)
class TestClipMLXEndToEnd:
    def test_clip_model(self, backend_type):
        backend = backend_type()
        config = get_config()
        img_path_cat = os.path.join(
            os.path.dirname(__file__), "..", "examples", "clip", "assets", "Cat.jpeg"
        )
        img_path_dog = os.path.join(
            os.path.dirname(__file__), "..", "examples", "clip", "assets", "Dog.jpeg"
        )

        m_model = clip_model(config, "vit-l/14")
        image_proc, tokenizer, o_model = load_hf_models(HF_PATH)
        text = ["a photo of a cat", "a photo of a dog"]

        image_input = image_proc(
            images=[Image.open(img_path_cat), Image.open(img_path_dog)],
            return_tensors="np",
        )["pixel_values"]
        tokens = cliptorch.tokenize(text)
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input_ids": (2, 77), "pixel_values": (2, 3, 224, 224)},
            data_keys={"input_ids", "pixel_values"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        input_ids = backend.array(torch.tensor(tokens).clone().numpy())
        pixel_values = backend.array(torch.tensor(image_input).clone().numpy())

        expected_result = o_model(
            input_ids=torch.tensor(tokens), pixel_values=torch.tensor(image_input)
        )
        outs = pm.evaluate(
            params, {"input_ids": input_ids, "pixel_values": pixel_values}
        )

        np.testing.assert_allclose(
            np.array(outs["text_embeds"]),
            expected_result.text_embeds.cpu().detach().numpy(),
            1e-5,
            1e-5,
        )  # type: ignore
        np.testing.assert_allclose(
            np.array(outs["image_embeds"]),
            expected_result.image_embeds.cpu().detach().numpy(),
            1e-5,
            1e-5,
        )  # type: ignore


def build_attention_mask():
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(77, 77)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


def load_hf_models(path):
    image_proc = transformers.CLIPImageProcessor.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    clip = transformers.CLIPModel.from_pretrained(path)
    return image_proc, tokenizer, clip
