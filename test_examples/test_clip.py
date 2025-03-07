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
    bottleneck,
    attention_pool2d,
    multi_head_attention_forward,
    modified_resnet
)
# noqa: N801
import torch.nn.functional as F
from .clip_torch import CLIP, ResidualAttentionBlock, Transformer, VisionTransformer
from .resnet_torch import Bottleneck, ModifiedResNet , AttentionPool2d, CLIP_RN
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

    def test_clip_vitl14(self, backend_type):
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
        
    def test_bottleneck(self,backend_type):
        m_model = bottleneck(64,64)
        o_model = Bottleneck(64,64)
        backend = backend_type()
        input_shape = (1,64,56,56)
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": input_shape}, # check shape
            data_keys={"input"},
            use_short_namings=False,
        )
        
        params = load_weights(pm.shapes, o_model, backend)
        torch_input = torch.randn(input_shape)
        expected_result = o_model(torch_input)
        input_backend = backend.array(torch_input.numpy())
        outs = pm(params, {"input": input_backend})
      
        res = outs["output"]
        np.testing.assert_allclose(
            np.array(res),
            expected_result.cpu().detach().numpy(),
            atol=1e-4,
            rtol=1e-4
        )
    
    def test_attentionpool2d(self,backend_type):
        m_model = attention_pool2d(7,2048,32,512)
        o_model = AttentionPool2d(7, 2048, 32, 512)
        
        backend = backend_type()
        input_shape = (1, 2048, 7,7)


        shapes = {
            "input": input_shape
        }

        data_keys = {
            "input"
        }
        torch_input = torch.randn(input_shape)
        
        
        backend = backend_type()
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes=shapes,
            data_keys=data_keys,
            use_short_namings=False,inference=True
        )
        
       
        input_backend = backend.array(torch_input.numpy())
        params = load_weights(pm.shapes, o_model, backend)
        expected_result = o_model(torch_input)
        print(expected_result)
        
        outs = pm.evaluate(params, {"input": input_backend})
        

        res = outs["output"]
        np.testing.assert_allclose(
            np.array(res),
            expected_result.cpu().detach().numpy(),
            atol=1e-4,
            rtol=1e-4
        )
    
    def test_multi_head_forward(self, backend_type):
        m_model = multi_head_attention_forward(2048, 32, 0.0)
        
        backend = backend_type()
        shapes = {
            "query": (1, 1, 2048),
            "key": (50, 1, 2048),
            "value": (50, 1, 2048),
            "q_proj_weight": (2048, 2048),
            "k_proj_weight": (2048, 2048),
            "v_proj_weight": (2048, 2048),
            "in_proj_bias": (6144,),
            "out_proj_weight": (512, 2048),
            "out_proj_bias": (512,),
        }
        data_keys = {
            "query",
            "key",
            "value",
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
            "in_proj_bias",
            "out_proj_weight",
            "out_proj_bias",
        }

        torch_inputs = {
            key: torch.randn(*shape)
            for key, shape in shapes.items()
        }
        backend_inputs = {
            key: backend.array(tensor.numpy())
            for key, tensor in torch_inputs.items()
        }
        
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes=shapes,
            data_keys=data_keys,
            use_short_namings=False,
            inference=True
        )
        params = pm.randomize_params()
        outs = pm.evaluate(params, backend_inputs)

        
        expected_result, attn_weights = F.multi_head_attention_forward(
            query=backend_inputs["query"],           
            key=backend_inputs["key"],               
            value=backend_inputs["value"],           
            embed_dim_to_check=2048,                 
            num_heads=32,                            
            in_proj_weight=None,                     
            in_proj_bias=backend_inputs["in_proj_bias"],  
            bias_k=None,                             
            bias_v=None,                             
            add_zero_attn=False,                     
            dropout_p=0.0,                           
            out_proj_weight=backend_inputs["out_proj_weight"],  
            out_proj_bias=backend_inputs["out_proj_bias"],      
            training=True,                           
            key_padding_mask=None,                   
            need_weights=True,                       
            attn_mask=None,                          
            use_separate_proj_weight=True,           
            q_proj_weight=backend_inputs["q_proj_weight"],  
            k_proj_weight=backend_inputs["k_proj_weight"],  
            v_proj_weight=backend_inputs["v_proj_weight"],  
            static_k=None,                           
            static_v=None,                           
            average_attn_weights=True,               
            is_causal=False                          
        )

        
        np.testing.assert_allclose(
            np.array(outs["output"]),
            expected_result.cpu().detach().numpy(),
            atol=1e-3,
            rtol=1e-3
        )
        
    def test_modified_resnet(self, backend_type):
        m_model = modified_resnet((3, 4, 6, 3),512,32,224,64)
        o_model = ModifiedResNet((3, 4, 6, 3),512,32,224,64)
        
        backend = backend_type()
        input_shape = (1, 3, 224, 224)


        shapes = {
            "input": input_shape
        }

        data_keys = {
            "input"
        }

        torch_input = torch.randn(input_shape)
        
        
        backend = backend_type()
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes=shapes,
            data_keys=data_keys,
            use_short_namings=False,inference=True
        )
        
        params = load_weights(pm.shapes, o_model, backend)
        input_backend = backend.array(torch_input.numpy())
        inputs = {"input": input_backend}
        outs = pm.evaluate(params, inputs)
        expected_result = o_model(torch_input)
        
        
        
        np.testing.assert_allclose(
            np.array(outs["output"]),
            expected_result.cpu().detach().numpy(),
            atol=1e-5,
            rtol=1e-5
        )
        
    def test_clip_resnet50(self, backend_type):
        torch_image = torch.randn(2, 3, 224, 224)
        torch_text = torch.randint(0, 49408, size=(2, 77))
        m_model = clip(
            embed_dim=512,                # Embedding dimension
            image_resolution=224,         # Input image resolution
            vision_layers=(3, 4, 6, 3),   # ResNet-50 layers configuration
            vision_width=64,              # Width of the vision model
            vision_patch_size=0,          # Patch size (not used for ResNet)
            context_length=77,            # Maximum length of text input
            vocab_size=49408,             # Size of the text tokenizer's vocabulary
            transformer_width=512,        # Width of the text transformer
            transformer_heads=8,          # Number of attention heads in the transformer
            transformer_layers=12,        # Number of transformer layers
            name="RN50_CLIP"              # Name of the model instance
        )
        o_model = CLIP_RN(512, 224, (3, 4, 6, 3), 64, 0, 77, 49408, 512, 8, 12)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"image": (2, 3, 224, 224), "text": (2, 77)},
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
