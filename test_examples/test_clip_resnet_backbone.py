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

import torch.nn.functional as F
import mithril as ml
from examples.clip.model import (bottleneck,attention_pool2d,multi_head_attention_forward,modified_resnet
    
)
from .resnet_torch import Bottleneck, ModifiedResNet , AttentionPool2d



installed_backends = [ml.TorchBackend, ml.JaxBackend]
sys.setrecursionlimit(3500)

if platform.system() == "Darwin":
    installed_backends.append(ml.MlxBackend)
    

def load_weights(
    param_shapes: Mapping, torch_model: torch.nn.Module, backend: ml.Backend
):
    ml_params = {}
    torch_state_dict = torch_model.state_dict()
    for i in torch_state_dict:
        print(i)
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
    def test_mha(self, backend_type):
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
    def test_mod_res(self, backend_type):
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
        
    
        