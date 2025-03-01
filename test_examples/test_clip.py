import torch
import torch.nn  as nn
import mithril as ml
from examples.clip.model import multi_head_attention,transformer, residual_attention_block, vision_transformer, clip
import numpy as np
import platform
from collections.abc import Mapping
from .clip_torch import Transformer, ResidualAttentionBlock, VisionTransformer, CLIP
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
    def test_multi_head(self):
        m_model = multi_head_attention(768,12,True)
        o_model = nn.MultiheadAttention(768,12)

        backend = ml.TorchBackend(device="cuda")

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"queries": (77,1,768),"mask":(77,77)},
            data_keys={"queries","mask"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_query = torch.randn(77,1,768)
        torch_mask = torch.randn(77,77)

        expected_result = o_model(torch_query,torch_query,torch_query,need_weights=False,attn_mask = torch_mask)[0]

        qr = backend.array(torch_query.numpy())
        msk = backend.array(torch_mask.numpy())
        outs = pm(params, {"queries": qr,"mask":msk})
       
        res = outs["output"]
        np.testing.assert_allclose(res.cpu().detach().numpy(), expected_result.cpu().detach().numpy(), 1e-5, 1e-5)  # type: ignore

    def test_res_block(self):
        torch_query = torch.randn(77,1,768)
        torch_mask = torch.randn(77,77)
        m_model = residual_attention_block(768,12,True)
        o_model = ResidualAttentionBlock(768,12,attn_mask=torch_mask)

        backend = ml.TorchBackend(device="cuda")

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": (77,1,768),"mask":(77,77)},
            data_keys={"input","mask"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        

        expected_result = o_model(torch_query)

        qr = backend.array(torch_query.numpy())
        msk = backend.array(torch_mask.numpy())
        outs = pm(params, {"input": qr,"mask":msk})
        res = outs["output"]
        np.testing.assert_allclose(res.cpu().detach().numpy(), expected_result.cpu().detach().numpy(), 1e-5, 1e-5)  # type: ignore
    def test_transformer(self):
        torch_query = torch.randn(77,1,768)
        torch_mask = torch.randn(77,77)
        m_model = transformer(768,12,12,True)
        o_model = Transformer(768,12,12,attn_mask=torch_mask)

        backend = ml.TorchBackend(device="cuda")

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": (77,1,768),"mask":(77,77)},
            data_keys={"input","mask"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        

        expected_result = o_model(torch_query)

        qr = backend.array(torch_query.numpy())
        msk = backend.array(torch_mask.numpy())
        outs = pm(params, {"input": qr,"mask":msk})
        res = outs["output"]
        np.testing.assert_allclose(res.cpu().detach().numpy(), expected_result.cpu().detach().numpy(), 1e-5, 1e-5)  # type: ignore
   
    def test_vison_transformer(self):
        torch_input = torch.randn(1,3,224,224)
        m_model = vision_transformer(224,14,1024,24,16,768,True)
        o_model = VisionTransformer(224,14,1024,24,16,768)

        backend = ml.TorchBackend(device="cuda")

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": (1,3,224,224)},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        

        expected_result = o_model(torch_input)

        qr = backend.array(torch_input.numpy())
        outs = pm(params, {"input": qr})
        res = outs["output"]
        np.testing.assert_allclose(res.cpu().detach().numpy(), expected_result.cpu().detach().numpy(), 1e-5, 1e-5)  # type: ignore
    
    def test_clip(self):
        torch_image = torch.randn(1,3,224,224)
        torch_text = torch.randint(0, 49408, size=(1, 77))
        torch_mask = build_attention_mask()
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
                    transformer_layers=12  # Number of transformer layers for the text model
                )
        o_model = CLIP(768,224,24,1024,14,77,49408,768,12,12)

        backend = ml.TorchBackend(device="cuda")

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"image": (1,3,224,224),"text":(1,77),"mask":(77,77)},
            data_keys={"image","text","mask"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        _,expected_result = o_model(torch_image,torch_text)

        image = backend.array(torch_image.numpy())
        text = backend.array(torch_text.numpy())
        mask = backend.array(torch_mask.numpy())
        outs = pm(params, {"image": image,"text": text, "mask":mask})
        res = outs["logits_per_text"]
        np.testing.assert_allclose(res.cpu().detach().numpy(), expected_result.cpu().detach().numpy(), 1e-5, 1e-5)  # type: ignore



def build_attention_mask():
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(77, 77)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask