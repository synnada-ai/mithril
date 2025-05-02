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

import platform
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pytest
import torch

import mithril as ml
from examples.flux.auto_encoder import (
    attn_block,
    decoder,
    downsample,
    encoder,
    resnet_block,
    upsample,
)
from examples.flux.layers import apply_rope as apply_rope_mithril
from examples.flux.layers import attention as attention_mithril
from examples.flux.layers import (
    double_stream_block,
    embed_nd,
    last_layer,
    modulation,
    qk_norm,
    rearrange,
    rms_norm,
    rope,
    single_stream_block,
)
from examples.flux.layers import mlp_embedder as mlp_embedder_mithril
from examples.flux.layers import timestep_embedding as timestep_embedding_mithril
from mithril.models import Concat, Gelu, IOKey, LayerNorm, Linear, Model

from .flux_torch import (
    AttnBlock,
    Decoder,
    DoubleStreamBlock,
    Downsample,
    EmbedND,
    Encoder,
    LastLayer,
    MLPEmbedder,
    Modulation,
    QKNorm,
    ResnetBlock,
    RMSNorm,
    SingleStreamBlock,
    Upsample,
    apply_rope,
    attention,
    timestep_embedding,
)
from .flux_torch import rope as torch_rope

installed_backends: list[type[ml.Backend]] = [
    ml.TorchBackend,
    ml.JaxBackend,
    # ml.NumpyBackend, #slow
]

if platform.system() == "Darwin":
    installed_backends.append(ml.MlxBackend)


def double_stream_block_functional(
    hidden_size: int,
    num_heads: int,
    mlp_ratio: float,
    qkv_bias: bool = False,
    *,
    name: str | None = None,
):
    # Define inputs
    img = IOKey("img", shape=[1, 4096, 3072])
    txt = IOKey("txt", shape=(1, 512, 3072))
    vec = IOKey("vec", shape=(1, 3072))
    pe = IOKey("pe", shape=(1, 1, 4608, 64, 2, 2))

    mlp_hidden_dim = int(hidden_size * mlp_ratio)

    # Image branch
    img_mod1, img_mod2 = modulation(hidden_size, double=True, name="img_mod")(input=vec)
    img_norm = LayerNorm(use_scale=False, use_bias=False, eps=1e-6, name="img_norm1")(
        input=img
    )
    img_modulated = (1 + img_mod1[1]) * img_norm + img_mod1[0]
    img_qkv = Linear(hidden_size * 3, use_bias=qkv_bias, name="img_attn_qkv")(
        input=img_modulated
    )
    img_rearrange_out = rearrange(num_heads=num_heads)(input=img_qkv)

    img_q_norm, img_k_norm = qk_norm(hidden_size // num_heads, name="img_attn_norm")(
        q_in=img_rearrange_out[0], k_in=img_rearrange_out[1]
    )

    # Text branch
    txt_mod1, txt_mod2 = modulation(hidden_size, double=True, name="txt_mod")(input=vec)
    txt_norm = LayerNorm(use_scale=False, use_bias=False, eps=1e-6, name="txt_norm1")(
        input=txt
    )
    txt_modulated = (1 + txt_mod1[1]) * txt_norm + txt_mod1[0]
    txt_qkv = Linear(hidden_size * 3, use_bias=qkv_bias, name="txt_attn_qkv")(
        input=txt_modulated
    )
    txt_rearrange_out = rearrange(num_heads=num_heads)(input=txt_qkv)

    txt_q_norm, txt_k_norm = qk_norm(hidden_size // num_heads, name="txt_attn_norm")(
        q_in=txt_rearrange_out[0], k_in=txt_rearrange_out[1]
    )

    # Cross attention
    q_concat = Concat(axis=2)(input=[txt_q_norm, img_q_norm])
    k_concat = Concat(axis=2)(input=[txt_k_norm, img_k_norm])
    v_concat = Concat(axis=2)(input=[txt_rearrange_out[2], img_rearrange_out[2]])
    attn = attention_mithril()(q=q_concat, k=k_concat, v=v_concat, pe=pe)

    # Split attention outputs: txt corresponds to the first txt.shape[1] tokens.
    txt_len = txt.shape[1]
    txt_attn = attn[:, :txt_len]  # type: ignore
    img_attn = attn[:, txt_len:]  # type: ignore

    # Image projection & residual
    img_proj = Linear(hidden_size, name="img_attn_proj")(input=img_attn)
    img = img + img_mod1[2] * img_proj
    img_norm_2 = LayerNorm(use_scale=False, use_bias=False, eps=1e-6, name="img_norm2")(
        input=img
    )

    # Image MLP
    mlp_out = Linear(mlp_hidden_dim, name="0")(input=IOKey("input"))
    mlp_out = Gelu(approximate=True)(input=mlp_out)
    mlp_out = Linear(hidden_size, name="2")(input=mlp_out)
    img_mlp_model = Model.create(name="img_mlp", output=mlp_out)
    txt_mlp_model = deepcopy(img_mlp_model)
    txt_mlp_model.name = "txt_mlp"

    img_mlp = img_mlp_model(input=(1 + img_mod2[1]) * img_norm_2 + img_mod2[0])
    img = img + img_mod2[2] * img_mlp

    # Text projection & residual
    txt_proj = Linear(hidden_size, name="txt_attn_proj")(input=txt_attn)
    txt = txt + txt_mod1[2] * txt_proj
    txt_norm_2 = LayerNorm(use_scale=False, use_bias=False, eps=1e-6, name="txt_norm2")(
        input=txt
    )

    # Text MLP
    txt_mlp = txt_mlp_model(input=(1 + txt_mod2[1]) * txt_norm_2 + txt_mod2[0])
    txt = txt + txt_mod2[2] * txt_mlp  # type: ignore[attr-defined]

    # Final outputs
    return Model.create(name=name, img_out=img, txt_out=txt)


class TestLayers:
    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_resnet_block(self, backend_type: type[ml.Backend]):
        input_shape = [2, 64, 32, 48]
        m_model = resnet_block(64, 64)
        o_model = ResnetBlock(64, 64)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": input_shape},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn(input_shape)
        expected_result = o_model(torch_inp)

        inp = backend.array(torch_inp.numpy())
        res = pm.evaluate(params, {"input": inp})["output"]
        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_attn_block(self, backend_type: type[ml.Backend]):
        m_model = attn_block(512)
        o_model = AttnBlock(512)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [1, 512, 32, 32]},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn([1, 512, 32, 32])
        expected_result = o_model(torch_inp)

        inp = backend.array(torch_inp.numpy())
        res = pm.evaluate(params, {"input": inp})["output"]
        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_downsample(self, backend_type: type[ml.Backend]):
        m_model = downsample(512)
        o_model = Downsample(512)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [2, 512, 32, 48]},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn([8, 512, 32, 48])
        expected_result = o_model(torch_inp)

        inp = backend.array(torch_inp.numpy())
        res = pm.evaluate(params, {"input": inp})["output"]
        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_upsample(self, backend_type: type[ml.Backend]):
        m_model = upsample(64)
        o_model = Upsample(64)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [1, 64, 32, 48]},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn([1, 64, 32, 48])
        expected_result = o_model(torch_inp)

        inp = backend.array(torch_inp.numpy())
        res = pm.evaluate(params, {"input": inp})["output"]
        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_encoder(self, backend_type: type[ml.Backend]):
        params: dict[str, int | list[int]] = {
            "resolution": 256,
            "in_channels": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "z_channels": 16,
        }
        m_model = encoder(**params)  # type: ignore
        o_model = Encoder(**params)  # type: ignore

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [1, 3, 256, 256]},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn([1, 3, 256, 256])
        expected_result = o_model(torch_inp)

        inp = backend.array(torch_inp.numpy())
        res = pm.evaluate(params, {"input": inp})["output"]
        np.testing.assert_allclose(res, expected_result.detach(), 5e-5, 5e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_decoder(self, backend_type: type[ml.Backend]):
        params: dict[str, int | list[int]] = {
            "ch": 128,
            "out_ch": 3,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "in_channels": 3,
            "resolution": 256,
            "z_channels": 16,
        }
        m_model = decoder(**params)  # type: ignore
        o_model = Decoder(**params)  # type: ignore

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [1, 16, 32, 32]},
            data_keys={"input"},
            jit=False,
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn([1, 16, 32, 32])
        expected_result = o_model(torch_inp)

        inp = backend.array(torch_inp.numpy())
        res = pm.evaluate(params, {"input": inp})["output"]
        np.testing.assert_allclose(res, expected_result.detach(), 1e-4, 1e-4)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_rms_norm(self, backend_type: type[ml.Backend]):
        B, L, H, D = 1, 24, 4080, 128
        input_ref = torch.randn(B, L, H, D)
        o_model = RMSNorm(dim=128)

        backend = backend_type()

        pm = ml.compile(
            rms_norm(dim=128),
            backend=backend,
            shapes={"input": [B, L, H, D]},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        expected_result = o_model(input_ref)

        inp = backend.array(input_ref.numpy())
        res = pm.evaluate(params, {"input": inp})["output"]
        np.testing.assert_allclose(res, expected_result.detach(), 1e-6, 1e-6)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_qk_norm(self, backend_type: type[ml.Backend]):
        B, L, H, D = 1, 24, 4080, 128
        q_ref = torch.randn(B, L, H, D)
        k_ref = torch.randn(B, L, H, D)
        v_ref = torch.randn(B, L, H, D)
        o_model = QKNorm(dim=128)

        backend = backend_type()

        pm = ml.compile(
            qk_norm(128),
            backend=backend,
            shapes={"q_in": [B, L, H, D], "k_in": [B, L, H, D]},
            data_keys={"q_in", "k_in"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        q_out_ref, k_out_ref = o_model(q_ref, k_ref, v_ref)

        q_in = backend.array(q_ref.numpy())
        k_in = backend.array(k_ref.numpy())
        res = pm.evaluate(params, {"q_in": q_in, "k_in": k_in})

        np.testing.assert_allclose(res["q_out"], q_out_ref.detach(), 1e-6, 1e-6)  # type: ignore
        np.testing.assert_allclose(res["k_out"], k_out_ref.detach(), 1e-6, 1e-6)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_modulation(self, backend_type: type[ml.Backend]):
        H, W = 1, 3072
        input_ref = torch.randn(H, W)
        o_model = Modulation(dim=3072, double=False)

        backend = backend_type()

        pm = ml.compile(
            modulation(3072, False),
            backend=backend,
            shapes={"input": [H, W]},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        out = o_model(input_ref)

        inp = backend.array(input_ref.numpy())
        res = pm.evaluate(params, {"input": inp})

        np.testing.assert_allclose(res["mod_1"][0], out[0].shift.detach(), 1e-5, 1e-5)  # type: ignore
        np.testing.assert_allclose(res["mod_1"][1], out[0].scale.detach(), 1e-5, 1e-5)  # type: ignore
        np.testing.assert_allclose(res["mod_1"][2], out[0].gate.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_single_stream_block(self, backend_type: type[ml.Backend]):
        hidden_size = 3072
        num_heads = 24
        mlp_ratio = 4.0

        input_ref = torch.randn(1, 4336, 3072)
        vec_ref = torch.randn(1, 3072)
        pe_ref = torch.randn(1, 1, 4336, 64, 2, 2)
        o_model = SingleStreamBlock(hidden_size, num_heads, mlp_ratio)

        backend = backend_type()

        pm = ml.compile(
            single_stream_block(hidden_size, num_heads, mlp_ratio),
            backend=backend,
            shapes={
                "input": [1, 4336, 3072],
                "vec": [1, 3072],
                "pe": [1, 1, 4336, 64, 2, 2],
            },
            data_keys={"input", "vec", "pe"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        expected_result = o_model(input_ref, vec_ref, pe_ref)

        inp = backend.array(input_ref.numpy())
        vec = backend.array(vec_ref.numpy())
        pe = backend.array(pe_ref.numpy())
        res = pm.evaluate(params, {"input": inp, "vec": vec, "pe": pe})

        np.testing.assert_allclose(res["output"], expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_double_stream_block(self, backend_type: type[ml.Backend]):
        hidden_size = 3072
        num_heads = 24
        mlp_ratio = 4.0

        dsb = double_stream_block(hidden_size, num_heads, mlp_ratio)
        dsb_func = double_stream_block_functional(hidden_size, num_heads, mlp_ratio)
        for model in [dsb, dsb_func]:
            img_ref = torch.randn(1, 4096, 3072)
            txt_ref = torch.randn(1, 512, 3072)
            vec_ref = torch.randn(1, 3072)
            pe_ref = torch.randn(1, 1, 4608, 64, 2, 2)
            o_model = DoubleStreamBlock(hidden_size, num_heads, mlp_ratio)

            backend = backend_type()

            pm = ml.compile(
                model,
                backend=backend,
                shapes={
                    "img": [1, 4096, 3072],
                    "txt": [1, 512, 3072],
                    "vec": [1, 3072],
                    "pe": [1, 1, 4608, 64, 2, 2],
                },
                data_keys={"img", "txt", "vec", "pe"},
                use_short_namings=False,
            )

            params = load_weights(pm.shapes, o_model, backend)

            img_out_ref, txt_out_ref = o_model(img_ref, txt_ref, vec_ref, pe_ref)

            img = backend.array(img_ref.numpy())
            txt = backend.array(txt_ref.numpy())
            vec = backend.array(vec_ref.numpy())
            pe = backend.array(pe_ref.numpy())
            res = pm.evaluate(params, {"img": img, "txt": txt, "vec": vec, "pe": pe})

            np.testing.assert_allclose(res["img_out"], img_out_ref.detach(), 1e-5, 1e-5)  # type: ignore
            np.testing.assert_allclose(res["txt_out"], txt_out_ref.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_apply_rope(self, backend_type: type[ml.Backend]):
        B, L, H, D = 1, 24, 4336, 128
        q_ref = torch.randn(B, L, H, D)
        k_ref = torch.randn(B, L, H, D)
        pe_ref = torch.randn(1, 1, H, D // 2, 2, 2)

        backend = backend_type()

        pm = ml.compile(
            apply_rope_mithril(),
            backend=backend,
            shapes={
                "xq": [B, L, H, D],
                "xk": [B, L, H, D],
                "freqs_cis": [1, 1, H, D // 2, 2, 2],
            },
            data_keys={"xq", "xk", "freqs_cis"},
            use_short_namings=False,
            inference=True,
        )

        expected_res = apply_rope(q_ref, k_ref, pe_ref)

        q = backend.array(q_ref.numpy())
        k = backend.array(k_ref.numpy())
        pe = backend.array(pe_ref.numpy())
        res = pm.evaluate({}, {"xq": q, "xk": k, "freqs_cis": pe})

        np.testing.assert_allclose(res["xq_out"], expected_res[0].detach(), 1e-6, 1e-6)  # type: ignore
        np.testing.assert_allclose(res["xk_out"], expected_res[1].detach(), 1e-6, 1e-6)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_time_embeddings(self, backend_type: type[ml.Backend]):
        input_ref = torch.ones([1])
        expected_res = timestep_embedding(input_ref, 256)

        backend = backend_type()

        pm = ml.compile(
            timestep_embedding_mithril(256),
            backend=backend,
            shapes={"input": [1]},
            data_keys={"input"},
            use_short_namings=False,
            inference=True,
        )

        input = backend.array(input_ref.numpy())
        res = pm.evaluate({}, {"input": input})

        np.testing.assert_allclose(res["output"], expected_res, 1e-4, 1e-4)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_mlp_embedder(self, backend_type: type[ml.Backend]):
        m_model = mlp_embedder_mithril(64)
        o_model = MLPEmbedder(48, 64)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [1, 64, 32, 48]},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn([1, 64, 32, 48])
        expected_result = o_model(torch_inp)

        inp = backend.array(torch_inp.numpy())
        res = pm.evaluate(params, {"input": inp})["output"]
        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_last_layer(self, backend_type: type[ml.Backend]):
        hidden_size = 256
        out_channels = 3
        patch_size = 2

        input_ref = torch.randn(1, 512, 256)
        vec_ref = torch.randn(1, 256)
        o_model = LastLayer(hidden_size, patch_size, out_channels)

        backend = backend_type()

        pm = ml.compile(
            last_layer(hidden_size, patch_size, out_channels),
            backend=backend,
            shapes={"input": [1, 512, 256], "vec": [1, 256]},
            data_keys={"input", "vec"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        expected_result = o_model(input_ref, vec_ref)

        input = backend.array(input_ref.numpy())
        vec = backend.array(vec_ref.numpy())
        res = pm.evaluate(params, {"input": input, "vec": vec})

        np.testing.assert_allclose(res["output"], expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_rope(self, backend_type: type[ml.Backend]):
        dim = 256
        theta = 4
        pos_ref = torch.rand(dim, 2)

        backend = backend_type()

        pm = ml.compile(
            rope(dim, theta),
            backend=backend,
            shapes={"input": [dim, 2]},
            data_keys={"input"},
            use_short_namings=False,
            jit=False,
            inference=True,
        )

        expected_result = torch_rope(pos_ref, dim, theta)

        pos = backend.array(pos_ref.numpy())
        res = pm.evaluate({}, {"input": pos})

        np.testing.assert_allclose(res["output"], expected_result, rtol=1e-5, atol=1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_embednd(self, backend_type: type[ml.Backend]):
        dim = 128
        theta = 10_000
        axes_dim = [16, 56, 56]
        input_ref = torch.rand(1, 4336, 3)
        o_model = EmbedND(dim=dim, theta=theta, axes_dim=axes_dim)

        backend = backend_type()

        pm = ml.compile(
            embed_nd(theta=theta, axes_dim=axes_dim),
            backend=backend,
            shapes={"input": [1, 4336, 3]},
            data_keys={"input"},
            use_short_namings=False,
            inference=True,
        )

        expected_result = o_model(input_ref)

        input = backend.array(input_ref.numpy())
        res = pm.evaluate({}, {"input": input})

        np.testing.assert_allclose(res["output"], expected_result, rtol=1e-5, atol=1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_attention(self, backend_type: type[ml.Backend]):
        B, L, H, D = 1, 24, 4336, 128
        q_ref = torch.randn(B, L, H, D)
        k_ref = torch.randn(B, L, H, D)
        v_ref = torch.randn(B, L, H, D)
        pe_ref = torch.randn(1, 1, H, D // 2, 2, 2)

        backend = backend_type()

        pm = ml.compile(
            attention_mithril(),
            backend=backend,
            shapes={
                "q": [B, L, H, D],
                "k": [B, L, H, D],
                "v": [B, L, H, D],
                "pe": [1, 1, H, D // 2, 2, 2],
            },
            data_keys={"q", "k", "v", "pe"},
            use_short_namings=False,
            inference=True,
        )

        expected_res = attention(q_ref, k_ref, v_ref, pe_ref)

        q = backend.array(q_ref.numpy())
        k = backend.array(k_ref.numpy())
        v = backend.array(v_ref.numpy())
        pe = backend.array(pe_ref.numpy())
        res = pm.evaluate({}, {"q": q, "k": k, "v": v, "pe": pe})

        np.testing.assert_allclose(res["output"], expected_res.detach(), 1e-5, 1e-5)  # type: ignore


## Utils


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
