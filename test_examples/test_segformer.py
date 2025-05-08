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

import numpy as np
import pytest
import torch
from transformers import AutoModelForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import (
    SegformerAttention,
    SegformerDecodeHead,
    SegformerDWConv,
    SegformerEfficientSelfAttention,
    SegformerEncoder,
    SegformerLayer,
    SegformerMixFFN,
    SegformerMLP,
    SegformerOverlapPatchEmbeddings,
    SegformerSelfOutput,
)

import mithril as ml
from examples.segformer.segformer_decode_head import (
    segformer_bilinear_interpolate,
    segformer_decode_head,
    segformer_mlp,
)
from examples.segformer.segformer_encoder import (
    segformer_attention,
    segformer_dwconv,
    segformer_efficient_self_attention,
    segformer_encoder,
    segformer_layer,
    segformer_mixffn,
    segformer_overlap_patch_embeddings,
    segformer_self_output,
)
from examples.segformer.segformer_semantic_segmentation import (
    segformer_semantic_segmentation,
)
from test_examples.test_flux import load_weights

installed_backends: list[type[ml.Backend]] = [
    ml.TorchBackend,
    ml.JaxBackend,
    # ml.NumpyBackend, #slow
]

if platform.system() == "Darwin":
    installed_backends.append(ml.MlxBackend)

SEG_MODEL_ID = "mattmdjaga/segformer_b2_clothes"


class TestSegformerLayers:
    segmentation_model = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL_ID)
    config = segmentation_model.segformer.config

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_dwconv(self, backend_type: type[ml.Backend]):
        input_shape = [1, 16384, 256]
        height, width = 128, 128
        m_model = segformer_dwconv(256)
        o_model = SegformerDWConv(256)
        o_model.eval()

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": input_shape},
            constant_keys={"height": height, "width": width},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn(input_shape)
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        expected_result = o_model(torch_inp, height, width)
        # Mithril results.
        res = pm.evaluate(params, {"input": input})["output"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_segformer_mixffn(self, backend_type: type[ml.Backend]):
        input_shape = [1, 16384, 64]
        height, width = 128, 128
        in_features, hidden_features = 64, 256
        m_model = segformer_mixffn(
            in_features=in_features, hidden_features=hidden_features
        )
        o_model = SegformerMixFFN(
            config=self.config, in_features=in_features, hidden_features=hidden_features
        )
        o_model.eval()

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": input_shape},
            constant_keys={"height": height, "width": width},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn(input_shape)
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        expected_result = o_model(torch_inp, height, width)
        # Mithril results.
        res = pm.evaluate(params, {"input": input})["output"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_segformer_self_output(self, backend_type: type[ml.Backend]):
        input_shape = [1, 16384, 64]
        hidden_size = 64
        m_model = segformer_self_output(hidden_size)
        o_model = SegformerSelfOutput(self.config, hidden_size)
        o_model.eval()

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
        dummy_input = torch.randn(input_shape)  # needed for the original torch model.
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        expected_result = o_model(torch_inp, dummy_input)
        # Mithril results.
        res = pm.evaluate(params, {"input": input})["output"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    @pytest.mark.parametrize("sequence_reduction_ratio", (1, 8))
    def test_segformer_efficient_self_attention(
        self, backend_type: type[ml.Backend], sequence_reduction_ratio: int
    ):
        input_shape = [1, 16384, 64]
        height, width = 128, 128
        hidden_size, num_attention_heads = 64, 1
        m_model = segformer_efficient_self_attention(
            hidden_size, num_attention_heads, sequence_reduction_ratio
        )
        o_model = SegformerEfficientSelfAttention(
            self.config, hidden_size, num_attention_heads, sequence_reduction_ratio
        )
        o_model.eval()

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": input_shape},
            constant_keys={"height": height, "width": width},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn(input_shape)
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        expected_result = o_model(torch_inp, height, width)[0]
        # Mithril results.
        res = pm.evaluate(params, {"input": input})["output"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    @pytest.mark.parametrize("sequence_reduction_ratio", (1, 8))
    def test_segformer_attention(
        self, backend_type: type[ml.Backend], sequence_reduction_ratio: int
    ):
        input_shape = [1, 16384, 64]
        height, width = 128, 128
        hidden_size, num_attention_heads = 64, 1
        m_model = segformer_attention(
            hidden_size, num_attention_heads, sequence_reduction_ratio
        )
        o_model = SegformerAttention(
            self.config, hidden_size, num_attention_heads, sequence_reduction_ratio
        )
        o_model.eval()

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": input_shape},
            constant_keys={"height": height, "width": width},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn(input_shape)
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        expected_result = o_model(torch_inp, height, width)[0]
        # Mithril results.
        res = pm.evaluate(params, {"input": input})["output"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    @pytest.mark.parametrize("sequence_reduction_ratio", (1, 8))
    def test_segformer_layer(
        self, backend_type: type[ml.Backend], sequence_reduction_ratio: int
    ):
        input_shape = [1, 16384, 64]
        height, width = 128, 128
        hidden_size, num_attention_heads = 64, 1
        mlp_ratio = 4.0
        drop_path = (
            0.9  # not used in inference, but needed for the original torch model.
        )

        m_model = segformer_layer(
            hidden_size, num_attention_heads, sequence_reduction_ratio, mlp_ratio
        )
        o_model = SegformerLayer(
            self.config,
            hidden_size,
            num_attention_heads,
            drop_path,
            sequence_reduction_ratio,
            mlp_ratio,
        )
        o_model.eval()

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": input_shape},
            constant_keys={"height": height, "width": width},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn(input_shape)
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        expected_result = o_model(torch_inp, height, width)[0]
        # Mithril results.
        res = pm.evaluate(params, {"input": input})["output"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_segformer_overlap_patch_embeddings(self, backend_type: type[ml.Backend]):
        input_shape = [1, 3, 512, 512]
        patch_size, stride, num_channels, hidden_size = 7, 4, 3, 64
        # Mithril does not need the num_channels parameter, automatically
        # infers from the input.
        m_model = segformer_overlap_patch_embeddings(patch_size, stride, hidden_size)
        o_model = SegformerOverlapPatchEmbeddings(
            patch_size, stride, num_channels, hidden_size
        )
        o_model.eval()

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
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        expected_embed, expected_height, expected_width = o_model(torch_inp)
        # Mithril results.
        res = pm.evaluate(params, {"input": input})

        np.testing.assert_allclose(
            res["embeddings"],  # type: ignore
            expected_embed.detach(),
            1e-5,
            1e-5,  # type: ignore
        )
        np.testing.assert_allclose(res["height"], expected_height, 1e-5, 1e-5)  # type: ignore
        np.testing.assert_allclose(res["width"], expected_width, 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_segformer_encoder(self, backend_type: type[ml.Backend]):
        pixel_values_shape = [1, 3, 512, 512]
        m_model = segformer_encoder(self.config)
        o_model = SegformerEncoder(self.config)
        o_model.eval()

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": pixel_values_shape},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn(pixel_values_shape)
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        total_result = o_model(torch_inp, output_hidden_states=True)
        expected_output = total_result.last_hidden_state
        expected_hidden_states = total_result.hidden_states
        # Mithril results.
        total_res = pm.evaluate(params, {"input": input})
        output = total_res["output"]
        hidden_states = total_res["hidden_states"]

        np.testing.assert_allclose(output, expected_output.detach(), 1e-5, 1e-5)  # type: ignore
        for expected_hidden, mithril_hidden in zip(
            expected_hidden_states,
            hidden_states,  # type: ignore
            strict=True,
        ):
            np.testing.assert_allclose(
                mithril_hidden, expected_hidden.detach(), 1e-5, 1e-5
            )

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_segformer_bilinear_interpolate(self, backend_type: type[ml.Backend]):
        pixel_values_shp = [1, 768, 64, 64]
        size = (128, 128)
        m_model = segformer_bilinear_interpolate()

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": pixel_values_shp},
            data_keys={"input", "size"},
            use_short_namings=False,
            inference=True,
            jit=False,
        )

        torch_inp = torch.randn(pixel_values_shp)
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        expected_result = torch.nn.functional.interpolate(
            torch_inp, size, mode="bilinear", align_corners=False
        )
        # Mithril results.
        res = pm.evaluate({}, {"input": input, "size": size})["output"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_segformer_mlp(self, backend_type: type[ml.Backend]):
        pixel_values_shp = [1, 320, 32, 32]
        m_model = segformer_mlp(self.config.decoder_hidden_size)
        o_model = SegformerMLP(self.config, input_dim=self.config.hidden_sizes[2])
        o_model.eval()

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": pixel_values_shp},
            data_keys={"input"},
            use_short_namings=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn(pixel_values_shp)
        input = backend.array(torch_inp.numpy())
        # Original Torch results.
        expected_result = o_model(torch_inp)
        # Mithril results.
        res = pm.evaluate(params, {"input": input})["output"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_segformer_decode_head(self, backend_type: type[ml.Backend]):
        hidden_shapes = [
            [1, 64, 128, 128],
            [1, 128, 64, 64],
            [1, 320, 32, 32],
            [1, 512, 16, 16],
        ]
        m_model = segformer_decode_head(self.config)
        o_model = SegformerDecodeHead(self.config)
        o_model.eval()

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            data_keys={"input"},
            use_short_namings=False,
            safe_names=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = tuple(torch.randn(shape) for shape in hidden_shapes)
        input = tuple(backend.array(inp.numpy()) for inp in torch_inp)
        # For this test, we need to move 'batch_norm_running_mean' and
        # 'batch_norm_running_var' to data dict since they are actually
        # state_keys.
        data = {
            "input": input,
            "batch_norm_running_mean": params.pop("batch_norm_running_mean"),
            "batch_norm_running_var": params.pop("batch_norm_running_var"),
        }
        # Original Torch results.
        expected_result = o_model(torch_inp)
        # Mithril results.
        res = pm.evaluate(params, data)["logits"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore

    @pytest.mark.parametrize("backend_type", installed_backends)
    def test_segformer_semantic_segmentation(self, backend_type: type[ml.Backend]):
        input_shape = [1, 3, 512, 512]
        o_model = self.segmentation_model
        m_model = segformer_semantic_segmentation(self.config)

        backend = backend_type()

        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": input_shape},
            data_keys={"input"},
            use_short_namings=False,
            safe_names=False,
        )

        params = load_weights(pm.shapes, o_model, backend)

        torch_inp = torch.randn(input_shape)
        input = backend.array(torch_inp.numpy())
        # For this test, we need to move 'batch_norm_running_mean' and
        # 'batch_norm_running_var' to data dict since they are actually
        # state_keys.
        data = {
            "input": input,
            "decode_head_batch_norm_running_mean": params.pop(
                "decode_head_batch_norm_running_mean"
            ),
            "decode_head_batch_norm_running_var": params.pop(
                "decode_head_batch_norm_running_var"
            ),
        }
        # Original Torch results.
        expected_result = o_model(torch_inp).logits
        # Mithril results.
        res = pm.evaluate(params, data)["logits"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)  # type: ignore
