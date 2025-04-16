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


import json
import sys
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sentencepiece import SentencePieceProcessor
from util import configs

import mithril as ml
from examples.t5 import t5_encode


def sanitize(weights):
    shared_replacement_patterns = [
        (".block.", ".layers."),
        (".k.", ".key_proj."),
        (".o.", ".out_proj."),
        (".q.", ".query_proj."),
        (".v.", ".value_proj."),
        ("shared.", "wte."),
        ("lm_head.", "lm_head.linear."),
        (".layer.0.layer_norm.", ".ln1."),
        (".layer.1.layer_norm.", ".ln2."),
        (".layer.2.layer_norm.", ".ln3."),
        (".final_layer_norm.", ".ln."),
        (
            "layers.0.layer.0.SelfAttention.relative_attention_bias.",
            "relative_attention_bias.embeddings.",
        ),
    ]

    encoder_replacement_patterns = [
        (".layer.0.SelfAttention.", ".attention."),
        (".layer.1.DenseReluDense.", ".dense."),
    ]

    decoder_replacement_patterns = [
        (".layer.0.SelfAttention.", ".self_attention."),
        (".layer.1.EncDecAttention.", ".cross_attention."),
        (".layer.2.DenseReluDense.", ".dense."),
    ]

    ignored_keys = ["decoder_layers_0_cross_attention_relative_attention_bias_weight"]

    def replace_key(key: str) -> str:
        for old, new in shared_replacement_patterns:
            key = key.replace(old, new)
        if key.startswith("encoder."):
            for old, new in encoder_replacement_patterns:
                key = key.replace(old, new)
        elif key.startswith("decoder."):
            for old, new in decoder_replacement_patterns:
                key = key.replace(old, new)
        return key.replace(".", "_")

    weights = {replace_key(k): v for k, v in weights.items()}
    for key in ignored_keys:
        if key in weights:
            del weights[key]
    return weights


class T5Tokenizer:
    def __init__(self, model_file, backend: ml.Backend, max_length=512):
        self._tokenizer = SentencePieceProcessor(model_file)
        self.max_length = max_length
        self.backend = backend

    @property
    def pad(self):
        try:
            return self._tokenizer.id_to_piece(self.pad_token)
        except IndexError:
            return None

    @property
    def pad_token(self):
        return self._tokenizer.pad_id()

    @property
    def bos(self):
        try:
            return self._tokenizer.id_to_piece(self.bos_token)
        except IndexError:
            return None

    @property
    def bos_token(self):
        return self._tokenizer.bos_id()

    @property
    def eos(self):
        try:
            return self._tokenizer.id_to_piece(self.eos_token)
        except IndexError:
            return None

    @property
    def eos_token(self):
        return self._tokenizer.eos_id()

    def tokenize(self, text, prepend_bos=True, append_eos=True, pad=True):
        if isinstance(text, list):
            return [self.tokenize(t, prepend_bos, append_eos, pad) for t in text]

        tokens = self._tokenizer.encode(text)

        if prepend_bos and self.bos_token >= 0:
            tokens = [self.bos_token] + tokens
        if append_eos and self.eos_token >= 0:
            tokens.append(self.eos_token)
        if pad and len(tokens) < self.max_length and self.pad_token >= 0:
            tokens += [self.pad_token] * (self.max_length - len(tokens))

        return tokens

    def encode(self, text, pad=True):
        if not isinstance(text, list):
            return self.encode([text], pad=pad)

        pad_token = self.pad_token if self.pad_token >= 0 else 0
        tokens = self.tokenize(text, pad=pad)
        length = max(len(t) for t in tokens)
        for t in tokens:
            t.extend([pad_token] * (length - len(t)))

        return self.backend.array(tokens)


sys.setrecursionlimit(3500)


def download_t5_encoder_weights(
    backend: ml.Backend, name: str
) -> dict[str, np.ndarray[Any, Any]]:
    model_index = hf_hub_download(
        configs[name].repo_id, "text_encoder_2/model.safetensors.index.json"
    )

    weight_files = set()
    with open(model_index) as f:
        for _, w in json.load(f)["weight_map"].items():
            weight_files.add(w)

    if backend.backend_type == "torch":
        target_lib = "pt"
    elif backend.backend_type == "jax":
        target_lib = "jax"
    else:
        # TODO Fix here
        raise NotImplementedError("T5 encoder only supported for Jax and Torch!")

    weights = {}
    for w in weight_files:
        w = hf_hub_download(configs[name].repo_id, f"text_encoder_2/{w}")
        safe_tensors = safe_open(w, target_lib)
        for key in safe_tensors.keys():  # type: ignore #noqa SIM118
            weights[key] = backend.array(safe_tensors.get_tensor(key))  # type: ignore

    return sanitize(weights)


def load_t5_encoder(name: str, max_len: int = 256) -> ml.models.PhysicalModel:
    config_path = hf_hub_download(configs[name].repo_id, "text_encoder_2/config.json")

    with open(config_path) as f:
        config = json.load(f)

    t5 = t5_encode(config, name="encoder")
    t5.set_shapes(input=[1, max_len])

    return t5


def load_t5_tokenizer(backend: ml.Backend, name: str, pad: bool = True):
    model_file = hf_hub_download(configs[name].repo_id, "tokenizer_2/spiece.model")
    return T5Tokenizer(
        model_file, backend, 256 if "schnell" in configs[name].repo_id else 512
    )
