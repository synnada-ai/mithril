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
from dataclasses import dataclass
from typing import Any

import numpy as np
import regex
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from util import configs

import mithril as ml
from examples.clip.model_mlx import clip_text_model

backend_torch = ml.TorchBackend()


class CLIPTokenizer:
    # A simple port of CLIPTokenizer from https://github.com/huggingface/transformers

    def __init__(self, bpe_ranks, vocab, backend: ml.Backend, max_length=77):
        self.max_length = max_length
        self.bpe_ranks = bpe_ranks
        self.vocab = vocab
        self.backend = backend
        self.pat = regex.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )

        self._cache = {self.bos: self.bos, self.eos: self.eos}

    @property
    def bos(self):
        return "<|startoftext|>"

    @property
    def bos_token(self):
        return self.vocab[self.bos]

    @property
    def eos(self):
        return "<|endoftext|>"

    @property
    def eos_token(self):
        return self.vocab[self.eos]

    def bpe(self, text):
        if text in self._cache:
            return self._cache[text]

        unigrams = list(text[:-1]) + [text[-1] + "</w>"]
        unique_bigrams = set(zip(unigrams, unigrams[1:], strict=False))

        if not unique_bigrams:
            return unigrams

        # In every iteration try to merge the two most likely bigrams. If none
        # was merged we are done.
        #
        # Ported from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/tokenization_clip.py
        while unique_bigrams:
            bigram = min(
                unique_bigrams, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break

            new_unigrams = []
            skip = False
            for a, b in zip(unigrams, unigrams[1:], strict=False):
                if skip:
                    skip = False
                    continue

                if (a, b) == bigram:
                    new_unigrams.append(a + b)
                    skip = True

                else:
                    new_unigrams.append(a)

            if not skip:
                new_unigrams.append(b)

            unigrams = new_unigrams
            unique_bigrams = set(zip(unigrams, unigrams[1:], strict=False))

        self._cache[text] = unigrams

        return unigrams

    def tokenize(self, text, prepend_bos=True, append_eos=True):
        if isinstance(text, list):
            return [self.tokenize(t, prepend_bos, append_eos) for t in text]

        # Lower case cleanup and split according to self.pat. Hugging Face does
        # a much more thorough job here but this should suffice for 95% of
        # cases.
        clean_text = regex.sub(r"\s+", " ", text.lower())
        tokens = regex.findall(self.pat, clean_text)

        # Split the tokens according to the byte-pair merge file
        bpe_tokens = [ti for t in tokens for ti in self.bpe(t)]

        # Map to token ids and return
        tokens = [self.vocab[t] for t in bpe_tokens]
        if prepend_bos:
            tokens = [self.bos_token] + tokens
        if append_eos:
            tokens.append(self.eos_token)

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
            if append_eos:
                tokens[-1] = self.eos_token

        return tokens

    def encode(self, text):
        if not isinstance(text, list):
            return self.encode([text])

        tokens = self.tokenize(text)
        for t in tokens:
            t.extend([self.eos_token] * (self.max_length - len(t)))

        return self.backend.array(tokens, dtype=ml.int32)


@dataclass
class CLIPTextModelConfig:
    num_layers: int = 23
    model_dims: int = 1024
    num_heads: int = 16
    max_length: int = 77
    vocab_size: int = 49408
    hidden_act: str = "quick_gelu"

    @classmethod
    def from_dict(cls, config):
        return cls(
            num_layers=config["num_hidden_layers"],
            model_dims=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            max_length=config["max_position_embeddings"],
            vocab_size=config["vocab_size"],
            hidden_act=config["hidden_act"],
        )


def load_clip_encoder(
    name: str,
) -> ml.models.PhysicalModel:
    config_path = hf_hub_download(configs[name].repo_id, "text_encoder/config.json")

    with open(config_path) as f:
        config = json.load(f)

    clip_model = clip_text_model(config)
    clip_model.set_shapes(input=[1, config["max_position_embeddings"]])

    return clip_model


def load_clip_tokenizer(backend: ml.Backend, name: str):
    vocab_file = hf_hub_download(configs[name].repo_id, "tokenizer/vocab.json")
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)

    merges_file = hf_hub_download(configs[name].repo_id, "tokenizer/merges.txt")
    with open(merges_file, encoding="utf-8") as f:
        bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
    bpe_merges = [tuple(m.split()) for m in bpe_merges]  # type: ignore
    bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))  # type: ignore

    return CLIPTokenizer(bpe_ranks, vocab, backend, max_length=77)


def download_clip_encoder_weights(
    backend: ml.Backend, name: str
) -> dict[str, np.ndarray[Any, Any]]:
    ckpt_path = hf_hub_download(configs[name].repo_id, "text_encoder/model.safetensors")

    if backend.backend_type == "torch":
        target_lib = "pt"
    elif backend.backend_type == "jax":
        target_lib = "jax"

    else:
        # TODO Fix here
        raise NotImplementedError("T5 encoder only supported for Jax and Torch!")

    safe_tensors = safe_open(ckpt_path, target_lib)
    weight_files = set(safe_tensors.keys())  # type: ignore

    weights = {}
    for key in weight_files:
        weights[key] = backend.array(safe_tensors.get_tensor(key))  # type: ignore
    weights = {
        key.replace("text_model.", "").replace(".", "_"): value
        for key, value in weights.items()
    }

    return weights
