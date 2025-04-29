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

import torch.nn as nn


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            from transformers import CLIPTextModel, CLIPTokenizer

            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                version, max_length=max_length
            )
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                version, **hf_kwargs
            )
        else:
            from transformers import T5EncoderModel, T5Tokenizer

            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(  # type: ignore
                version, max_length=max_length
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(  # type: ignore
                version, **hf_kwargs
            )

        self.hf_module = self.hf_module.eval().requires_grad_(False)  # type: ignore

    def __call__(self, text: list[str]):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(  # type: ignore
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),  # type: ignore
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
