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
import argparse
from typing import Any

import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from whisper_model import whisper_model

import mithril as ml
from mithril import Backend

backend_map: dict[str, type[Backend]] = {
    "torch": ml.TorchBackend,
    "jax": ml.JaxBackend,
    "numpy": ml.NumpyBackend,
    "mlx": ml.MlxBackend,
}
# %% Get required vocabulary and convert decoded id's to text.
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
vocab = tokenizer.get_vocab()
reverse_dict = {value: key for key, value in vocab.items()}


def convert_ids_to_text(decoded_ids: list[int], reverse_dict: dict[int, str]) -> str:
    decoded_text = ""
    for key in decoded_ids:
        current_token = reverse_dict[key]
        if current_token[0] != "<":
            if current_token[0] == "Ġ":
                decoded_text += " "
                decoded_text += current_token[1:]
            else:
                decoded_text += current_token
    return decoded_text


# %% Get weights for pre-trained model and adapt names to the logical model.
def get_weights(backend: Backend) -> dict[str, Any]:
    model_hf = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    sd_hf = model_hf.state_dict()
    params = {}
    for key in sd_hf:
        ml_key = key.replace(".", "_")
        if "conv" in ml_key and "bias" in ml_key:
            params[ml_key] = backend.array(np.array(sd_hf[key])).reshape(1, -1, 1)
        else:
            params[ml_key] = backend.array(np.array(sd_hf[key]))
    return params


# %%
# Generate mel-spectograms from raw audio file
def process_input(audio: np.ndarray) -> np.ndarray:
    sr = 16000  # Sampling rate
    target_samples = (3000 - 1) * 160 + 400  # Exactly 3000 samples is needed.
    n_mels = 80
    # Length adjustment for shorter sequence by padding
    y = librosa.util.fix_length(audio, size=target_samples)

    # Mel spectrogram generation
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels, power=2, center=False
    )

    # Converting to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalization steps
    mean = np.mean(mel_spec_db)
    std = np.std(mel_spec_db)
    standardized_mel = (mel_spec_db - mean) / std  # Zero mean, unit variance

    mel_min = np.min(standardized_mel)
    mel_max = np.max(standardized_mel)
    normalized_mel = (
        2 * (standardized_mel - mel_min) / (mel_max - mel_min) - 1
    )  # Normalize to [-1, 1]

    # Reshape for model input (a single sample is considered)
    input_features = normalized_mel.reshape((1, n_mels, 3000))
    return input_features


# Initialize and compile the whisper model
def run_inference(file_path: str, backend: str) -> str:
    backend_obj = backend_map[backend](dtype=ml.float32, device="cpu")
    sample, sample_rate = librosa.load(file_path, sr=16000)
    whisper = whisper_model(num_layers=4, input_dim=384, num_heads=6, ffn_dim=1536)
    compiled_model = ml.compile(
        whisper,
        backend_obj,
        data_keys={"input", "decoder_input_ids"},
        jit=False,
        use_short_namings=False,
    )
    params = get_weights(backend_obj)
    input_features = backend_obj.array(
        process_input(sample)
    )  # Generate mel-spectograms for sample input
    token = 50258  # End token for whisper
    decoded_text = [token]
    decoder_input_ids = backend_obj.array([decoded_text])  # Working
    # Simple greedy decoding
    while token != 50257:  # Start token for whisper
        outputs = compiled_model.evaluate(
            params,
            data={"input": input_features, "decoder_input_ids": decoder_input_ids},
        )
        logits = outputs["output"]
        logits = logits[:, -1, :]  # type: ignore
        token = backend_obj.argmax(logits)  # type: ignore
        decoded_text.append(int(token))
        decoder_input_ids = backend_obj.array([decoded_text])
    transcription = convert_ids_to_text(decoder_input_ids.tolist()[0], reverse_dict)
    print(transcription[1:])
    return transcription[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_path", type=str, default="1040-133433-0001.flac")
    ap.add_argument("--backend", type=str, default="torch")
    args = ap.parse_args()
    run_args = {k: v for k, v in vars(args).items() if k != "save"}
    run_inference(**run_args)


if __name__ == "__main__":
    main()
