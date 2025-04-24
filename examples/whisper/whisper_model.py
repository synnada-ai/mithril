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
from mithril.models import (
    Add,
    Arange,
    Convolution1D,
    Embedding,
    Gelu,
    LayerNorm,
    Linear,
    Model,
    ScaledDotProduct,
    Size,
)


# KV cache logic is not implemented and currently it is accepted as None
def whisper_attention(
    name: str,
    input_dim: int,
    num_heads: int,
    is_causal: bool = False,
    cross_attention: bool = False,
):
    """Attention implementation for self attention in encoder-decoder,
    and cross-attention in decoder"""
    model = Model(name=name)
    t_axes = (0, 2, 1, 3)
    v_project = Linear(input_dim, name="v_proj")
    k_project = Linear(input_dim, name="k_proj", use_bias=False)
    q_project = Linear(input_dim, name="q_proj")
    model |= q_project.connect(input="input", output="q")
    shp_con = model.input.shape  # type: ignore
    reshape_con = (shp_con[0], shp_con[1], num_heads, -1)
    tq = model.q.reshape(reshape_con).transpose(t_axes)  # type: ignore
    if cross_attention:
        model |= v_project.connect(input="xa", output="v")
        model |= k_project.connect(input="xa", output="k")
        shp_con_2 = model.xa.shape  # type: ignore
        reshape_kv = (shp_con_2[0], shp_con_2[1], num_heads, -1)
        tk = model.k.reshape(reshape_kv).transpose(t_axes)  # type: ignore
        tv = model.v.reshape(reshape_kv).transpose(t_axes)  # type: ignore
    else:
        model |= v_project.connect(input="input", output="v")
        model |= k_project.connect(input="input", output="k")  # type: ignore
        tk = model.k.reshape(reshape_con).transpose(t_axes)  # type: ignore
        tv = model.v.reshape(reshape_con).transpose(t_axes)  # type: ignore
    model |= ScaledDotProduct(is_causal=is_causal).connect(
        query=tq, key=tk, value=tv, output="sdp_out"
    )
    t_sdp = model.sdp_out.transpose(t_axes).reshape(shp_con[:3])  # type: ignore
    model |= (lin := Linear(input_dim, name="out_proj")).connect(t_sdp)
    model.set_cout(lin.output)
    return model


def encoder_block(num_layers: int, input_dim: int, num_heads: int, ffn_dim: int):
    layers = Model(name="layers")
    for idx in range(num_layers):
        block = Model(name=f"{idx}")
        block |= LayerNorm(name="self_attn_layer_norm").connect(
            "layer_in", output="layer_norm_out"
        )
        block += whisper_attention("self_attn", input_dim, num_heads)
        block += Add().connect("layer_in", output="attn_res")
        block.set_cout("attn_res")
        block += LayerNorm(name="final_layer_norm")
        block += Linear(ffn_dim, name="fc1")
        block += Gelu()
        block += Linear(input_dim, name="fc2").connect(output="ffn_out")
        block |= Add().connect("attn_res", block.ffn_out)  # type: ignore
        block.set_cin("layer_in")
        layers += block
    return layers


def decoder_block(num_layers: int, input_dim: int, num_heads: int, ffn_dim: int):
    layers = Model(name="layers")
    for idx in range(num_layers):
        block = Model(name=f"{idx}")
        block += LayerNorm(name="self_attn_layer_norm").connect(
            input="layer_in", output="layer_norm_out"
        )
        block += whisper_attention(
            "self_attn", input_dim, num_heads, is_causal=True
        )  # Self attention between decoder ids
        block |= Add().connect(left="layer_in", right=block.cout, output="attn_res")
        block.set_cout("attn_res")
        block += LayerNorm(name="encoder_attn_layer_norm")
        # Cross attention between audio and decoder ids
        block |= whisper_attention(
            "encoder_attn", input_dim, num_heads, is_causal=False, cross_attention=True
        ).connect(input=block.cout, xa="encoder_hidden_states")
        block |= Add().connect("attn_res", block.cout, output="cross_attn_out")
        block.set_cout("cross_attn_out")
        block += LayerNorm(name="final_layer_norm")
        block += Linear(ffn_dim, name="fc1")
        block += Gelu()
        block += Linear(input_dim, name="fc2").connect(output="ffn_out")
        block |= Add().connect("cross_attn_out", block.ffn_out)  # type: ignore
        block.set_cin("layer_in")
        layers += block.connect(encoder_hidden_states="encoder_hidden_states")
    return layers


# Creating encoder to generate representations from mel-spectograms
def whisper_encoder(num_layers: int, input_dim: int, num_heads: int, ffn_dim: int):
    model = Model(name="encoder")
    model |= Convolution1D(
        out_channels=input_dim, kernel_size=3, padding=1, name="conv1"
    ).connect(input="input", output="conv1_out")
    model |= Gelu().connect(input="conv1_out", output="gelu1_out")
    model |= Convolution1D(
        out_channels=input_dim, kernel_size=3, stride=2, padding=1, name="conv2"
    ).connect(input="gelu1_out", output="conv2_out")
    model |= Gelu().connect(input="conv2_out", output="gelu2_out")
    processed_out = model.gelu2_out.transpose((0, 2, 1))  # type: ignore
    model |= Arange().connect(stop=1500, output="embedding_in")
    model |= Embedding(
        name="embed_positions", num_embeddings=1500, dim=input_dim
    ).connect(
        input="embedding_in", output="pos_out"
    )  # Sinusiodal positional embeddings
    model |= Add().connect(
        left="pos_out", right=processed_out, output="attention_input"
    )
    model.set_cout("attention_input")
    encoder_layers = encoder_block(num_layers, input_dim, num_heads, ffn_dim)
    model += encoder_layers
    model += LayerNorm(name="layer_norm").connect(output="encoder_hidden_states")
    return model


# Create decoder for autoregressive generation using encoder states
def whisper_decoder(num_layers: int, input_dim: int, num_heads: int, ffn_dim: int):
    model = Model(name="decoder")
    model |= Embedding(
        name="embed_tokens", num_embeddings=51865, dim=input_dim
    ).connect(input="decoder_input_ids", output="embedded_tokens")
    model |= Size(dim=1).connect("embedded_tokens")  # Decoder id token embeddings
    model += Arange()
    model += Embedding(
        name="embed_positions", num_embeddings=448, dim=input_dim
    ).connect(output="embedded_positions")  # Positional embedding for decoder ids
    model |= Add().connect("embedded_tokens", model.cout)
    decoder_layers = decoder_block(num_layers, input_dim, num_heads, ffn_dim)
    model += decoder_layers(encoder_hidden_states="encoder_hidden_states")
    model += LayerNorm(name="layer_norm")
    return model


# Merge the blocks and create the model
def whisper_model(
    num_layers: int, input_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-5
):
    model = Model(name="model")
    model += whisper_encoder(num_layers, input_dim, num_heads, ffn_dim).connect(
        input="input", encoder_hidden_states="encoder_hidden_states"
    )
    decoder_model = whisper_decoder(num_layers, input_dim, num_heads, ffn_dim)
    model |= decoder_model(
        encoder_hidden_states="encoder_hidden_states",
        decoder_input_ids="decoder_input_ids",
    )
    whisper = Model()
    whisper |= model.connect(input="input", decoder_input_ids="decoder_input_ids")
    whisper += Linear(
        51865, name="proj_out", use_bias=False
    )  # Mapping decoder output to vocabulary tokens
    return whisper
