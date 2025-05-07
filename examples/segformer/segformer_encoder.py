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

from mithril.framework import IOKey, Tensor
from mithril.models import (
    Buffer,
    Convolution2D,
    Flatten,
    Gelu,
    LayerNorm,
    Linear,
    Model,
    Softmax,
    ToTuple,
)


def segformer_dwconv(dim=768, *, name: str | None = None) -> Model:
    hidden_states = IOKey("input", type=Tensor[float])
    height = IOKey("height")
    width = IOKey("width")

    shape = hidden_states.shape
    batch_size = shape[0]
    num_channels = shape[2]
    hidden_states = hidden_states.transpose(axes=(0, 2, 1))
    hidden_states = hidden_states.reshape(
        shape=(batch_size, num_channels, height, width)
    )
    hidden_states = Convolution2D(
        kernel_size=3,
        out_channels=dim,
        padding=1,
        use_bias=True,
        groups=dim,
        name="dwconv",
    )(input=hidden_states)
    hidden_states = Flatten(start_dim=2)(input=hidden_states).transpose(axes=(0, 2, 1))
    # Final outputs
    return Model.create(name=name, output=hidden_states)


def segformer_mixffn(
    in_features, hidden_features=None, out_features=None, *, name: str | None = None
):
    input = IOKey("input", type=Tensor[float])
    height = IOKey("height")
    width = IOKey("width")

    out_features = out_features or in_features
    hidden_states = Linear(hidden_features, name="dense1")(input=input)
    hidden_states = segformer_dwconv(dim=hidden_features, name="dwconv")(
        input=hidden_states, height=height, width=width
    )
    hidden_states = Gelu(name="intermediate_act_fn")(hidden_states)
    hidden_states = Linear(out_features, name="dense2")(input=hidden_states)
    return Model.create(name=name, output=hidden_states)


def segformer_self_output(hidden_size, *, name: str | None = None):
    input = IOKey("input", type=Tensor[float])
    hidden_states = Linear(hidden_size, name="dense")(input=input)
    return Model.create(name=name, output=hidden_states)


def transpose_for_scores(
    num_attention_heads, attention_head_size, *, name: str | None = None
):
    hidden_states = IOKey("input", type=Tensor[float])
    hidden_states_shp = hidden_states.shape
    new_shape = (
        hidden_states_shp[0],
        hidden_states_shp[1],
        num_attention_heads,
        attention_head_size,
    )
    hidden_states = hidden_states.reshape(new_shape).transpose(axes=(0, 2, 1, 3))
    return Model.create(name=name, output=hidden_states)


def segformer_efficient_self_attention(
    hidden_size,
    num_attention_heads,
    sequence_reduction_ratio,
    *,
    name: str | None = None,
):
    hidden_states = IOKey("input", type=Tensor[float])
    height = IOKey("height")
    width = IOKey("width")

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"The hidden size ({hidden_size}) is not a multiple of"
            f"the number of attention heads ({num_attention_heads})"
        )
    # NOTE: "height" and "width" keys are only used if sequence_reduction_ratio > 1.
    # We have to add them to the model regardless of sequence_reduction_ratio
    # to keep the API consistent.
    height = Buffer()(input=height)
    width = Buffer()(input=width)

    attention_head_size = int(hidden_size / num_attention_heads)
    all_head_size = num_attention_heads * attention_head_size

    query = Linear(hidden_size, name="query")(input=hidden_states)
    query_layer = transpose_for_scores(num_attention_heads, attention_head_size)(
        input=query
    )

    if sequence_reduction_ratio > 1:
        shape = hidden_states.shape
        batch_size = shape[0]
        num_channels = shape[2]
        hidden_states = hidden_states.transpose(axes=(0, 2, 1)).reshape(
            (batch_size, num_channels, height, width)
        )
        hidden_states = Convolution2D(
            kernel_size=sequence_reduction_ratio,
            out_channels=hidden_size,
            stride=sequence_reduction_ratio,
            name="sr",
        )(input=hidden_states)
        hidden_states = hidden_states.reshape((batch_size, num_channels, -1)).transpose(
            axes=(0, 2, 1)
        )
        hidden_states = LayerNorm(name="layer_norm")(input=hidden_states)

    key = Linear(hidden_size, name="key")(input=hidden_states)
    key_layer = transpose_for_scores(num_attention_heads, attention_head_size)(
        input=key
    )
    value = Linear(hidden_size, name="value")(input=hidden_states)
    value_layer = transpose_for_scores(num_attention_heads, attention_head_size)(
        input=value
    )

    attention_scores = query_layer @ key_layer.transpose(axes=(0, 1, 3, 2))
    attention_scores = attention_scores / (attention_head_size**0.5)
    attention_probs = Softmax()(input=attention_scores)
    context_layer = attention_probs @ value_layer
    context_layer = context_layer.transpose(axes=(0, 2, 1, 3))
    context_shp = context_layer.shape
    new_context_layer_shape = (context_shp[0], context_shp[1], all_head_size)
    # new_context_layer_shape = context_layer.shape[:-2] + (all_head_size,)
    context_layer = context_layer.reshape(new_context_layer_shape)
    return Model.create(
        name=name, output=context_layer, height_out=height, width_out=width
    )


def segformer_attention(
    hidden_size,
    num_attention_heads,
    sequence_reduction_ratio,
    *,
    name: str | None = None,
):
    hidden_states = IOKey("input", type=Tensor[float])
    height = IOKey("height")
    width = IOKey("width")

    self_output, _, _ = segformer_efficient_self_attention(
        hidden_size, num_attention_heads, sequence_reduction_ratio, name="self"
    )(input=hidden_states, height=height, width=width)
    self_output = segformer_self_output(hidden_size, name="output")(input=self_output)
    return Model.create(name=name, output=self_output)


def segformer_layer(
    hidden_size,
    num_attention_heads,
    sequence_reduction_ratio,
    mlp_ratio,
    *,
    name: str | None = None,
):
    hidden_states = IOKey("input", type=Tensor[float])
    height = IOKey("height")
    width = IOKey("width")

    attention_input = LayerNorm(name="layer_norm_1")(input=hidden_states)
    attention_output = segformer_attention(
        hidden_size, num_attention_heads, sequence_reduction_ratio, name="attention"
    )(input=attention_input, height=height, width=width)
    hidden_states = attention_output + hidden_states
    mlp_input = LayerNorm(name="layer_norm_2")(input=hidden_states)
    mlp_hidden_size = int(hidden_size * mlp_ratio)
    mlp_output = segformer_mixffn(hidden_size, mlp_hidden_size, name="mlp")(
        input=mlp_input, height=height, width=width
    )
    layer_output = mlp_output + hidden_states
    return Model.create(name=name, output=layer_output)


def segformer_overlap_patch_embeddings(
    patch_size, stride, hidden_size, *, name: str | None = None
):
    pixel_values = IOKey("input", type=Tensor[float])
    embeddings = Convolution2D(
        patch_size, hidden_size, stride, patch_size // 2, name="proj"
    )(input=pixel_values)
    shape = embeddings.shape
    height, width = shape[2], shape[3]
    embeddings = Flatten(start_dim=2)(input=embeddings).transpose(axes=(0, 2, 1))
    embeddings = LayerNorm(name="layer_norm")(input=embeddings)
    return Model.create(name=name, embeddings=embeddings, height=height, width=width)


def segformer_block(
    hidden_size,
    num_attention_heads,
    depth,
    sequence_reduction_ratio,
    mlp_ratio,
    *,
    name: str | None = None,
) -> Model:
    hidden_states = IOKey("input", type=Tensor[float])
    height = IOKey("height")
    width = IOKey("width")

    for i in range(depth):
        hidden_states = segformer_layer(
            hidden_size,
            num_attention_heads,
            sequence_reduction_ratio,
            mlp_ratio,
            name=f"{i}",
        )(input=hidden_states, height=height, width=width)
    return Model.create(name=name, output=hidden_states)


def segformer_encoder(config, *, name: str | None = None):
    pixel_values = IOKey("input", type=Tensor[float])
    hidden_states = pixel_values
    num_encoder_blocks = config.num_encoder_blocks
    batch_size = pixel_values.shape[0]
    all_hidden_states = []
    for i in range(num_encoder_blocks):
        # Patch Embeddings.
        hidden_states, height, width = segformer_overlap_patch_embeddings(
            config.patch_sizes[i],
            config.strides[i],
            config.hidden_sizes[i],
            name=f"patch_embeddings_{i}",
        )(input=hidden_states)
        # Transformers.
        hidden_states = segformer_block(
            config.hidden_sizes[i],
            config.num_attention_heads[i],
            config.depths[i],
            config.sr_ratios[i],
            config.mlp_ratios[i],
            name=f"block_{i}",
        )(input=hidden_states, height=height, width=width)
        # Layer Norm.
        hidden_states = LayerNorm(name=f"layer_norm_{i}")(input=hidden_states)
        # Reshape back to (batch_size, num_channels, height, width).
        if i != num_encoder_blocks - 1 or (
            i == num_encoder_blocks - 1 and config.reshape_last_stage
        ):
            hidden_states = hidden_states.reshape(
                (batch_size, height, width, -1)
            ).transpose(axes=(0, 3, 1, 2))
        all_hidden_states.append(hidden_states)

    hidden_states_tuple = ToTuple(n=num_encoder_blocks)(*all_hidden_states)
    return Model.create(
        name=name, output=hidden_states, hidden_states=hidden_states_tuple
    )
