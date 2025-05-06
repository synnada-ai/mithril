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


import mithril as ml
from mithril import IOKey
from mithril.models import (
    Arange,
    ArgMax,
    Buffer,
    Cast,
    Concat,
    Convolution2D,
    Embedding,
    Flatten,
    GroupNorm,
    LayerNorm,
    Linear,
    Max,
    MaxPool2D,
    Mean,
    Model,
    Power,
    Randn,
    Relu,
    Reshape,
    ScaledDotProduct,
    Sigmoid,
    Sqrt,
    Sum,
    Tensor,
    Transpose,
    Where,
    ZerosLike,
)

backend_torch = ml.TorchBackend()


def layer_norm(name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    block |= Cast().connect(input=input, dtype=ml.float32)
    block |= LayerNorm().connect(
        input=block.cout, weight=IOKey("weight"), bias=IOKey("bias")
    )  # type: ignore
    block |= Cast().connect(dtype=input.dtype(), input=block.cout, output="output")  # type: ignore
    block.expose_keys("output")
    block.set_cin("input")
    return block


def quick_gelu(name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    block |= Sigmoid().connect((1.702 * input), output="sigmoid")
    block |= Buffer().connect(input * block.sigmoid, output="output")  # type: ignore
    block.expose_keys("output")
    return block


def multi_head_attention(
    d_model: int, n_head: int, use_attn_mask: bool = False, name: str | None = None
):
    block = Model(name=name)
    assert d_model % n_head == 0, "d_model is not divisible by h"
    queries = IOKey("queries")
    head_dim = d_model // n_head
    B, L = queries.shape[0], queries.shape[1]
    block |= Linear(3 * d_model, name="in_proj").connect(queries, output="in_proj")

    in_proj = (
        block.in_proj.reshape((B, L, 3, -1))  # type: ignore
        .reshape((1, B, L, 3, d_model))
        .transpose((3, 1, 2, 0, 4))
        .reshape((3, B, L, -1))
    )

    queries = (
        in_proj[0, :, :, :].reshape((B, L, n_head, head_dim)).transpose((1, 2, 0, 3))
    )
    keys = in_proj[1, :, :, :].reshape((B, L, n_head, head_dim)).transpose((1, 2, 0, 3))
    values = (
        in_proj[2, :, :, :].reshape((B, L, n_head, head_dim)).transpose((1, 2, 0, 3))
    )

    if use_attn_mask:
        block |= (mask_model := build_attention_mask())
        block |= ScaledDotProduct(is_causal=False, use_attn_mask=True).connect(
            query=queries,
            key=keys,
            value=values,
            attn_mask=mask_model.cout,
            output="attention",
        )
    else:
        block |= ScaledDotProduct(is_causal=False, use_attn_mask=False).connect(
            query=queries, key=keys, value=values, output="attention"
        )
    block |= Buffer().connect(input=block.attention, output="buffer_output")  # type: ignore
    values_hat = block.attention.transpose((2, 0, 1, 3)).reshape((B * L, d_model))  # type: ignore
    block |= Linear(d_model, name="out_proj").connect(values_hat, output="out")
    block |= Buffer().connect(
        input=block.out.reshape((B, L, d_model)),  # type: ignore
        output="output",
    )
    block.expose_keys("output")
    return block


def mlp_resblock(d_model: int, name: str | None = None):
    block = Model(name=name)
    block |= Linear(d_model * 4, name="c_fc").connect(
        input="input", output="c_fc_output"
    )
    block |= quick_gelu(name="gelu").connect(
        input=block.c_fc_output,  # type: ignore
        output="gelu_output",
    )
    block |= Linear(d_model, name="c_proj").connect(
        input=block.gelu_output,  # type: ignore
        output="output",
    )
    block.expose_keys("output")
    return block


def residual_attention_block(
    d_model: int, n_head: int, use_attn_mask: bool = False, name: str | None = None
):
    block = Model(name=name)
    assert d_model % n_head == 0, "d_model is not divisible by h"
    input = IOKey("input")
    block += layer_norm(name="ln_1").connect(input="input", output="ln_1")

    attn = multi_head_attention(d_model, n_head, use_attn_mask, name="attn")
    block |= attn.connect(queries=block.ln_1, output="attention")  # type: ignore

    block |= layer_norm(name="ln_2").connect(
        input=input + block.attention,  # type: ignore
        output="ln_2",
    )
    mlp = mlp_resblock(d_model, name="mlp")

    block |= mlp.connect(input=block.ln_2, output="mlp_output")  # type: ignore

    result = input + block.attention + block.mlp_output  # type: ignore
    block |= Buffer().connect(result, output="output")
    block.expose_keys("output")
    return block


def seq_resblocks(
    width: int,
    layers: int,
    heads: int,
    use_attn_mask: bool = False,
    name: str | None = None,
):
    block = Model(name=name)
    input_key = "input"
    for idx in range(layers):
        block |= residual_attention_block(
            width, heads, use_attn_mask=use_attn_mask, name=f"{idx}"
        ).connect(input=input_key, output=f"attn_output_{idx}")
        input_key = f"attn_output_{idx}"
    block |= Buffer().connect(input=f"attn_output_{idx}", output="output")
    block.expose_keys("output")
    return block


def transformer(
    width: int,
    layers: int,
    heads: int,
    use_attn_mask: bool = False,
    name: str | None = None,
):
    block = Model(name=name)
    input = IOKey("input")

    resblocks = seq_resblocks(
        width=width,
        layers=layers,
        heads=heads,
        use_attn_mask=use_attn_mask,
        name="resblocks",
    )
    block |= resblocks.connect(input=input, output="resblocks_output")

    block |= Buffer().connect(input=block.resblocks_output, output="output")  # type: ignore
    block.expose_keys("output")

    return block


def vision_transformer(
    input_resolution: int,
    patch_size: int,
    width: int,
    layers: int,
    heads: int,
    output_dim: int,
    use_proj: bool = False,
    name: str | None = None,
):
    block = Model(name=name)
    input = IOKey("input")

    block |= Convolution2D(
        kernel_size=patch_size,
        out_channels=width,
        stride=patch_size,
        use_bias=False,
        name="conv1",
    ).connect(input=input, output="conv1")
    shape_conv1 = block.conv1.shape  # type: ignore
    block |= Reshape().connect(
        shape=(shape_conv1[0], shape_conv1[1], -1),
        input=block.conv1,  # type: ignore
        output="conv1_r",
    )
    conv1_rt = block.conv1_r.transpose((0, 2, 1))  # type: ignore
    conv1_rt_shape = conv1_rt.shape

    # TODO: Implement zeros primitive and replace it with following two lines.
    block |= Randn().connect(
        shape=(conv1_rt_shape[0], 1, conv1_rt_shape[-1]), output="rand_1"
    )
    block |= ZerosLike().connect(input=block.rand_1, output="zeros_out")  # type: ignore
    class_embedding = IOKey("class_embedding", differentiable=True, shape=[width])

    block |= Concat(axis=1).connect(
        input=[class_embedding + block.zeros_out, conv1_rt],  # type: ignore
        output="cat1",
    )
    positional_embedding = IOKey(
        "positional_embedding",
        differentiable=True,
        shape=[(input_resolution // patch_size) ** 2 + 1, width],
    )

    block |= layer_norm(name="ln_pre").connect(
        input=block.cat1 + positional_embedding,  # type: ignore
        output="ln_1",
    )
    block.set_shapes(positional_embedding=["a", "b"], cat1=["n", "a", "b"])
    transformer_visual = transformer(width, layers, heads, name="transformer")

    block |= transformer_visual.connect(
        input=block.ln_1.transpose((1, 0, 2)),  # type: ignore
        output="transformer",
    )
    block |= Transpose(axes=(1, 0, 2)).connect(
        input=block.transformer,  # type: ignore
        output="transformer_p",
    )
    block |= layer_norm(name="ln_post").connect(
        input=block.transformer_p,  # type: ignore
        output="ln_post",
    )
    if use_proj:
        block |= Buffer().connect(
            block.ln_post[:, 0, :]  # type: ignore
            @ IOKey("proj", differentiable=True, shape=(width, output_dim)),
            output="output",
        )
        block.expose_keys("output")
        return block

    block |= Buffer().connect(input=block.ln_post[:, 0, :], output="output")  # type: ignore
    block.expose_keys("output")
    return block


def multi_head_attention_forward(
    embed_dim_to_check: int, num_heads: int, dropout_p: float
):
    block = Model()
    query = IOKey("query", type=ml.Tensor)
    key = IOKey("key", type=ml.Tensor)
    value = IOKey("value", type=ml.Tensor)
    q_proj_weight = IOKey("q_proj_weight", type=ml.Tensor)
    k_proj_weight = IOKey("k_proj_weight", type=ml.Tensor)
    v_proj_weight = IOKey("v_proj_weight", type=ml.Tensor)
    in_proj_bias = IOKey("in_proj_bias", type=ml.Tensor)
    out_proj_weight = IOKey("out_proj_weight", type=ml.Tensor)
    out_proj_bias = IOKey("out_proj_bias", type=ml.Tensor)

    tgt_len, bsz, embed_dim = query.shape[0], query.shape[1], query.shape[2]

    # assert embed_dim == embed_dim_to_check, "Embedding dimension mismatch."

    head_dim = embed_dim // num_heads
    # assert (head_dim * num_heads) == embed_dim,
    # "embed_dim must be divisible by num_heads"

    q = query @ q_proj_weight.transpose() + in_proj_bias[0:embed_dim]  # type: ignore
    block |= Buffer().connect(input=q)
    k = (
        key @ k_proj_weight.transpose()
        + in_proj_bias[embed_dim_to_check : 2 * embed_dim_to_check]
    )
    block |= Buffer().connect(input=k)

    v = (
        value @ v_proj_weight.transpose()
        + in_proj_bias[2 * embed_dim_to_check : 3 * embed_dim_to_check]
    )
    block |= Buffer().connect(input=v)

    q_r = q.reshape((tgt_len, bsz * num_heads, head_dim)).transpose((1, 0, 2))
    block |= Buffer().connect(input=q_r)

    k_r = k.reshape((-1, bsz * num_heads, head_dim)).transpose((1, 0, 2))
    block |= Buffer().connect(input=k_r)

    v_r = v.reshape((-1, bsz * num_heads, head_dim)).transpose((1, 0, 2))
    block |= Buffer().connect(input=v_r)

    block |= ScaledDotProduct(is_causal=False).connect(
        query=q_r, key=k_r, value=v_r, output="attention"
    )

    attn_output = block.attention.transpose((1, 0, 2)).reshape(  # type: ignore
        (tgt_len, bsz, embed_dim)
    )
    attn_output = attn_output @ out_proj_weight.transpose() + out_proj_bias

    block |= Buffer().connect(input=attn_output, output="output")
    block.expose_keys("output")

    return block


def attention_pool2d(
    spacial_dim: int,
    embed_dim: int,
    num_heads: int,
    output_dim: int | None = None,
    name: str | None = None,
):
    block = Model(name=name)
    input = IOKey("input")
    output_dim = output_dim or embed_dim
    """
    self.positional_embedding = 
      nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5) 
    self.k_proj = nn.Linear(embed_dim, embed_dim)
    self.q_proj = nn.Linear(embed_dim, embed_dim)
    self.v_proj = nn.Linear(embed_dim, embed_dim)
    self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
    """
    q_proj_weight = IOKey(
        "q_proj_weight", differentiable=True, shape=(embed_dim, embed_dim)
    )
    k_proj_weight = IOKey(
        "k_proj_weight", differentiable=True, shape=(embed_dim, embed_dim)
    )
    v_proj_weight = IOKey(
        "v_proj_weight", differentiable=True, shape=(embed_dim, embed_dim)
    )
    q_proj_bias = IOKey("q_proj_bias", differentiable=True, shape=(embed_dim,))
    k_proj_bias = IOKey("k_proj_bias", differentiable=True, shape=(embed_dim,))
    v_proj_bias = IOKey("v_proj_bias", differentiable=True, shape=(embed_dim,))
    block |= Concat().connect(
        input=[q_proj_bias, k_proj_bias, v_proj_bias], output="in_proj_bias"
    )
    out_proj_weight = IOKey(
        "c_proj_weight", differentiable=True, shape=(output_dim, embed_dim)
    )
    out_proj_bias = IOKey(
        "c_proj_bias", differentiable=True, type=ml.Tensor, shape=(output_dim,)
    )
    positional_embedding = IOKey(
        "positional_embedding",
        differentiable=True,
        type=ml.Tensor,
        shape=(spacial_dim**2 + 1, embed_dim),
    )
    block |= Flatten(start_dim=2).connect(input=input, output="flatten_output")
    block |= Transpose(axes=(2, 0, 1)).connect(
        input=block.flatten_output,  # type: ignore
        output="transpose_output",
    )
    block |= Mean(axis=0, keepdim=True).connect(
        input=block.transpose_output,  # type: ignore
        output="mean_output",
    )
    block |= Concat(axis=0).connect(
        input=[block.mean_output, block.transpose_output],  # type: ignore
        output="cn1",
    )
    block |= Buffer().connect(block.cn1 + positional_embedding[:, None, :], output="x")  # type: ignore

    _multi_head_attention_forward = multi_head_attention_forward(
        embed_dim, num_heads, 0.0
    )

    block |= _multi_head_attention_forward.connect(
        query=block.x[:1],  # type: ignore
        key=block.x,  # type: ignore
        value=block.x,  # type: ignore
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        in_proj_bias=block.in_proj_bias,  # type: ignore
        out_proj_weight=out_proj_weight,
        out_proj_bias=out_proj_bias,
        output="attention",
    )
    attn_shape = block.attention.shape  # type: ignore

    block |= Reshape().connect(
        input=block.attention,  # type: ignore
        shape=(attn_shape[-2], attn_shape[-1]),
        output="output",
    )
    block.expose_keys("output")
    return block


def bottleneck(inplanes: int, planes: int, stride: int = 1, name: str | None = None):
    block = Model(name=name)
    expansion = 4
    input = IOKey("input")

    block += Convolution2D(
        kernel_size=1, out_channels=planes, use_bias=False, name="conv1"
    ).connect(input="input")
    # nn.BatchNorm2d(planes)
    block += GroupNorm(num_groups=1, name="bn1")
    block += Relu(name="relu1")
    block += Convolution2D(
        kernel_size=3, out_channels=planes, padding=1, use_bias=False, name="conv2"
    )
    # nn.BatchNorm2d(planes)
    block += GroupNorm(num_groups=1, name="bn2")
    block += Relu(name="relu2")
    if stride > 1:
        # nn.AvgPool2d(stride)
        # nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        block += MaxPool2D(stride, name="avgpool")

    block += Convolution2D(
        kernel_size=1, out_channels=planes * expansion, use_bias=False, name="conv3"
    )
    # nn.BatchNorm2d(planes)
    block += GroupNorm(num_groups=1, name="bn3").connect(output="out1")

    if stride > 1 or inplanes != planes * expansion:
        # nn.AvgPool2d(stride)
        block |= MaxPool2D(stride, name=f"downsample_{-1}").connect(
            input, output="downsample_pool"
        )
        block |= Convolution2D(
            kernel_size=1,
            out_channels=planes * expansion,
            use_bias=False,
            name=f"downsample_{0}",
        ).connect(block.downsample_pool, output="downsample_conv")  # type: ignore
        # nn.BatchNorm2d(planes)
        block |= GroupNorm(num_groups=1, name=f"downsample_{1}").connect(
            block.downsample_conv,  # type: ignore
            output="out2",
        )
        out = block.out1 + block.out2  # type: ignore

    else:
        out = block.out1 + input  # type: ignore
    block |= Relu(name="relu3").connect(out, output="output")
    block.expose_keys("output")
    block.set_cout("output")

    return block


def make_layer(inplanes, planes, blocks, stride=1, name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    block |= bottleneck(inplanes, planes, stride, name="0").connect(
        input=input, output="bottle_neck0"
    )
    _inplanes = 4 * planes
    input_key = "bottle_neck0"
    for i in range(1, blocks):
        block |= bottleneck(_inplanes, planes, name=f"{i}").connect(
            input=input_key, output=f"bottle_neck{i}"
        )
        input_key = f"bottle_neck{i}"
    block |= Buffer().connect(input=f"bottle_neck{i}", output="output")
    block.expose_keys("output")
    return block


def modified_resnet(
    layers, output_dim, heads, input_resolution=224, width=64, name: str | None = None
):
    block = Model(name=name)
    input = IOKey("input")

    # x = x.type(self.conv1.weight.dtype)
    block |= Convolution2D(
        kernel_size=3,
        out_channels=width // 2,
        stride=2,
        padding=1,
        use_bias=False,
        name="conv1",
    ).connect(input=input, output="conv_out_1")
    # nn.BatchNorm2d(width // 2)
    block |= GroupNorm(num_groups=1, name="bn1").connect(
        input="conv_out_1", output="norm_out_1"
    )
    block |= Relu(name="relu1").connect(input="norm_out_1", output="rl_out_1")
    block |= Convolution2D(
        kernel_size=3, out_channels=width // 2, padding=1, use_bias=False, name="conv2"
    ).connect(input="rl_out_1", output="conv_out_2")

    # nn.BatchNorm2d(width // 2)
    block |= GroupNorm(num_groups=1, name="bn2").connect(
        input="conv_out_2", output="norm_out_2"
    )
    block |= Relu(name="relu2").connect(input="norm_out_2", output="rl_out_2")
    block |= Convolution2D(
        kernel_size=3, out_channels=width, padding=1, use_bias=False, name="conv3"
    ).connect(input="rl_out_2", output="conv_out_3")

    # nn.BatchNorm2d(width)
    block |= GroupNorm(num_groups=1, name="bn3").connect(
        input="conv_out_3", output="norm_out_3"
    )
    block |= Relu(name="relu3").connect(input="norm_out_3", output="rl_out_3")
    # nn.AvgPool2d(2)

    block |= MaxPool2D(kernel_size=2, name="avgpool").connect(
        input="rl_out_3", output="avgpool_out"
    )
    make_layer_block = make_layer(width, width, layers[0], name="layer1")
    input_key = "make_layer_0"
    block |= make_layer_block.connect(input=block.avgpool_out, output=input_key)  # type: ignore

    for idx in range(1, 4):
        make_layer_block = make_layer(
            width, width * (2**idx), layers[idx], stride=2, name=f"layer{idx+1}"
        )
        block |= make_layer_block.connect(input=input_key, output=f"make_layer_{idx}")
        input_key = f"make_layer_{idx}"
    attnpool = attention_pool2d(
        input_resolution // 32, width * 32, heads, output_dim, name="attnpool"
    )
    block |= attnpool.connect(
        input=input_key,
        output="attn_output",
    )
    block |= Buffer().connect(input=block.attn_output, output="output")  # type: ignore
    block.expose_keys("output")
    return block


# for torch.tensor.norm
def norm(
    p: str | int = 2,
    axis: int | None = None,
    keepdim: bool = False,
    name: str | None = None,
):
    block = Model(name=name)
    input = IOKey("input")
    if p == "inf":
        block += Max(axis=axis, keepdim=keepdim).connect(input=input.abs())
    elif p == 1:
        block += Sum(axis=axis, keepdim=keepdim).connect(input=input.abs())
    elif p == 2:
        block += Sum(axis=axis, keepdim=keepdim).connect(input=(input**2))
        block += Sqrt()
    else:
        assert isinstance(p, int)
        block += Sum(axis=axis, keepdim=keepdim).connect(input=(input.abs() ** p))
        block += Power(exponent=(1 / p))
    block += Buffer().connect(output="output")
    block.expose_keys("output")
    return block


def clip(
    embed_dim: int,
    # vision
    image_resolution: int,
    vision_layers: tuple[int, int, int, int] | int,
    vision_width: int,
    vision_patch_size: int,
    # text
    context_length: int,
    vocab_size: int,
    transformer_width: int,
    transformer_heads: int,
    transformer_layers: int,
    name: str | None = None,
):
    block = Model(name=name)
    image = IOKey(
        "image", type=ml.Tensor, shape=["N", 3, image_resolution, image_resolution]
    )
    text = IOKey("text", type=ml.Tensor, shape=["M", context_length])

    if isinstance(vision_layers, tuple | list):
        vision_heads = vision_width * 32 // 64
        visual = modified_resnet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width,
            name="visual",
        )
        block |= visual.connect(input=image, output="image_features")

    else:
        vision_heads = vision_width // 64
        visual = vision_transformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            use_proj=True,
            name="visual",
        )
        block |= visual.connect(
            input="image",
            class_embedding="visual_class_embedding",
            positional_embedding="visual_positional_embedding",
            proj="visual_proj",
            output="image_features",
        )

    block |= Embedding(vocab_size, transformer_width, name="token_embedding").connect(
        input=text, output="token_embedding"
    )

    positional_embedding = IOKey(
        "positional_embedding",
        type=ml.Tensor,
        differentiable=True,
        shape=(context_length, transformer_width),
    )
    embedding = block.token_embedding + positional_embedding  # type: ignore
    transformer_main = transformer(
        width=transformer_width,
        layers=transformer_layers,
        heads=transformer_heads,
        use_attn_mask=True,
        name="transformer",
    )
    block |= transformer_main.connect(
        input=embedding.transpose((1, 0, 2)),  # type: ignore
        output="transformer",
    )

    block |= layer_norm(name="ln_final").connect(
        input=block.transformer.transpose((1, 0, 2)),  # type: ignore
        output="ln_final",
    )
    block |= Arange().connect(stop=block.ln_final.shape[0], output="arange_out")  # type: ignore
    block |= ArgMax(axis=-1).connect(input=text, output="argmax_out")
    block |= Buffer().connect(
        input=block.ln_final[block.arange_out, block.argmax_out],  # type: ignore
        output="eot_tokens",
    )

    # TODO: This set_shape call occurs because of a missing feature
    # in indexer_constraint. It should be removed once the feature is
    # implemented.
    block.set_shapes(
        eot_tokens=["N", transformer_width],
        ln_final=["N", context_length, transformer_width],
    )

    text_projection = IOKey(
        "text_projection",
        type=ml.Tensor,
        differentiable=True,
        shape=(transformer_width, embed_dim),
    )

    block |= Buffer().connect(
        input=block.eot_tokens @ text_projection,  # type: ignore
        output="text_features",
    )

    norm1 = norm(p=2, axis=1, keepdim=True)
    norm2 = norm(p=2, axis=1, keepdim=True)

    block |= norm1.connect(input=block.image_features, output="image_features_norm")  # type: ignore
    block |= norm2.connect(input=block.text_features, output="text_features_norm")  # type: ignore

    image_features = block.image_features / block.image_features_norm  # type: ignore
    text_features = block.text_features / block.text_features_norm  # type: ignore

    # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logit_scale = IOKey("logit_scale", type=ml.Tensor, differentiable=True, shape=(1,))
    logits_per_image = logit_scale.exp() * (image_features @ text_features.transpose())
    logits_per_text = logits_per_image.transpose()
    block |= Buffer().connect(input=logits_per_image, output="logits_per_image")
    block |= Buffer().connect(input=logits_per_text, output="logits_per_text")
    block.expose_keys("logits_per_image", "logits_per_text")

    return block


def build_attention_mask() -> Model:
    block = Model()
    block |= Arange(stop=77).connect(output="arange_out_1")
    block |= Arange(stop=77).connect(output="arange_out_2")
    upper_bool_triu = block.arange_out_1[..., None] >= block.arange_out_2[None, ...]  # type: ignore
    block |= Where().connect(
        cond=upper_bool_triu,
        input1=Tensor(0.0),
        input2=Tensor(float("-inf")),
        output="output",
    )
    block.expose_keys("output")
    return block
