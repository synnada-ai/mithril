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

import sys
from copy import deepcopy

import pytest

import mithril as ml
from mithril import IOKey
from mithril.framework.physical.model import FlatModel
from mithril.models import (
    Add,
    Arange,
    AtLeast1D,
    Buffer,
    Concat,
    Convolution2D,
    GroupNorm,
    Linear,
    Model,
    Multiply,
    Power,
    Reshape,
    ScaledDotProduct,
    Square,
    Tensor,
    ToList,
    Transpose,
    Where,
    functional,
)

from .helper import assert_models_equal


def test_extend_canonicals_for_main_model():
    block = Model()
    buffer = Buffer()
    input = IOKey()
    block |= buffer.connect(input=input)
    assert block.cout.metadata == buffer.output.metadata
    input.sqrt()
    assert block.cout.metadata == buffer.output.metadata


def test_extend_canonicals_for_extract_model():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    output = input1 * input2
    assert output.model is not None
    mult_model = list(output.model.dag.keys())[0]
    assert output.model.cout.metadata == mult_model.output.metadata  # type: ignore


def test_extend_two_connections():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    output = input1 * input2

    model = Model()
    model += Multiply().connect(left="input1", right="input2")
    assert output.model is not None
    assert_models_equal(output.model, model)


def test_extend_error_shp_mismatch():
    input1 = IOKey("input1", shape=[3, 3])
    input2 = IOKey("input2", shape=[3, 2])
    with pytest.raises(ValueError) as err:
        input1 * input2

    assert (
        str(err.value)
        == "Inputs shape mismatch at dimension 1. Shapes are inconsistent."
    )


def test_extend_and_extraction():
    input1 = IOKey()
    input2 = IOKey()
    input3 = IOKey()
    mult_output = input1 * input2
    output = mult_output + input3

    model = Model()
    model |= (mult := Multiply())
    model |= Add().connect(left=mult.output)
    assert output.model is not None
    assert_models_equal(output.model, model)


def test_extend_and_extraction_named():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    input3 = IOKey("input3")
    mult_output = input1 * input2
    output = mult_output + input3

    model = Model()
    model |= (mult := Multiply()).connect(left="input1", right="input2")
    model |= Add().connect(left=mult.output, right="input3")
    assert output.model is not None
    assert_models_equal(output.model, model)


def test_extend_and_extraction_via_extend_api():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    input3 = IOKey("input3")
    mult_output = input1 * input2
    model1 = Model()
    model1 |= Add().connect(left=mult_output, right=input3)

    model2 = Model()
    model2 |= (mult := Multiply()).connect(left="input1", right="input2")
    model2 |= Add().connect(left=mult.output, right="input3")
    assert_models_equal(model1, model2)


def test_extend_connection_with_model():
    add = Add()
    input1 = IOKey()
    output = add.output * input1

    model = Model()
    model |= (add2 := Add())
    model |= Multiply().connect(add2.output)
    assert output.model is not None
    assert_models_equal(output.model, model)


def test_extend_multiple_models():
    add = Add()
    input1 = IOKey()
    add.output * input1

    input2 = IOKey()
    output2 = add.output * input2

    model = Model()
    model |= (add2 := Add())
    model |= Multiply().connect(add2.output)
    model |= Multiply().connect(add2.output)
    assert output2.model is not None
    assert_models_equal(output2.model, model)


def test_extend_to_model_connection_nested():
    add = Add()
    m1 = Model()
    m1 |= add
    m2 = Model()
    m2 |= m1
    m3 = Model()
    m3 |= m2

    input1 = IOKey()
    output = add.output * input1
    model = Model()
    model |= m3
    model |= Buffer().connect(output)

    _add = Add()
    _m1 = Model()
    _m1 |= _add
    _m2 = Model()
    _m2 |= _m1
    _m3 = Model()
    _m3 |= _m2

    _model = Model()
    _model |= _m3
    _model |= (mult := Multiply()).connect(_add.output)
    _model |= Buffer().connect(mult.output)
    assert_models_equal(model, _model)


def test_extend_and_extraction_same_inputs():
    input1 = IOKey()
    input2 = IOKey()
    add_output = input1 + input2
    mult_output = input1 * input2
    assert add_output.model == mult_output.model == input1.model == input2.model

    _input1 = IOKey()
    _input2 = IOKey()

    model = Model()
    model |= Add().connect(left=_input1, right=_input2)
    model |= Multiply().connect(left=_input1, right=_input2)
    assert_models_equal(model, mult_output.model)  # type: ignore


def test_extend_extraction_frozen_models():
    add_output = Add().output * Add().output
    mult_output = Add().output * Add().output
    output = add_output + mult_output

    model = Model()
    model |= (add1 := Add())
    model |= (add2 := Add())
    model |= (mult1 := Multiply()).connect(left=add1.output, right=add2.output)
    model |= (add3 := Add())
    model |= (add4 := Add())
    model |= (mult2 := Multiply()).connect(left=add3.output, right=add4.output)
    model |= Add().connect(left=mult1.output, right=mult2.output)
    assert output.model is not None
    assert_models_equal(model, output.model)


def test_extend_extraction_immediate_values():
    model = Model()
    model |= (add := Add())
    output = add.output + 2
    model |= Buffer().connect(output)

    model1 = Model()
    model1 |= (add1 := Add())
    model1 |= (add2 := Add()).connect(left=add1.output, right=2)
    model1 |= Buffer().connect(add2.output)

    assert output.model is not None
    assert_models_equal(model1, model)


def test_extend_single_frozen_single_non_frozen_model():
    model1 = Model()
    model1 |= (add1 := Add())
    model1._freeze()

    model2 = Model()
    model2 |= (add2 := Add())
    model2 |= Buffer().connect(add1.output * add2.output)

    _model1 = Model()
    _model1 |= (_add1 := Add())
    _model1._freeze()

    _model2 = Model()
    _model2 |= _model1
    _model2 |= (_add2 := Add())
    _model2 |= (mult := Multiply()).connect(left=_add1.output, right=_add2.output)
    _model2 |= Buffer().connect(mult.output)

    assert_models_equal(_model2, model2)


def test_extend_test_extend_multiple_non_frozen_models_error():
    model = Model()
    model |= (add := Add())

    model1 = Model()
    model1 |= (add2 := Add())

    with pytest.raises(ValueError) as err:
        add.output + add2.output
    assert str(err.value) == "Multiple non-frozen active models found in connections!"


def test_extend_test_extend_multiple_non_frozen_models_with_connection_error():
    out1 = IOKey("out1")
    out2 = IOKey("out2")

    model1 = Model()
    model1 |= Add().connect(output=out1)
    model2 = Model()
    model2 |= Add().connect(output=out2)

    with pytest.raises(ValueError) as err:
        out1 + out2
    assert str(err.value) == "Multiple non-frozen active models found in connections!"


def test_extend_non_frozen_model_and_frozen_model():
    out1 = IOKey("out1")
    out2 = IOKey("out2")

    model1 = Model()
    model1 |= Add().connect(output=out1)
    model2 = Model()
    model2 |= Add().connect(output=out2)
    model2._freeze()

    output = out1 + out2
    model1 |= Buffer().connect(output)

    _out1 = IOKey("out1")
    _out2 = IOKey("out2")
    _model1 = Model()
    _model1 |= (add := Add()).connect(output=_out1)
    _model2 = Model()
    _model2 |= Add().connect(output=_out2)

    _model1 |= _model2
    _model1 |= (add := Add()).connect(_out1, _out2)
    _model1 |= Buffer().connect(add.output)
    assert_models_equal(model1, _model1)  # type: ignore


def test_extend_check_metadata():
    weight_key = IOKey("weight")
    t_w = weight_key.transpose()
    m = Model()
    m |= Buffer().connect(t_w)
    assert list(m.dag.keys())[0].input.metadata == m.weight.metadata  # type: ignore

    model = Model()
    model |= m.connect(weight=IOKey("weight"))
    assert list(m.dag.keys())[0].input.metadata == m.weight.metadata  # type: ignore

    _weight_key = IOKey("weight")
    _m = Model()
    _m |= Transpose().connect(_weight_key)
    _m += Buffer()
    assert list(_m.dag.keys())[0].input.metadata == _m.weight.metadata  # type: ignore

    _model = Model()
    _model |= _m.connect(weight=IOKey("weight"))
    assert list(_m.dag.keys())[0].input.metadata == _m.weight.metadata  # type: ignore

    assert_models_equal(model, _model)


def test_extend_metadata_linear():
    lin1 = Linear()
    assert list(lin1.dag.keys())[0].input.metadata is lin1.weight.metadata  # type: ignore

    model = Model()
    model += lin1.connect(weight=IOKey("w"))
    assert list(lin1.dag.keys())[0].input.metadata is lin1.weight.metadata  # type: ignore
    assert lin1.weight.metadata is model.w.metadata  # type: ignore


def test_extend_provisional_model():
    model = Model()
    model |= Add().connect(left="left", right="right", output="output")
    _model = deepcopy(model)
    pow = model.output**2  # type: ignore
    assert_models_equal(model, _model)

    assert pow.model.provisional_source == model
    buf_model = Buffer()
    model |= buf_model.connect(pow)

    model2 = Model()
    model2 |= Add().connect(left="left", right="right", output="output")
    model2 |= Power().connect(model2.output, 2)  # type: ignore
    model2 += Buffer()
    assert_models_equal(model, model2)


def test_extend_concat():
    model = Model()
    model |= (buff1 := Buffer())
    model |= (buff2 := Buffer())
    buff1_1d = buff1.output.atleast_1d()
    buff2_1d = buff2.output.atleast_1d()
    model |= Concat().connect(input=[buff1_1d, buff2_1d])

    _model = Model()
    _model |= (_buff1 := Buffer())
    _model |= (_buff2 := Buffer())
    _model |= (_buff1_1d := AtLeast1D()).connect(_buff1.output)
    _model |= (_buff2_1d := AtLeast1D()).connect(_buff2.output)
    _model |= (list_m := ToList(2)).connect(
        input1=_buff1_1d.output, input2=_buff2_1d.output
    )
    _model |= Concat().connect(input=list_m.output)
    assert_models_equal(model, _model)


def test_extend_only_dependent_submodels():
    model = Model()
    model |= (buff1 := Buffer())
    a = buff1.output**2
    b = buff1.output / 3
    c = a + 4
    assert a.model is b.model is c.model and a.model is not None
    provisional_model = b.model
    assert provisional_model is not None
    dag = provisional_model.dag
    assert {m.__class__.__name__ for m in dag} == {"PowerOp", "AddOp", "DivideOp"}

    model |= Buffer().connect(c)

    assert b.model is provisional_model

    _model = Model()
    _model |= (buff := Buffer())
    _model |= (pow := Power()).connect(buff.output, 2)
    _model |= (add := Add()).connect(pow.output, 4)
    _model |= Buffer().connect(add.output)
    assert_models_equal(model, _model)


def test_extend_merge_while_provisional_model_created():
    model = Model()
    model |= (add := Add())
    a = add.output**2
    _ = add.output / 3
    c = a + 4
    model.merge_connections(add.left, add.right)
    model |= Buffer().connect(c)

    con = IOKey()
    _model = Model()
    _model |= (add := Add()).connect(con, con)
    _model |= (pow := Power()).connect(add.output, 2)
    _model |= (add := Add()).connect(pow.output, 4)
    _model |= Buffer().connect(add.output)
    assert_models_equal(model, _model)


def test_extend_error_by_constraint_solver():
    model = Model()
    buff = Buffer()
    model |= buff.connect(input="input1")
    model |= Add().connect(buff.output, IOKey(shape=[4, 4]))
    with pytest.raises(ValueError) as err:
        t = buff.input.T
        t.set_shapes([3, 4, 5])
    assert str(err.value) == "Possible values mismatch!"


def test_extend_error_by_constraint_solver_nested_model():
    model = Model()
    buff = Buffer()
    model |= buff.connect(input="input1")
    model |= Add().connect(buff.output, IOKey(shape=[4, 4]))
    parent_m = Model()
    parent_m |= model
    grand_parent_m = Model()
    grand_parent_m |= parent_m

    with pytest.raises(ValueError) as err:
        t = buff.input.T
        t.set_shapes([3, 4, 5])
    assert str(err.value) == "Possible values mismatch!"


def test_immediate_extend_integration():
    model = Model()
    query = IOKey("query", type=ml.Tensor)
    key = IOKey("key", type=ml.Tensor)
    bsz = query.transpose()

    q = query + 2
    model |= Buffer().connect(input=q)
    _key = key + 1
    k_r = _key + bsz
    model |= Buffer().connect(input=k_r)

    for con in model.conns.input_connections:
        assert con.model is model


def test_immediate_extend_integration_reshape():
    model = Model()
    queries = IOKey("queries")
    B = queries.shape[1]
    model |= Linear().connect(queries, output="in_proj")

    _ = model.in_proj.reshape((B, B, 3, -1))  # type: ignore

    for con in model.conns.input_connections:
        assert con.model is model


def test_immediate_extend_integration_str_matching():
    block = Model()
    input = IOKey("input")
    block += Buffer().connect(input="input", output="b_out")

    block |= Buffer().connect(input=input + block.b_out)  # type: ignore

    result = block.b_out + input  # type: ignore
    block |= Buffer().connect(result, output=IOKey("output"))

    for con in block.conns.input_connections:
        assert con.model is block


def test_immediate_extend_integration_str_matching2():
    block = Model()
    input = IOKey("input")
    block |= Buffer().connect(input="input", output="b_out")
    block |= Buffer().connect(input=input, output="b_odsfut")


def test_apply_rope():
    block = Model()
    # We define the input connections
    xq = IOKey("xq", type=ml.Tensor)
    freqs_cis = IOKey("freqs_cis", type=ml.Tensor)

    xq_shape = xq.shape
    a, b, c, d = freqs_cis[..., 0], xq[..., 0], freqs_cis[..., 1], xq[..., 1]
    e = a * b
    f = c * d
    _ = e + f

    block |= Reshape().connect(shape=xq_shape, output="xq_out_raw")

    for con in block.conns.input_connections:
        assert con.model is block


def apply_rope(*, name: str | None = None) -> Model:
    block = Model(name=name)
    # We define the input connections
    xq = IOKey("xq", type=ml.Tensor)
    freqs_cis = IOKey("freqs_cis", type=ml.Tensor)

    xq_shape = xq.shape
    # Do the math
    a = (_a1 := freqs_cis[..., 0]) * (_a2 := xq[..., 0])
    b = freqs_cis[..., 1] * xq[..., 1]
    xq_out = a + b

    block |= Reshape().connect(xq_out, shape=xq_shape, output="xq_out_raw")
    return block


def test_apply_rope_2():
    block = apply_rope()
    for con in block.conns.input_connections:
        assert con.model is block


def build_attention_mask() -> Model:
    block = Model()
    block |= Arange(stop=77).connect(output="arange_out_1")
    block |= Arange(stop=77).connect(output="arange_out_2")
    upper_bool_triu = block.arange_out_1[..., None] >= block.arange_out_2[None, ...]  # type: ignore
    block |= Where().connect(
        cond=upper_bool_triu,
        input1=Tensor(0.0),
        input2=Tensor(float("-inf")),
        output=IOKey("output"),
    )
    return block


def test_multihead():
    d_model = 768
    n_head = 12
    block = Model()
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

    block |= (mask_model := build_attention_mask())
    block |= ScaledDotProduct(is_causal=False, use_attn_mask=True).connect(
        query=queries,
        key=keys,
        value=values,
        attn_mask=mask_model.cout,
        output="attention",
    )
    _ = B * L
    for con in block.conns.input_connections:
        assert con.model is block


def test_extend_new_model_with_provisional_ref_count():
    submodel = Model()
    submodel |= (mult := Multiply())
    mult.output + 3
    sub_pro_model = submodel.provisional_model
    assert isinstance(sub_pro_model, Model)
    model = Model()
    model |= (buff := Buffer())
    buff.output - 2
    pro_model = model.provisional_model
    model |= submodel
    assert submodel.provisional_model is None
    assert sub_pro_model.provisional_source is False
    assert pro_model is model.provisional_model
    assert isinstance(model.provisional_model, Model)
    assert len(model.provisional_model.dag) == 2  # AddOp and SubtractOp
    assert sys.getrefcount(sub_pro_model.conns) == 4


def test_extend_new_model_with_provisional_model_connection_ref_count():
    submodel = Model()
    submodel |= (mult := Multiply())
    mult.output + 3
    sub_pro_model = submodel.provisional_model
    assert sys.getrefcount(sub_pro_model) == 7
    model = Model()
    model |= (buff := Buffer())
    buff.output - 2
    model |= submodel
    assert sys.getrefcount(sub_pro_model) == 2


def test_extend_child_provisional_extraction():
    submodel = Model()
    submodel |= (mult := Multiply())
    add_output = mult.output + 3

    model = Model()
    model |= (buff := Buffer())
    buff.output - 2
    model |= submodel
    pow = add_output**2
    model |= Buffer().connect(pow)
    for con in model.conns.input_connections:
        assert con.model is model


def test_flat_model_key_naming_matching():
    # This test is added to check that the key naming is consistent.
    # Cleaning conns objects was leading an error in the FlatModel
    # because of the metadata mismatch.
    submodel = Model()
    input = IOKey("input", type=Tensor)
    submodel |= Buffer().connect(output="arange")
    omega = submodel.arange + 2  # type: ignore
    out = input[..., None] * omega
    submodel |= Buffer().connect(input=out, output="dummy_out")

    model = Model()
    input = IOKey("input")

    model |= submodel.connect(input=input)

    metadata = list(list(model.dag.keys())[0].dag.keys())[2].input.metadata  # type: ignore
    assert model.conns.get_con_by_metadata(metadata) is not None  # type: ignore
    flat_model = FlatModel(
        model,
        ml.JaxBackend().primitive_function_dict,
        short_namings=False,
    )
    assert flat_model.queued_models == {}


def test_functional_model():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = Add()(input1, input2)
    x = Multiply()(x, x)
    x = x**2  # type: ignore

    model = Model()
    model |= (add := Add()).connect("input1", "input2")
    model |= (mult := Multiply()).connect(add.output, add.output)
    model |= Power().connect(mult.output, 2)

    assert x.model is not None
    assert_models_equal(x.model, model)


def test_functional_model_with_lin():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = Add()(left=input1, right=input2)
    x = Multiply()(left=x, right=x)
    x = x**2  # type: ignore
    x = Linear()(input=x)

    model = Model()
    model |= (add := Add()).connect("input1", "input2")
    model |= (mult := Multiply()).connect(add.output, add.output)
    model |= (pow := Power()).connect(mult.output, 2)
    model |= Linear().connect(input=pow.output)

    assert_models_equal(x.model.parent, model)  # type: ignore


def test_functional_model_with_create_api():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = Add()(left=input1, right=input2)
    x = Multiply()(left=x, right=x)
    x = x**2  # type: ignore
    x = Linear()(input=x)
    f_model = Model.create(x)  # type: ignore

    model = Model()
    model |= (add := Add()).connect("input1", "input2")
    model |= (mult := Multiply()).connect(add.output, add.output)
    model |= (pow := Power(exponent=2)).connect(mult.output)
    model |= Linear().connect(input=pow.output)
    model._freeze()

    assert_models_equal(f_model, model)


def test_functional_model_with_create_api_no_key_namings():
    input1 = IOKey()
    input2 = IOKey()
    x = Add()(left=input1, right=input2)
    x = Multiply()(left=x, right=x)
    x = x**2  # type: ignore
    x = Linear()(input=x)
    f_model = Model.create(x)  # type: ignore

    model = Model()
    model |= (add := Add())
    model |= (mult := Multiply()).connect(add.output, add.output)
    model |= (pow := Power(exponent=2)).connect(mult.output)
    model |= Linear().connect(input=pow.output)
    model._freeze()

    assert_models_equal(f_model, model)


def test_functional_model_with_create_api_with_immediate_in_call():
    input1 = IOKey()
    x = Add()(left=input1, right=3)
    x = Multiply()(left=x, right=x)
    x = x**2  # type: ignore
    x = Linear()(input=x)
    f_model = Model.create(x)  # type: ignore

    model = Model()
    model |= (add := Add(right=3)).connect()
    model |= (mult := Multiply()).connect(add.output, add.output)
    model |= (pow := Power(exponent=2)).connect(mult.output)
    model |= Linear().connect(input=pow.output)
    model._freeze()

    assert_models_equal(f_model, model)


def test_functional_model_with_create_api_with_immediate_in_model_init():
    input1 = IOKey()
    x = Add(right=3)(left=input1)
    f_model = Model.create(x)  # type: ignore

    model = Model()
    model |= Add(right=3)
    model._freeze()

    assert_models_equal(f_model, model)


def test_functional_partial_model_creation():
    input = IOKey("input")
    t_input = input.transpose((0, 2, 3, 1))
    _ = t_input**2
    model = Model()
    model |= Buffer().connect(input=t_input)
    model._freeze()

    input = IOKey("input")
    t_input = Transpose(axes=(0, 2, 3, 1))(input=input)  # type: ignore
    _ = t_input**2
    b_out = Buffer()(input=t_input)
    functional_model = Model.create(b_out)  # type: ignore

    input = IOKey("input")
    t_input = input.transpose((0, 2, 3, 1))
    _ = t_input**2
    b_out = Buffer()(input=t_input)
    functional_model2 = Model.create(b_out)  # type: ignore

    assert_models_equal(functional_model, model)
    assert_models_equal(functional_model2, model)


def test_functional_model_unnamed_input_keys():
    x = IOKey("input", shape=[None, 512, None, None])
    normalized = GroupNorm(num_groups=32, eps=1e-6, name="norm")(x)
    functional_model = Model.create(normalized=normalized)  # type: ignore

    x = IOKey("input", shape=[None, 512, None, None])
    model = Model()
    model |= GroupNorm(num_groups=32, eps=1e-6, name="norm").connect(x, "normalized")
    model._freeze()

    assert_models_equal(functional_model, model)


def attn_block_functional(n_channels: int, *, name: str | None = None):
    # Keep the original input for the residual connection.
    x = IOKey("input", shape=[None, 512, None, None])
    normalized = GroupNorm(num_groups=32, eps=1e-6, name="norm")(input=x)
    query = Convolution2D(1, n_channels, name="q")(input=normalized)
    key = Convolution2D(1, n_channels, name="k")(input=normalized)
    value = Convolution2D(1, n_channels, name="v")(input=normalized)
    shape = query.shape  # type: ignore

    query = query.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))  # type: ignore
    key = key.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))  # type: ignore
    value = value.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))  # type: ignore
    sdp_out = ScaledDotProduct(is_causal=False)(query=query, key=key, value=value)

    reshaped = Reshape()(input=sdp_out, shape=(shape[0], shape[2], shape[3], shape[1]))
    transposed = Transpose(axes=(0, 3, 1, 2))(input=reshaped)
    proj_out = Convolution2D(1, n_channels, name="proj_out")(input=transposed)

    return Model.create(output=proj_out + x, name=name)  # type: ignore


def attn_block(n_channels: int, *, name: str | None = None):
    block = Model(name=name)
    block |= GroupNorm(num_groups=32, eps=1e-6, name="norm").connect(
        IOKey("input", shape=[None, 512, None, None]), "normalized"
    )
    block |= Convolution2D(1, n_channels, name="q").connect(
        "normalized", output="query"
    )
    block |= Convolution2D(1, n_channels, name="k").connect("normalized", output="key")
    block |= Convolution2D(1, n_channels, name="v").connect(
        "normalized", output="value"
    )

    query = block.query  # type: ignore[attr-defined]
    key = block.key  # type: ignore[attr-defined]
    value = block.value  # type: ignore[attr-defined]

    shape = query.shape  # type: ignore[attr-defined]

    query = query.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    key = key.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    value = value.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    block |= ScaledDotProduct(is_causal=False).connect(
        query, key, value, output="sdp_out"
    )
    block.set_cout("sdp_out")

    block += Reshape().connect(shape=(shape[0], shape[2], shape[3], shape[1]))
    block += Transpose(axes=(0, 3, 1, 2))
    block += Convolution2D(1, n_channels, name="proj_out")
    block += Add().connect(right="input", output=IOKey("output"))

    return block


def test_functional_attn_block():
    n_channels = 512
    functional_model = attn_block_functional(n_channels)
    model = attn_block(n_channels)
    model._freeze()
    assert_models_equal(functional_model, model)


@functional
def my_lin(left, right):
    scale = IOKey("scale")
    add_output = Add()(left=left, right=right)
    mult_out = Multiply()(left=left, right=add_output)
    return mult_out * scale  # type: ignore


def manual_functional_lin(left, right, name: str | None = None):
    _l, _r = IOKey(), IOKey()
    m = Model.create(my_lin(_l, _r), name=name)
    m.rename_key(_l, "left")
    m.rename_key(_r, "right")
    return m(left, right)


def test_functional_model_naming():
    # Functional API with name
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = my_lin(input1, input2, name="my_lin")
    functional_model = Model.create(x)
    assert list(functional_model.dag)[0].name == "my_lin"

    # Functional API without name
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = my_lin(input1, input2)
    functional_model_no_name = Model.create(x)
    assert list(functional_model_no_name.dag)[0].name == "my_lin"
    assert_models_equal(functional_model, functional_model_no_name)


def test_functional_model_with_decorator():
    # Functional API with decorator
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = my_lin(input1, input2, name="my_lin")

    # Equivalent model using the |= operator

    # Create Square model
    model = Model(name="my_lin")
    model |= (add := Add()).connect(left="left", right="right")
    model |= (mult1 := Multiply()).connect("left", add.output)
    model |= Multiply().connect(mult1.output, "scale")
    # Create wrapper of lin_nested model
    parent_model = Model()
    parent_model |= model.connect(left="input1", right="input2")

    assert_models_equal(x.model.parent, parent_model)

    functional_model = Model.create(x)
    parent_model._freeze()
    assert_models_equal(functional_model, parent_model)

    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = manual_functional_lin(input1, input2, name="my_lin")
    manual_functional_model = Model.create(x)
    assert_models_equal(manual_functional_model, functional_model)


@functional
def square(arg):
    return arg**2


@functional
def lin_nested(left, right, *, scale_shp=None):
    scale = IOKey("scale", shape=scale_shp)
    add_output = Add()(left=left, right=right)
    mult_out = Multiply()(left=left, right=add_output)
    return mult_out * square(scale, name="my_square")


def test_functional_model_with_decorator_nested():
    # Functional API with nested functional model
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = lin_nested(input1, input2, name="my_lin")
    functional_model = Model.create(x)

    # Equivalent model using the |= operator
    # Create Square model
    sq_model = Model("my_square")
    sq_model |= Square().connect("arg")

    # Create lin_nested model
    model = Model(name="my_lin")
    model |= (add := Add()).connect(left="left", right="right")
    model |= (mult1 := Multiply()).connect("left", add.output)
    model |= sq_model.connect(arg="scale")
    model |= Multiply().connect(mult1.output, sq_model.cout)

    # Create wrapper of lin_nested model
    parent_model = Model()
    parent_model |= model.connect(left="input1", right="input2")
    parent_model._freeze()

    assert_models_equal(functional_model, parent_model)


def test_functional_model_with_call_concat():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = Concat()(input=[input1**2, Add(right=1)(input2)])
    functional_model = Model.create(x)

    # Equivalent model using the |= operator
    model = Model()
    model |= (pow := Power(exponent=2)).connect("input1")
    model |= (add := Add(right=1)).connect("input2")
    model |= Concat().connect(input=[pow.output, add.output])
    model._freeze()

    assert_models_equal(functional_model, model)  # type: ignore
