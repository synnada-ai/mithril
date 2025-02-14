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

from copy import deepcopy

import pytest

import mithril as ml
from mithril.framework.common import Tensor
from mithril.models import (
    Add,
    Buffer,
    Convolution2D,
    Divide,
    Gelu,
    IOKey,
    Linear,
    LogisticRegression,
    Model,
    Relu,
    Sigmoid,
    Sine,
    Tanh,
)


def test_canonical_output_1():
    # extend to the canvas model
    model = Model()
    conv = Convolution2D(3, 4)
    model += conv()
    assert model.cin == model.conns.get_con_by_metadata(conv.input.metadata)
    assert model.cout == model.conns.get_con_by_metadata(conv.output.metadata)

    model = Model()
    conv = Convolution2D(3, 4)
    model += conv(input="input")

    assert model.cin.key == "input"
    assert model.cout == model.conns.get_con_by_metadata(conv.output.metadata)

    model = Model()
    conv = Convolution2D(3, 4)
    model += conv(output="output")

    assert model.cin == model.conns.get_con_by_metadata(conv.input.metadata)
    assert model.cout.key == "output"

    model = Model()
    conv = Convolution2D(3, 4)
    model += conv(input="input", output="output")

    assert model.cin.key == "input"
    assert model.cout.key == "output"


def test_canonical_output_2():
    # iadd operator to the canvas model
    model = Model()
    model += (c1 := Convolution2D(3, 4))
    # += operator defaultly sets input="input" if there is not any input

    assert model.cin == model.conns.get_con_by_metadata(c1.input.metadata)
    assert model.cout == model.conns.get_con_by_metadata(c1.output.metadata)


def test_canonical_output_3():
    # First iadd operator then extend
    model = Model()
    c1 = Convolution2D(3, 4)
    c2 = Convolution2D(3, 4)
    model |= c1
    model |= c2(input=c1.output)

    assert model.cin == model.conns.get_con_by_metadata(c1.input.metadata)
    assert model.cout == model.conns.get_con_by_metadata(c2.output.metadata)


def test_canonical_output_4():
    # First iadd operator then extend but use canonical_output to extend
    model = Model()
    c1 = Convolution2D(3, 4)
    c2 = Convolution2D(3, 4)
    model |= c1
    model |= c2(input=model.cout)

    assert model.cin == model.conns.get_con_by_metadata(c1.input.metadata)
    assert model.cout == model.conns.get_con_by_metadata(c2.output.metadata)


def test_canonical_output_5():
    # First extend then iadd operator
    model = Model()
    model += Convolution2D(3, 4)(input="input")
    model += (c2 := Convolution2D(3, 4))

    assert model.cin.key == "input"
    assert model.cout == model.conns.get_con_by_metadata(c2.output.metadata)


def test_canonical_output_6():
    # Don't use canonical output in extend
    model = Model()
    l1 = LogisticRegression()
    l2 = Linear()
    model |= l1(input="input")
    model |= l2(input=l1.probs_output)

    assert model.cin == model.conns.get_con_by_metadata(l1.input.metadata)

    assert model.conns.cins == {model.conns.get_con_by_metadata(l1.input.metadata)}

    assert model.conns.couts == {
        model.conns.get_con_by_metadata(l1.output.metadata),
        model.conns.get_con_by_metadata(l2.output.metadata),
    }

    model = Model()
    l1 = LogisticRegression()
    l2 = Linear()

    model |= l1(input="input")
    model |= l2(input=l1.output)

    assert model.cin == model.conns.get_con_by_metadata(l1.input.metadata)

    assert model.conns.cins == {model.conns.get_con_by_metadata(l1.input.metadata)}

    assert model.conns.couts == {model.conns.get_con_by_metadata(l2.output.metadata)}


def test_canonical_output_8():
    modelsub = Model()
    modelsub |= Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub |= Sigmoid()(input="in2", output=IOKey(name="out2"))
    modelsub.set_cin("in2")
    modelsub.set_cout("out2")

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model |= modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model |= modelsub2(in2="out2", in1="out1", out1=IOKey(name="out4"))

    assert model.cin == model.conns.get_con_by_metadata(
        modelsub.in2.metadata  # type: ignore
    )
    assert model.cout == model.conns.get_con_by_metadata(
        modelsub2.out2.metadata  # type: ignore
    )


def test_canonical_output_9():
    modelsub = Model()
    modelsub |= Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub |= Sigmoid()(input="in2", output=IOKey(name="out2"))
    modelsub.set_cin("in2")
    modelsub.set_cout("out2")

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model |= modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model |= modelsub2(in1="out2", in2="out1", out1=IOKey(name="out4"))

    assert model.cin == model.conns.get_con_by_metadata(
        modelsub.in2.metadata  # type: ignore
    )
    assert model.cout == model.conns.get_con_by_metadata(
        modelsub2.out2.metadata  # type: ignore
    )


def test_canonical_output_10():
    # Canonical output is None
    modelsub = Model()
    modelsub |= Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub |= Sigmoid()(input="in2", output=IOKey(name="out2"))
    modelsub.set_cin("in2")
    modelsub.set_cout("out2")

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model |= modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model |= modelsub2(in2="out2", out2="in1")

    assert model.cin == model.conns.get_con_by_metadata(
        modelsub.in2.metadata  # type: ignore
    )
    assert model.conns.couts == set()


def test_canonical_output_11():
    # Canonical input is None
    modelsub = Model()
    modelsub |= Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub |= Sigmoid()(input="in2", output=IOKey(name="out2"))
    modelsub.set_cin("in2")
    modelsub.set_cout("out2")

    model = Model()
    model |= modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model |= Sigmoid()(input="out1", output="in2")

    assert model.conns.cins == set()
    assert model.cout == model.conns.get_con_by_metadata(
        modelsub.out2.metadata  # type: ignore
    )


def test_canonical_output_14():
    # Canonical output is NOT_AVAILABLE for a while then redetermined
    modelsub = Model()
    modelsub |= Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub |= Sigmoid()(input="in2", output=IOKey(name="out2"))
    modelsub.set_cin("in2")
    modelsub.set_cout("out2")

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model |= modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model |= modelsub2(in2="out2", out2="in1")

    model |= (relu := Relu())(input=IOKey("input"), output=IOKey("output"))

    assert model.conns.cins == {
        model.conns.get_con_by_metadata(relu.input.metadata),
        model.conns.get_con_by_metadata(modelsub.in2.metadata),  # type: ignore
    }

    assert model.conns.couts == {model.conns.get_con_by_metadata(relu.output.metadata)}


def test_canonical_output_exposed_1():
    model1 = Model()
    model1 |= Linear(dimension=32)
    model1 += Relu()
    model1 |= Linear(dimension=16)(input=model1.cout, output="output1")

    model1._freeze()

    assert list(model1.conns.output_keys) == ["output1"]
    assert "output1" in model1.external_keys


def test_canonical_output_exposed_2():
    # Canonical output should be considered as exposed in extend info
    model1 = Model()
    model1 |= Linear(dimension=32)
    model1 += Relu()
    model1 |= Linear(dimension=16)(input=model1.cout, output="output1")

    extend_info = model1(output1="output1")
    assert extend_info.connections == {"output1": "output1"}


def test_canonical_output_exposed_3():
    model1 = Model()
    model1 |= Linear(dimension=32)
    model1 += Relu()
    model1 |= Linear(dimension=16)(input=model1.cout, output="output1")

    model = Model()
    model |= model1(output1="output1")

    model._freeze()
    assert list(model.output_keys) == ["output1"]


def test_canonical_input_1():
    # Override canonical input keep canonical output same
    model = Model()
    linear = Linear()
    model |= linear(input="input1")
    model |= LogisticRegression()(input="input2", output="input1")

    assert model.cin.key == "input2"
    assert model.cout == model.conns.get_con_by_metadata(linear.output.metadata)
    # assert model.cout.key == 'Linear_0_output'


def test_canonical_input_2():
    # Override canonical input and canonical output
    model = Model()
    logistic = LogisticRegression()
    model |= (lin1 := Linear())(input="input1")
    model |= logistic(input="input2", probs_output="input1")

    assert model.cin.key == "input2"

    assert model.conns.couts == {
        model.conns.get_con_by_metadata(lin1.output.metadata),
        model.conns.get_con_by_metadata(logistic.output.metadata),
    }


def test_canonical_input_3():
    # Override canonical input keep canonical output same but complex
    model = Model()
    linear1 = Linear()
    linear2 = Linear()
    logistic = LogisticRegression()

    model |= linear1(input="input1")
    model |= linear2(input="input2")
    model |= logistic(input="input3", output="input1")

    assert model.conns.cins == {
        model.conns.get_con_by_metadata(linear2.input.metadata),
        model.conns.get_con_by_metadata(logistic.input.metadata),
    }

    assert model.conns.couts == {
        model.conns.get_con_by_metadata(linear1.output.metadata),
        model.conns.get_con_by_metadata(linear2.output.metadata),
    }


def test_canonical_input_5():
    # Override canonical input keep canonical output same but complex
    model = Model()
    model |= (l1 := Linear())
    model += Linear()
    model += (l2 := Linear())
    model |= Linear()(input=l2.output, output="my_output")

    assert model.cin == model.conns.get_con_by_metadata(l1.input.metadata)
    assert model.cout.key == "my_output"


def test_canonical_input_7():
    model = Model()
    model_1 = Model()
    model_1 |= Relu()(input="input1", output=IOKey(name="output1"))
    model_1 |= Sigmoid()(input="input2", output=IOKey(name="output2"))
    model_1.set_cin("input2")
    model_1.set_cout("output2")
    gelu5 = Gelu()

    model_2 = Model()
    model_2 |= Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 |= Sine()(input="input2", output=IOKey(name="output2"))
    model_1.set_cin("input2")
    model_1.set_cout("output2")
    model |= gelu5()
    model |= model_1(input1="input", output1=gelu5.input)
    model |= model_2(
        input2=gelu5.output,
        output2=model_1.input2,  # type: ignore
        input1=model_1.output2,  # type: ignore
        output1=IOKey(name="output"),
    )

    assert model.conns.cins == set()
    assert model.conns.couts == {
        model.conns.get_con_by_metadata(model_2.output1.metadata)  # type: ignore
    }


def test_canonical_input_8():
    model = Model()

    model |= Tanh()(input="input1", output="output1")
    model |= Sine()(input="input2", output="input1")

    assert model.cin.key == "input2"
    assert model.cout.key == "output1"


def test_canonical_input_9():
    # Canonical input is NOT_AVAILABLE for a while then redetermined
    modelsub = Model()
    modelsub |= Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub |= Sigmoid()(input="in2", output=IOKey(name="out2"))
    modelsub.set_cin("in2")
    modelsub.set_cout("out2")

    model = Model()
    model |= modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model |= Sigmoid()(input="out1", output="in2")

    model |= (relu := Relu())(input="input", output="output")

    assert model.conns.cins == {
        model.conns.get_con_by_metadata(relu.input.metadata),
    }

    assert model.conns.couts == {
        model.conns.get_con_by_metadata(relu.output.metadata),
        model.conns.get_con_by_metadata(modelsub.out2.metadata),  # type: ignore
    }


def test_canonical_input_10():
    # Valued cannection cannot be canonical input
    model = Model()
    model |= Divide()(numerator=3, denominator="input2", output="output")

    assert model.conns.cins == set()


def test_canonical_input_11():
    # Valued cannection cannot be canonical input
    model = Model()
    model |= (buff := Buffer())
    model |= Divide()(numerator=3, denominator="input2", output="output")

    canonical_input = model.cin
    assert canonical_input.metadata == buff.input.metadata


def test_canonical_input_12():
    # Valued cannection cannot be canonical input
    model = Model()
    model |= (buff := Buffer())
    model |= Buffer()(input=3 / buff.output)

    canonical_input = model.cin
    assert canonical_input.metadata == buff.input.metadata


def test_canonical_dual_iadd_op():
    model1 = Model()
    model1 |= (c1 := Convolution2D(3, 4))
    model1 += Convolution2D(3, 4)

    model = Model()
    model |= model1
    model += Convolution2D(3, 4)
    model += (c4 := Convolution2D(3, 4))

    assert model.cin == model.conns.get_con_by_metadata(c1.input.metadata)
    assert model.cout == model.conns.get_con_by_metadata(c4.output.metadata)


def test_set_cin():
    model = LogisticRegression()
    # Change cin to weight.
    model.set_cin("weight")
    assert model.cin == model.weight
    # Change cin to bias.
    model.set_cin("bias")
    assert model.cin == model.bias
    # Set previous cin ("weight") again.
    model.set_cin("weight")
    assert model.cin == model.weight
    # Try setting multiple cin
    model.set_cin("weight", "bias")
    assert model.conns.cins == {model.weight, model.bias}
    # TODO: we can directly raise an error in cin property
    with pytest.raises(KeyError) as err_info:
        assert model.cin
    assert (
        str(err_info.value) == "'Currently, there exists 2 canonical inputs, "
        "model should have exactly one canonical input!'"
    )


def test_set_cout():
    model = LogisticRegression()
    # Change cout to probs_output.
    model.set_cout("probs_output")
    assert model.cout == model.probs_output
    # Change cout to output.
    model.set_cout("output")
    assert model.cout == model.output
    # Set previous cout ("probs_output") again.
    model.set_cout("probs_output")
    assert model.cout == model.probs_output
    # Try setting multiple cout
    model.set_cout("probs_output", "output")
    assert model.conns.couts == {model.probs_output, model.output}
    # TODO: we can directly raise an error in cout property
    with pytest.raises(KeyError) as err_info:
        assert model.cout

    assert (
        str(err_info.value) == "'Currently, there exists 2 canonical outputs, "
        "model should have exactly one canonical output!'"
    )


def test_set_values():
    model1 = Model() + Add()(left="left", right="right")

    assert model1.conns.cins == {model1.left, model1.right}  # type: ignore
    model1.set_values(left=Tensor([[3]]))
    assert model1.conns.cins == {model1.right}  # type: ignore
    model1.set_values(right=Tensor([[3]]))
    assert model1.conns.cins == set()

    model2 = Add(left=Tensor([[3]]))
    assert model2.conns.cins == {model2.right}

    model3 = Add(right=Tensor([[3]]))
    assert model3.conns.cins == {model3.left}

    model4 = Add(left=Tensor([[3]]), right=Tensor([[3]]))
    assert model4.conns.cins == set()


# TODO: Add more tests for available single_canonical_input
def test_child_single_available_canonical_input_with_name():
    model = Model()
    model |= Linear(input=Tensor([[3]]))

    model += (add := Add())(right="right")

    assert model.conns.cins == {
        model.conns.get_con_by_metadata(add.right.metadata),
    }


def test_child_single_available_canonical_input_with_values():
    model = Model()
    model |= Linear(input=Tensor([[3]]))

    model += Add()(right=Tensor([[3]]))
    assert model.conns.cins == set()


def test_child_single_available_canonical_input_with_io_key():
    model = Model()
    model |= Linear(input=Tensor([[3]]))
    model += (add := Add())(right=IOKey("right"))
    assert model.conns.cins == {
        model.conns.get_con_by_metadata(add.right.metadata),
    }


def test_child_single_available_canonical_input_with_valued_io_key():
    model = Model()
    model |= Linear(input=Tensor([[3]]))

    model += Add()(right=IOKey("right", Tensor([[3]])))
    assert model.conns.cins == set()


def test_child_single_available_canonical_input_with_connection():
    model = Model()
    model |= Linear()(input=Tensor([[3]]), output="lin_out1")
    model |= Linear()(input=Tensor([[3]]), output="lin_out2")
    model.set_cout("lin_out1")
    model += Add()(right="lin_out2")
    assert model.conns.cins == set()


def test_child_zero_available_canonical_input():
    model = Model()
    model |= Linear()(input=Tensor([[3]]), output="lin_out1")
    model |= Linear()(input=Tensor([[3]]), output="lin_out2")
    model.set_cout("lin_out1")

    model |= Add()(left=model.cout, right="lin_out2")
    assert model.conns.cins == set()


def test_child_zero_canonical_input():
    model = Model()
    model |= Linear()(input=Tensor([[3]]), output="lin_out1")
    model.set_cout("lin_out1")

    add_model = Add()
    add_model.set_cin()
    with pytest.raises(KeyError) as err_info:
        model += add_model()
    ref = "'No existing canonical input is found to extension model! Use |= operator.'"
    assert str(err_info.value) == ref


def test_child_single_canonical_input():
    model = Model()
    assert model.conns.cins == set()

    model |= Buffer()(input="input1", output="output1")
    assert model.conns.cins == {model.input1}  # type: ignore

    model |= Buffer()(input="input2", output="input1")
    assert model.conns.cins == {model.input2}  # type: ignore

    model |= Buffer()(input="input3", output="input2")
    assert model.conns.cins == {model.input3}  # type: ignore


def test_parent_single_canonical_output():
    model = Model()
    assert model.conns.couts == set()

    model |= Buffer()(input="input1", output="output1")
    assert model.conns.couts == {model.output1}  # type: ignore

    model += Buffer()(output="output2")
    assert model.conns.couts == {model.output2}  # type: ignore

    model += Buffer()(output="output3")
    assert model.conns.couts == {model.output3}  # type: ignore


def test_child_multi_canonical_input_error():
    child = Model()
    child |= Add()
    child |= Add()

    parent = Model()
    parent |= Relu()
    with pytest.raises(KeyError) as err_info:
        parent += child
    ref = (
        "'Submodel must have single available canonical input! "
        "Set canonical input or use |= operator.'"
    )
    assert str(err_info.value) == ref


def test_parent_multi_canonical_output_error():
    parent = Model()
    parent |= Add()
    parent |= Add()

    with pytest.raises(KeyError) as err_info:
        parent += Buffer()
    ref = (
        "'Currently, there exists 2 canonical outputs, "
        "model should have exactly one canonical output!'"
    )

    assert str(err_info.value) == ref


def test_child_no_canonical_input_error():
    child = Model()
    child |= Add()
    child.set_cin()

    parent = Model()
    parent |= Relu()
    with pytest.raises(KeyError) as err_info:
        parent += child
    ref = "'No existing canonical input is found to extension model! Use |= operator.'"
    assert str(err_info.value) == ref


def test_parent_no_canonical_output_error():
    parent = Model()
    parent |= Divide()
    parent += Divide()
    parent.set_cout()

    with pytest.raises(KeyError) as err_info:
        parent += Buffer()
    ref = (
        "'Currently, there exists 0 canonical outputs, "
        "model should have exactly one canonical output!'"
    )
    assert str(err_info.value) == ref


def test_new_connection_unconnected_input():
    model = Model()
    assert model.conns.cins == set()

    model |= Relu()(input="input1", output="output1")
    assert model.conns.cins == {model.input1}  # type: ignore

    model |= Relu()(input="input2", output="output2")
    assert model.conns.cins == {model.input1, model.input2}  # type: ignore

    model |= Relu()(input="input3", output="output3")
    assert model.conns.cins == {
        model.input1,  # type: ignore
        model.input2,  # type: ignore
        model.input3,  # type: ignore
    }


def test_new_connection_exposed_internal_output():
    model = Model()
    assert model.conns.couts == set()

    model |= Relu()(input="input1", output=IOKey("output1", expose=True))
    assert model.conns.couts == {model.output1}  # type: ignore

    model |= Relu()(input="input2", output=IOKey("input1", expose=True))
    assert model.conns.couts == {model.output1, model.input1}  # type: ignore

    model |= Relu()(input="input3", output=IOKey("input2", expose=True))
    assert model.conns.couts == {
        model.output1,  # type: ignore
        model.input1,  # type: ignore
        model.input2,  # type: ignore
    }


def test_new_connection_multi_output_without_call():
    model = Model()
    model |= (submodel := LogisticRegression())
    assert model.conns.couts == {
        model.conns.get_con_by_metadata(submodel.output.metadata)
    }


def test_unexposed_canonical_output_connection_after_freeze():
    model = Model()
    model += Linear()
    model += Linear()
    model += Linear()
    assert len(model.conns.output_connections) == 0
    model._freeze()
    assert set(model.conns.output_connections) == {model.cout}


def test_unexposed_canonical_output_connection_after_extension():
    model = Model()
    model += Linear()
    model += Linear()
    model += Linear()

    bigger_model = Model()
    bigger_model |= model

    assert bigger_model.cout == bigger_model.conns.get_con_by_metadata(
        model.cout.metadata
    )


def test_new_connection_multi_output_exposed_noncanonical_output():
    submodel1 = Model()
    submodel1 |= (l1 := LogisticRegression())(
        probs_output=IOKey("output1", expose=True)
    )

    submodel2 = Model()
    submodel2 |= (l2 := LogisticRegression())(
        probs_output=IOKey("output1", expose=True)
    )

    bigger_model = Model()
    bigger_model += submodel1
    bigger_model |= submodel2

    assert bigger_model.conns.couts == {
        bigger_model.conns.get_con_by_metadata(l1.output.metadata),
        bigger_model.conns.get_con_by_metadata(l2.output.metadata),
    }


def test_new_connection_multi_output_expose_false_canonical_output():
    submodel1 = Model()
    submodel1 |= (l1 := LogisticRegression())(
        probs_output=IOKey("output1", expose=True),
        output=IOKey("output2", expose=False),
    )

    submodel2 = Model()
    submodel2 |= (l2 := LogisticRegression())(
        probs_output=IOKey("output1", expose=True),
        output=IOKey("output2", expose=False),
    )

    bigger_model = Model()
    bigger_model += submodel1
    bigger_model |= submodel2

    # assert bigger_model.conns.couts == set()

    assert bigger_model.conns.couts == {
        bigger_model.conns.get_con_by_metadata(l1.output.metadata),
        bigger_model.conns.get_con_by_metadata(l2.output.metadata),
    }


def test_new_connection_multi_output_set_name():
    submodel = LogisticRegression()
    model = Model()
    model |= submodel(probs_output="output1", output="output2")
    assert model.conns.couts == {
        model.conns.get_con_by_metadata(submodel.output.metadata),
    }


def test_existing_connection_parent_input_updated_to_input():
    model = Model()
    assert model.conns.cins == set()

    model |= Relu()(input="input1")
    assert model.conns.cins == {model.input1}  # type: ignore

    # Add Relu input to existing input1
    model |= Relu()(input="input1")
    assert model.conns.cins == {model.input1}  # type: ignore


def test_existing_connection_parent_input_updated_to_internal():
    model = Model()
    assert model.conns.cins == set()

    model |= Relu()(input="input1")
    assert model.conns.cins == {model.input1}  # type: ignore

    # Add Relu input to existing input1
    model |= Relu(input=Tensor(3))(output="input1")  # input1 is now internal
    assert model.conns.cins == set()


def test_existing_connection_parent_internal_updated_to_internal():
    model = Model()
    model |= Relu()(input="input1", output="output1")
    assert model.conns.couts == {model.output1}  # type: ignore

    model += Relu()(output="output2")
    assert model.conns.couts == {model.output2}  # type: ignore

    # Make internal output1 stay internal
    model |= Relu()(model.output1, output="output3")  # type: ignore
    assert model.conns.couts == {model.output2, model.output3}  # type: ignore


def test_existing_connection_parent_internal_updated_to_output():
    model = Model()
    model |= Relu()(input="input1", output="output1")
    assert model.conns.couts == {model.output1}  # type: ignore

    model += Relu()(output="output2")
    assert model.conns.couts == {model.output2}  # type: ignore

    # Make internal output1 exposed
    in_key = IOKey(connections={model.output1}, expose=True)  # type: ignore
    model |= Relu()(input=in_key, output="output3")
    assert model.conns.couts == {model.output2, model.output3}  # type: ignore


def test_existing_connection_parent_internal_updated_to_output2():
    model = Model()
    model |= Relu()(input="input1", output="output1")
    assert model.conns.couts == {model.output1}  # type: ignore

    # Make internal output1 exposed
    in_key = IOKey(connections={model.output1}, expose=True)  # type: ignore
    model |= Relu()(input=in_key, output="output3")
    assert model.conns.couts == {model.output3}  # type: ignore


def test_existing_connection_parent_internal_updated_to_output3():
    model = Model()
    model |= Relu()(input="input1", output=IOKey("output1"))
    model += Relu()(output="output3")
    assert model.conns.couts == {model.output3}  # type: ignore

    model = Model()
    model |= Relu()(input="input1", output=IOKey("output1"))
    model |= Relu()(input=model.output1, output="output3")  # type: ignore
    assert model.conns.couts == {model.output3}  # type: ignore


def test_existing_connection_parent_output_updated_to_internal():
    model = Model()
    model |= Relu()(input="input1", output="output1")
    assert model.conns.couts == {model.output1}  # type: ignore

    # output1 is now internal
    model |= Relu()(input=model.output1, output="output2")  # type: ignore
    assert model.conns.couts == {model.output2}  # type: ignore


# TODO: Add tests with IOKey with multiple connections for input and output


def test_compile_multi_canonical_output_no_exposed_output():
    model = Model()
    model |= Relu()("input1")
    model |= Relu()("input2")
    model |= Relu()("input3")

    backend = ml.JaxBackend()
    pm = ml.compile(model, backend)
    assert pm.output_keys == ["__output", "_output", "output"]


def test_error_compile_no_canonical_output_no_exposed_output():
    model = Model()
    model |= Relu()
    model |= Relu()
    model |= Relu()
    model.set_cout()

    backend = ml.JaxBackend()
    with pytest.raises(KeyError) as err_info:
        ml.compile(model, backend)
    assert str(err_info.value) == "'Models with no output keys can not be compiled.'"


def test_error_strict_no_available_canonical_input():
    # Strict
    add = Add()
    model = Model()
    model |= Relu()
    with pytest.raises(KeyError) as err_info:
        model += add(model.cout, model.cout)  # -> raises error

    assert str(err_info.value) == (
        "'Submodel must have single available canonical input! "
        "Set canonical input or use |= operator.'"
    )
