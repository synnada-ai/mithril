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

import re

import mithril
from mithril import JaxBackend, TorchBackend
from mithril.framework.common import TBD, IOKey
from mithril.framework.constraints import squeeze_constraints
from mithril.models import (
    L2,
    MLP,
    Add,
    Connect,
    Convolution2D,
    CrossEntropy,
    CustomPrimitiveModel,
    Layer,
    Linear,
    Mean,
    Model,
    Relu,
    Scalar,
    Sigmoid,
    SquaredError,
    TensorType,
    TrainModel,
)
from mithril.utils import dict_conversions

from .helper import assert_evaluations_equal, assert_models_equal


def test_linear_expose():
    model = Model()
    model += Linear(dimension=42)(input="input", w="w", output=IOKey(name="output"))
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_linear_expose_2():
    model = Model()
    model += Linear(dimension=42)(w="w", output=IOKey(name="output"))
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_linear_not_expose():
    model = Model()
    model += Linear(dimension=42)(input="input")
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_constant_key():
    model = Model()
    model += Add()(left="input", right=3, output=IOKey(name="output"))
    model2 = Model()
    model2 += model()

    model_dict_created = dict_conversions.model_to_dict(model2)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model2, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model2, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_constant_key_2():
    model = Model()
    model += (add := Add())(left="input", right=3, output=IOKey(name="output"))
    model += Add()(left="input2", right=add.right, output=IOKey(name="output2"))
    model2 = Model()
    model2 += model(output=IOKey(name="output"), output2=IOKey(name="output2"))

    model_dict_created = dict_conversions.model_to_dict(model2)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model2, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model2, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_linear_directly():
    model = Linear(dimension=42)
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_mlp_directly():
    model = MLP(dimensions=[11, 76], activations=[Sigmoid(), Relu()])

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_1():
    model = Model()
    model += Linear(dimension=10)(input="input", w="w", output=IOKey(name="output"))
    model += Linear(dimension=71)(input="output", w="w1", output=IOKey(name="output2"))

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_2():
    model = Model()
    model += Linear(dimension=10)(input="input", w="w", output=IOKey(name="output"))
    model += Linear(dimension=71)(
        input=model.output,  # type: ignore
        w="w1",
        output=IOKey(name="output2"),
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_2_1():
    model = Model()
    model += (l1 := Linear(dimension=10))(
        input="input", w="w", output=IOKey(name="output")
    )
    model += Linear(dimension=71)(input=l1.output, w="w1", output=IOKey(name="output2"))

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_2_2():
    model = Model()
    model += (l1 := Linear(dimension=10))(input="input", w="w")
    model += Linear(dimension=71)(input=l1.output, w="w1", output=IOKey(name="output2"))

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_2_3():
    model = Model()
    model += (l1 := Linear())(input="input", w="w")
    model += Linear()(input=l1.output, w=l1.w, b=l1.b, output=IOKey(name="output2"))

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_3():
    model = Model()
    model += (l1 := Linear(dimension=10))(
        input="input", w="w", output=IOKey(name="output")
    )
    model += Linear(dimension=71)(input=l1.output, w="w1", output=IOKey(name="output2"))
    model += Linear(dimension=71)(input="input2", w="w1", output=IOKey(name="output3"))

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 256]), "input2": backend.ones([4, 10])},
    )


def test_composite_4():
    model = Model()
    model += (l1 := Linear(dimension=10))(
        input="input", w="w", output=IOKey(name="output")
    )
    model += Linear(dimension=71)(input=l1.output, w="w1", output=IOKey(name="output2"))
    model += Linear(dimension=71)(input=l1.output, w="w1", output=IOKey(name="output3"))

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_5():
    model = Model()
    model += Linear(dimension=10)(w="w", output=IOKey(name="output"))
    model += Linear(dimension=71)(
        input=model.canonical_output, w="w1", output=IOKey(name="output2")
    )
    model += Linear(dimension=71)(
        input=model.canonical_output, w="w2", output=IOKey(name="output3")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 256])},
    )


def test_composite_6():
    model = Model()
    model += Linear(dimension=10)(w="w", output=IOKey(name="output"))
    model += Linear(dimension=71)(
        input=model.canonical_output, w="w1", output=IOKey(name="output2")
    )
    model += Layer(dimension=71, activation=Sigmoid())(
        input="output2", w="w2", output=IOKey(name="output3")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 256])},
    )


def test_composite_7():
    model = Model()
    model += (l1 := Linear(dimension=10))(
        input="my_input", w="w", output=IOKey(name="output")
    )
    model += Linear(dimension=71)(input="input2", w="w1", output=l1.input)

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input2": backend.ones([4, 256])}
    )


def test_composite_8():
    model = Model()
    model += (l1 := Linear(dimension=10))(w="w", output=IOKey(name="output"))
    model += Linear(dimension=71)(input="input2", w="w1", output=l1.input)

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input2": backend.ones([4, 256])},
    )


def test_composite_9():
    model = Model()
    model += (l1 := Linear(dimension=10))(w="w", output=IOKey(name="output"))
    model += (l2 := Linear(dimension=10))(
        input="", w="w1", output=IOKey(name="output2")
    )
    model += Linear(dimension=71)(
        input="input", w="w2", output=Connect(l1.input, l2.input)
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_10():
    model = Model()
    model += Linear(dimension=10)(input="input2", w="w", output=IOKey(name="output"))
    model += Linear(dimension=10)(input="input1", w="w1", output=IOKey(name="output2"))
    model += Linear(dimension=71)(
        input="input",
        w="w2",
        output=Connect("input1", "input2", key=IOKey(name="my_input")),
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_10_expose_false():
    model = Model()
    model += Linear(dimension=10)(input="input2", w="w", output=IOKey(name="output"))
    model += Linear(dimension=10)(input="input1", w="w1", output=IOKey(name="output2"))
    model += Linear(dimension=71)(
        input="input",
        w="w2",
        output=Connect("input1", "input2", key=IOKey(name="my_input", expose=False)),
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_11():
    mlp_model = MLP(activations=[Relu(), Relu(), Relu()], dimensions=[12, 24, None])
    model = TrainModel(mlp_model)
    model.add_loss(
        SquaredError(),
        input=mlp_model.canonical_output,
        target=[[2.2, 4.2], [2.2, 4.2]],
        reduce_steps=[Mean()],
    )

    context_dict = dict_conversions.model_to_dict(model)
    context_recreated = dict_conversions.dict_to_model(context_dict)
    context_dict_recreated = dict_conversions.model_to_dict(context_recreated)

    assert context_dict == context_dict_recreated
    assert_models_equal(model, context_recreated)


def test_composite_12():
    # Case where submodel output keys only named
    model = Model()
    model.extend(Linear(dimension=10), input="input2", w="w", output="output")
    model.extend(Linear(dimension=10), input="input1", w="w1", output="output2")
    model.extend(
        Linear(dimension=71),
        input="input",
        w="w2",
        output=Connect("input1", "input2", key=IOKey(name="my_input")),
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_13():
    # Case where submodel output keys IOKey but not exposed
    model = Model()
    model.extend(
        Linear(dimension=10),
        input="input2",
        w="w",
        output=IOKey("output", expose=False),
    )
    model.extend(
        Linear(dimension=10),
        input="input1",
        w="w1",
        output=IOKey("output2", expose=False),
    )
    model.extend(
        Linear(dimension=71),
        input="input",
        w="w2",
        output=Connect("input1", "input2", key=IOKey(name="my_input")),
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_auto_iadd_1():
    model = Model()
    model += Sigmoid()(input="input", output=IOKey(name="output"))
    model += Sigmoid()(output="output2")
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_auto_iadd_2():
    model = Model()
    model += Sigmoid()(input="input", output=IOKey(name="output"))
    model += Sigmoid()(input="", output="output2")
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_convolution():
    model = Model()
    model += Convolution2D(kernel_size=3, out_channels=20)(output=IOKey(name="output"))

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 3, 32, 32])},
    )


def test_tbd():
    model = Model()
    model += Convolution2D(kernel_size=3, out_channels=20, stride=TBD)(
        output=IOKey(name="output"), stride=(1, 1)
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 3, 32, 32])},
    )


def test_train_context_1():
    model = Model()
    layer1 = Linear(dimension=16)
    layer2 = Linear(dimension=10)

    model += layer1(w="w0", b="b0", input="input")
    model += layer2(input=layer1.output, w="w1", b="b1", output=IOKey(name="output"))

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(), [Mean()], target="target", input=model.canonical_output
    )
    context_dict = dict_conversions.model_to_dict(context)
    context_recreated = dict_conversions.dict_to_model(context_dict)
    context_dict_recreated = dict_conversions.model_to_dict(context_recreated)

    assert context_dict == context_dict_recreated
    assert_models_equal(context, context_recreated)

    backend = TorchBackend(precision=64)
    assert_evaluations_equal(
        context,
        context_recreated,
        backend,
        static_keys={
            "input": backend.ones([4, 32]),
            "target": backend.ones([4], dtype=mithril.int64),
        },
    )


def test_train_context_2():
    model = Model()
    layer1 = Linear(dimension=16)
    layer2 = Linear(dimension=10)

    model += layer1(w="w0", b="b0", input="input")
    model += layer2(input=layer1.output, w="w1", b="b1", output=IOKey(name="output"))

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(), [Mean()], target="target", input=model.canonical_output
    )
    context.add_regularization(model=L2(), coef=1e-1, input=re.compile("w\\d"))
    context_dict = dict_conversions.model_to_dict(context)
    context_recreated = dict_conversions.dict_to_model(context_dict)
    context_dict_recreated = dict_conversions.model_to_dict(context_recreated)
    assert context_dict == context_dict_recreated
    assert_models_equal(context, context_recreated)

    backend = TorchBackend(precision=64)
    assert_evaluations_equal(
        context,
        context_recreated,
        backend,
        static_keys={
            "input": backend.ones([4, 32]),
            "target": backend.ones([4], dtype=mithril.core.Dtype.int64),
        },
    )


def test_set_values_constant_1():
    # Set value using IOKey
    model = Model()
    model += Linear(10)(
        w="w0",
        b="b0",
        input="input",
        output=IOKey(name="output", expose=False),
    )
    model += Linear(1)(
        w="w1",
        b=IOKey(value=[123], name="b1"),
        input="input2",
        output=IOKey(name="output2"),
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([10, 4]), "input2": backend.ones([10, 4])},
    )


def test_set_values_constant_2():
    # Set value using set_values api
    model = Model()
    model.extend(
        Linear(10),
        w="w0",
        b="b0",
        input="input",
        output=IOKey(name="output", expose=False),
    )
    model.extend(
        Linear(1), w="w1", b="b1", input="input2", output=IOKey(name="output2")
    )
    model.set_values({"b1": [123]})

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(precision=64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([10, 4]), "input2": backend.ones([10, 4])},
    )


def test_set_values_tbd_1():
    model = Model()
    model.extend(Linear(10), w="w0", b="b0", input="input", output=IOKey(name="output"))
    model.extend(Linear(1), w="w1", b=IOKey(value=TBD, name="b1"), input="input2")

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)


def test_set_values_ellipsis_2():
    model = Model()
    model.extend(Linear(10), w="w0", b="b0", input="input", output=IOKey(name="output"))
    model.extend(Linear(1), w="w1", b="b1", input="input2")
    model.set_values({"b1": TBD})

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)


def test_make_shape_constrain():
    model = Model()

    def my_adder(input, rhs):
        return input + rhs

    TorchBackend.register_primitive(my_adder)  # After serialization is this available?

    class MyAdder(CustomPrimitiveModel):
        def __init__(self, threshold=3) -> None:
            threshold *= 2
            super().__init__(
                formula_key="my_adder",
                output=TensorType([("Var_out", ...)]),
                input=TensorType([("Var_1", ...)]),
                rhs=Scalar(int, threshold),
            )
            self.set_constraint(
                fn=squeeze_constraints, keys=[CustomPrimitiveModel.output_key, "input"]
            )

    model += MyAdder()(input="input")
    # model.extend(MyAdder(), input = "input")
    model.set_shapes({"input": [1, 128, 1, 8, 16]})

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    # TODO: Handle TensorType and Scalar conversions!!!
    assert model_dict_created == model_dict_recreated
    assert model.shapes == model_recreated.shapes
    assert_models_equal(model, model_recreated)
    TorchBackend.registered_primitives.pop("my_adder")
