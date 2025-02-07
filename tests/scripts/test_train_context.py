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

import numpy as np
import pytest

import mithril
from mithril import JaxBackend, NumpyBackend
from mithril.models import (
    L1,
    L2,
    MLP,
    Buffer,
    CrossEntropy,
    IOKey,
    LeakyRelu,
    Linear,
    Max,
    Mean,
    Min,
    Model,
    QuadraticFormRegularizer,
    Relu,
    Sigmoid,
    Softmax,
    SquaredError,
    Subtract,
    Sum,
    Tensor,
    TrainModel,
)

from .test_utils import assert_metadata_equal, assert_results_equal


def test_add_loss_case_1():
    model = Model()
    buff1 = Buffer()
    relu1 = Relu()
    relu2 = Relu()
    model += buff1(input="input")
    model += relu1(input=buff1.output)
    model += relu2(input=relu1.output)

    ctx2 = TrainModel(model)
    ctx2.add_loss(
        rel1 := Relu(), [Max(axis=-1), Min(axis=-1), Sum(axis=-1)], input=relu2.output
    )
    assert_metadata_equal(relu2.output, rel1.input)


def test_add_loss_case_2():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()

    inputs = {"input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])}

    model += relu1(input="input")
    model += relu2(input=relu1.output)
    model += relu3(input=relu2.output, output=IOKey(name="output"))

    ctx1 = TrainModel(model)
    ctx1.add_loss(
        Subtract(),
        [Min(axis=-1), Max(axis=-1), Mean(axis=-1)],
        left="output",
        right=Tensor(0.0),
        key_name="abcd",
    )
    compiled_ctx1 = mithril.compile(model=ctx1, backend=NumpyBackend())
    outputs, grads = compiled_ctx1.evaluate_all(inputs)
    ref_outputs = {
        "abcd": np.array(5.0),
        "final_cost": np.array(5.0),
        "output": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    }
    ref_grads = {
        "input": np.array([[[0.0, 0.0], [0.5, 0.0]], [[0.0, 0.0], [0.5, 0.0]]])
    }
    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


def test_add_loss_case_2_exception_2():
    """Throws exception since trying to add additional loss to a finalized
    TrainModel.
    """
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    relu4 = Relu()

    model += relu1(input="input")
    model += relu2(input=relu1.output)
    model += relu3(input=relu2.output, output=IOKey(name="output1"))
    model += relu4(input=relu3.output, output=IOKey(name="output2"))

    ctx1 = TrainModel(model)
    ctx1.add_loss(
        Subtract(),
        [Min(axis=-1), Max(axis=-1), Mean(axis=-1)],
        left="output1",
        right=Tensor(0.0),
        key_name="abcd",
    )

    # Finalize train model.
    ctx1.finalize()
    with pytest.raises(Exception) as err_info:
        ctx1.add_loss(
            Subtract(),
            [Min(axis=-1), Max(axis=-1), Mean(axis=-1)],
            left=relu4.output,
            right=0.0,
            key_name="abcde",
        )

    assert (
        str(err_info.value) == "No modifications can be made to a finalized TrainModel!"
    )


def test_add_loss_case_3():
    backend = JaxBackend(dtype=mithril.float64)
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()

    inputs = {
        "input": backend.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    }

    model += relu1(input="input")
    model += relu2(input=relu1.output)
    model += relu3(input=relu2.output, output=IOKey(name="output"))

    ctx1 = TrainModel(model)
    ctx1.add_loss(Relu(), [Min(axis=-1), Sum()], input="output")

    compiled_train_model = mithril.compile(
        model=ctx1, backend=JaxBackend(dtype=mithril.float64)
    )
    outputs, grads = compiled_train_model.evaluate_all(inputs)
    ref_outputs = {
        "output": backend.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "final_cost": backend.array(16.0),
    }
    ref_grads = {
        "input": backend.array([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
    }
    assert_results_equal(ref_outputs, outputs)
    assert_results_equal(grads, ref_grads)


def test_add_loss_case_4():
    model = Model()
    sigmoid1 = Sigmoid()
    sigmoid2 = Sigmoid()
    sigmoid3 = Sigmoid()

    model += sigmoid1(input="input")
    model += sigmoid2(input=sigmoid1.output)
    model += sigmoid3(input=sigmoid2.output, output=IOKey(name="final_cost"))

    with pytest.raises(Exception) as err_info:
        TrainModel(model)

    assert (
        str(err_info.value)
        == "\"'final_cost' could not be used as an external key in TrainModel!\""
    )


def test_add_loss_case_5():
    model = Model()
    lrelu1 = LeakyRelu()
    lrelu2 = LeakyRelu()
    lrelu3 = LeakyRelu()

    model += lrelu1(input="input")
    model += lrelu2(input=lrelu1.output)
    model += lrelu3(input=lrelu2.output, output=IOKey(name="output"))

    ctx1 = TrainModel(model)

    with pytest.raises(KeyError) as err_info:
        ctx1.add_loss(
            LeakyRelu(),
            [Min(axis=-1)],
            input=lrelu3.output,
            output=IOKey(name="output1"),
        )

    assert str(err_info.value) == "'Output of the loss model cannot be defined!'"


def test_add_loss_case_6():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()

    model += relu1(input="input")
    model += relu2(input=relu1.output)
    model += relu3(input=relu2.output, output=IOKey(name="output"))

    ctx1 = TrainModel(model)

    with pytest.raises(KeyError) as err_info:
        ctx1.add_loss(
            Relu(), [Min(axis=-1)], input=relu3.output, output=IOKey(name="output1")
        )

    assert str(err_info.value) == "'Output of the loss model cannot be defined!'"


def test_add_loss_case_7():
    LossModel = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()

    LossModel += relu1(input="input")
    LossModel += relu2(input=relu1.output, output=IOKey(name="output1"))
    LossModel += relu3(input=relu1.output, output=IOKey(name="output2"))

    model = Model()
    relu4 = Relu()
    relu5 = Relu()
    relu6 = Relu()

    model += relu4(input="input")
    model += relu5(input=relu4.output)
    model += relu6(input=relu5.output, output=IOKey(name="output"))

    ctx1 = TrainModel(model)
    with pytest.raises(Exception) as err_info:
        ctx1.add_loss(LossModel, [Min(axis=-1)], input="output")

    assert str(err_info.value) == "'All models in steps require single output.'"


def test_add_loss_case_8():
    backend = NumpyBackend()

    inputs = {
        "input": backend.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    }

    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()

    model += relu1(input="input")
    model += relu2(input=relu1.output, output=IOKey(name="output1"))
    model += relu3(input=relu1.output, output=IOKey(name="output2"))

    ctx1 = TrainModel(model)
    ctx1.add_loss(Relu(), [Sum(), Sum()], input="output1")
    ctx1.add_loss(Relu(), [Sum(), Sum()], input="output2")

    compiled_train_model = mithril.compile(model=ctx1, backend=backend)
    outputs, grads = compiled_train_model.evaluate_all(inputs)
    ref_outputs = {
        "output1": backend.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "output2": backend.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "final_cost": backend.array(72.0),
    }
    ref_grads = {
        "input": backend.array([[[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]])
    }
    assert_results_equal(ref_outputs, outputs)
    assert_results_equal(grads, ref_grads)


def test_add_loss_case_9():
    backend = NumpyBackend(dtype=mithril.float64)
    inputs = {
        "input": backend.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    }

    model = Model()
    sigmoid1 = Relu()
    sigmoid2 = Relu()
    sigmoid3 = Relu()

    model += sigmoid1(input="input")
    model += sigmoid2(input=sigmoid1.output, output=IOKey(name="output1"))
    model += sigmoid3(input=sigmoid1.output, output=IOKey(name="output2"))

    ctx1 = TrainModel(model)
    ctx1.add_loss(Relu(), [Sum(axis=(1,))], input="output1")
    ctx1.add_loss(Relu(), [Sum(axis=(1,))], input="output2")

    compiled_train_model = mithril.compile(model=ctx1, backend=backend)
    outputs = compiled_train_model.evaluate(inputs)
    grads = compiled_train_model.evaluate_gradients(inputs)
    ref_outputs = {
        "output1": backend.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "output2": backend.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "final_cost": backend.array(72.0),
    }
    ref_grads = {
        "input": backend.array([[[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]])
    }
    assert_results_equal(ref_outputs, outputs)
    assert_results_equal(grads, ref_grads)


def test_add_metric_1():
    model = Model()
    buff1 = Buffer()
    model += buff1(input="input")

    ctx2 = TrainModel(model)
    ctx2.add_loss(Relu(), input=buff1.output)
    ctx2.add_metric(Relu(), key_name="metric", input=buff1.output)

    backend = NumpyBackend()
    input = backend.randn(5, 5)
    c_model = mithril.compile(ctx2, backend, data_keys={"input"})
    result = c_model.evaluate({}, {"input": input})
    res_metric = result["metric"]
    assert isinstance(res_metric, np.ndarray)

    assert "metric" in ctx2.output_keys
    assert res_metric.shape == input.shape
    np.testing.assert_almost_equal(res_metric, np.where(input > 0, input, 0))


def test_add_metric_2():
    model = Model()
    buff1 = Buffer()
    model += buff1(input="input")

    ctx2 = TrainModel(model)
    ctx2.add_loss(Relu(), input=buff1.output)
    ctx2.add_metric(
        Relu(), key_name="metric", reduce_steps=[Mean()], input=buff1.output
    )

    backend = NumpyBackend()
    input = backend.randn(5, 5)
    c_model = mithril.compile(ctx2, backend, data_keys={"input"})
    result = c_model.evaluate({}, {"input": input})

    expected_metric = np.array(np.mean(np.where(input > 0, input, 0)))
    assert "metric" in ctx2.output_keys
    res_metric = result["metric"]
    assert isinstance(res_metric, np.ndarray)
    assert res_metric.shape == expected_metric.shape
    np.testing.assert_almost_equal(res_metric, expected_metric)


def test_add_regularization_case_1():
    model = Model()
    linear_1 = Linear()
    model += linear_1(
        output="output",
        **{key: key for key in linear_1.input_keys if not key.startswith("$")},
    )
    ctx = TrainModel(model)
    ctx.add_regularization(model=L2(), coef=1e-1, input="weight")
    with pytest.raises(Exception) as err_info:
        mithril.compile(model=ctx, backend=NumpyBackend(dtype=mithril.float64))

    assert str(err_info.value) == "Requires at least 1 attached loss!"


def test_add_regularization_case_2():
    model = Model()
    linear_1 = Linear()

    model.extend(
        linear_1,
        output="output",
        **{key: key for key in linear_1.input_keys if not key.startswith("$")},
    )
    ctx = TrainModel(model)
    with pytest.raises(KeyError) as err_info:
        ctx.add_regularization(model=L2(), coef=1e-1, input="weight", output="output1")

    assert (
        str(err_info.value)
        == "'The provided keys do not match the regularization model keys!'"
    )


def test_add_regularization_case_3():
    model = Model()
    linear_1 = Linear()
    model += linear_1(
        output="output",
        **{key: key for key in linear_1.input_keys if not key.startswith("$")},
    )

    ctx = TrainModel(model)
    ctx.add_regularization(l2_model_1 := L2(), coef=1e-1, input="weight")
    ctx.add_regularization(l2_model_2 := L2(), coef=1e-1, input="weight")

    assert_metadata_equal(l2_model_1.input, l2_model_2.input, linear_1.weight)


def test_add_regularization_case_6():
    model = Model()
    linear_1 = Linear()
    model += linear_1(
        output="output",
        **{key: key for key in linear_1.input_keys if not key.startswith("$")},
    )

    ctx = TrainModel(model)
    ctx.add_regularization(model_1 := L2(), coef=1e-1, input=linear_1.weight)
    ctx.add_regularization(model_2 := L2(), coef=1e-1, input=linear_1.weight)

    assert_metadata_equal(model_1.input, model_2.input, linear_1.weight)


def test_add_regularization_case_7():
    model = Model()
    linear_1 = Linear()
    model += linear_1(
        output="output",
        **{key: key for key in linear_1.input_keys if not key.startswith("$")},
    )

    ctx = TrainModel(model)
    ctx.add_regularization(model=L2(), coef=1e-1, input="weight")
    ctx.add_regularization(
        model_1 := QuadraticFormRegularizer(),
        coef=1e-1,
        input="weight",
        kernel="kernel",
    )

    assert_metadata_equal(linear_1.weight, model_1.input)


def test_add_regularization_case_7_exception():
    """Throws exception since trying to adding additional regularization
    after finalizing TrainModel.
    """
    model = Model()
    linear_1 = Linear()
    model += linear_1(
        output=IOKey(name="output"),
        **{key: key for key in linear_1.input_keys if not key.startswith("$")},
    )

    ctx = TrainModel(model)
    ctx.add_regularization(model=L2(), coef=Tensor(1e-1), input="weight")
    ctx.add_loss(SquaredError(), input="output", target="target")

    # Finalize train model.
    ctx.finalize()
    with pytest.raises(Exception) as err_info:
        ctx.add_regularization(
            model=QuadraticFormRegularizer(), coef=Tensor(1e-1), input="weight"
        )

    assert (
        str(err_info.value) == "No modifications can be made to a finalized TrainModel!"
    )


def test_autogenerated_key_regularization_integrated_linear_9():
    model = Model()
    model += Linear()(input="input", bias="b", output=IOKey(name="output"))

    ctx = TrainModel(model)
    # Here, user did not provide any naming for the weight to be regularized.
    # So he/she can find correpsonding autogenerated key from the summary and
    # provide the full-name or regex pattern for the key/keys.
    # TODO: Uncomment below summary call.
    ctx.add_regularization(model=L1(), coef=Tensor(1e-1), input=re.compile("weight$"))
    ctx.add_loss(SquaredError(), [Mean()], input="output", target="target")
    ctx.set_loss_combiner(Mean())

    backend = NumpyBackend(dtype=mithril.float64)
    data = {
        "input": backend.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.2]]),
        "target": backend.array([[1.0], [2.0], [3.0]]),
    }
    params = {"weight": backend.array([[2.0, 1.0]]), "b": backend.array([0.2])}

    # train_model = ctx.finalize_model()
    comp_train_model = mithril.compile(
        model=ctx,
        backend=backend,
        constant_keys=data,
    )

    result = comp_train_model.evaluate(params, data)
    gradients = comp_train_model.evaluate_gradients(params, data=data)
    res_out = result["output"]
    res_cost = result["final_cost"]
    assert isinstance(res_out, np.ndarray)
    assert isinstance(res_cost, np.ndarray)

    assert (backend.abs(res_out - backend.array([[0.5], [0.8], [1.0]])) <= 1e-14).all()
    assert (res_cost - 1.996666666666666666666666666666666666666667) <= 1e-14
    assert (
        backend.abs(
            gradients["weight"]
            - backend.array(
                [
                    [
                        -0.5599999999999999999666666666666666666667,
                        -0.42666666666666666666666667,
                    ],
                ]
            )
        )
        <= 1e-14
    ).all()
    assert (
        backend.abs(gradients["b"] - backend.array([-2.466666666666666666666666667]))
        <= 1e-14
    ).all()


def test_autogenerated_key_regularization_integrated_nn_7_regex():
    model = Model()
    model += MLP(dimensions=[3, 2], activations=[Sigmoid(), Softmax()])(
        input="input", output=IOKey(name="output")
    )

    ctx = TrainModel(model)
    # Here, user did not provide any naming for the weight to be regularized.
    # So he/she can find correpsonding autogenerated key from the summary and
    # provide the full-name or regex pattern for the key/keys.

    ctx.add_regularization(model=L2(), coef=Tensor(1e-1), input=re.compile("weight\\d"))
    ctx.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], input="output", target="target"
    )
    ctx.set_loss_combiner(Mean())

    backend = NumpyBackend(dtype=mithril.float64)
    data = {"input": backend.array([[1.0]]), "target": backend.array([0])}
    params = {
        "weight0": backend.array([[1.0], [2], [3]]),
        "weight1": backend.array([[-1.0, 0, 1], [-2, 0, 2]]),
        "bias0": backend.array([-2.0, -3, 0]),
        "bias1": backend.array([-5.0, 5]),
    }

    # train_model = ctx.finalize_model()
    comp_train_model = mithril.compile(
        model=ctx,
        backend=backend,
        constant_keys=data,
    )

    result = comp_train_model.evaluate(params, data)
    res_final_cost = result["final_cost"]
    res_out = result["output"]
    assert isinstance(res_final_cost, np.ndarray)
    assert isinstance(res_out, np.ndarray)
    gradients = comp_train_model.evaluate_gradients(params, data=data)

    assert (
        backend.abs(
            res_out
            - backend.array(
                [
                    [
                        0.000022916448682582498287877106161,
                        0.9999770835513174175017121228938392776607400542093695578010433420,
                    ]
                ]
            )
        )
        <= 1e-14
    ).all()
    assert (res_final_cost - 11.883655622163706) <= 1e-14
    assert (
        backend.abs(
            gradients["weight0"]
            - backend.array([[-0.09660742759420338], [0.2], [0.34517562444230764]])
        )
        <= 1e-14
    ).all()
    assert (
        backend.abs(
            gradients["weight1"]
            - backend.array(
                [
                    [-0.36893525818771367, -0.26893525818771363, -0.8525522972063397],
                    [0.06893525818771368, 0.2689352581877137, 1.15255229720634],
                ]
            )
        )
        <= 1e-14
    ).all()
    assert (
        backend.abs(
            gradients["bias0"]
            - backend.array([-0.19660742759420338, 0, 0.04517562444230764])
        )
        <= 1e-14
    ).all()
    assert (
        backend.abs(
            gradients["bias1"]
            - backend.array([-0.9999770835513174, 0.9999770835513175])
        )
        <= 1e-14
    ).all()


def test_train_model_extend():
    model = Model()
    model += MLP(dimensions=[3, 2], activations=[Sigmoid(), Softmax()])(
        input="input", output="output"
    )

    ctx = TrainModel(model)

    with pytest.raises(NotImplementedError) as err_info:
        ctx += Relu()

    assert str(err_info.value) == "TrainModel could not be extended!"


def test_train_model_extended():
    model = Model()
    model += MLP(dimensions=[3, 2], activations=[Sigmoid(), Softmax()])(
        input="input", output="output"
    )

    ctx = TrainModel(model)
    model = Model()

    with pytest.raises(AttributeError) as err_info:
        model += ctx

    assert str(err_info.value) == "TrainModel could extend any other model!"


def test_add_loss_compile_shape_1():
    model = Model()
    relu = Relu()
    model += relu(input="input")
    ctx1 = TrainModel(model)
    ctx1.add_loss(SquaredError(), input=relu.output, target="target")

    model = Model()
    relu = Relu()
    model += relu(input="input")
    ctx2 = TrainModel(model)
    ctx2.add_loss(SquaredError(), input=relu.output, target="target")

    initial_shapes = {
        "$_Model_0_output": ["(V1, ...)"],
        "$_SquaredError_1_output": ["(V2, ...)"],
        "$_Mean_2_output": [],
        "input": ["(V1, ...)"],
        "target": ["(V3, ...)"],
        "$_Mean_2_axis": None,
        "$_Mean_2_keepdim": None,
    }

    assert ctx2.shapes == ctx1.shapes == initial_shapes

    pm1 = mithril.compile(
        ctx1,
        safe_shapes=True,
        data_keys={"input", "target"},
        backend=JaxBackend(),
    )
    pm2 = mithril.compile(
        ctx2,
        safe_shapes=False,
        data_keys={"input", "target"},
        backend=JaxBackend(),
    )

    assert pm1.shapes == {
        "input": [None, "..."],
        "output_0": [None, "..."],
        "target": [None, "..."],
        "output_1": [None, "..."],
        "axis": None,
        "keepdim": None,
        "output_2": [],
    }

    assert pm2.shapes == {
        "input": ["..."],
        "output_0": ["..."],
        "target": ["..."],
        "output_1": ["..."],
        "axis": None,
        "keepdim": None,
        "output_2": [],
    }
