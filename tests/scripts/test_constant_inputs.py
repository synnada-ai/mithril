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

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from types import EllipsisType

import numpy as np
import pytest
import torch

import mithril
from mithril import JaxBackend, MlxBackend, NumpyBackend, TorchBackend
from mithril.framework.common import (
    NOT_GIVEN,
    TBD,
    Connect,
    Connection,
    ConnectionType,
    IOKey,
    NotAvailable,
    ToBeDetermined,
)
from mithril.framework.logical import ExtendInfo
from mithril.models import (
    MLP,
    Add,
    Arange,
    BinaryCrossEntropy,
    Buffer,
    Concat,
    Convolution1D,
    Convolution2D,
    CrossEntropy,
    Eye,
    Greater,
    LeakyRelu,
    Linear,
    Log,
    LogicalAnd,
    LogicalNot,
    MaxPool1D,
    MaxPool2D,
    Mean,
    Model,
    Multiply,
    PaddingConverter1D,
    PaddingConverter2D,
    PolynomialFeatures,
    Power,
    Relu,
    ScalarItem,
    Shape,
    Sigmoid,
    Sum,
    Tanh,
    ToTensor,
    TrainModel,
    Transpose,
    Where,
)
from mithril.utils.utils import PaddingType

from .test_utils import assert_results_equal


def get_array_device(array, type):
    match type:
        case "numpy":
            return "cpu"
        case "jax":
            return next(iter(array.devices())).platform
        case "torch":
            return array.device.type
        case "mlx":
            return "gpu"


def get_array_precision(array, type):
    if type == "mlx":
        return 8 * array.itemsize
    else:
        return 8 * array.dtype.itemsize


def check_if_installed(backend):
    try:
        backend()
        return True
    except RuntimeError:
        return False


def assert_all_backends_device_precision(model: Model):
    """This function tests that whether all precision and device
    handling algorithms of the library is working successfully.
    This function compiles the given model, randomizes the inputs with
    all possible devices and precisions that backend has,
    evaluates the output and evaluates the gradient of outputs.
    This function tests if all created outputs have correct device and precision.


    Args:
        model (Model): Model to be compiled
    """
    # Detect installed backends
    installed_backends: Iterable[
        type[NumpyBackend] | type[TorchBackend] | type[JaxBackend] | type[MlxBackend]
    ] = filter(check_if_installed, [NumpyBackend, JaxBackend, TorchBackend, MlxBackend])
    # Detect their supported device and precision
    backends_with_device_precision = (
        backend
        for backends in installed_backends
        for backend in product(
            [backends], backends.get_available_devices(), backends.supported_precisions
        )
    )
    unsupported_device_precisions = [
        (TorchBackend, "mps:0", 64),
        (MlxBackend, "cpu", 16),
        (MlxBackend, "cpu", 32),
        (TorchBackend, "cpu:0", 16),
    ]

    for backend_class, device, precision in backends_with_device_precision:
        # remove unsupported backend, device and precision trios
        if (backend_class, device, precision) in unsupported_device_precisions:
            continue
        _type = backend_class.type
        backend = backend_class(device=device, precision=precision)

        comp_model = mithril.compile(
            model=model,
            backend=backend,  # type: ignore
            jit=False,
            safe=False,
        )

        randomized_inputs = comp_model.randomize_params()  # type: ignore # (check after DataType update)
        initial_randomized_inputs = randomized_inputs.copy()

        if device[-2] == ":":
            device = device[:-2]

        # Check if randomized inputs have correct device and precision
        for randomized_input in randomized_inputs.values():
            assert get_array_device(randomized_input, _type) == device
            assert get_array_precision(randomized_input, _type) == precision

        outputs = comp_model.evaluate(randomized_inputs)
        initial_outputs = outputs.copy()

        # Check if outputs have correct device and precision
        for output in outputs.values():
            assert get_array_device(output, _type) == device
            assert get_array_precision(output, _type) == precision

        grads = comp_model.evaluate_gradients(
            output_gradients=outputs, params=randomized_inputs
        )

        # Check if gradients have correct device and precision
        for grad in grads.values():
            assert get_array_device(grad, _type) == device
            assert get_array_precision(grad, _type) == precision

        # In final step. we compare used inputs (used inputs are given as input to the
        # either to comp_model.evaluate() or comp_model.evaluate_gradients()) with their
        # non-used copies. It is expected that their values are exactly the same. Aim
        # of this check is to make sure that no in-place changes are occurred in given
        # inputs.
        if device == "cpu":
            for val1, val2 in zip(
                randomized_inputs.values(),
                initial_randomized_inputs.values(),
                strict=False,
            ):
                np.testing.assert_array_equal(np.array(val1), np.array(val2))

            for val1, val2 in zip(
                outputs.values(), initial_outputs.values(), strict=False
            ):
                np.testing.assert_array_equal(np.array(val1), np.array(val2))


class ReduceMult(Model):
    input: Connection
    axis: Connection
    output: Connection

    def __init__(
        self,
        dimension: int | None = None,
        axis: int | tuple[int, ...] | None | ToBeDetermined = TBD,
    ) -> None:
        super().__init__()
        rdc = Mean(axis=axis)
        self += rdc(input="input", axis="axis")
        self += Multiply()(left=rdc.output, right=2.0, output=IOKey(name="output"))
        shapes: Mapping[str, Sequence[str | tuple[str, EllipsisType]]] = {
            "input": ["N", ("Var_inter", ...), "d_in"]
        }
        self.set_shapes(shapes)

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "axis": axis, "output": output}
        return ExtendInfo(self, kwargs)


np_input = np.random.randn(10, 3, 3).astype(np.float32)


def test_default_in_numpy_error():
    """This test should raise ValueError because we are providing a new
    value in compile for "axis" key which is set as 0 before.
    """
    model = Model()
    model1 = ReduceMult(axis=0)
    model2 = Mean(axis=TBD)

    model += model1(input="input", axis="axis")
    model += model2(input=model1.output, axis=model1.axis, output=IOKey(name="output"))
    static_inputs: dict[str, np.ndarray | ToBeDetermined] = {
        "input": np_input,
        "axis": TBD,
    }
    backend = NumpyBackend()
    with pytest.raises(ValueError) as err_info:
        mithril.compile(
            model=model,
            backend=backend,
            static_keys=static_inputs,
            # file_path = "constant_inputs"
        )
    assert (
        str(err_info.value)
        == "Statically given key: axis has been already set as static with a value!"
    )


def test_make_static_numpy_error():
    """We are providing a new Ellipsis value in compile
    for "axis" key which is set as 0 before.
    """
    model = Model()
    mean_model = Mean(axis=TBD)

    mult_out = IOKey(name="mult_out")

    rdc = Mean(axis=0)
    model += rdc(input="input", axis="axis")
    model += Multiply()(left=rdc.output, right=0, output=mult_out)
    model += mean_model(input=mult_out, axis="axis", output=IOKey(name="output"))
    static_inputs: dict[str, np.ndarray | ToBeDetermined] = {
        "input": np_input,
        "axis": TBD,
    }
    with pytest.raises(ValueError) as err_info:
        mithril.compile(model=model, backend=NumpyBackend(), static_keys=static_inputs)
    assert (
        str(err_info.value)
        == "Statically given key: axis has been already set as static with a value!"
    )


def test_default_given_extend_1_numpy():
    """Multiwrite error since "axis" is an input and connected to a valued
    connection."""
    # TODO: In this test actually there exists a multiwrite for axis since
    # "axis" is exposed as input for the model but connected to an inner input
    # with a value 0. This should not be allowed.
    model = Model()
    model1 = ReduceMult()
    model2 = Mean(axis=0)
    model += model1(input="input", axis="axis")
    with pytest.raises(ValueError) as err_info:
        model += model2(
            input=model1.output, axis=model1.axis, output=IOKey(name="output")
        )
    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_default_given_extend_3_numpy_error():
    """This test should raise ValueError since output shape_map of model1 is [] and
    required input shape_map of model2 is ["x", (Var, TBD)].
    NOTE: This test is not related to this file. It's only a trial.
    """
    model = Model()
    model1 = ReduceMult(axis=None)
    model2 = Mean(axis=0)
    model += model1(input="input", axis="axis")
    with pytest.raises(ValueError) as err_info:
        model += model2(input=model1.output, output=IOKey(name="output"))
    assert str(err_info.value) == "Requires minimum of 1 dimensionality, got 0."


def test_default_given_compile_numpy():
    """This test should work properly and generate the same results as in
    'test_default_given_extend_2_numpy' test. We provide "axis" input in compile.
    """
    model = Model()
    model1 = ReduceMult()
    model2 = Mean(axis=TBD)
    model += model1(input="input", axis="axis")
    model += model2(input=model1.output, axis=model1.axis, output=IOKey(name="output"))
    static_inputs: dict[str, np.ndarray | int] = {"input": np_input, "axis": 0}
    expected_result = (np_input.mean(0) * 2).mean(0)
    compiled_model = mithril.compile(
        model=model,
        backend=NumpyBackend(),
        static_keys=static_inputs,
        # file_path = "constant_inputs.py"
    )
    inputs = compiled_model.randomize_params()
    data = {"axis": None}

    result = compiled_model.evaluate(inputs, data)
    output_gradients = {"output": np.ones_like(result["output"])}
    compiled_model.evaluate_gradients(
        params=inputs, data=data, output_gradients=output_gradients
    )
    np.testing.assert_array_equal(expected_result, result["output"])


def test_default_given_extend_numpy_3():
    """This test should work properly and generate the same results as in
    'test_default_given_extend_2_numpy' test. We provide "axis" input
    in extend after wrapping all model in a new model.
    """
    model = Model()
    model1 = ReduceMult()
    model2 = Mean(axis=TBD)
    model += model1(input=IOKey(name="input", shape=[*np_input.shape]), axis="axis")
    model += model2(input=model1.output, axis=model1.axis, output=IOKey(name="output"))
    final_model = Model()
    final_model += model(axis=0, input="input", output=IOKey(name="output"))
    expected_result = (np_input.mean(0) * 2).mean(0)
    compiled_model = mithril.compile(
        model=final_model,
        backend=NumpyBackend(),
        static_keys={"input": TBD},
        safe=False,
    )
    inputs = compiled_model.randomize_params()
    data = {"input": np_input}

    result = compiled_model.evaluate(inputs, data)
    output_gradients = {"output": np.ones_like(result["output"])}
    compiled_model.evaluate_gradients(
        params=inputs, data=data, output_gradients=output_gradients
    )
    np.testing.assert_array_equal(expected_result, result["output"])


def test_default_given_extend_numpy_3_set_values():
    """Same with test_default_given_extend_numpy_3 but
    set_values is used instead of providing axis in extend.
    """
    model = Model()
    model1 = ReduceMult()
    model2 = Mean(axis=TBD)
    model += model1(input=IOKey(name="input", shape=[*np_input.shape]), axis="axis")
    model += model2(input=model1.output, axis=model1.axis, output=IOKey(name="output"))
    final_model = Model()
    final_model += model(axis="axis", input="input", output=IOKey(name="output"))
    final_model.set_values({"axis": 0})
    expected_result = (np_input.mean(0) * 2).mean(0)
    compiled_model = mithril.compile(
        model=final_model,
        backend=NumpyBackend(),
        static_keys={"input": TBD},
        safe=False,
    )
    inputs = compiled_model.randomize_params()
    data = {"input": np_input}

    result = compiled_model.evaluate(inputs, data)
    output_gradients = {"output": np.ones_like(result["output"])}
    compiled_model.evaluate_gradients(
        params=inputs, data=data, output_gradients=output_gradients
    )
    np.testing.assert_array_equal(expected_result, result["output"])


def test_constant_given_data_numpy():
    """This test should work properly and generate the same results as in
    'test_default_given_compile_numpy' test. We provide "axis" input in evaluate.
    """
    model = Model()
    model1 = ReduceMult()
    model2 = Mean(axis=TBD)
    model += model1(input="input", axis="axis")
    model += model2(input=model1.output, axis=model1.axis, output=IOKey(name="output"))
    static_inputs = {
        "input": np_input,
    }
    expected_result = (np_input.mean(0) * 2).mean(0)
    compiled_model = mithril.compile(
        model=model,
        backend=NumpyBackend(),
        static_keys=static_inputs,
        # file_path = "constant_inputs.py"
    )

    inputs = compiled_model.randomize_params()
    data = {"axis": 0}

    result = compiled_model.evaluate(inputs, data)
    output_gradients = {"output": np.ones_like(result["output"])}
    compiled_model.evaluate_gradients(
        params=inputs, data=data, output_gradients=output_gradients
    )
    np.testing.assert_array_equal(expected_result, result["output"])


def test_constant_numpy():
    """Tests if right error is thrown for the case when Ellipsis Scalar value (axis)
    is changed by another model's axis which is 0. ValueError must be raised in
    reduce_constraint function because shape_map can not be updated with new axis
    value for Mean model which had axis = TBD before. Since shape map didn't updated
    for the new axis value, we should check minimum input rank requirement in
    reduce_constraint for the BackendVar which is "rhs" of Multiply model. It's rank
    is 0 since it is initialized as 2.0.
    """
    model = Model()
    mean_model = Mean(axis=TBD)
    mult_out = IOKey(name="mult_out")

    rdc = Mean(axis=0)
    model += rdc(input="input", axis="axis")
    model += Multiply()(
        left=rdc.output, right=IOKey(value=2.0, name="rhs"), output=mult_out
    )
    model += mean_model(input=mult_out, axis="axis", output=IOKey(name="output"))
    other_model = Model()
    other_model += Mean(axis=TBD)(input="input", axis="axis")
    with pytest.raises(ValueError) as err_info:
        model += other_model(input=model.rhs, axis=model.axis)  # type: ignore
    assert (
        str(err_info.value)
        == "Input rank is 0. Minimum rank 1 input is required for axis = (0,)."
    )


def test_constant_numpy_set_values():
    """Same with test_constant_numpy but set_values
    is used instead of providing rhs value in extend.
    """
    model = Model()
    mean_model = Mean(axis=TBD)
    rdc = Mean(axis=0)

    mult_out = IOKey(name="mult_out")
    model += rdc(input="input", axis="axis")
    model += Multiply()(left=rdc.output, right=IOKey(name="rhs"), output=mult_out)
    model.set_values({"rhs": 2.0})
    model += mean_model(input=mult_out, axis="axis", output=IOKey(name="output"))
    other_model = Model()
    other_model += Mean(axis=TBD)(input="input", axis="axis")
    with pytest.raises(ValueError) as err_info:
        model += other_model(input=model.rhs, axis=model.axis)  # type: ignore
    assert (
        str(err_info.value)
        == "Input rank is 0. Minimum rank 1 input is required for axis = (0,)."
    )


def test_axis():
    model = Model()
    relu = LeakyRelu(slope=2.3)
    rob_pow = Power(robust=True, threshold=TBD)
    model += relu(input="input")
    model += rob_pow(base=relu.output, exponent="exponent", threshold=relu.slope)

    backend = NumpyBackend()
    compiled_model = mithril.compile(
        model=model,
        backend=backend,
        safe=False,
        jit=False,
        shapes={"input": [4, 5, 8], "exponent": [4, 5, 8]},
    )
    input = {"input": np.random.rand(4, 5, 8), "exponent": np.random.rand(4, 5, 8)}

    expected_result = compiled_model.backend.leaky_relu(
        input["input"], backend.array(2.3)
    )
    expected_result = expected_result ** input["exponent"]

    compiled_model.evaluate(input)
    compiled_model.evaluate_gradients(
        input, output_gradients={"output": np.random.rand(4, 5, 8)}
    )
    assert type(backend.array(2.3)), type(
        compiled_model.data_store.cached_data["slope_1"].value
    )


def test_axis_1():
    model = Model()
    relu = LeakyRelu(slope=TBD)
    rob_pow = Power(robust=True, threshold=2.3)
    model += rob_pow(base="base", exponent="exponent")
    model += relu(input=rob_pow.output, slope=rob_pow.threshold)
    # Check required value transfer occured in logical model
    # assert relu.conns.get_data("slope").value == 2.3

    backend = NumpyBackend()
    compiled_model = mithril.compile(
        model=model,
        backend=backend,
        safe=False,
        jit=False,
        shapes={"base": [4, 5, 8], "exponent": [4, 5, 8]},
    )
    input = {"base": np.random.rand(4, 5, 8), "exponent": np.random.rand(4, 5, 8)}
    compiled_model.evaluate(input)
    compiled_model.evaluate_gradients(
        input, output_gradients={"output": np.random.rand(4, 5, 8)}
    )
    assert type(backend.array(2.3)), type(
        compiled_model.data_store.cached_data["threshold_1"].value
    )


def test_mean_1():
    """This test contains two model, first model consists three consecutive mean
    models that share same axis (1) second model ise a buffer model with staticly
    initialized 2x2 shape input, second model's output is connected to first model's
    input. It is expected an error to raise when connectiong two models becuase three
    consecutive mean model means that it will reduce input's dimensionality by 3,
    However, since static input has already shape of 2x2, this cannot be possible
    """
    mean_model = Model()
    mean_1 = Mean(axis=TBD)
    mean_2 = Mean(axis=TBD)
    mean_3 = Mean(axis=TBD)
    mean_model += mean_1(axis=1, input="input")
    mean_model += mean_2(input=mean_1.output, axis=mean_1.axis)
    mean_model += mean_3(
        input=mean_2.output, axis=mean_2.axis, output=IOKey(name="output")
    )

    model = Model()
    buff1 = Buffer()
    model += buff1(
        input=IOKey(value=[[2.0, 3.0], [1.0, 7.0]], name="input"),
        output=IOKey(name="output"),
    )
    with pytest.raises(ValueError) as err_info:
        model += mean_model(input=buff1.output, output=IOKey(name="output1"))
    assert str(err_info.value) == "Requires minimum of 4 dimensionality, got 2."


def test_mean_1_set_values_1():
    """Same as test_mean_1 but set_values is used
    on mean_model instead of providing axis in extend.
    """
    mean_model = Model()
    mean_1 = Mean(axis=TBD)
    mean_2 = Mean(axis=TBD)
    mean_3 = Mean(axis=TBD)
    mean_model += mean_1(input="input")
    mean_model += mean_2(input=mean_1.output, axis=mean_1.axis)
    mean_model += mean_3(
        input=mean_2.output, axis=mean_2.axis, output=IOKey(name="output")
    )
    mean_model.set_values({mean_1.axis: 1})

    model = Model()
    buff1 = Buffer()
    model += buff1(
        input=IOKey(value=[[2.0, 3.0], [1.0, 7.0]], name="input"),
        output=IOKey(name="output"),
    )
    # model.make_static("input", [[2.0, 3.0], [1.0, 7.0]])
    with pytest.raises(ValueError) as err_info:
        model += mean_model(input=buff1.output, output=IOKey(name="output1"))
    assert str(err_info.value) == "Requires minimum of 4 dimensionality, got 2."


def test_mean_1_set_values_2():
    """Same as test_mean_1 but set_values is used
    on mean_model instead of providing axis in extend.
    """
    mean_model = Model()
    mean_1 = Mean(axis=TBD)
    mean_2 = Mean(axis=TBD)
    mean_3 = Mean(axis=TBD)
    mean_model += mean_1(input="input")
    mean_model += mean_2(input=mean_1.output, axis=mean_1.axis)
    mean_model += mean_3(
        input=mean_2.output, axis=mean_2.axis, output=IOKey(name="output")
    )
    mean_1.set_values({"axis": 1})

    model = Model()
    buff1 = Buffer()
    model += buff1(
        input=IOKey(value=[[2.0, 3.0], [1.0, 7.0]], name="input"),
        output=IOKey(name="output"),
    )
    with pytest.raises(ValueError) as err_info:
        model += mean_model(input=buff1.output, output=IOKey(name="output1"))
    assert str(err_info.value) == "Requires minimum of 4 dimensionality, got 2."


def test_scalar_mean_2_1():
    mean_model = Model()
    mean_1 = Mean()
    with pytest.raises(ValueError) as err_info:
        mean_model += mean_1(axis=1, input="input")
    assert (
        str(err_info.value)
        == "Value is set before as None. A scalar value can not be reset."
    )


def test_scalar_mean_2_2():
    mean_model = Model()
    rob_pow = Power(robust=True, threshold=1.3)
    with pytest.raises(ValueError) as err_info:
        mean_model += rob_pow(threshold=1.5, base="input")
    assert (
        str(err_info.value) == "Value of Power's threshold given as 1.5. "
        "But the value is already initialized as 1.3"
    )


def test_scalar_threshold():
    model = Model()
    rob_pow = Power(robust=True)
    with pytest.raises(ValueError) as err_info:
        model += rob_pow(threshold=1.5, base="input")
    assert (
        str(err_info.value) == "Value of Power's threshold given as 1.5. "
        "But the value is already initialized as Constant.MIN_POSITIVE_NORMAL"
    )


def test_scalar_mean_2_set_values():
    mean_model = Model()
    mean_1 = Mean()

    with pytest.raises(ValueError) as err_info_1:
        mean_model += mean_1(input="input")
        mean_1.set_values({"axis": 1})
    assert (
        str(err_info_1.value)
        == "Value is set before as None. A scalar value can not be reset."
    )

    # TODO: Complete this test after CONSTANT handling is implemented.
    # with pytest.raises(ValueError) as err_info_2:
    #     mean_model.extend(rob_pow, threshold = 1.5, base = "input")


def test_reduce_axis_error():
    model = Model()
    mean_1 = Mean(axis=TBD)
    add_2 = Sum(axis=TBD)
    mean_2 = Mean(axis=1)

    model += mean_1(input="input", axis="axis", output="out1")
    model += add_2(input=mean_1.input, axis=mean_1.axis, output="out2")
    with pytest.raises(ValueError) as err_info:
        model += mean_2(input="input2", axis=mean_1.axis, output="out3")

    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_scalar_1():
    """This test should raise an error since we are trying to connect an output
    connection to a valued input connection. If model1 is initialized with
    enforce_jit=True, jittabilty error would be raised since a Tensor to
    Scalar conversion is needed from left input of add_1 which is Tensor,
    to left_2 which is valued Scalar.
    """
    model1 = Model(enforce_jit=False)
    model2 = Model()
    add_1 = Add()
    add_2 = Add()
    model1 += add_1(left=[4.0, 5.0], right=[8.0, 9.0])
    model2 += add_2(
        left=IOKey(name="left_2", value=[7.0, 11.0]), output=IOKey(name="output")
    )
    with pytest.raises(ValueError) as err_info:
        model1 += model2(left_2=add_1.output)
    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_scalar_1_set_values():
    model1 = Model(enforce_jit=False)
    model2 = Model()
    add_1 = Add()
    add_2 = Add()
    model1 += add_1
    model1.set_values({add_1.left: [4.0, 5.0], add_1.right: [8.0, 9.0]})
    model2 += add_2(
        left=IOKey(name="left_2", value=[7.0, 11.0]), output=IOKey(name="output")
    )
    with pytest.raises(ValueError) as err_info:
        model1 += model2(left_2=add_1.output)
    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_scalar_2():
    model = Model()
    add = Add()
    with pytest.raises(KeyError) as err_info:
        model += add(left=[4.0, 5.0], right=[8.0, 9.0], output=[7.0, 8.0])
    assert str(err_info.value) == (
        "'output key is an output of the model, output values could not be "
        "set in extend.'"
    )


def test_scalar_2_set_values():
    model = Model()
    add = Add()
    model += add(left="left", right="right", output="output")
    with pytest.raises(Exception) as err_info:
        model.set_values(
            {"left": [4.0, 5.0], "right": [8.0, 9.0], "output": [7.0, 8.0]}
        )

    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_scalar_3():
    model1 = Model()
    add_2 = Add()
    add_1 = Add()
    model1 += add_2
    with pytest.raises(KeyError) as err_info:
        model1 += add_1(left="left", right="right", output=[add_2.left, 4.0])
    assert str(err_info.value) == (
        "'output key is an output of the model, output values could not be "
        "set in extend.'"
    )


def test_scalar_3_set_values():
    model1 = Model(enforce_jit=False)
    add_2 = Add()
    add_1 = Add()
    model1 += add_2
    with pytest.raises(Exception) as err_info:
        model1 += add_1(left="left", right="right", output="output")
        model1.set_values({"output": [add_2.left, 4.0]})

    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_scalar_4():
    model1 = Model()
    add_1 = Add()
    with pytest.raises(Exception) as err_info:
        model1 += add_1(left="left", right="right", output=3.0)
    assert str(err_info.value) == (
        "'output key is an output of the model, output values could not be "
        "set in extend.'"
    )


def test_scalar_4_set_values():
    model1 = Model(enforce_jit=False)
    add_1 = Add()
    with pytest.raises(Exception) as err_info:
        model1 += add_1(left="left", right="right", output="output")
        model1.set_values({"output": 3.0})

    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_static_1():
    """When add_1.left is set as static a ToTensor operation is added to the model,
    so this connection is an output of a model. When we try to set a new value for
    this connection, it should raise an error since another ToTensor output is
    tried to connect to the same connection. This is a multi-write error for an
    internal key!
    """
    model1 = Model()
    add_1 = Add()
    model1 += add_1(left=[2.0, 3.0], right="right", output=IOKey(name="output"))
    with pytest.raises(Exception) as err_info:
        model1.set_values({add_1.left: [3.0, 4.0]})
    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_static_2():
    model1 = Model()
    model2 = Model()
    add_1 = Add()
    model1 += add_1(left=[2.0, 3.0], right="right", output=IOKey(name="output"))
    model2 += model1
    comp_model = mithril.compile(model=model2, backend=NumpyBackend())
    import numpy as np

    infered_value = comp_model.data_store._cached_data[
        "_Model_0_ToTensor_0_output"
    ].value

    assert isinstance(infered_value, np.ndarray)
    np.testing.assert_almost_equal(
        infered_value,
        np.array([2.0, 3.0]),
    )
    params = {"right": np.array(1.0)}
    output_grads = {"output": np.array([1.0, 1.0])}
    ref_output = {"output": np.array([3.0, 4.0])}
    ref_grads = {"right": np.array(2.0)}
    outputs, grads = comp_model.evaluate_all(params, output_gradients=output_grads)
    assert_results_equal(ref_output, outputs)
    assert_results_equal(ref_grads, grads)


def test_static_2_set_values():
    model1 = Model()
    model2 = Model()
    add_1 = Add()
    model1 += add_1(right="right", output=IOKey(name="output"))
    model1.set_values({add_1.left: [2.0, 3.0]})
    model2 += model1
    comp_model = mithril.compile(model=model2, backend=NumpyBackend())

    infered_value = comp_model.data_store._cached_data[
        "_Model_0_ToTensor_0_output"
    ].value

    assert isinstance(infered_value, np.ndarray)
    np.testing.assert_almost_equal(
        infered_value,
        np.array([2.0, 3.0]),
    )
    params = {"right": np.array(1.0)}
    output_grads = {"output": np.array([1.0, 1.0])}
    ref_output = {"output": np.array([3.0, 4.0])}
    ref_grads = {"right": np.array(2.0)}
    outputs, grads = comp_model.evaluate_all(params, output_gradients=output_grads)
    assert_results_equal(ref_output, outputs)
    assert_results_equal(ref_grads, grads)


def test_static_3():
    model1 = Model()
    model2 = Model()
    add_1 = Add()
    model1 += add_1(left=[2.0, 3.0], right="right", output=IOKey(name="output"))
    model2 += model1
    assert not isinstance(model2.canonical_input, NotAvailable)
    key = model2.canonical_input.data.key
    with pytest.raises(Exception) as err:
        mithril.compile(
            model=model2, backend=NumpyBackend(), static_keys={key: [3.0, 4.0]}
        )

    assert str(err.value) == (
        "Statically given key: input has been already set as static with a value!"
    )


def test_static_3_set_values():
    model1 = Model()
    model2 = Model()
    add_1 = Add()
    model1 += add_1(right="right", output=IOKey(name="output"))
    model1.set_values({add_1.left: [2.0, 3.0]})
    model2 += model1
    assert not isinstance(model2.canonical_input, NotAvailable)
    key = model2.canonical_input.data.key
    with pytest.raises(Exception) as err:
        mithril.compile(
            model=model2, backend=NumpyBackend(), static_keys={key: [3.0, 4.0]}
        )

    assert str(err.value) == (
        "Statically given key: input has been already set as static with a value!"
    )


def test_static_4():
    model = Model()
    model += Greater()(left="input", right=0.6)
    model += Where()(cond=model.canonical_output, input1=1, input2=0)

    backend = TorchBackend()
    compiled_model = mithril.compile(
        model, backend, static_keys={"input": TBD}, inference=True
    )

    expected = {
        "_ToTensor_0_output": backend.array(0.6),
        "_ToTensor_2_output": backend.array(1),
        "_ToTensor_3_output": backend.array(0),
    }
    for key, value in expected.items():
        assert compiled_model.data_store.cached_data[key].value == value


def test_static_4_set_values():
    model = Model()
    model += (gr := Greater())(left="input")
    model.set_values({gr.right: 0.6})
    model += Where()(cond=model.canonical_output, input1=1, input2=0)

    backend = TorchBackend()
    compiled_model = mithril.compile(
        model, backend, static_keys={"input": TBD}, inference=True
    )

    expected = {
        "_ToTensor_0_output": backend.array(0.6),
        "_ToTensor_2_output": backend.array(1),
        "_ToTensor_3_output": backend.array(0),
    }
    for key, value in expected.items():
        assert compiled_model.data_store.cached_data[key].value == value


def test_str_axis():
    with pytest.raises(ValueError) as err_info:
        Mean(axis="axis")  # type: ignore

    assert str(err_info.value) == "Requires valid axis type!"


def test_str_axis_set_shapes():
    mean = Mean(axis=TBD)
    with pytest.raises(TypeError) as err_info:
        mean.set_values({"axis": "axis"})  # type: ignore

    assert str(err_info.value) == (
        "Acceptable types are int | tuple[int, ...] | None | list[int], "
        "but <class 'str'> type value is provided!"
    )


def test_float_axis_2():
    model1 = Model()
    mean1 = Mean(axis=TBD)
    with pytest.raises(TypeError) as err_info:
        model1 += mean1(axis=3.0)
    assert str(err_info.value) == (
        "Acceptable types are int | tuple[int, ...] | None | list[int], but "
        "<class 'float'> type value is provided!"
    )


def test_float_axis_2_set_values():
    mean1 = Mean(axis=TBD)
    with pytest.raises(TypeError) as err_info:
        mean1.set_values({"axis": 3.0})
    assert str(err_info.value) == (
        "Acceptable types are int | tuple[int, ...] | None | list[int], but "
        "<class 'float'> type value is provided!"
    )


def test_float_axis_3():
    with pytest.raises(ValueError) as err_info:
        Mean(axis=2.3)  # type: ignore
    assert str(err_info.value) == "Requires valid axis type!"


def test_static_type():
    model1 = Model()
    poly_feat_1 = PolynomialFeatures(degree=TBD)
    conv2d = Convolution2D(stride=TBD, kernel_size=3)
    model1 += conv2d(input="", stride=(2, 3))
    with pytest.raises(Exception) as err:
        model1 += poly_feat_1(input="", degree=conv2d.stride)

    assert str(err.value) == (
        "Acceptable types are tuple[int, int], but <class 'int'> "
        "type value is provided!"
    )


def test_static_type_set_value():
    model1 = Model()
    poly_feat_1 = PolynomialFeatures(degree=TBD)
    conv2d = Convolution2D(stride=TBD, kernel_size=3)
    model1 += conv2d(input="")
    model1.set_values({conv2d.stride: (2, 3)})
    with pytest.raises(Exception) as err:
        model1 += poly_feat_1(input="", degree=conv2d.stride)

    assert str(err.value) == (
        "Acceptable types are tuple[int, int], but <class 'int'> "
        "type value is provided!"
    )


# TODO: why this test here?
def test_nontensor_extend_from_input_multiple_connection():
    model = Model()
    mean1, mean2, mean3, mean4 = (
        Mean(axis=TBD),
        Mean(axis=TBD),
        Mean(axis=TBD),
        Mean(axis=TBD),
    )
    model += mean1
    model += mean2
    model += mean3
    model += mean4(axis=Connect(mean1.axis, mean2.axis, mean3.axis))
    assert (
        mean1.axis.data.metadata
        == mean2.axis.data.metadata
        == mean3.axis.data.metadata
        == mean4.axis.data.metadata
    )


# TODO: Why this test here?
def test_bool_tensor():
    model = Model()
    and1 = LogicalAnd()
    model += and1(left="in1", right="in2", output=IOKey(name="output"))
    comp_model = mithril.compile(
        model=model, backend=NumpyBackend(), safe=False, inference=True
    )
    assert comp_model.ignore_grad_keys == {"output"}


def test_bool_tensor_numpy_32():
    model = Model()
    not_1 = LogicalNot()
    add_1 = Add()
    ref = np.array([8.0, 9.0])
    model += not_1(input=IOKey(value=[False, False], name="input"))
    model += add_1(left=[7.0, 8.0], right=not_1.output, output=IOKey(name="output"))
    comp_model = mithril.compile(model=model, backend=NumpyBackend())
    output = comp_model.evaluate()["output"]
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_bool_tensor_numpy_32_set_values():
    model = Model()
    not_1 = LogicalNot()
    add_1 = Add()
    ref = np.array([8.0, 9.0])
    model += not_1(input=IOKey(name="input", value=TBD))
    model += add_1(left=[7.0, 8.0], right=not_1.output, output=IOKey(name="output"))
    model.set_values({model.input: [False, False]})  # type: ignore
    comp_model = mithril.compile(model=model, backend=NumpyBackend())
    output = comp_model.evaluate()["output"]
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_bool_tensor_numpy_64():
    model = Model()
    not_1 = LogicalNot()
    add_1 = Add()
    ref = np.array([8.0, 9.0])
    model += not_1(input=IOKey(value=[False, False], name="input"))
    model += add_1(left=[7.0, 8.0], right=not_1.output, output=IOKey(name="output"))
    comp_model = mithril.compile(model=model, backend=NumpyBackend(precision=64))
    output = comp_model.evaluate()["output"]
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float64


def test_bool_tensor_torch_32():
    model = Model()
    not_1 = LogicalNot()
    add_1 = Add()
    ref = np.array([8.0, 9.0])
    model += not_1(input=IOKey(value=[False, False], name="input"))
    model += add_1(left=[7.0, 8.0], right=not_1.output, output=IOKey(name="output"))
    comp_model = mithril.compile(model=model, backend=TorchBackend(precision=32))
    output = comp_model.evaluate()["output"].numpy()
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_bool_tensor_torch_64():
    model = Model()
    not_1 = LogicalNot()
    add_1 = Add()
    ref = np.array([8.0, 9.0])
    model += not_1(input=IOKey(value=[False, False], name="input"))
    model += add_1(left=[7.0, 8.0], right=not_1.output, output=IOKey(name="output"))
    comp_model = mithril.compile(model=model, backend=TorchBackend(precision=64))
    output = comp_model.evaluate()["output"].numpy()
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float64


def test_bool_tensor_jax_32():
    model = Model()
    not_1 = LogicalNot()
    add_1 = Add()
    ref = np.array([8.0, 9.0])
    model += not_1(input=IOKey(value=[False, False], name="input"))
    model += add_1(left=[7.0, 8.0], right=not_1.output, output=IOKey(name="output"))
    comp_model = mithril.compile(model=model, backend=JaxBackend(precision=32))
    output = np.array(comp_model.evaluate()["output"])
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_bool_tensor_jax_64():
    model = Model()
    not_1 = LogicalNot()
    add_1 = Add()
    ref = np.array([8.0, 9.0])
    model += not_1(input=IOKey(value=[False, False], name="input"))
    model += add_1(left=[7.0, 8.0], right=not_1.output, output=IOKey(name="output"))
    comp_model = mithril.compile(model=model, backend=JaxBackend(precision=64))
    output = np.array(comp_model.evaluate()["output"])
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float64


def test_bool_tensor_mlx_32():
    model = Model()
    not_1 = LogicalNot()
    add_1 = Add()
    ref = np.array([8.0, 9.0])
    model += not_1(input=IOKey(value=[False, False], name="input"))
    model += add_1(left=[7.0, 8.0], right=not_1.output, output=IOKey(name="output"))
    comp_model = mithril.compile(model=model, backend=JaxBackend(precision=32))
    output = np.array(comp_model.evaluate()["output"])
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_bool_tensor_mlx_64():
    model = Model()
    not_1 = LogicalNot()
    add_1 = Add()
    ref = np.array([8.0, 9.0])
    model += not_1(input=IOKey(value=[False, False], name="input"))
    model += add_1(left=[7.0, 8.0], right=not_1.output, output=IOKey(name="output"))
    comp_model = mithril.compile(model=model, backend=JaxBackend(precision=64))
    output = np.array(comp_model.evaluate()["output"])
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float64


def test_static_input_1():
    model = Model()
    add_1 = Add()
    ref = np.array(5.0)
    model += add_1(left=TBD, right=TBD)
    comp_model = mithril.compile(
        model=model, backend=NumpyBackend(precision=32), jit=False
    )

    output = comp_model.evaluate(
        data={
            "input": np.array(2.0, dtype=np.float32),
            "right": np.array(3.0, dtype=np.float32),
        }
    )["output"]
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_static_input_2():
    model = Model()
    add_1 = Add()
    ref = np.array(5.0)
    model += add_1(left=TBD, right=TBD)
    comp_model = mithril.compile(
        model=model,
        backend=NumpyBackend(precision=32),
        jit=False,
        static_keys={
            "input": np.array(2.0, dtype=np.float32),
            "right": np.array(3.0, dtype=np.float32),
        },
    )

    output = comp_model.evaluate()["output"]
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_static_input_3():
    backend = NumpyBackend(precision=32)
    model = Model()
    add_1 = Add()
    ref = np.array(5.0)
    model += add_1(left=TBD, right=TBD)
    comp_model = mithril.compile(
        model=model,
        backend=backend,
        jit=False,
        static_keys={"input": backend.array(2.0), "right": backend.array(3.0)},
    )

    output = comp_model.evaluate()["output"]
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_static_input_4():
    backend = NumpyBackend(precision=32)
    model = Model()
    add_1 = Add()
    ref = np.array(5.0)
    model += add_1(left="in1", right="in2")
    comp_model = mithril.compile(
        model=model, backend=backend, jit=False, static_keys={"in1": TBD, "in2": TBD}
    )

    output = comp_model.evaluate(
        data={
            "in1": np.array(2.0, dtype=np.float32),
            "in2": np.array(3.0, dtype=np.float32),
        }
    )["output"]
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_static_input_5():
    model = Model()
    add_1 = Add()
    ref = np.array(5.0)
    model += add_1(left=TBD, right=TBD)
    comp_model = mithril.compile(
        model=model,
        backend=NumpyBackend(precision=32),
        jit=False,
        static_keys={
            "input": np.array(2.0, dtype=np.float64),
            "right": np.array(3.0, dtype=np.float64),
        },
    )

    output = comp_model.evaluate()["output"]
    np.testing.assert_allclose(output, ref)
    assert output.dtype == np.float32


def test_static_input_6():
    model_1 = Model()
    model_2 = Model(enforce_jit=False)
    add_1 = Add()
    add_2 = Add()
    add_3 = Add()

    model_1 += add_1(
        left=IOKey(value=TBD, name="left"),
        right=IOKey(value=TBD, name="right"),
        output=IOKey(name="out1"),
    )
    model_1 += add_2(
        left=add_1.left + 1.0, right=add_1.right, output=IOKey(name="out2")
    )

    model_2 += add_3(
        left=IOKey(value=3.0, name="left"),
        right=IOKey(value=4.0, name="right"),
        output=IOKey(name="output"),
    )
    model_2 += model_1(left=add_3.left, right=add_3.right, out2=IOKey(name="output_1"))

    backend = JaxBackend()
    comp_model = mithril.compile(model=model_2, backend=backend, safe=False, jit=False)
    output = comp_model.evaluate()

    # It is Tensor type.
    assert model_1.left.metadata.data.value is None  # type: ignore

    assert (
        model_1.right.metadata.data.value is None  # type: ignore
    )  # It is Tensor type.
    assert (
        model_2.left.metadata.data.value == 3.0  # type: ignore
    )  # It is Scalar type with a defined value.
    assert (
        model_2.right.metadata.data.value == 4.0  # type: ignore
    )  # It is Scalar type with a defined value.
    assert output["output"] == backend.array(7.0)
    assert output["output_1"] == backend.array(8.0)


def test_static_input_6_error():
    """Raises ValueError since multiwrite occurs for left key of model_1
    with values 3.0 and 1.0
    """
    model_1 = Model()
    model_2 = Model(enforce_jit=False)
    add_1 = Add()
    add_2 = Add()
    add_3 = Add()

    model_1 += add_1(
        left=IOKey(value=1.0, name="left"),
        right=IOKey(value=TBD, name="right"),
        output=IOKey(name="out1"),
    )
    model_1 += add_2(left=add_1.left, right=add_1.right, output=IOKey(name="out2"))

    model_2 += add_3(
        left=IOKey(value=3.0, name="left"),
        right=IOKey(value=4.0, name="right"),
        output=IOKey(name="output"),
    )
    with pytest.raises(ValueError) as err_info:
        model_2 += model_1(
            left=add_3.left, right=add_3.right, out2=IOKey(name="output_1")
        )
    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_static_input_7():
    model_1 = Model()
    model_2 = Model()
    add_1 = Add()
    add_2 = Add()
    add_3 = Add()

    model_1 += add_1(
        left=IOKey(value=3.0, name="left"),
        right=IOKey(value=4.0, name="right"),
        output=IOKey(name="out1"),
    )
    model_1 += add_2(left=add_1.left, right=add_1.right, output="out2")

    model_2 += model_1(left="left", right="right")
    model_2 += add_3(
        left=model_1.left,  # type: ignore
        right=model_1.right,  # type: ignore
        output=IOKey(name="output"),
    )


def test_linear_1():
    model = Model()
    lin1 = Linear()
    lin1.set_shapes({"w": [2, 2], "input": [2, 2]})
    model += lin1(input="input", output=IOKey(name="output"))
    assert_all_backends_device_precision(model)


def test_mlp():
    mlp_model = MLP(
        activations=[Buffer(), LeakyRelu(), Sigmoid()], dimensions=[2, 1, 1]
    )
    mlp_model.set_shapes({"input": [1, 1]})
    assert_all_backends_device_precision(mlp_model)


def test_add_1():
    model = Model()
    add_model = Add()
    model += add_model(left=1, right="right", output=IOKey(name="output"))
    model.set_shapes({"right": [1, 1, 1]})
    assert_all_backends_device_precision(model)


def test_composite_1():
    model = Model()
    add_model = Add()
    shape_model = Shape()
    index_model = ScalarItem()
    red_model = Mean(axis=TBD)
    model += add_model(left=[[[1]]], right="right")
    model += shape_model(input=add_model.output)
    model += index_model(input=shape_model.output, index=1)
    model += red_model(
        input=add_model.output, axis=index_model.output, output=IOKey(name="output")
    )
    model.set_shapes({"right": [1, 1, 1, 1, 1]})
    mithril.compile(model=model, backend=NumpyBackend(), jit=False, safe=False)
    assert_all_backends_device_precision(model)


def test_composite_1_set_values():
    model = Model()
    add_model = Add()
    shape_model = Shape()
    index_model = ScalarItem()
    red_model = Mean(axis=TBD)
    model += add_model(right="right")
    model.set_values({add_model.left: [[[1]]]})
    model += shape_model(input=add_model.output)
    model += index_model(input=shape_model.output, index=1)
    model += red_model(
        input=add_model.output, axis=index_model.output, output=IOKey(name="output")
    )
    model.set_shapes({"right": [1, 1, 1, 1, 1]})
    mithril.compile(model=model, backend=NumpyBackend(), jit=False, safe=False)
    assert_all_backends_device_precision(model)


def test_composite_2():
    model = Model()
    conv1 = Convolution2D(kernel_size=2, out_channels=4)
    leaky_relu = LeakyRelu(slope=TBD)
    model += conv1(input="input")
    model += leaky_relu(input=conv1.output, output=IOKey(name="output"), slope=0.3)
    model.set_shapes({"input": [1, 1, 4, 4]})
    assert_all_backends_device_precision(model)


def test_composite_2_set_values():
    model = Model()
    conv1 = Convolution2D(kernel_size=2, out_channels=4)
    leaky_relu = LeakyRelu(slope=TBD)
    model += conv1(input="input")
    model += leaky_relu(input=conv1.output, output=IOKey(name="output"))
    model.set_values({leaky_relu.slope: 0.3})
    model.set_shapes({"input": [1, 1, 4, 4]})
    assert_all_backends_device_precision(model)


def test_composite_3():
    model = Model()
    conv1 = Convolution2D(kernel_size=2, out_channels=1, stride=TBD)
    leaky_relu = LeakyRelu(slope=TBD)
    mean_model = Mean(axis=TBD)
    model += conv1(input="input", stride=(2, 3))
    model += leaky_relu(input=conv1.output, slope=0.3)
    model += mean_model(axis=conv1.stride)
    assert not isinstance(conv1.canonical_output, NotAvailable)
    model.set_canonical_output(conv1.canonical_output)
    model.set_shapes({"input": [1, 1, 8, 8]})
    assert_all_backends_device_precision(model)


def test_composite_3_set_values():
    model = Model()
    conv1 = Convolution2D(kernel_size=2, out_channels=1, stride=TBD)
    leaky_relu = LeakyRelu(slope=TBD)
    mean_model = Mean(axis=TBD)
    model += conv1(input="input", stride=(2, 3))
    model.set_values({conv1.stride: (2, 3)})
    model += leaky_relu(input=conv1.output)
    model.set_values({leaky_relu.slope: 0.3})
    model += mean_model(axis=conv1.stride)
    assert not isinstance(conv1.canonical_output, NotAvailable)
    model.set_canonical_output(conv1.canonical_output)

    model.set_shapes({"input": [1, 1, 8, 8]})
    assert_all_backends_device_precision(model)


def test_composite_4():
    model = Model()
    conv1 = Convolution2D(kernel_size=2, out_channels=1, stride=TBD)
    leaky_relu = LeakyRelu(slope=TBD)
    mean_model = Mean(axis=TBD)
    model += conv1(input="input", stride=(2, 3))
    model += leaky_relu(input=conv1.output, slope=0.3)
    model += mean_model(axis=conv1.stride)
    model.set_shapes({"input": [1, 1, 8, 8]})
    assert not isinstance(conv1.canonical_output, NotAvailable)
    model.set_canonical_output(conv1.canonical_output)
    assert_all_backends_device_precision(model)


def test_composite_4_set_values():
    model = Model()
    conv1 = Convolution2D(kernel_size=2, out_channels=1, stride=TBD)
    leaky_relu = LeakyRelu(slope=TBD)
    mean_model = Mean(axis=TBD)
    model += conv1(input="input")
    model.set_values({conv1.stride: (2, 3)})
    model += leaky_relu(input=conv1.output)
    model.set_values({leaky_relu.slope: 0.3})
    model += mean_model(axis=conv1.stride)
    model.set_shapes({"input": [1, 1, 8, 8]})
    assert not isinstance(conv1.canonical_output, NotAvailable)
    model.set_canonical_output(conv1.canonical_output)
    assert_all_backends_device_precision(model)


def test_composite_5():
    list1 = np.random.randn(2, 3, 4).tolist()
    list2 = np.random.randn(1, 3, 4).tolist()
    list3 = np.random.randn(2, 2, 1, 1, 1).tolist()
    model = Model()
    add_model_1 = Add()
    add_model_2 = Add()
    add_model_3 = Add()
    model += add_model_1(left=IOKey(value=list1, name="left1"), right=list1)
    model += add_model_2(left=add_model_1.output, right=list2)
    model += add_model_3(left=add_model_2.output, right=list3)

    assert_all_backends_device_precision(model)


def test_composite_5_set_values():
    list1 = np.random.randn(2, 3, 4).tolist()
    list2 = np.random.randn(1, 3, 4).tolist()
    list3 = np.random.randn(2, 2, 1, 1, 1).tolist()
    model = Model()
    add_model_1 = Add()
    add_model_2 = Add()
    add_model_3 = Add()
    model += add_model_1(left=IOKey(name="left1"))
    model.set_values({add_model_1.left: list1, add_model_1.right: list1})
    model += add_model_2(left=add_model_1.output)
    model.set_values({add_model_2.right: list2})
    model += add_model_3(left=add_model_2.output)
    model.set_values({add_model_3.right: list3})

    assert_all_backends_device_precision(model)


def test_composite_6():
    list1 = np.random.randn(2, 3, 4).tolist()
    list2 = np.random.randn(1, 3, 4).tolist()
    list3 = np.random.randn(2, 2, 1, 1, 1).tolist()
    model = Model()
    add_model_1 = Add()
    add_model_2 = Add()
    add_model_3 = Add()
    model += add_model_1(left=IOKey(value=1, name="left1"), right=list1)
    model += add_model_2(left=add_model_1.output, right=list2)
    model += add_model_3(left=add_model_2.output, right=list3)
    assert_all_backends_device_precision(model)


def test_composite_6_set_values():
    list1 = np.random.randn(2, 3, 4).tolist()
    list2 = np.random.randn(1, 3, 4).tolist()
    list3 = np.random.randn(2, 2, 1, 1, 1).tolist()
    model = Model()
    add_model_1 = Add()
    add_model_2 = Add()
    add_model_3 = Add()
    model += add_model_1(left=IOKey(name="left1"))
    model.set_values({add_model_1.left: 1, add_model_1.right: list1})
    model += add_model_2(left=add_model_1.output)
    model.set_values({add_model_2.right: list2})
    model += add_model_3(left=add_model_2.output)
    model.set_values({add_model_3.right: list3})
    assert_all_backends_device_precision(model)


def test_composite_7():
    list1 = np.random.randn(2, 3, 4).tolist()
    list2 = np.random.randn(1, 3, 4).tolist()
    list3 = np.random.randn(2, 2, 1, 1, 1).tolist()
    model = Model()
    add_model_1 = Add()
    add_model_2 = Add()
    add_model_3 = Add()
    model += add_model_1(left=IOKey(name="left1", value=[[1]]), right=list1)
    model += add_model_2(left=add_model_1.output, right=list2)
    model += add_model_3(left=add_model_2.output, right=list3)
    assert_all_backends_device_precision(model)


def test_composite_7_set_values():
    list1 = np.random.randn(2, 3, 4).tolist()
    list2 = np.random.randn(1, 3, 4).tolist()
    list3 = np.random.randn(2, 2, 1, 1, 1).tolist()
    model = Model()
    add_model_1 = Add()
    add_model_2 = Add()
    add_model_3 = Add()
    model += add_model_1(left=IOKey(name="left1"))
    model.set_values({add_model_1.left: [[1]], add_model_1.right: list1})
    model += add_model_2(left=add_model_1.output)
    model.set_values({add_model_2.right: list2})
    model += add_model_3(left=add_model_2.output)
    model.set_values({add_model_3.right: list3})
    assert_all_backends_device_precision(model)


def test_composite_conv_mean():
    list1 = np.random.randn(1, 1, 8, 8).tolist()
    model = Model()
    conv_model = Convolution2D(kernel_size=2, out_channels=1, stride=(2, 3))
    reduce_model = Mean(axis=TBD)
    model += conv_model(input=IOKey(value=list1, name="input"))
    model += reduce_model(axis=conv_model.stride)
    assert not isinstance(conv_model.canonical_output, NotAvailable)
    model.set_canonical_output(conv_model.canonical_output)
    assert_all_backends_device_precision(model)


def test_composite_conv_mean_set_values():
    list1 = np.random.randn(1, 1, 8, 8).tolist()
    model = Model()
    conv_model = Convolution2D(kernel_size=2, out_channels=1, stride=(2, 3))
    reduce_model = Mean(axis=TBD)
    model += conv_model(input=IOKey(name="input"))
    model.set_values({"input": list1})
    model += reduce_model(axis=conv_model.stride)
    assert not isinstance(conv_model.canonical_output, NotAvailable)
    model.set_canonical_output(conv_model.canonical_output)
    assert_all_backends_device_precision(model)


def test_composite_conv_mean_2():
    list1 = np.ones((1, 1, 8, 8)).tolist()
    model = Model()
    conv_model = Convolution2D(kernel_size=2, out_channels=1, stride=TBD)
    reduce_model = Sum(axis=TBD)
    model += conv_model(input=IOKey(value=list1, name="input"))
    model += reduce_model(axis=conv_model.stride, input=conv_model.output)
    comp_model = mithril.compile(model=model, backend=NumpyBackend(), jit=False)
    inputs = {"kernel": np.ones((1, 1, 2, 2)), "bias": np.ones((1, 1, 1, 1))}
    outputs = comp_model.evaluate(params=inputs, data={"stride_1": (1, 2)})
    ref_outputs = {"output": np.ones((1, 4)) * 35.0}
    assert_results_equal(outputs, ref_outputs)


def test_composite_conv_mean_2_set_values():
    list1 = np.ones((1, 1, 8, 8)).tolist()
    model = Model()
    conv_model = Convolution2D(kernel_size=2, out_channels=1, stride=TBD)
    reduce_model = Sum(axis=TBD)
    model += conv_model(input=IOKey(name="input"))
    model.set_values({"input": list1})
    model += reduce_model(axis=conv_model.stride, input=conv_model.output)
    comp_model = mithril.compile(model=model, backend=NumpyBackend(), jit=False)
    inputs = {"kernel": np.ones((1, 1, 2, 2)), "bias": np.ones((1, 1, 1, 1))}
    outputs = comp_model.evaluate(params=inputs, data={"stride_1": (1, 2)})
    ref_outputs = {"output": np.ones((1, 4)) * 35.0}
    assert_results_equal(outputs, ref_outputs)


def test_unused_cached_values_1():
    """Tests for the proper functioning of data_store object of model.
    Unused or pre-used static data should not be hold in any of data/cache dict.
    """
    model = Model()
    linear_model = Linear(dimension=2)
    model += linear_model(input=[[3.0], [2.0]], w=[[1.0, 2.0]], b=[3.0, 1.0])
    comp_model = mithril.compile(model=model, backend=(backend := NumpyBackend()))
    dtype = backend.get_backend_array_type()
    cache = comp_model.data_store.data_values
    expected_cache = {"output": np.array([[6.0, 7.0], [5.0, 5.0]], dtype=dtype)}
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    assert data_keys == set()
    # Try evaluate and evaluate gradients once.
    result = comp_model.evaluate(params={}, data={})
    gradients = comp_model.evaluate_gradients(
        params={}, data={}, output_gradients={"output": np.ones_like(result["output"])}
    )
    assert np.all(result["output"] == np.array([[6.0, 7.0], [5.0, 5.0]], dtype=dtype))
    assert gradients == {}


def test_unused_cached_values_1_set_values():
    """Tests for the proper functioning of data_store object of model.
    Unused or pre-used static data should not be hold in any of data/cache dict.
    """
    model = Model()
    linear_model = Linear(dimension=2)
    model += linear_model()
    model.set_values(
        {
            linear_model.w: [[1.0, 2.0]],
            linear_model.b: [3.0, 1.0],
            linear_model.input: [[3.0], [2.0]],
        }
    )
    comp_model = mithril.compile(model=model, backend=(backend := NumpyBackend()))
    dtype = backend.get_backend_array_type()
    cache = comp_model.data_store.data_values
    expected_cache = {"output": np.array([[6.0, 7.0], [5.0, 5.0]], dtype=dtype)}
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    assert data_keys == set()
    # Try evaluate and evaluate gradients once.
    result = comp_model.evaluate(params={}, data={})
    gradients = comp_model.evaluate_gradients(
        params={}, data={}, output_gradients={"output": np.ones_like(result["output"])}
    )
    assert np.all(result["output"] == np.array([[6.0, 7.0], [5.0, 5.0]], dtype=dtype))
    assert gradients == {}


def test_unused_cached_values_2():
    """Tests for the proper functioning of data_store object of model.
    Unused or pre-used static data should not be hold in any of data/cache dict.
    """
    model = Model()
    linear_model = Linear(dimension=2)
    model += linear_model(input=TBD, w=[[1.0, 2.0]], b=[3.0, 1.0])
    comp_model = mithril.compile(model=model, backend=(backend := NumpyBackend()))
    dtype = backend.get_backend_array_type()
    cache = comp_model.data_store.data_values

    expected_cache = {
        "_ToTensor_0_output": np.array([[1.0, 2.0]], dtype=dtype),
        "_ToTensor_1_output": np.array([3.0, 1.0], dtype=dtype),
        "output_cache": {},
        "_Linear_2_MatrixMultiply_0_output_cache": {},
    }
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    expected_data_keys = {"input"}
    assert data_keys == expected_data_keys
    # Try evaluate and evaluate gradients once.
    data = {"input": np.array([[3.0], [2.0]], dtype=dtype)}
    result = comp_model.evaluate(params={}, data=data)
    gradients = comp_model.evaluate_gradients(
        params={},
        data=data,
        output_gradients={"output": np.ones_like(result["output"])},
    )
    assert np.all(result["output"] == np.array([[6.0, 7.0], [5.0, 5.0]], dtype=dtype))
    assert gradients == {}


def test_unused_cached_values_2_set_values():
    """Tests for the proper functioning of data_store object of model.
    Unused or pre-used static data should not be hold in any of data/cache dict.
    """
    model = Model()
    linear_model = Linear(dimension=2)
    model += linear_model()
    model.set_values(
        {
            linear_model.input: TBD,
            linear_model.w: [[1.0, 2.0]],
            linear_model.b: [3.0, 1.0],
        }
    )
    comp_model = mithril.compile(model=model, backend=(backend := NumpyBackend()))
    dtype = backend.get_backend_array_type()
    cache = comp_model.data_store.data_values

    expected_cache = {
        "_ToTensor_0_output": np.array([[1.0, 2.0]], dtype=dtype),
        "_ToTensor_1_output": np.array([3.0, 1.0], dtype=dtype),
        "output_cache": {},
        "_Linear_2_MatrixMultiply_0_output_cache": {},
    }
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    expected_data_keys = {"input"}
    assert data_keys == expected_data_keys
    # Try evaluate and evaluate gradients once.
    data = {"input": np.array([[3.0], [2.0]], dtype=dtype)}
    result = comp_model.evaluate(params={}, data=data)
    gradients = comp_model.evaluate_gradients(
        params={},
        data=data,
        output_gradients={"output": np.ones_like(result["output"])},
    )
    assert np.all(result["output"] == np.array([[6.0, 7.0], [5.0, 5.0]], dtype=dtype))
    assert gradients == {}


def test_unused_cached_values_3():
    """Tests for the proper functioning of data_store object of model.
    Unused or pre-used static data should not be hold in any of data/cache dict.
    """
    model = Model()
    linear_model = Linear(dimension=2)
    model += linear_model(input=[[3.0], [2.0]], w=[[1.0, 2.0]], b=TBD)
    comp_model = mithril.compile(model=model, backend=(backend := NumpyBackend()))
    dtype = backend.get_backend_array_type()
    cache = comp_model.data_store.data_values

    expected_cache = {
        "output_cache": {},
        "_Linear_2_MatrixMultiply_0_output": np.array([[3.0, 6], [2, 4]], dtype=dtype),
    }
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    expected_data_keys = {"b"}
    assert data_keys == expected_data_keys
    # Try evaluate and evaluate gradients once.
    data = {"b": np.array([3.0, 1.0], dtype=dtype)}
    result = comp_model.evaluate(params={}, data=data)
    gradients = comp_model.evaluate_gradients(
        params={},
        data=data,
        output_gradients={"output": np.ones_like(result["output"])},
    )
    assert np.all(result["output"] == np.array([[6.0, 7.0], [5.0, 5.0]], dtype=dtype))
    assert gradients == {}


def test_unused_cached_values_3_set_values():
    """Tests for the proper functioning of data_store object of model.
    Unused or pre-used static data should not be hold in any of data/cache dict.
    """
    model = Model()
    linear_model = Linear(dimension=2)
    model += linear_model()
    model.set_values(
        {
            linear_model.input: [[3.0], [2.0]],
            linear_model.w: [[1.0, 2.0]],
            linear_model.b: TBD,
        }
    )
    comp_model = mithril.compile(model=model, backend=(backend := NumpyBackend()))
    dtype = backend.get_backend_array_type()
    cache = comp_model.data_store.data_values

    expected_cache = {
        "output_cache": {},
        "_Linear_2_MatrixMultiply_0_output": np.array([[3.0, 6], [2, 4]], dtype=dtype),
    }
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    expected_data_keys = {"b"}
    assert data_keys == expected_data_keys
    # Try evaluate and evaluate gradients once.
    data = {"b": np.array([3.0, 1.0], dtype=dtype)}
    result = comp_model.evaluate(params={}, data=data)
    gradients = comp_model.evaluate_gradients(
        params={},
        data=data,
        output_gradients={"output": np.ones_like(result["output"])},
    )
    assert np.all(result["output"] == np.array([[6.0, 7.0], [5.0, 5.0]], dtype=dtype))
    assert gradients == {}


def test_static_shape_model_1():
    comp_model = mithril.compile(
        model=Shape(),
        backend=NumpyBackend(),
        shapes={"input": [8, 8]},
        inference=True,
        safe=False,
    )
    cache = comp_model.data_store.data_values
    expected_cache = {"output": (8, 8)}
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    assert data_keys == set()
    # Try evaluate and evaluate gradients once.
    result = comp_model.evaluate(params={}, data={})

    assert result["output"] == (8, 8)


def test_static_shape_model_2():
    model = Model()
    model += Shape()
    model += ToTensor()
    model += Relu()
    comp_model = mithril.compile(
        model=model,
        backend=NumpyBackend(),
        shapes={"input": [8, 8]},
        safe=False,  # Because "input" key will be unused.
    )
    cache = comp_model.data_store.data_values
    expected_cache = {"output": np.array([8, 8], dtype=np.int32)}
    # Check cached_data.
    # assert cache is not None and cache.keys() == expected_cache.keys()
    assert isinstance(cache, dict)
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    assert data_keys == set()
    # Try evaluate and evaluate gradients once.
    result = comp_model.evaluate(params={}, data={})
    gradients = comp_model.evaluate_gradients(
        params={},
        data={},
        output_gradients={"output": np.ones_like(result["output"])},
    )
    assert np.all(result["output"] == np.array([8, 8], dtype=np.int32))
    assert gradients == {}


def test_static_shape_model_2_error():
    model = Model()
    model += Shape()
    model += ToTensor()
    model += Relu()
    with pytest.raises(ValueError) as err_info:
        mithril.compile(
            model=model,
            backend=NumpyBackend(),
            shapes={"input": [8, 8]},
            static_keys={"input": TBD},
        )
    assert (
        str(err_info.value)
        == "Given 'input' key is unused for the model, no need to provide data for it."
    )


def test_static_shape_model_3():
    model = Model()
    model += Tanh()
    model += Shape()
    model += ToTensor()
    model += Relu()

    backend = NumpyBackend()
    comp_model = mithril.compile(
        model=model, backend=backend, static_keys={"input": backend.ones(8, 8)}
    )
    cache = comp_model.data_store.data_values
    expected_cache = {"output": np.array([8, 8], dtype=np.int32)}
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    assert data_keys == set()
    # Try evaluate and evaluate gradients once.
    result = comp_model.evaluate(params={}, data={})
    gradients = comp_model.evaluate_gradients(
        params={},
        data={},
        output_gradients={"output": np.ones_like(result["output"])},
    )
    assert np.all(result["output"] == np.array([8, 8], dtype=np.int32))
    assert gradients == {}


def test_static_shape_model_4():
    model = Model()
    model += Relu()
    model += Log(robust=True, cutoff=TBD)
    model += Shape()
    model += ToTensor()
    model += Relu()

    backend = NumpyBackend()
    comp_model = mithril.compile(
        model=model, backend=backend, static_keys={"input": backend.ones(8, 8)}
    )
    cache = comp_model.data_store.data_values
    expected_cache = {"output": np.array([8, 8], dtype=np.int32)}
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    assert data_keys == set()
    # Try evaluate and evaluate gradients once.
    result = comp_model.evaluate(params={}, data={})
    gradients = comp_model.evaluate_gradients(
        params={},
        data={},
        output_gradients={"output": np.ones_like(result["output"])},
    )
    assert np.all(result["output"] == np.array([8, 8], dtype=np.int32))
    assert gradients == {}


def test_static_shape_model_5():
    model = Model()
    model += Relu()
    model += (log := Log(robust=True, cutoff=TBD))
    model += Shape()
    model += ToTensor()
    model += Relu()(input=model.canonical_output, output=IOKey(name="output1"))
    model += Relu()(input=log.output, output=IOKey(name="output2"))

    backend = NumpyBackend()
    static_keys: dict[str, np.ndarray | ToBeDetermined] = {
        "input": backend.ones(8, 8),
        "cutoff": TBD,
    }
    comp_model = mithril.compile(
        model=model,
        backend=backend,
        static_keys=static_keys,
    )
    cache = comp_model.data_store.data_values
    expected_cache = {
        "output1": np.array([8, 8], dtype=np.int32),
        "_Relu_0_output": backend.ones(8, 8),
        "_Log_1_output_cache": {},
        "output2_cache": {},
    }
    # Check cached_data.
    assert cache is not None and cache.keys() == expected_cache.keys()
    assert all([np.all(value == expected_cache[key]) for key, value in cache.items()])
    # Check runtime data keys.
    data_keys = comp_model.data_store.runtime_static_keys
    expected_data_keys = {"cutoff"}
    assert data_keys == expected_data_keys
    # Try evaluate and evaluate gradients once.
    data = {"cutoff": 0.00005}
    result = comp_model.evaluate(params={}, data=data)
    gradients = comp_model.evaluate_gradients(
        params={},
        data=data,
        output_gradients={
            "output1": np.ones_like(result["output1"]),
            "output2": np.ones_like(result["output2"]),
        },
    )
    assert np.all(result["output1"] == np.array([8, 8], dtype=np.int32))
    assert np.all(result["output2"] == backend.zeros(8, 8))
    assert gradients == {}


def test_nontensor_gradient():
    backend = NumpyBackend(precision=64)
    model = Model()
    shape_model = Shape()
    to_tensor_model = ToTensor()
    relu = Relu()
    add_model = Add()

    model += shape_model(input="input")
    model += relu(input="input")
    model += to_tensor_model(input=shape_model.output, output=IOKey(name="out1"))
    model += add_model(left="in1", right=relu.output, output=IOKey(name="out2"))

    ctx = TrainModel(model)
    ctx.add_loss(Buffer(), input="out1", reduce_steps=[Sum()])
    ctx.add_loss(Buffer(), input="out2", reduce_steps=[Sum()])

    comp_model = mithril.compile(model=ctx, backend=backend, safe=False, jit=False)

    input = backend.array([[1.0, 2.0, 3.0], [1.0, 4.0, 2.0], [3.0, 2.0, 1.0]])
    in1 = backend.array(1.0)
    outputs, grads = comp_model.evaluate_all({"input": input, "in1": in1})
    np.testing.assert_allclose(np.array(outputs["final_cost"]), np.array(34.0))
    np.testing.assert_allclose(np.array(outputs["out1"]), np.array([3, 3]))
    np.testing.assert_allclose(
        np.array(outputs["out2"]),
        np.array([[2.0, 3.0, 4.0], [2.0, 5.0, 3.0], [4.0, 3.0, 2.0]]),
    )
    np.testing.assert_allclose(np.array(grads["in1"]), np.array(9.0))
    np.testing.assert_allclose(
        np.array(grads["input"]),
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )


def test_nontensor_gradient_2():
    backend = NumpyBackend()
    mlp_model = MLP(activations=[Sigmoid(), Relu(), Sigmoid()], dimensions=[12, 13, 2])
    shape_model = Shape()
    to_tensor_model = ToTensor()
    add_model = Add()
    mult_model = Multiply()
    relu_model = Relu()

    model = Model()
    model += mlp_model(input="input")
    model += shape_model(input=mlp_model.output)
    model += to_tensor_model(input=shape_model.output)
    model += add_model(
        left="", right=to_tensor_model.output, output=IOKey(name="output")
    )
    model += mult_model(left="", right="right1", output=add_model.left)
    model += relu_model(input="in1", output=mult_model.left)
    static_keys = {
        "input": backend.array([[10.0, 2.0], [1.0, 1.0]]),
    }
    comp_model = mithril.compile(
        model=model,
        backend=backend,
        static_keys=static_keys,
        shapes={"right1": [1], "in1": [1]},
    )

    trainable_keys = {"right1": backend.array([2.0]), "in1": backend.array([1.0])}

    trainable_keys = comp_model.randomize_params() | trainable_keys
    output_grads = {"output": backend.array([1.0, 1.0])}
    outputs, grads = comp_model.evaluate_all(
        params=trainable_keys, output_gradients=output_grads
    )
    np.testing.assert_allclose(np.array(outputs["output"]), np.array([4.0, 4.0]))
    np.testing.assert_allclose(np.array(grads["right1"]), np.array([2.0]))
    np.testing.assert_allclose(np.array(grads["in1"]), np.array([4.0]))


def test_nontensor_gradient_3():
    backend = NumpyBackend()
    model = Model()
    shape_model = Shape()
    to_tensor_model = ToTensor()
    model += shape_model(input="input")
    model += to_tensor_model(input=shape_model.output, output=IOKey(name="output"))
    ctx = TrainModel(model)
    ctx.add_loss(Buffer(), input="output", reduce_steps=[Sum()])
    input = backend.randn(3, 4, 5, 6, 5)
    comp_model = mithril.compile(
        model=ctx,
        backend=backend,
        safe=False,
        jit=False,
    )
    comp_model.evaluate({"input": input})
    outputs, grads = comp_model.evaluate_all({"input": input})
    ref_outputs = {"output": backend.array([3, 4, 5, 6, 5]), "final_cost": np.array(23)}
    ref_grads = {"input": backend.zeros(3, 4, 5, 6, 5, dtype=mithril.float32)}

    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


def test_numpy_without_shape():
    backend = NumpyBackend(precision=32)
    model = Model()
    add_model = Add()
    model += add_model(left="left", right="right", output=IOKey(name="output"))
    model.set_shapes({"left": [], "right": []})
    ctx = TrainModel(model)
    ctx.add_loss(Buffer(), input="output", reduce_steps=[Mean()])
    inputs = {"left": backend.array(1.2), "right": backend.array(1.0)}
    comp_model = mithril.compile(
        model=ctx,
        backend=backend,
        safe=False,
        jit=False,
    )
    outputs, grads = comp_model.evaluate_all(inputs)
    np.testing.assert_allclose(np.array(outputs["output"]), np.array(2.2))
    np.testing.assert_allclose(np.array(grads["left"]), np.array(1.0))
    np.testing.assert_allclose(np.array(grads["right"]), np.array(1.0))


def test_multiple_to_tensor():
    backend = NumpyBackend(precision=32)
    tt_1 = ToTensor()
    tt_2 = ToTensor()
    shp_1 = Shape()
    shp_2 = Shape()
    add_model = Add()
    add_model_2 = Add()
    model = Model()
    model_1 = Model()
    model_2 = Model()
    model += shp_1
    model += tt_1
    model += add_model(
        left=model.canonical_output, right="right", output=IOKey(name="output")
    )
    model_1 += shp_2
    model_1 += tt_2
    model_1 += add_model_2(
        left=model_1.canonical_output, right="right", output=IOKey(name="output")
    )
    model_2 += model
    model_2 += model_1
    comp_model = mithril.compile(
        model=model_2,
        backend=backend,
        safe=False,
        jit=False,
        shapes={"input": [3, 4, 5, 5, 2, 7, 9]},
    )
    params = {"right_1": backend.array([1.0]), "right_0": backend.array([2.0])}
    outputs = comp_model.evaluate(params)
    np.testing.assert_allclose(np.array(outputs["output"]), np.array([8.0]))


def test_concat_axis_ellipsis_1():
    backend = NumpyBackend()
    model = Model()
    concat_model = Concat(n=2, axis=TBD)
    model += concat_model(input1="input1", input2="input2")
    comp_model = mithril.compile(model=model, backend=backend, safe=False)

    in1 = backend.array([[2.0]])
    in2 = backend.array([[2.0]])

    inputs = {"input1": in1, "input2": in2}

    data = {"axis": 1}
    ref_results = {"output": [[2.0, 2.0]]}

    result = comp_model.evaluate(params=inputs, data=data)
    assert_results_equal(result, ref_results)


def test_concat_axis_ellipsis_2():
    backend = NumpyBackend()
    model = Model()
    concat_model = Concat(n=2, axis=TBD)
    model += concat_model(input1="input1", input2="input2", axis="axis")
    comp_model = mithril.compile(model=model, backend=backend, safe=False)

    in1 = backend.array([[2.0]])
    in2 = backend.array([[2.0]])

    ref_results = {"output": [[2.0, 2.0]]}

    inputs = {"input1": in1, "input2": in2}

    data = {"axis": 1}

    result = comp_model.evaluate(params=inputs, data=data)
    assert_results_equal(result, ref_results)


def test_polyfeatures_degree_ellipsis():
    backend = NumpyBackend()
    model = Model()
    poly_feat_model = PolynomialFeatures(degree=TBD)
    model += poly_feat_model(
        input="input", output=IOKey(name="output"), degree="degree"
    )

    comp_model = mithril.compile(model=model, backend=backend, safe=False)

    params = {"input": backend.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])}

    data = {"degree": 2}

    ref_results = {
        "output": [
            [1.0, 2.0, 1.0, 2.0, 4.0],
            [2.0, 1.0, 4.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    }

    result = comp_model.evaluate(params=params, data=data)
    assert_results_equal(result, ref_results)


def test_eye_ellipsis_1():
    backend = NumpyBackend()
    model = Model()
    eye_model = Eye(N=TBD)
    model += eye_model(N="N", output=IOKey(name="output"))
    comp_model = mithril.compile(model=model, backend=backend, safe=False)

    data = {"N": 5}

    ref_results = {
        "output": [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    }

    result = comp_model.evaluate(params={}, data=data)
    assert_results_equal(result, ref_results)


def test_eye_ellipsis_2():
    backend = NumpyBackend()
    model = Model()
    eye_model = Eye(N=TBD, M=TBD)
    model += eye_model(N="N", output=IOKey(name="output"), M="M")

    comp_model = mithril.compile(model=model, backend=backend, safe=False)

    data = {"N": 5, "M": 5}

    ref_results = {
        "output": [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    }

    result = comp_model.evaluate(params={}, data=data)
    assert_results_equal(result, ref_results)


def test_cross_entropy_robust_ellipsis():
    backend = TorchBackend()
    model = Model()
    ce_model = CrossEntropy(robust=TBD, input_type="probs")
    model += ce_model(
        input="input", target="target", output=IOKey(name="output"), robust="robust"
    )

    comp_model = mithril.compile(
        model=model,
        backend=backend,
        static_keys={"input": TBD, "target": TBD},
        jit=False,
    )

    data: dict[str, torch.Tensor | bool] = {
        "input": backend.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "target": backend.array([0, 1, 0, 1]),
        "robust": False,
    }
    outputs = comp_model.evaluate(params={}, data=data)
    ref_results = {"output": backend.array([0.0, 0.0, 0.0, 0.0])}
    assert_results_equal(outputs, ref_results)


def test_bce_ellipsis():
    backend = NumpyBackend()
    model_1 = Model()
    ce_model_1 = BinaryCrossEntropy(
        robust=TBD, pos_weight=TBD, cutoff=TBD, input_type="probs"
    )
    model_1 += ce_model_1(
        input="input",
        target="target",
        output=IOKey(name="output"),
        robust="robust",
        pos_weight="pos_weight",
        cutoff="cutoff",
    )

    comp_model_1 = mithril.compile(
        model=model_1,
        backend=backend,
        static_keys={
            "input": TBD,
            "target": TBD,
            "robust": TBD,
            "pos_weight": TBD,
            "cutoff": TBD,
        },
    )

    model_2 = Model()
    ce_model_2 = BinaryCrossEntropy(input_type="probs")
    model_2 += ce_model_2(input="input", target="target")

    comp_model_2 = mithril.compile(
        model=model_2, backend=backend, static_keys={"input": TBD, "target": TBD}
    )

    data_1: dict[str, np.ndarray | bool | float] = {
        "input": backend.array([0.5, 0.5]),
        "target": backend.array([0.5, 0.5]),
        "robust": False,
        "pos_weight": 1.0,
        "cutoff": 1e-300,
    }

    data_2: dict[str, np.ndarray] = {
        "input": backend.array([0.5, 0.5]),
        "target": backend.array([0.5, 0.5]),
    }

    res_1 = comp_model_1.evaluate(data=data_1)
    res_2 = comp_model_2.evaluate(data=data_2)

    assert_results_equal(res_1, res_2)


def test_arange_ellipsis():
    backend = TorchBackend()
    model = Model()
    arange_model = Arange(start=TBD, stop=TBD, step=TBD)
    model += arange_model(
        output=IOKey(name="output"), start="start", stop="stop", step="step"
    )
    pm = mithril.compile(model=model, backend=backend, static_keys={})
    ref_outputs = {"output": backend.array([3, 4, 5, 6, 7, 8, 9])}
    outputs = pm.evaluate(data={"start": 3, "stop": 10, "step": 1})
    assert_results_equal(outputs, ref_outputs)


def test_transpose_axis_ellipsis_1():
    backend = TorchBackend()
    model_1 = Model()
    transpose_model_1 = Transpose(axes=TBD)
    model_1 += transpose_model_1(
        input="input", output=IOKey(name="output"), axes=(2, 3, 0, 1)
    )

    static_input = {"input": backend.randn(4, 3, 6, 7)}

    pm_1 = mithril.compile(model=model_1, backend=backend, static_keys=static_input)

    model_2 = Model()
    transpose_model_2 = Transpose(axes=(2, 3, 0, 1))
    model_2 += transpose_model_2(
        input="input", output=IOKey(name="output"), axes=(2, 3, 0, 1)
    )

    pm_2 = mithril.compile(model=model_2, backend=backend, static_keys=static_input)

    out_1 = pm_1.evaluate()
    out_2 = pm_2.evaluate()

    assert_results_equal(out_1, out_2)


def test_maxpool_1d_padding_type_input():
    backend = TorchBackend()
    model_1 = Model()
    maxpool = MaxPool1D(kernel_size=2, padding=TBD)
    model_1 += maxpool(padding=PaddingType.VALID)

    pm = mithril.compile(model=model_1, backend=backend, static_keys={"input": TBD})
    out_1 = pm.evaluate(
        data={"input": backend.array([[[10.0, 11.0, 12.0, 13.0, 14.0]]])}
    )
    assert (out_1["output"] == backend.array([[[11.0, 13.0]]])).all()


def test_maxpool_1d_padding_input_in_evaluate():
    backend = TorchBackend()
    maxpool = MaxPool1D(kernel_size=2, padding=TBD)

    pm = mithril.compile(
        model=maxpool,
        backend=backend,
        static_keys={"input": TBD},
    )
    out_1 = pm.evaluate(
        data={
            "input": backend.array([[[10.0, 11.0, 12.0, 13.0, 14.0]]]),
            "padding": PaddingType.VALID,
        }
    )
    assert (out_1["output"] == backend.array([[[11.0, 13.0]]])).all()


def test_maxpool_1d_padding_input_solved_in_constraint():
    model_1 = Model()
    maxpool = MaxPool1D(kernel_size=2, padding=TBD)
    assert maxpool.padding.metadata.data.value is TBD
    # Find PaddingConverter model
    for _model in maxpool.dag:
        if isinstance(_model, PaddingConverter1D):
            pad_model = _model
            break
    assert pad_model.output.metadata.data.value is TBD
    model_1 += maxpool(padding=PaddingType.VALID)
    assert pad_model.output.metadata.data.value == (0, 0)


def test_maxpool_2d_padding_input_solved_in_constraint():
    model_1 = Model()
    maxpool = MaxPool2D(kernel_size=2, padding=TBD)
    assert maxpool.padding.metadata.data.value is TBD
    # Find PaddingConverter model
    for _model in maxpool.dag:
        if isinstance(_model, PaddingConverter2D):
            pad_model = _model
            break
    assert pad_model.output.metadata.data.value is TBD
    model_1 += maxpool(padding=PaddingType.VALID)
    assert pad_model.output.metadata.data.value == (0, 0)


def test_conv_1d_padding_input_solved_in_constraint():
    model_1 = Model()
    maxpool = Convolution1D(kernel_size=2, padding=TBD)
    assert maxpool.padding.metadata.data.value is TBD
    # Find PaddingConverter model
    for _model in maxpool.dag:
        if isinstance(_model, PaddingConverter1D):
            pad_model = _model
            break
    assert pad_model.output.metadata.data.value is TBD
    model_1 += maxpool(padding=PaddingType.VALID)
    assert pad_model.output.metadata.data.value == (0, 0)


def test_conv_2d_padding_input_solved_in_constraint():
    model_1 = Model()
    maxpool = Convolution2D(kernel_size=2, padding=TBD)
    assert maxpool.padding.metadata.data.value is TBD
    # Find PaddingConverter model
    for _model in maxpool.dag:
        if isinstance(_model, PaddingConverter2D):
            pad_model = _model
            break
    assert pad_model.output.metadata.data.value is TBD
    model_1 += maxpool(padding=PaddingType.VALID)
    assert pad_model.output.metadata.data.value == (0, 0)
