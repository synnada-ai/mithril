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
from collections.abc import Mapping, Sequence
from types import EllipsisType

import jax.numpy as jnp
import numpy as np
import pytest

from mithril import JaxBackend, NumpyBackend, compile
from mithril.backends.with_autograd.jax_backend.ops import (
    add,
    partial,
    reduce_mean,
    shape,
    to_tensor,
)
from mithril.framework import NOT_GIVEN, ConnectionType, ExtendInfo
from mithril.framework.constraints import bcast
from mithril.models import (
    TBD,
    Add,
    Connect,
    CustomPrimitiveModel,
    IOKey,
    MatrixMultiply,
    Mean,
    Model,
    Multiply,
    PrimitiveSlice,
    PrimitiveUnion,
    Reshape,
    ScalarItem,
    Shape,
    TensorType,
    ToTensor,
)

from .test_utils import assert_results_equal

to_tensor = partial(to_tensor, precision=32, device="cpu")

############################################################################################
# In this file some of our models are tested to see if they are jittable
# in all possible cases.
############################################################################################


class MyModel(Model):
    def __init__(self, dimension: int | None = None) -> None:
        """This model implements above model.

        mult_model = MatrixMultiplication()
        sum_model = Add()
        self.extend(mult_model, input = "input", rhs = "w")
        self.extend(sum_model, input = mult_model.output, rhs = "b")
        self.extend((reshp := Reshape(shape = [sum_model.output.shape[:], 1, 1])),
        input = sum_model.output) self.extend((sum2 := Sum()), input = sum_model.
        output.shape[mult_model.output.shape[reshp.output.shape[-1]]], rhs = 3.0)
        self.extend(Multiplication(), input = sum2.output, rhs = 2.0, output =
        IOKey(name = "output"))
        """
        super().__init__()
        mult_model = MatrixMultiply()
        sum_model = Add()
        self += mult_model(left="input", right="w")  # (10, 1)
        self += sum_model(left=mult_model.output, right="b")  # (10, 1)
        self += (sum_shp := Shape())(input=sum_model.output)  # (10, 1)
        self += (sum_slc := PrimitiveSlice())(input=sum_shp.output)  # (10, 1)
        self += (uni := PrimitiveUnion(n=3))(
            input1=sum_slc.output, input2=1, input3=1
        )  # (10, 1, 1, 1)
        self += (reshp_1 := Reshape())(
            input=sum_model.output, shape=uni.output
        )  # (10, 1, 1, 1)
        self += (reshp_shp := Shape())(input=reshp_1.output)  # (10, 1, 1, 1)
        self += (idx_1 := ScalarItem())(index=-1, input=reshp_shp.output)  # 1
        self += (mult_shp := Shape())(input=mult_model.output)  # (10, 1)
        self += (idx_2 := ScalarItem())(index=idx_1.output, input=mult_shp.output)  # 1
        self += (idx_3 := ScalarItem())(index=idx_2.output, input=sum_shp.output)  # 1
        self += (tens := ToTensor())(input=idx_3.output)  # array(1)
        self += (sum := Add())(left=tens.output, right=3.0)  # array(4)
        self += Multiply()(
            left=sum.output, right=2.0, output=IOKey(name="output")
        )  # array(8)

        shapes: Mapping[str, Sequence[str | tuple[str, EllipsisType] | int | None]] = {
            "input": ["N", ("Var_inter", ...), "d_in"],
            "w": ["d_in", dimension],
            "b": [dimension],
        }
        self.set_shapes(shapes)
        ...


class MyModel2(Model):
    def __init__(self, dimension: int | None = None) -> None:
        """This model implements above model.

        mult_model = MatrixMultiplication()
        sum_model = Add()
        self.extend(mult_model, input = "input", rhs = "w")
        self.extend(sum_model, input = mult_model.output, rhs = "b")
        self.extend((reshp := Reshape(shape = [sum_model.output.shape[:], 1, 1])),
        input = sum_model.output) self.extend((sum2 := Sum()), input = sum_model.
        output.shape[mult_model.output.shape[reshp.output.shape[-1]]], rhs = 3.0)
        self.extend(Multiplication(), input = sum2.output, rhs = 2.0, output =
        IOKey(name = "output"))
        """
        super().__init__()
        mult_model = MatrixMultiply()
        sum_model = Add()
        self += mult_model(left="input", right="w")  # (10, 1)
        self += sum_model(left=mult_model.output, right="b")  # (10, 1)
        self += (sum_shp := Shape())(input=sum_model.output)  # (10, 1)
        self += (uni := PrimitiveUnion(n=3))(
            input1=sum_shp.output, input2=1, input3=3
        )  # (10, 1, 1, 1)
        self += (idx_1 := ScalarItem())(index=-1, input=uni.output)  # 1
        self += (tens := ToTensor())(input=idx_1.output)  # array(1)
        self += Multiply()(
            left=tens.output, right=2.0, output=IOKey(name="output")
        )  # array(8)

        shapes: Mapping[str, Sequence[str | tuple[str, EllipsisType] | int | None]] = {
            "input": ["N", ("Var_inter", ...), "d_in"],
            "w": ["d_in", dimension],
            "b": [dimension],
        }
        self.set_shapes(shapes)


np_input = np.random.randn(10, 3).astype(np.float32)


def test_mymodel_numpy():
    model = MyModel(dimension=1)
    static_inputs = {"input": np_input}
    compiled_model = compile(
        model=model, backend=NumpyBackend(), static_keys=static_inputs, jit=False
    )
    inputs = compiled_model.randomize_params()
    result = compiled_model.evaluate(inputs)
    output_gradients = {"output": np.ones_like(result["output"])}
    outputs, grads = compiled_model.evaluate_all(
        params=inputs, output_gradients=output_gradients
    )
    ref_output = {"output": np.array(8.0)}
    assert_results_equal(outputs, ref_output)
    assert_results_equal(grads, {})


def test_mymodel_jax_1():
    model = MyModel(dimension=1)
    static_inputs = {"input": jnp.array(np_input)}
    compiled_model = compile(
        model=model, backend=JaxBackend(), static_keys=static_inputs, jit=False
    )
    inputs = compiled_model.randomize_params()
    result = compiled_model.evaluate(inputs)
    output_gradients = {"output": jnp.ones_like(result["output"])}
    outputs, grads = compiled_model.evaluate_all(
        params=inputs, output_gradients=output_gradients
    )
    ref_output = {"output": jnp.array(8.0)}
    assert_results_equal(outputs, ref_output)
    assert_results_equal(grads, {})


def test_mymodel_jax_2():
    model = MyModel2(dimension=1)
    static_inputs = {"input": jnp.array(np_input)}
    compiled_model = compile(
        model=model, backend=JaxBackend(), static_keys=static_inputs, jit=False
    )
    inputs = compiled_model.randomize_params()
    result = compiled_model.evaluate(inputs)
    output_gradients = {"output": jnp.ones_like(result["output"])}
    outputs, grads = compiled_model.evaluate_all(
        params=inputs, output_gradients=output_gradients
    )
    # assert_results_equal(outputs, ref_output)
    assert_results_equal(grads, {})


def test_mymodel_jax():
    """This function tests if jax model is
    properly jitted.
    """
    static_inputs = {"input": jnp.array(np_input)}

    # set a dict_counter dict, if the function is properly jitted,
    # We only
    jit_counter = {"jit_counter": 0}

    def adder(left, right):
        jit_counter["jit_counter"] += 1
        return left + right

    JaxBackend.register_primitive(adder)

    class Adder(CustomPrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="adder",
                output=TensorType([("Var_out", ...)]),
                left=TensorType([("Var_1", ...)]),
                right=TensorType([("Var_2", ...)]),
            )
            self.set_constraint(fn=bcast, keys=["output", "left", "right"])

        def __call__(  # type: ignore[override]
            self,
            left: ConnectionType = NOT_GIVEN,
            right: ConnectionType = NOT_GIVEN,
            output: ConnectionType = NOT_GIVEN,
        ) -> ExtendInfo:
            kwargs = {"left": left, "right": right, "output": output}
            return ExtendInfo(self, kwargs)

    model = MyModel(dimension=1)
    model += Adder()(left="output", right="r1", output=IOKey(name="o1"))
    compiled_model = compile(
        model=model, backend=JaxBackend(), static_keys=static_inputs, jit=True
    )
    inputs = compiled_model.randomize_params()
    compiled_model.evaluate(inputs)
    compiled_model.evaluate(inputs)
    compiled_model.evaluate(inputs)
    assert jit_counter["jit_counter"] == 1


def test_logical_model_jittable_1():
    """Tests for jittablity in Logical domain. Since this model
    requires TensorToList operation before ToTensor, it breaks the
    jit.
    """
    model = Model()
    model += (add1 := Add())(left="l1", right="l2", output=IOKey(name="out1"))
    model += (add2 := Add())(left="l3", right="l4")
    with pytest.raises(Exception) as error_info:
        model += ToTensor()(
            input=Connect(add1.left, add2.left, key=IOKey(name="input"))
        )
    modified_msg = re.sub("\\s*", "", str(error_info.value))
    expected_msg = (
        "Model with enforced Jit can not be extended by a non-jittable model! \
                    Jit can be unforced by setting enforce_jit = False"
    )
    assert modified_msg == re.sub("\\s*", "", expected_msg)


def test_logical_model_jittable_2():
    """Tests for jittablity in Logical domain. Since this model
    sets enforce_jit to False, no error will be thrown.
    """
    model = Model()
    model += (add1 := Add())(left="l1", right="l2", output=IOKey(name="out1"))
    model += (add2 := Add())(left="l3", right="l4")
    model.enforce_jit = False
    model += ToTensor()(input=Connect(add1.left, add2.left, key=IOKey(name="input")))
    assert not model.enforce_jit


def test_logical_model_jittable_3():
    """Tests for jittablity in Logical domain. Since this model
    does not enforce Jit in its init, no error will be thrown.
    """
    model = Model(enforce_jit=False)
    model += (add1 := Add())(left="l1", right="l2", output=IOKey(name="out1"))
    model += (add2 := Add())(left="l3", right="l4")
    model.enforce_jit = False
    model += ToTensor()(input=Connect(add1.left, add2.left, key=IOKey(name="input")))
    assert not model.enforce_jit


def test_physical_model_jit_1():
    """Tests for jittablity in Physical domain. Since compilation is done
    with jit = False, no errors will be raised when model is not jittable.
    """
    model = Model(enforce_jit=False)
    model += (add1 := Add())(left="l1", right="l2", output=IOKey(name="out1"))
    model += (add2 := Add())(left="l3", right="l4")
    model.enforce_jit = False
    model += ToTensor()(input=Connect(add1.left, add2.left, key=IOKey(name="input")))

    backend = JaxBackend()
    compiled_model = compile(model=model, backend=backend, safe=False, jit=False)
    inputs = compiled_model.randomize_params()
    output_gradients = {"out1": backend.ones_like(inputs["input"])}
    outputs, grads = compiled_model.evaluate_all(
        inputs, output_gradients=output_gradients
    )


def test_physical_model_jit_2():
    """Tests for jittablity in Physical domain. Since compilation is done
    with jit = True, exception will be raised because model is not jittable.
    """
    model = Model(enforce_jit=False)
    model += (add1 := Add())(left="l1", right="l2", output=IOKey(name="out1"))
    model += (add2 := Add())(left="l3", right="l4")
    model.enforce_jit = False
    model += ToTensor()(input=Connect(add1.left, add2.left, key=IOKey(name="input")))

    backend = JaxBackend()

    with pytest.raises(Exception) as error_info:
        compile(model=model, backend=backend, safe=False, jit=True)

    expected_msg = "Model is not jittable. Can only be compiled with jit = False."
    assert str(error_info.value) == expected_msg


def test_jit_1():
    jit_counter = {"jit_counter": 0}

    def adder(left, right):
        jit_counter["jit_counter"] += 1
        return left + right

    JaxBackend.register_primitive(adder)

    class Adder(CustomPrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="adder",
                output=TensorType([("Var_out", ...)]),
                left=TensorType([("Var_1", ...)]),
                right=TensorType([("Var_2", ...)]),
            )
            self.set_constraint(fn=bcast, keys=["output", "left", "right"])

    add_model = Add()
    mean_model = Mean(axis=TBD)
    model = Model()
    model += add_model(left="left", right="right")
    with pytest.raises(Exception) as err_info:
        model += mean_model(
            input="input", axis=add_model.output, output=IOKey(name="output")
        )
    assert str(err_info.value) == (
        "Model with enforced Jit can not be extended by a non-jittable model!     "
        "                        Jit can be unforced by setting enforce_jit = False"
    )


def test_jit_2():
    backend = JaxBackend()
    model = Model(enforce_jit=False)
    model += (add_model := Add())(left="left", right="right")
    in1 = add_model.output
    out1 = in1.shape()
    out2 = out1.sum()
    mean_model = Mean(axis=TBD)
    model += mean_model(input="input", axis=out2, output=IOKey(name="output"))
    pm = compile(model=model, backend=backend, safe=False, jit=False)
    params = {
        "left": backend.randn(1, 1),
        "right": backend.randn(1, 1),
        "input": backend.randn(1, 1, 1, 1, 1, 1, 1, 1, 1),
    }
    pm.evaluate(params=params)
    # TODO: Make required assertions!!!
    ...


def test_jit_3():
    backend = JaxBackend()
    model = Model()
    model += Mean(axis=TBD)(input="input", output=IOKey(name="output"), axis="axis")
    pm = compile(model=model, backend=backend, safe=False, jit=False)

    inputs = {"input": backend.randn(1, 2, 3, 2, 3, 2, 3, 2)}
    data = {"axis": 3}

    pm.evaluate(params=inputs, data=data)


def test_jit_4():
    backend = JaxBackend()
    model = Model()
    model += Mean(axis=TBD)(input="input", output=IOKey(name="output"), axis="axis")
    pm = compile(
        model=model, backend=backend, safe=False, jit=True, static_keys={"axis": 3}
    )

    inputs = {"input": backend.randn(1, 2, 3, 2, 3, 2, 3, 2)}
    data = {"axis": 3}

    pm.evaluate(params=inputs, data=data)


def test_jit_5():
    backend = JaxBackend()
    import jax

    @jax.jit
    def evaluate(params):
        input = params["input"]
        keepdim_1 = False
        left = params["left"]
        right = params["right"]
        _Add_0_output = add(left, right)
        _Shape_1_output = shape(_Add_0_output)
        sum_shape = sum(_Shape_1_output)
        idx = (sum_shape**2) * 2 + 3
        output = reduce_mean(input, axis=sum(_Shape_1_output), keepdim=keepdim_1)
        for _ in range(idx):
            output = output + 1
        return {"output": output}

    params = {
        "left": backend.randn(1, 1),
        "right": backend.randn(1, 1),
        "input": backend.randn(1, 1, 1, 1, 1, 1, 1, 1, 1),
    }
    evaluate(params)
    evaluate(params)
    evaluate(params)
