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


import pytest

import mithril as ml
from mithril.framework.logical.operators import (
    CosineOp,
    MultiplyOp,
    SineOp,
    TransposeOp,
)
from mithril.framework.physical.flat_graph import FlatGraph
from mithril.models import Add, Buffer, ConstraintSolver, Model, Relu, Sigmoid, Tanh


def test_flatgraph_1():
    graph = FlatGraph(
        {"input1", "input2"}, {"output"}, ml.JaxBackend(), ConstraintSolver(), []
    )
    graph.add_value(Relu().submodel, {"input": "input1", "output": "relu_out"})
    graph.add_value(Buffer().submodel, {"input": "relu_out", "output": "buffer_output"})
    graph.add_value(Buffer().submodel, {"input": "buffer_output", "output": "output"})
    for op in graph.all_models:
        graph.prune_duplicate_operation(op, {}, {})

    expected_connections = ["input1", "relu_out"]

    assert sorted(graph.connections.keys()) == sorted(expected_connections)
    assert sorted(graph.get_target_keys("relu_out", True)) == (["output"])


def test_flatgraph_2():
    graph = FlatGraph(
        {"input1", "input2"},
        {"output1", "output2", "output3", "output4"},
        ml.JaxBackend(),
        ConstraintSolver(),
        [],
    )
    graph.add_value(Relu().submodel, {"input": "input1", "output": "relu_out"})
    graph.add_value(Buffer().submodel, {"input": "relu_out", "output": "output1"})
    graph.add_value(Buffer().submodel, {"input": "output1", "output": "output2"})
    graph.add_value(Buffer().submodel, {"input": "output2", "output": "output3"})
    graph.add_value(Buffer().submodel, {"input": "output3", "output": "output4"})
    for op in graph.all_models:
        graph.prune_duplicate_operation(op, {}, {})

    expected_connections = ["input1", "relu_out"]

    assert sorted(graph.connections.keys()) == sorted(expected_connections)
    assert graph.output_dict["output4"] == "relu_out"
    assert sorted(graph.get_target_keys("relu_out", True)) == (
        ["output1", "output2", "output3", "output4"]
    )


def test_flatgraph_3():
    graph = FlatGraph(
        {"input1", "input2"},
        {"output1", "output2", "output3", "output4"},
        ml.JaxBackend(),
        ConstraintSolver(),
        [],
    )
    graph.add_value(Relu().submodel, {"input": "input1", "output": "relu_out"})
    graph.add_value(Relu().submodel, {"input": "relu_out", "output": "output1"})
    graph.add_value(Relu().submodel, {"input": "output1", "output": "output2"})
    for op in graph.all_models:
        graph.prune_duplicate_operation(op, {}, {})

    expected_connections = ["input1", "output1", "output2", "relu_out"]

    assert sorted(graph.connections.keys()) == sorted(expected_connections)
    assert sorted(graph.connections["output2"].source_keys) == (["output1"])
    assert sorted(graph.connections["relu_out"].target_keys) == (["output1"])


def test_flatgraph_4():
    backend = ml.TorchBackend(dtype=ml.float64)
    model_1 = Model()
    model_1 |= Relu().connect(input="relu_1", output="output_1")
    model_1 |= Relu().connect(input="relu_2", output="output_2")
    model_1.expose_keys("output_1", "output_2")

    model_2 = Model()
    model_2 |= Relu().connect(input="relu_1", output="output_1")
    model_2 |= Relu().connect(input="relu_2", output="output_2")
    model_1.expose_keys("output_1", "output_2")

    model = Model()
    model |= model_1.connect()
    model |= model_2.connect(
        relu_2="input",
        output_2=model_1.relu_2,  # type: ignore
        relu_1=model_1.output_2,  # type: ignore
        output_1=ml.IOKey(name="output"),
    )
    model.expose_keys("output")

    pm = ml.compile(model=model, backend=backend, inference=True)
    assert pm.input_keys == {"input"}
    assert len(pm.flat_graph.all_source_keys) == 3
    assert len(pm.flat_graph.all_target_keys) == 3


def test_infer_static():
    # Infer the only primitive of the model
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Relu().connect(input="input", output=ml.IOKey(name="output"))

    pm = ml.compile(
        model=model,
        backend=backend,
        constant_keys={"input": backend.randn(5, 5)},
        inference=True,
    )

    assert pm.flat_graph.all_source_keys == set()
    assert pm.flat_graph.all_target_keys == set()
    assert list(pm.flat_graph.topological_order) == []


def test_infer_static_2():
    # Infer only one step, output of infered primitives needed
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Relu().connect(input="input1", output="relu_out")
    model |= (add := Add()).connect("relu_out", "input2", ml.IOKey(name="output"))

    pm = ml.compile(
        model=model,
        backend=backend,
        constant_keys={"input1": backend.randn(5, 5)},
        inference=True,
    )

    assert (
        len(pm.flat_graph.topological_order) == 1
        and add.submodel in pm.flat_graph.all_models
    )
    assert pm.flat_graph.all_source_keys == {"relu_out", "input2"}
    assert pm.flat_graph.all_target_keys == {"output"}
    assert list(pm.flat_graph.topological_order) == ["output"]


def test_infer_static_3():
    # Infer only one step, output of infered primitives needed
    # also infered output needed
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Relu().connect(input="input1", output="relu_out")
    model |= (add := Add()).connect("relu_out", "input2", "output")
    model.expose_keys("relu_out", "output")

    pm = ml.compile(
        model=model,
        backend=backend,
        constant_keys={"input1": backend.randn(5, 5)},
        inference=True,
    )

    assert (
        len(pm.flat_graph.topological_order) == 1
        and add.submodel in pm.flat_graph.all_models
    )
    assert pm.flat_graph.all_source_keys == {"relu_out", "input2"}
    assert pm.flat_graph.all_target_keys == {"output"}
    assert list(pm.flat_graph.topological_order) == ["output"]
    assert pm.flat_graph.output_dict == {"output": "output", "relu_out": "relu_out"}


def test_infer_static_4():
    # Infer all primitives
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Relu().connect(input="input1", output="relu_out")
    model |= Add().connect("relu_out", "input2", "output")
    model.expose_keys("relu_out", "output")

    pm = ml.compile(
        model=model,
        backend=backend,
        constant_keys={"input1": backend.randn(5, 5), "input2": backend.ones((5, 5))},
        inference=True,
    )

    assert pm.flat_graph.all_source_keys == set()
    assert pm.flat_graph.all_target_keys == set()
    assert list(pm.flat_graph.topological_order) == []
    assert pm.flat_graph.output_dict == {"output": "output", "relu_out": "relu_out"}


def test_discard_primitive():
    # Discard one of the primitives
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Sigmoid().connect(input="input1", output=ml.IOKey(name="output1"))
    model |= (relu := Relu()).connect(input="input2", output=ml.IOKey(name="output2"))

    pm = ml.compile(
        model=model,
        backend=backend,
        discard_keys={"output1"},
        inference=True,
    )

    assert (
        len(pm.flat_graph.topological_order) == 1
        and relu.submodel in pm.flat_graph.all_models
    )
    assert pm.flat_graph.all_source_keys == {"input2"}
    assert pm.flat_graph.all_target_keys == {"output2"}
    assert list(pm.flat_graph.topological_order) == ["output2"]


def test_discard_partial_of_sequence():
    # Discard partial of a sequence
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= (sig := Sigmoid()).connect(input="input1", output="output1")
    model |= Tanh().connect(input="output1", output="output3")
    model |= (relu2 := Relu()).connect(input="input2", output="output2")
    model.expose_keys("output1", "output2", "output3")

    pm = ml.compile(
        model=model,
        backend=backend,
        discard_keys={"output3"},
        inference=True,
    )

    assert (
        len(pm.flat_graph.topological_order) == 2
        and relu2.submodel in pm.flat_graph.all_models
        and sig.submodel in pm.flat_graph.all_models
    )
    assert pm.flat_graph.all_source_keys == {"input1", "input2"}
    assert pm.flat_graph.all_target_keys == {"output1", "output2"}
    assert list(pm.flat_graph.topological_order) == ["output2", "output1"]


def test_discard_whole_sequence():
    # Discard whole sequence
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Sigmoid().connect(input="input1", output="output1")
    model |= Tanh().connect(input="output1", output=ml.IOKey(name="output3"))
    model |= (relu := Relu()).connect(input="input2", output=ml.IOKey(name="output2"))

    pm = ml.compile(
        model=model,
        backend=backend,
        discard_keys={"output3"},
        inference=True,
    )

    assert (
        len(pm.flat_graph.topological_order) == 1
        and relu.submodel in pm.flat_graph.all_models
    )
    assert pm.flat_graph.all_source_keys == {"input2"}
    assert pm.flat_graph.all_target_keys == {"output2"}
    assert list(pm.flat_graph.topological_order) == ["output2"]


def test_discard_everthing():
    # Discard everything
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Sigmoid().connect(input="input1", output="output1")
    model |= Tanh().connect(input="output1", output=ml.IOKey(name="output3"))
    model |= Relu().connect(input="input2", output=ml.IOKey(name="output2"))

    pm = ml.compile(
        model=model,
        backend=backend,
        discard_keys={"output3", "output2"},
        inference=True,
    )

    assert pm.flat_graph.all_source_keys == set()
    assert pm.flat_graph.all_target_keys == set()
    assert list(pm.flat_graph.topological_order) == []


def test_discard_from_middle():
    # Discard everything
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Sigmoid().connect(input="input1", output="output1")
    model |= Tanh().connect(input="output1", output=ml.IOKey(name="output3"))
    model |= Relu().connect(input="input2", output=ml.IOKey(name="output2"))

    with pytest.raises(KeyError) as e:
        ml.compile(
            model=model,
            backend=backend,
            discard_keys={"output1"},
            inference=True,
        )

    assert str(e.value) == (
        "'Provided discard keys must be subset of the input keys and output keys. "
        "Invalid keys: output1.'"
    )


def test_insert_to_input_siso_used_by_single_op():
    # Insert before SISO operator
    sine_op = SineOp()
    cosine_op = CosineOp()
    transpose_op = TransposeOp()
    fg = FlatGraph({"input"}, {"output"}, ml.TorchBackend(), ConstraintSolver(), [])
    fg.add_value(sine_op, {"input": "input", "output": "sin_out"})
    fg.add_value(cosine_op, {"input": "sin_out", "output": "output"})

    assert fg.input_keys == {"input"}
    assert fg.output_keys == {"output"}
    assert fg.all_keys == {"input", "sin_out", "output"}
    assert fg.all_models == [sine_op, cosine_op]
    assert fg.get_source_keys("output") == ["sin_out"]
    assert fg.get_target_keys("sin_out") == ["output"]

    fg.insert_operator_before(
        transpose_op,
        {"input": "sin_out", "output": "transpose_out"},
        cosine_op,
        "sin_out",
    )
    assert fg.all_keys == {"input", "sin_out", "transpose_out", "output"}
    assert fg.all_models == [sine_op, cosine_op, transpose_op]
    assert fg.get_source_keys("output") == ["transpose_out"]
    assert fg.get_target_keys("transpose_out") == ["output"]
    assert fg.get_source_keys("transpose_out") == ["sin_out"]
    assert fg.get_target_keys("sin_out") == ["transpose_out"]


def test_insert_to_input_siso_used_by_multiple_ops():
    # Insert before SISO operator
    # The input is used by multiple operators
    sine_op = SineOp()
    cosine_op = CosineOp()
    multiply_op = MultiplyOp()
    transpose_op = TransposeOp()
    fg = FlatGraph({"input"}, {"output"}, ml.TorchBackend(), ConstraintSolver(), [])
    fg.add_value(sine_op, {"input": "input", "output": "sin_out"})
    fg.add_value(cosine_op, {"input": "sin_out", "output": "output"})
    fg.add_value(
        multiply_op, {"left": "sin_out", "right": "input", "output": "multiply_out"}
    )

    assert fg.input_keys == {"input"}
    assert fg.output_keys == {"output"}
    assert fg.all_keys == {"input", "sin_out", "output", "multiply_out"}
    assert fg.all_models == [sine_op, cosine_op, multiply_op]
    assert fg.get_source_keys("output") == ["sin_out"]
    assert fg.get_source_keys("multiply_out") == ["sin_out", "input"]
    assert fg.get_target_keys("sin_out") == ["output", "multiply_out"]

    fg.insert_operator_before(
        transpose_op,
        {"input": "sin_out", "output": "transpose_out"},
        cosine_op,
        "sin_out",
    )
    assert fg.all_keys == {
        "input",
        "sin_out",
        "transpose_out",
        "output",
        "multiply_out",
    }
    assert fg.all_models == [sine_op, cosine_op, multiply_op, transpose_op]
    assert fg.get_source_keys("output") == ["transpose_out"]
    assert fg.get_target_keys("transpose_out") == ["output"]
    assert fg.get_source_keys("transpose_out") == ["sin_out"]
    assert fg.get_target_keys("sin_out") == ["multiply_out", "transpose_out"]
    assert fg.get_source_keys("multiply_out") == ["sin_out", "input"]


def test_insert_to_input_miso_used_by_single_op():
    # Insert before MISO operator
    # The input is used by single operators
    sine_op = SineOp()
    multiply_op = MultiplyOp()
    transpose_op = TransposeOp()
    fg = FlatGraph({"input"}, {"output"}, ml.TorchBackend(), ConstraintSolver(), [])
    fg.add_value(sine_op, {"input": "input", "output": "sin_out"})
    fg.add_value(multiply_op, {"left": "sin_out", "right": "input", "output": "output"})

    assert fg.input_keys == {"input"}
    assert fg.output_keys == {"output"}
    assert fg.all_keys == {"input", "sin_out", "output"}
    assert fg.all_models == [sine_op, multiply_op]
    assert fg.get_source_keys("output") == ["sin_out", "input"]
    assert fg.get_target_keys("sin_out") == ["output"]

    fg.insert_operator_before(
        transpose_op,
        {"input": "sin_out", "output": "transpose_out"},
        multiply_op,
        "sin_out",
    )
    assert fg.all_keys == {"input", "sin_out", "transpose_out", "output"}
    assert fg.all_models == [sine_op, multiply_op, transpose_op]
    assert fg.get_source_keys("output") == ["transpose_out", "input"]
    assert fg.get_target_keys("transpose_out") == ["output"]
    assert fg.get_source_keys("transpose_out") == ["sin_out"]
    assert fg.get_target_keys("sin_out") == ["transpose_out"]


def test_insert_to_input_miso_used_by_multiple_ops():
    # Insert before MISO operator
    # The input is used by multiple operators
    sine_op = SineOp()
    multiply_op = MultiplyOp()
    cosine_op = CosineOp()
    transpose_op = TransposeOp()
    fg = FlatGraph({"input"}, {"output"}, ml.TorchBackend(), ConstraintSolver(), [])
    fg.add_value(sine_op, {"input": "input", "output": "sin_out"})
    fg.add_value(
        multiply_op, {"left": "sin_out", "right": "input", "output": "multiply_out"}
    )
    fg.add_value(cosine_op, {"input": "sin_out", "output": "output"})

    assert fg.input_keys == {"input"}
    assert fg.output_keys == {"output"}
    assert fg.all_keys == {"input", "sin_out", "multiply_out", "output"}
    assert fg.all_models == [sine_op, multiply_op, cosine_op]
    assert fg.get_source_keys("output") == ["sin_out"]
    assert fg.get_source_keys("multiply_out") == ["sin_out", "input"]
    assert fg.get_target_keys("sin_out") == ["multiply_out", "output"]

    fg.insert_operator_before(
        transpose_op,
        {"input": "sin_out", "output": "transpose_out"},
        multiply_op,
        "sin_out",
    )
    assert fg.all_keys == {
        "input",
        "sin_out",
        "transpose_out",
        "multiply_out",
        "output",
    }
    assert fg.all_models == [sine_op, multiply_op, cosine_op, transpose_op]
    assert fg.get_source_keys("output") == ["sin_out"]
    assert fg.get_target_keys("transpose_out") == ["multiply_out"]
    assert fg.get_source_keys("transpose_out") == ["sin_out"]
    assert fg.get_target_keys("sin_out") == ["output", "transpose_out"]
    assert fg.get_source_keys("multiply_out") == ["transpose_out", "input"]


def test_insert_to_output_siso_used_by_single_op():
    # Insert after SISO operator
    sine_op = SineOp()
    cosine_op = CosineOp()
    transpose_op = TransposeOp()
    fg = FlatGraph({"input"}, {"output"}, ml.TorchBackend(), ConstraintSolver(), [])
    fg.add_value(sine_op, {"input": "input", "output": "sin_out"})
    fg.add_value(cosine_op, {"input": "sin_out", "output": "output"})

    assert fg.input_keys == {"input"}
    assert fg.output_keys == {"output"}
    assert fg.all_keys == {"input", "sin_out", "output"}
    assert fg.all_models == [sine_op, cosine_op]
    assert fg.get_source_keys("output") == ["sin_out"]
    assert fg.get_target_keys("sin_out") == ["output"]

    fg.insert_operator_after(
        transpose_op,
        {"input": "transpose_input", "output": "sin_out"},
        "transpose_input",
        sine_op,
        "sin_out",
    )
    assert fg.all_keys == {"input", "sin_out", "transpose_input", "output"}
    assert fg.all_models == [sine_op, cosine_op, transpose_op]
    assert fg.get_source_keys("output") == ["sin_out"]
    assert fg.get_target_keys("transpose_input") == ["sin_out"]
    assert fg.get_source_keys("transpose_input") == ["input"]
    assert fg.get_target_keys("sin_out") == ["output"]


def test_insert_to_output_siso_used_by_multiple_ops():
    # Insert before SISO operator
    # The output is used by multiple operators
    sine_op = SineOp()
    cosine_op = CosineOp()
    transpose_op = TransposeOp()
    multiply_op = MultiplyOp()
    fg = FlatGraph({"input"}, {"output"}, ml.TorchBackend(), ConstraintSolver(), [])
    fg.add_value(sine_op, {"input": "input", "output": "sin_out"})
    fg.add_value(cosine_op, {"input": "sin_out", "output": "output"})
    fg.add_value(
        multiply_op, {"left": "sin_out", "right": "input", "output": "multiply_out"}
    )

    assert fg.input_keys == {"input"}
    assert fg.output_keys == {"output"}
    assert fg.all_keys == {"input", "sin_out", "output", "multiply_out"}
    assert fg.all_models == [sine_op, cosine_op, multiply_op]
    assert fg.get_source_keys("output") == ["sin_out"]
    assert fg.get_target_keys("sin_out") == ["output", "multiply_out"]

    fg.insert_operator_after(
        transpose_op,
        {"input": "transpose_input", "output": "sin_out"},
        "transpose_input",
        sine_op,
        "sin_out",
    )
    assert fg.all_keys == {
        "input",
        "sin_out",
        "transpose_input",
        "multiply_out",
        "output",
    }
    assert fg.all_models == [sine_op, cosine_op, multiply_op, transpose_op]
    assert fg.get_source_keys("output") == ["sin_out"]
    assert fg.get_target_keys("sin_out") == ["output", "multiply_out"]
    assert fg.get_source_keys("sin_out") == ["transpose_input"]
    assert fg.get_target_keys("transpose_input") == ["sin_out"]
    assert fg.get_source_keys("transpose_input") == ["input"]
    assert fg.get_source_keys("multiply_out") == ["sin_out", "input"]


def test_insert_errors():
    # Insert after SISO operator
    # The output is used by multiple operators
    sine_op = SineOp()
    cosine_op = CosineOp()
    transpose_op = TransposeOp()
    fg = FlatGraph({"input"}, {"output"}, ml.TorchBackend(), ConstraintSolver(), [])
    fg.add_value(sine_op, {"input": "input", "output": "sin_out"})

    # Insert operator that is not in the graph
    with pytest.raises(ValueError) as e:
        fg.insert_operator_after(
            transpose_op,
            {"input": "transpose_input", "output": "sin_out"},
            "transpose_input",
            cosine_op,
            "sin_out",
        )

    assert str(e.value) == ("Base operator `cos` must already be in the graph")

    with pytest.raises(ValueError) as e:
        fg.insert_operator_before(
            transpose_op,
            {"input": "transpose_input", "output": "sin_out"},
            cosine_op,
            "sin_out",
        )

    assert str(e.value) == ("Base operator `cos` must already be in the graph")

    # Insert with wrong key
    with pytest.raises(ValueError) as e:
        fg.insert_operator_after(
            transpose_op,
            {"input": "transpose_input", "output": "sin_out"},
            "transpose_input",
            sine_op,
            "asd",
        )

    assert str(e.value) == ("Inserted key `asd` must be in the keys dictionary")

    with pytest.raises(ValueError) as e:
        fg.insert_operator_after(
            transpose_op,
            {"input": "transpose_input", "output": "sin_out"},
            "sin_out",
            sine_op,
            "sin_out",
        )

    assert str(e.value) == ("Source key `sin_out` must be in the keys dictionary")

    with pytest.raises(ValueError) as e:
        fg.insert_operator_before(
            transpose_op,
            {"input": "transpose_input", "output": "sin_out"},
            sine_op,
            "asd",
        )

    assert str(e.value) == ("Inserted key `asd` must be in the keys dictionary")

    with pytest.raises(ValueError) as e:
        fg.insert_operator_before(
            transpose_op,
            {"input": "transpose_input", "output": "qwerty"},
            sine_op,
            "sin_out",
        )

    assert str(e.value) == ("Inserted key `sin_out` must be in the keys dictionary")


def test_insert_to_output_key_not_in_keys_err():
    # If the target inserted key not in the keys raise error
    sine_op = SineOp()
    cosine_op = CosineOp()
    fg = FlatGraph({"input"}, {"output"}, ml.TorchBackend(), ConstraintSolver(), [])
    fg.add_value(sine_op, {"input": "input", "output": "sin_out"})
    with pytest.raises(ValueError) as e:
        fg.insert_operator_after(
            cosine_op,
            {"input": "sin_out", "output": "cos_out"},
            "sin_out",
            sine_op,
            "not_in_keys",
        )

    assert str(e.value) == ("Inserted key `not_in_keys` must be in the keys dictionary")
