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
from mithril.framework.physical.flat_graph import FlatGraph
from mithril.models import Add, Buffer, ConstraintSolver, Model, Relu, Sigmoid, Tanh


def test_flatgraph_1():
    graph = FlatGraph(
        {"input1", "input2"}, {"output"}, ml.JaxBackend(), ConstraintSolver()
    )
    graph.add_value(Relu().submodel, {"input": "input1", "output": "relu_out"})
    graph.add_value(Buffer().submodel, {"input": "relu_out", "output": "buffer_output"})
    graph.add_value(Buffer().submodel, {"input": "buffer_output", "output": "output"})
    graph.prune_duplicate_nodes({}, {})

    expected_connections = ["input1", "relu_out"]

    assert sorted(graph.connections.keys()) == sorted(expected_connections)
    assert sorted(graph.get_target_keys("relu_out", True)) == (["output"])


def test_flatgraph_2():
    graph = FlatGraph(
        {"input1", "input2"},
        {"output1", "output2", "output3", "output4"},
        ml.JaxBackend(),
        ConstraintSolver(),
    )
    graph.add_value(Relu().submodel, {"input": "input1", "output": "relu_out"})
    graph.add_value(Buffer().submodel, {"input": "relu_out", "output": "output1"})
    graph.add_value(Buffer().submodel, {"input": "output1", "output": "output2"})
    graph.add_value(Buffer().submodel, {"input": "output2", "output": "output3"})
    graph.add_value(Buffer().submodel, {"input": "output3", "output": "output4"})
    graph.prune_duplicate_nodes({}, {})

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
    )
    graph.add_value(Relu().submodel, {"input": "input1", "output": "relu_out"})
    graph.add_value(Relu().submodel, {"input": "relu_out", "output": "output1"})
    graph.add_value(Relu().submodel, {"input": "output1", "output": "output2"})
    graph.prune_duplicate_nodes({}, {})

    expected_connections = ["input1", "output1", "output2", "relu_out"]

    assert sorted(graph.connections.keys()) == sorted(expected_connections)
    assert sorted(graph.connections["output2"].source_keys) == (["output1"])
    assert sorted(graph.connections["relu_out"].target_keys) == (["output1"])


def test_flatgraph_4():
    backend = ml.TorchBackend(dtype=ml.float64)
    model_1 = Model()
    model_1 |= Relu()(input="relu_1", output=ml.IOKey(name="output_1"))
    model_1 |= Relu()(input="relu_2", output=ml.IOKey(name="output_2"))

    model_2 = Model()
    model_2 |= Relu()(input="relu_1", output=ml.IOKey(name="output_1"))
    model_2 |= Relu()(input="relu_2", output=ml.IOKey(name="output_2"))

    model = Model()
    model |= model_1()
    model |= model_2(
        relu_2="input",
        output_2=model_1.relu_2,  # type: ignore
        relu_1=model_1.output_2,  # type: ignore
        output_1=ml.IOKey(name="output"),
    )

    pm = ml.compile(model=model, backend=backend, inference=True)
    assert pm.input_keys == {"input"}
    assert len(pm.flat_graph.all_source_keys) == 3
    assert len(pm.flat_graph.all_target_keys) == 3


def test_infer_static():
    # Infer the only primitive of the model
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Relu()(input="input", output=ml.IOKey(name="output"))

    pm = ml.compile(
        model=model,
        backend=backend,
        constant_keys={"input": backend.randn(5, 5)},
        inference=True,
    )

    assert pm.flat_graph.all_source_keys == set()
    assert pm.flat_graph.all_target_keys == set()
    assert pm.flat_graph.topological_order == []


def test_infer_static_2():
    # Infer only one step, output of infered primitives needed
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Relu()(input="input1", output="relu_out")
    model |= (add := Add())("relu_out", "input2", ml.IOKey(name="output"))

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
    assert pm.flat_graph.topological_order == ["output"]


def test_infer_static_3():
    # Infer only one step, output of infered primitives needed
    # also infered output needed
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Relu()(input="input1", output=ml.IOKey("relu_out"))
    model |= (add := Add())("relu_out", "input2", ml.IOKey(name="output"))

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
    assert pm.flat_graph.topological_order == ["output"]
    assert pm.flat_graph.output_dict == {"output": "output", "relu_out": "relu_out"}


def test_infer_static_4():
    # Infer all primitives
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Relu()(input="input1", output=ml.IOKey("relu_out"))
    model |= Add()("relu_out", "input2", ml.IOKey(name="output"))

    pm = ml.compile(
        model=model,
        backend=backend,
        constant_keys={"input1": backend.randn(5, 5), "input2": backend.ones((5, 5))},
        inference=True,
    )

    assert pm.flat_graph.all_source_keys == set()
    assert pm.flat_graph.all_target_keys == set()
    assert pm.flat_graph.topological_order == []
    assert pm.flat_graph.output_dict == {"output": "output", "relu_out": "relu_out"}


def test_discard_primitive():
    # Discard one of the primitives
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Sigmoid()(input="input1", output=ml.IOKey(name="output1"))
    model |= (relu := Relu())(input="input2", output=ml.IOKey(name="output2"))

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
    assert pm.flat_graph.topological_order == ["output2"]


def test_discard_partial_of_sequence():
    # Discard partial of a sequence
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= (sig := Sigmoid())(input="input1", output=ml.IOKey(name="output1"))
    model |= Tanh()(input="output1", output=ml.IOKey(name="output3"))
    model |= (relu2 := Relu())(input="input2", output=ml.IOKey(name="output2"))

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
    assert pm.flat_graph.topological_order == ["output1", "output2"]


def test_discard_whole_sequence():
    # Discard whole sequence
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Sigmoid()(input="input1", output="output1")
    model |= Tanh()(input="output1", output=ml.IOKey(name="output3"))
    model |= (relu := Relu())(input="input2", output=ml.IOKey(name="output2"))

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
    assert pm.flat_graph.topological_order == ["output2"]


def test_discard_everthing():
    # Discard everything
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Sigmoid()(input="input1", output="output1")
    model |= Tanh()(input="output1", output=ml.IOKey(name="output3"))
    model |= Relu()(input="input2", output=ml.IOKey(name="output2"))

    pm = ml.compile(
        model=model,
        backend=backend,
        discard_keys={"output3", "output2"},
        inference=True,
    )

    assert pm.flat_graph.all_source_keys == set()
    assert pm.flat_graph.all_target_keys == set()
    assert pm.flat_graph.topological_order == []


def test_discard_from_middle():
    # Discard everything
    backend = ml.TorchBackend(dtype=ml.float32)
    model = Model()
    model |= Sigmoid()(input="input1", output="output1")
    model |= Tanh()(input="output1", output=ml.IOKey(name="output3"))
    model |= Relu()(input="input2", output=ml.IOKey(name="output2"))

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
