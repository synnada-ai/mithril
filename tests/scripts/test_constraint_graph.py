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

from mithril.framework.common import (
    ConstrainResultType,
    Constraint,
    IOHyperEdge,
    Updates,
    UpdateType,
)
from mithril.models import Model, Sigmoid


class ConstrGraphTestBase:
    conditions: tuple[bool, ...]
    trace_list: list[str]
    result_map: dict[tuple[bool, ...], list[str]]

    def model(self) -> Model:
        # model with constraint graph
        raise NotImplementedError()

    def reset_state(self):
        # resets state of the class variables
        # to be used again
        self.conditions = (False,) * len(self.conditions)
        self.trace_list = []

    def assert_results(self, *conds: bool):
        self.reset_state()
        model = self.model()
        self.conditions = conds  # set conditions
        model.set_shapes(input=["a"])  # trigger constraint solver loop
        assert sorted(self.trace_list) == sorted(self.result_map[conds])


class ThreeConstraints(ConstrGraphTestBase):
    # class that defines three constraints
    # each constraint returns True or False based on the condition
    # if returns true, it appends the constraint name to the trace_list
    conditions: tuple[bool, ...] = (False, False, False)

    def constraint_1(
        self, input: IOHyperEdge, output: IOHyperEdge
    ) -> ConstrainResultType:
        if self.conditions[0]:
            self.trace_list.append("c1")
            return True, Updates()
        else:
            return False, Updates()

    def constraint_2(
        self, input: IOHyperEdge, output: IOHyperEdge
    ) -> ConstrainResultType:
        if self.conditions[1]:
            self.trace_list.append("c2")
            return True, Updates()
        else:
            return False, Updates()

    def constraint_3(
        self, input: IOHyperEdge, output: IOHyperEdge
    ) -> ConstrainResultType:
        if self.conditions[2]:
            self.trace_list.append("c3")
            return True, Updates()
        else:
            return False, Updates()


@pytest.mark.parametrize("cond3", [True, False])
@pytest.mark.parametrize("cond2", [True, False])
@pytest.mark.parametrize("cond1", [True, False])
class ThreeConstraintsTest(ThreeConstraints):
    """Test base class for testing dependency
    graphs including three constraints,

    In order to use this Test base, create a subclass that inherits from this class

    subclass should have two fields defined:
       1. result_map: dict[tuple[bool, ...], list[str]]
       2. model: Callable[[], PrimitiveModel]

    result_map defines result for each possible condition of constraints

    model should return a PrimitiveModel object with constraints defined

    in each constraints, if its conditions true, appends its name to trace_list
    (namely, c1, c2, c3). However, if a constraint has dependencies, it cannot be
    called and therefore, its name will not be appended to trace_list.

    keys of result_map defines each conditions, and values are the expected results
    of trace list. (see examples of TestThreeSequential, TestThreeOneToMany,
    TestThreeManyToOne)
    """

    # runs test for all possible conditions
    def test_conditions(self, cond1: bool, cond2: bool, cond3: bool):
        self.assert_results(cond1, cond2, cond3)


class FourConstraints(ThreeConstraints):
    # adds another constraint to the ThreeConstraints
    conditions: tuple[bool, ...] = (False, False, False, False)

    def constraint_4(
        self, input: IOHyperEdge, output: IOHyperEdge
    ) -> ConstrainResultType:
        if self.conditions[3]:
            self.trace_list.append("c4")
            return True, Updates()
        else:
            return False, Updates()


@pytest.mark.parametrize("cond4", [True, False])
@pytest.mark.parametrize("cond3", [True, False])
@pytest.mark.parametrize("cond2", [True, False])
@pytest.mark.parametrize("cond1", [True, False])
class FourConstraintsTest(FourConstraints):
    # runs test for all possible conditions
    def test_conditions(self, cond1: bool, cond2: bool, cond3: bool, cond4: bool):
        self.assert_results(cond1, cond2, cond3, cond4)


class TestThreeSequential(ThreeConstraintsTest):
    result_map: dict[tuple[bool, ...], list[str]] = {
        (True, True, True): ["c1", "c2", "c3"],
        (True, True, False): ["c1", "c2"],
        (True, False, True): ["c1"],
        (True, False, False): ["c1"],
        (False, True, True): [],
        (False, True, False): [],
        (False, False, True): [],
        (False, False, False): [],
    }

    def model(self) -> Model:
        # constr_1 ----> constr_2 ----> constr_3

        model = Sigmoid()
        c1 = model.add_constraint(fn=self.constraint_1, keys=["input", "output"])
        c2 = model.add_constraint(
            fn=self.constraint_2, keys=["input", "output"], dependencies={c1}
        )
        model.add_constraint(
            fn=self.constraint_3, keys=["input", "output"], dependencies={c2}
        )
        return model


class TestThreeOneToMany(ThreeConstraintsTest):
    result_map: dict[tuple[bool, ...], list[str]] = {
        (True, True, True): ["c1", "c2", "c3"],
        (True, True, False): ["c1", "c2"],
        (True, False, True): ["c1", "c3"],
        (True, False, False): ["c1"],
        (False, True, True): [],
        (False, True, False): [],
        (False, False, True): [],
        (False, False, False): [],
    }

    def model(self) -> Model:
        #              + ---> constr_2
        #              |
        # constr_1 --- +
        #              |
        #              + ---> constr_3

        model = Sigmoid()
        c1 = model.add_constraint(fn=self.constraint_1, keys=["input", "output"])
        model.add_constraint(
            fn=self.constraint_2, keys=["input", "output"], dependencies={c1}
        )
        model.add_constraint(
            fn=self.constraint_3, keys=["input", "output"], dependencies={c1}
        )
        return model


class TestThreeManyToOne(ThreeConstraintsTest):
    result_map: dict[tuple[bool, ...], list[str]] = {
        (True, True, True): ["c1", "c2", "c3"],
        (True, True, False): ["c1", "c2"],
        (True, False, True): ["c1"],
        (True, False, False): ["c1"],
        (False, True, True): ["c2"],
        (False, True, False): ["c2"],
        (False, False, True): [],
        (False, False, False): [],
    }

    def model(self) -> Model:
        #  constr_1 --- +
        #               |
        #               + ---> constr_3
        #               |
        #  constr_2 --- +

        model = Sigmoid()
        c1 = model.add_constraint(fn=self.constraint_1, keys=["input", "output"])
        c2 = model.add_constraint(fn=self.constraint_2, keys=["input", "output"])
        model.add_constraint(
            fn=self.constraint_3, keys=["input", "output"], dependencies={c1, c2}
        )
        return model


class TestFourDiamond(FourConstraintsTest):
    result_map: dict[tuple[bool, ...], list[str]] = {
        (True, True, True, True): ["c1", "c2", "c3", "c4"],
        (True, True, True, False): ["c1", "c2", "c3"],
        (True, True, False, True): ["c1", "c2"],
        (True, True, False, False): ["c1", "c2"],
        (True, False, True, True): ["c1", "c3"],
        (True, False, True, False): ["c1", "c3"],
        (True, False, False, True): ["c1"],
        (True, False, False, False): ["c1"],
        (False, True, True, True): [],
        (False, True, True, False): [],
        (False, True, False, True): [],
        (False, True, False, False): [],
        (False, False, True, True): [],
        (False, False, True, False): [],
        (False, False, False, True): [],
        (False, False, False, False): [],
    }

    def model(self) -> Model:
        #              + ---> constr_2 --- +
        #              |                   |
        # constr_1 --- +                   + ---> constr_4
        #              |                   |
        #              + ---> constr_3 --- +

        model = Sigmoid()

        c1 = model.add_constraint(fn=self.constraint_1, keys=["input", "output"])
        c2 = model.add_constraint(
            fn=self.constraint_2, keys=["input", "output"], dependencies={c1}
        )

        c3 = model.add_constraint(
            fn=self.constraint_3, keys=["input", "output"], dependencies={c1}
        )

        model.add_constraint(
            fn=self.constraint_4, keys=["input", "output"], dependencies={c2, c3}
        )
        return model


class TestTwoPhaseDiamond(TestFourDiamond):
    result_map: dict[tuple[bool, ...], list[str]] = {
        (True, True, True, True): ["c1", "c2", "c3", "c4"],
        (True, True, True, False): ["c1", "c2", "c3"],
        (True, True, False, True): ["c1", "c2"],
        (True, True, False, False): ["c1", "c2"],
        (True, False, True, True): ["c1", "c2", "c3", "c4"],
        (True, False, True, False): ["c1", "c2", "c3"],
        (True, False, False, True): ["c1", "c2"],
        (True, False, False, False): ["c1", "c2"],
        (False, True, True, True): ["c1", "c2", "c3", "c4"],
        (False, True, True, False): ["c1", "c2", "c3"],
        (False, True, False, True): ["c1", "c2"],
        (False, True, False, False): ["c1", "c2"],
        (False, False, True, True): ["c1", "c2", "c3", "c4"],
        (False, False, True, False): ["c1", "c2", "c3"],
        (False, False, False, True): ["c1", "c2"],
        (False, False, False, False): ["c1", "c2"],
    }

    def model(self) -> Model:
        model = super().model()
        self.conditions = (True, True, False, False)
        model.set_shapes(input=[("V1", ...), "a"])
        return model


class TestTwoPhaseDiamondAtInit(TestFourDiamond):
    result_map: dict[tuple[bool, ...], list[str]] = {
        (True, True, True, True): ["c1", "c2", "c3", "c4"],
        (True, True, True, False): ["c1", "c2", "c3"],
        (True, True, False, True): ["c1", "c2"],
        (True, True, False, False): ["c1", "c2"],
        (True, False, True, True): ["c1", "c2", "c3", "c4"],
        (True, False, True, False): ["c1", "c2", "c3"],
        (True, False, False, True): ["c1", "c2"],
        (True, False, False, False): ["c1", "c2"],
        (False, True, True, True): ["c1", "c2", "c3", "c4"],
        (False, True, True, False): ["c1", "c2", "c3"],
        (False, True, False, True): ["c1", "c2"],
        (False, True, False, False): ["c1", "c2"],
        (False, False, True, True): ["c1", "c2", "c3", "c4"],
        (False, False, True, False): ["c1", "c2", "c3"],
        (False, False, False, True): ["c1", "c2"],
        (False, False, False, False): ["c1", "c2"],
    }

    def model(self) -> Model:
        self.conditions = (True, True, False, False)
        model = super().model()
        return model


class TestFourManyToMany(FourConstraintsTest):
    result_map: dict[tuple[bool, ...], list[str]] = {
        (True, True, True, True): ["c1", "c2", "c3", "c4"],
        (True, True, True, False): ["c1", "c2", "c3"],
        (True, True, False, True): ["c1", "c2", "c4"],
        (True, True, False, False): ["c1", "c2"],
        (True, False, True, True): ["c1"],
        (True, False, True, False): ["c1"],
        (True, False, False, True): ["c1"],
        (True, False, False, False): ["c1"],
        (False, True, True, True): ["c2"],
        (False, True, True, False): ["c2"],
        (False, True, False, True): ["c2"],
        (False, True, False, False): ["c2"],
        (False, False, True, True): [],
        (False, False, True, False): [],
        (False, False, False, True): [],
        (False, False, False, False): [],
    }

    def model(self) -> Model:
        #  constr_1 --- +     + ---> constr_3
        #               |     |
        #               + --- +
        #               |     |
        #  constr_2 --- +     + ---> constr_4

        model = Sigmoid()

        c1 = model.add_constraint(fn=self.constraint_1, keys=["input", "output"])
        c2 = model.add_constraint(fn=self.constraint_2, keys=["input", "output"])

        model.add_constraint(
            fn=self.constraint_3, keys=["input", "output"], dependencies={c1, c2}
        )

        model.add_constraint(
            fn=self.constraint_4, keys=["input", "output"], dependencies={c1, c2}
        )
        return model


class TestFourTwoSequential(FourConstraintsTest):
    result_map: dict[tuple[bool, ...], list[str]] = {
        (True, True, True, True): ["c1", "c2", "c3", "c4"],
        (True, True, True, False): ["c1", "c2", "c3"],
        (True, True, False, True): ["c1", "c2"],
        (True, True, False, False): ["c1", "c2"],
        (True, False, True, True): ["c1", "c3", "c4"],
        (True, False, True, False): ["c1", "c3"],
        (True, False, False, True): ["c1"],
        (True, False, False, False): ["c1"],
        (False, True, True, True): ["c3", "c4"],
        (False, True, True, False): ["c3"],
        (False, True, False, True): [],
        (False, True, False, False): [],
        (False, False, True, True): ["c3", "c4"],
        (False, False, True, False): ["c3"],
        (False, False, False, True): [],
        (False, False, False, False): [],
    }

    def model(self) -> Model:
        # constr_1 ----> constr_2
        #
        # constr_3 ----> constr_4

        model = Sigmoid()

        c1 = model.add_constraint(fn=self.constraint_1, keys=["input", "output"])
        model.add_constraint(
            fn=self.constraint_2, keys=["input", "output"], dependencies={c1}
        )

        c3 = model.add_constraint(fn=self.constraint_3, keys=["input", "output"])

        model.add_constraint(
            fn=self.constraint_4, keys=["input", "output"], dependencies={c3}
        )
        return model


def mock_constraint_fn(input: IOHyperEdge, output: IOHyperEdge) -> ConstrainResultType:
    return True, Updates()


def test_constraint_objects_single_dependency():
    a = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])
    b = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])

    b.add_dependencies(a)

    assert a.children == {b}
    assert a.parents == set()
    assert b.children == set()
    assert b.parents == {a}


def test_constraint_objects_multiple_sequential_dependency():
    a = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])
    b = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])
    c = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])

    c.add_dependencies(b)
    b.add_dependencies(a)

    assert a.parents == set()
    assert a.children == {b}

    assert b.parents == {a}
    assert b.children == {c}

    assert c.parents == {b}
    assert c.children == set()


def test_constraint_objects_multiple_sequential_dependency_clear():
    a = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])
    b = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])
    c = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])

    c.add_dependencies(b)
    b.add_dependencies(a)

    a.clear()

    assert a.parents == set()
    assert a.children == set()

    assert b.parents == set()
    assert b.children == {c}

    assert c.parents == {b}
    assert c.children == set()


def test_constraint_objects_multiple_sequential_dependency_clear_two_constraint():
    a = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])
    b = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])
    c = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])

    c.add_dependencies(b)
    b.add_dependencies(a)

    a.clear()
    b.clear()

    assert a.parents == set()
    assert a.children == set()

    assert b.parents == set()
    assert b.children == set()

    assert c.parents == set()
    assert c.children == set()


def test_constraint_objects_multiple_sequential_dependency_clear_call():
    a = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])
    b = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])
    c = Constraint(fn=mock_constraint_fn, types=[UpdateType.SHAPE])

    c.add_dependencies(b)
    b.add_dependencies(a)

    a([IOHyperEdge(), IOHyperEdge()])
    b([IOHyperEdge(), IOHyperEdge()])

    assert a.parents == set()
    assert a.children == set()

    assert b.parents == set()
    assert b.children == set()

    assert c.parents == set()
    assert c.children == set()
