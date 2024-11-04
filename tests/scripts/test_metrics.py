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

import numpy as np

from mithril import TorchBackend, compile
from mithril.framework.common import TBD
from mithril.models import (
    AUC,
    F1,
    Accuracy,
    IOKey,
    Metric,
    Model,
    Precision,
    Recall,
)

TOLERANCE = 1e-6


# In this tests, for the expected_results, Scikit-learn metrics are used.


def test_metrics_1():
    model = Model()
    model += AUC(3, False)("pred", "label", IOKey("AUC_OvR"))
    model += Accuracy(is_pred_one_hot=True, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = np.array(
        [
            [0.4, 0.4, 0.2],
            [0.1, 0.6, 0.3],
            [0.2, 0.4, 0.4],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.3, 0.6],
        ]
    )
    label = np.array([1, 1, 2, 0, 1, 2])

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 0.6666666666666666,
        "Macro_Precision": 0.7222222222222222,
        "Weighted_Precision": 0.75,
        "Micro_Recall": 0.6666666666666666,
        "Macro_Recall": 0.7222222222222222,
        "Weighted_Recall": 0.6666666666666666,
        "Micro_F1": 0.6666666666666666,
        "Macro_F1": 0.6666666666666666,
        "Weighted_F1": 0.6666666666666666,
        "Accuracy": 0.6666666666666666,
        "AUC_OvR": 0.98148148148148,
    }

    for key in expected_results:
        if key in result:
            np.testing.assert_allclose(
                result[key], expected_results[key], atol=TOLERANCE
            )


def test_metrics_2():
    model = Model()
    model += Accuracy(is_pred_one_hot=True, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.6, 0.2, 0.2],
            [0.95, 0.03, 0.02],
        ]
    )
    label = np.array([0, 0, 0, 0, 0])

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 1.0,
        "Macro_Precision": 1.0,
        "Weighted_Precision": 1.0,
        "Micro_Recall": 1.0,
        "Macro_Recall": 1.0,
        "Weighted_Recall": 1.0,
        "Micro_F1": 1.0,
        "Macro_F1": 1.0,
        "Weighted_F1": 1.0,
        "Accuracy": 1.0,
    }

    for key in result:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)


def test_metrics_3():
    model = Model()
    model += Accuracy(is_pred_one_hot=True, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = np.array(
        [
            [0.33, 0.33, 0.33],
            [0.33, 0.33, 0.33],
            [0.33, 0.33, 0.33],
            [0.33, 0.33, 0.33],
            [0.33, 0.33, 0.33],
        ]
    )
    label = np.array([0, 1, 2, 0, 1])

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 0.4,
        "Macro_Precision": 0.13333333333333333,
        "Weighted_Precision": 0.16,
        "Micro_Recall": 0.4,
        "Macro_Recall": 0.3333333333333333,
        "Weighted_Recall": 0.4,
        "Micro_F1": 0.4,
        "Macro_F1": 0.19047619047619047,
        "Weighted_F1": 0.22857142857142856,
        "Accuracy": 0.4,
        "AUC_OvR": 0.98148148148148,
    }

    for key in result:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)


def test_metrics_4():
    model = Model()
    model += Accuracy(is_pred_one_hot=True, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.85, 0.05],
            [0.1, 0.05, 0.85],
            [0.95, 0.03, 0.02],
            [0.05, 0.9, 0.05],
        ]
    )
    label = np.array([0, 1, 2, 0, 1])

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 1.0,
        "Macro_Precision": 1.0,
        "Weighted_Precision": 1.0,
        "Micro_Recall": 1.0,
        "Macro_Recall": 1.0,
        "Weighted_Recall": 1.0,
        "Micro_F1": 1.0,
        "Macro_F1": 1.0,
        "Weighted_F1": 1.0,
        "Accuracy": 1.0,
        "AUC_OvR": 1.0,
    }

    for key in result:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)


def test_metrics_5():
    model = Model()
    model += Accuracy(is_pred_one_hot=True, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=True)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = np.array(
        [
            [0.05, 0.9, 0.05],
            [0.1, 0.05, 0.85],
            [0.85, 0.1, 0.05],
            [0.02, 0.95, 0.03],
            [0.05, 0.8, 0.15],
        ]
    )
    label = np.array([0, 1, 2, 0, 1])

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 0.2,
        "Macro_Precision": 0.1111111111111111,
        "Weighted_Precision": 0.13333333333333333,
        "Micro_Recall": 0.2,
        "Macro_Recall": 0.16666666666666666,
        "Weighted_Recall": 0.2,
        "Micro_F1": 0.2,
        "Macro_F1": 0.13333333333333333,
        "Weighted_F1": 0.16,
        "Accuracy": 0.2,
        "AUC_OvR": 0.20833333333333,
    }

    for key in result:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)


def test_metrics_6():
    model = Model()
    model += Metric()(pred="pred", label="label", output=IOKey("output"))

    pred = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    label = [[0, 1, 0], [1, 0, 0], [0, 1, 0]]

    backend = TorchBackend()

    compiled_model = compile(
        model, backend, static_keys={"pred": TBD, "label": TBD}, inference=True
    )

    expected_result = np.array([0, 0, 1])
    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )["output"]

    np.testing.assert_allclose(result, expected_result, atol=TOLERANCE)


def test_metrics_7():
    model = Model()
    model += Metric()(pred="pred", label="label", output=IOKey("output"))
    pred = [[0, 1, 0], [0, 1, 0], [1, 0, 0]]
    label = [[0, 1, 0], [1, 0, 0], [0, 1, 0]]

    backend = TorchBackend()

    compiled_model = compile(
        model, backend, static_keys={"pred": TBD, "label": TBD}, inference=True
    )

    expected_result = np.array([0, 1, -1])
    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )["output"]

    np.testing.assert_allclose(result, expected_result, atol=TOLERANCE)


def test_metrics_8():
    model = Model()
    model += Metric(is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("output")
    )
    pred = [[0, 1, 0], [0, 1, 0], [1, 0, 0]]
    label = [1, 0, 1]

    backend = TorchBackend()

    compiled_model = compile(
        model, backend, static_keys={"pred": TBD, "label": TBD}, inference=True
    )

    expected_result = np.array([0, 1, -1])
    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )["output"]

    np.testing.assert_allclose(result, expected_result, atol=TOLERANCE)


def test_metrics_9():
    model = Model()
    model += Metric(is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("output")
    )
    pred = [1, 1, 0]
    label = [1, 0, 1]

    backend = TorchBackend()

    compiled_model = compile(
        model, backend, static_keys={"pred": TBD, "label": TBD}, inference=True
    )

    expected_result = np.array([0, 1, -1])
    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )["output"]

    np.testing.assert_allclose(result, expected_result, atol=TOLERANCE)


def test_metrics_10():
    model = Model()
    model += Metric(
        threshold=0.3, is_binary=True, is_label_one_hot=False, is_pred_one_hot=False
    )(pred="pred", label="label", output=IOKey("output"))
    pred = [0.5, 0.2, 0.1]
    label = [1, 0, 1]

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    expected_result = np.array([0, 0, -1])
    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )["output"]

    np.testing.assert_allclose(result, expected_result, atol=TOLERANCE)


def test_metrics_11():
    model = Model()
    model += Metric(
        threshold=0.3, is_binary=True, is_label_one_hot=True, is_pred_one_hot=False
    )(pred="pred", label="label", output=IOKey("output"))
    pred = [0.5, 0.2, 0.1]
    label = [[0, 1], [1, 0], [0, 1]]

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    expected_result = np.array([0, 0, -1])
    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )["output"]

    np.testing.assert_allclose(result, expected_result, atol=TOLERANCE)


def test_metrics_12():
    model = Model()
    model += Accuracy(is_pred_one_hot=False, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = [0, 0, 0, 2, 0, 2, 0, 0]
    label = [0, 1, 0, 2, 0, 1, 0, 2]

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 0.625,
        "Macro_Precision": 0.38888888888888884,
        "Weighted_Precision": 0.4583333333333333,
        "Micro_Recall": 0.625,
        "Macro_Recall": 0.5,
        "Weighted_Recall": 0.625,
        "Micro_F1": 0.625,
        "Macro_F1": 0.43333333333333335,
        "Weighted_F1": 0.525,
        "Accuracy": 0.625,
    }

    for key in expected_results:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)


def test_metrics_13():
    model = Model()
    model += Accuracy(is_pred_one_hot=False, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = [1, 2, 0, 2, 1, 2, 2, 1]
    label = [0, 1, 0, 2, 0, 1, 0, 2]

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 0.25,
        "Macro_Precision": 0.4166666666666667,
        "Weighted_Precision": 0.5625,
        "Micro_Recall": 0.25,
        "Macro_Recall": 0.25,
        "Weighted_Recall": 0.25,
        "Accuracy": 0.25,
        "Micro_F1": 0.25,
        "Macro_F1": 0.24444444444444446,
        "Weighted_F1": 0.2833333333333333,
    }

    for key in expected_results:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)


def test_metrics_14():
    model = Model()
    model += Accuracy(is_pred_one_hot=False, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = [0, 0, 0, 0, 0, 0, 0, 0]
    label = [0, 1, 0, 2, 0, 1, 0, 2]

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 0.5,
        "Macro_Precision": 0.16666666666666666,
        "Weighted_Precision": 0.25,
        "Micro_Recall": 0.5,
        "Macro_Recall": 0.3333333333333333,
        "Weighted_Recall": 0.5,
        "Micro_F1": 0.5,
        "Macro_F1": 0.2222222222222222,
        "Weighted_F1": 0.3333333333333333,
        "Accuracy": 0.5,
    }

    for key in expected_results:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)


def test_metrics_15():
    model = Model()
    model += Accuracy(is_pred_one_hot=False, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = [0, 1, 0, 2, 0, 1, 0, 2]
    label = [0, 1, 0, 2, 0, 1, 0, 2]

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 1.0,
        "Macro_Precision": 1.0,
        "Weighted_Precision": 1.0,
        "Micro_Recall": 1.0,
        "Macro_Recall": 1.0,
        "Weighted_Recall": 1.0,
        "Micro_F1": 1.0,
        "Macro_F1": 1.0,
        "Weighted_F1": 1.0,
        "Accuracy": 1.0,
    }

    for key in expected_results:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)


def test_metrics_16():
    model = Model()
    model += Accuracy(is_pred_one_hot=False, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = [0, 0, 0, 2, 0, 2, 0, 2]
    label = [0, 1, 0, 2, 0, 1, 0, 2]

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 0.75,
        "Macro_Precision": 0.48888888888888893,
        "Weighted_Precision": 0.5666666666666667,
        "Micro_Recall": 0.75,
        "Macro_Recall": 0.6666666666666666,
        "Weighted_Recall": 0.75,
        "Micro_F1": 0.75,
        "Macro_F1": 0.562962962962963,
        "Weighted_F1": 0.6444444444444444,
        "Accuracy": 0.75,
    }

    for key in expected_results:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)


def test_metrics_17():
    model = Model()
    model += Accuracy(is_pred_one_hot=False, is_label_one_hot=False)(
        pred="pred", label="label", output=IOKey("Accuracy")
    )

    model += F1("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_F1")
    )
    model += F1("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_F1")
    )
    model += F1("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_F1")
    )

    model += Precision("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Precision")
    )
    model += Precision("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Precision")
    )
    model += Precision("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Precision")
    )

    model += Recall("micro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Micro_Recall")
    )
    model += Recall("macro", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Macro_Recall")
    )
    model += Recall("weighted", 3, is_label_one_hot=False, is_pred_one_hot=False)(
        pred="pred", label="label", output=IOKey("Weighted_Recall")
    )

    pred = [2, 2, 2, 1, 2, 0, 2, 0]
    label = [0, 1, 0, 2, 0, 1, 0, 2]

    backend = TorchBackend()

    compiled_model = compile(
        model,
        backend,
        static_keys={"pred": TBD, "label": TBD},
        inference=True,
        jit=False,
    )

    result = compiled_model.evaluate(
        params={},
        data={"pred": backend.array(pred), "label": backend.array(label)},
    )

    expected_results = {
        "Micro_Precision": 0.0,
        "Macro_Precision": 0.0,
        "Weighted_Precision": 0.0,
        "Micro_Recall": 0.0,
        "Macro_Recall": 0.0,
        "Weighted_Recall": 0.0,
        "Micro_F1": 0.0,
        "Macro_F1": 0.0,
        "Weighted_F1": 0.0,
        "Accuracy": 0.0,
    }

    for key in expected_results:
        np.testing.assert_allclose(result[key], expected_results[key], atol=TOLERANCE)
