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

import platform
from collections.abc import Callable
from itertools import product

import pytest

import mithril as ml
from mithril import JaxBackend, MlxBackend, NumpyBackend, TorchBackend

from .test_utils import get_array_device, get_array_precision

# Create instances of each backend
backends = [ml.NumpyBackend, ml.JaxBackend, ml.TorchBackend, ml.MlxBackend]


testing_fns: dict[type[ml.Backend], Callable] = {}
array_fns: dict[type[ml.Backend], Callable] = {}
installed_backends: list[
    type[TorchBackend] | type[JaxBackend] | type[MlxBackend] | type[NumpyBackend]
] = []

try:
    import torch

    testing_fns[TorchBackend] = torch.allclose
    installed_backends.append(TorchBackend)
except ImportError:
    pass

try:
    import jax

    testing_fns[JaxBackend] = jax.numpy.allclose
    installed_backends.append(JaxBackend)
except ImportError:
    pass

try:
    import numpy

    testing_fns[NumpyBackend] = numpy.allclose
    installed_backends.append(NumpyBackend)
except ImportError:
    pass

try:
    import mlx.core as mx

    if platform.system() != "Darwin":
        raise ImportError
    testing_fns[MlxBackend] = mx.allclose
    installed_backends.append(MlxBackend)
except ImportError:
    pass


def assert_backend_results_equal(
    backend: ml.Backend,
    fn: Callable,
    fn_args: list,
    fn_kwargs: dict,
    ref_output,
    ref_output_device,
    ref_output_precision,
    rtol,
    atol,
):
    ref_output_device = ref_output_device.split(":")[0]
    testing_fn = testing_fns[backend.__class__]

    output = fn(*fn_args, **fn_kwargs)
    output_device = get_array_device(output, backend.type)
    output_precision = get_array_precision(output, backend.type)

    assert tuple(output.shape) == tuple(ref_output.shape)
    assert output_device == ref_output_device
    assert output_precision == ref_output_precision
    assert testing_fn(output, ref_output, rtol=rtol, atol=atol)


unsupported_device_precisions = [
    (ml.TorchBackend, "mps:0", 64),
    (ml.MlxBackend, "cpu", 16),
    (ml.MlxBackend, "cpu", 32),
    (ml.TorchBackend, "cpu:0", 16),
]

# find all backends with their device and precision
backends_with_device_precision = list(
    backend_device_precision
    for backends in installed_backends
    for backend_device_precision in product(
        [backends], backends.get_available_devices(), backends.supported_precisions
    )
    if backend_device_precision
    not in unsupported_device_precisions  # filter out unsupported combinations
)

...
names = [
    backend.__name__ + "-" + device + "-" + str(precision)
    for backend, device, precision in backends_with_device_precision
]

tolerances = {16: 1e-2, 32: 1e-5, 64: 1e-6}


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestArray:
    def test_array(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.array
        fn_args = [[1, 2, 3]]
        fn_kwargs: dict = {}

        ref_output = backend.array([1, 2, 3])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_array_edge_case(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.array
        fn_args = [1]
        fn_kwargs: dict = {}

        ref_output = backend.array(1)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestZeros:
    def test_zeros(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.zeros
        fn_args = [(2, 3)]
        fn_kwargs: dict = {}

        ref_output = backend.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_zeros_int(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.zeros
        fn_args = [(2, 3)]
        dtype = getattr(ml, f"int{precision}")
        fn_kwargs: dict = {"dtype": dtype}

        ref_output = backend.array([[0, 0, 0], [0, 0, 0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_zeros_edge(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.zeros
        fn_args = [()]
        fn_kwargs: dict = {}
        ref_output = backend.array(0.0)

        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestOnes:
    def test_ones(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.ones
        fn_args = [(2, 3)]
        fn_kwargs: dict = {}
        ref_output = backend.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_ones_int(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.ones
        fn_args = [(2, 3)]
        dtype = getattr(ml, f"int{precision}")
        fn_kwargs: dict = {"dtype": dtype}

        ref_output = backend.array([[1, 1, 1], [1, 1, 1]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_zeros_edge(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.ones
        fn_args = [()]
        fn_kwargs: dict = {}

        ref_output = backend.array(1.0)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestArange:
    def test_arange(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.arange
        fn_args: list = [-3, 5, 2]
        fn_kwargs: dict = {}

        ref_output = backend.array([-3, -1, 1, 3])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_arange_float(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.arange
        fn_args = [-3, 5, 2]
        dtype = getattr(ml, f"float{precision}")
        fn_kwargs: dict = {"dtype": dtype}

        ref_output = backend.array([-3.0, -1.0, 1.0, 3.0])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_arange_negative(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.arange
        fn_args = [3, 1, -1]
        fn_kwargs: dict = {}

        ref_output = backend.array([3, 2])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestFlatten:
    def test_flatten(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.flatten
        fn_args: list = [backend.array([[1, 2], [3, 4]])]
        fn_kwargs: dict = {}

        ref_output = backend.array([1, 2, 3, 4])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_flatten_float(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.flatten
        fn_args: list = [backend.array([[1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}

        ref_output = backend.array([1.0, 2.0, 3.0, 4.0])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_flatten_edge(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.flatten
        fn_args: list = [backend.array(1)]
        fn_kwargs: dict = {}
        ref_output = backend.array([1])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestTranspose:
    def test_transpose(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.transpose
        fn_args: list = [backend.array([[1, 2], [3, 4]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[1, 3], [2, 4]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_transpose_float(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.transpose
        fn_args: list = [backend.array([[1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[1.0, 3.0], [2.0, 4.0]])

        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_transpose_with_axes(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)

        fn = backend.transpose
        fn_args: list = [backend.array([[1, 2, 3, 4]]), [1, 0]]
        fn_kwargs: dict = {}

        ref_output = backend.array([[1], [2], [3], [4]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestRelu:
    def test_relu_int(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.relu
        fn_args: list = [backend.array([[-1, 2], [3, 4]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[0, 2], [3, 4]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_relu_edge(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.relu
        fn_args: list = [backend.array([[0.0, 1e10], [-1e10, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[0.0, 1e10], [0.0, 4.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_relu_float(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.relu
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[0.0, 2.0], [3.0, 4.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestSigmoid:
    def test_sigmoid_float(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.sigmoid
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [
                [0.2689414322376251, 0.8807970285415649],
                [0.9525741338729858, 0.9820137619972229],
            ]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestSign:
    def test_sign_float(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.sign
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[-1.0, 1.0], [1.0, 1.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_sign_int(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.sign
        fn_args: list = [backend.array([[-1, 2], [3, 4]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[-1, 1], [1, 1]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestAbs:
    def test_abs_float(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.abs
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[1.0, 2.0], [3.0, 4.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_abs_int(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.abs
        fn_args: list = [backend.array([[-1, 2], [3, 4]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[1, 2], [3, 4]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_abs_edge(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.abs
        fn_args: list = [backend.array(0.0)]
        fn_kwargs: dict = {}
        ref_output = backend.array(0.0)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestOnesLike:
    def test_ones_like(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.ones_like
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[1.0, 1.0], [1.0, 1.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_ones_edge(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.ones_like
        fn_args: list = [backend.array(0.0)]
        fn_kwargs: dict = {}
        ref_output = backend.array(1.0)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestZerosLike:
    def test_zeros_like(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.zeros_like
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[0.0, 0.0], [0.0, 0.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_zeros_edge(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.zeros_like
        fn_args: list = [backend.array(0.0)]
        fn_kwargs: dict = {}
        ref_output = backend.array(0.0)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestSin:
    def test_sin(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.sin
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [
                [-0.8414709848079, 0.9092974268256817],
                [0.1411200080598672, -0.7568024953079282],
            ]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestCos:
    def test_cos(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.cos
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [
                [0.5403023058681398, -0.4161468365471424],
                [-0.9899924966004454, -0.6536436208636119],
            ]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestTanh:
    def test_tanh(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.tanh
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [
                [-0.7615941559557649, 0.9640275800758169],
                [0.9950547536867305, 0.999329299739067],
            ]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestLeakyRelu:
    def test_leaky_relu(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.leaky_relu
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]]), 0.1]
        fn_kwargs: dict = {}
        ref_output = backend.array([[-0.1, 2.0], [3.0, 4.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestSoftplus:
    def test_softplus(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.softplus
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [
                [0.3132616877555847, 2.1269280910491943],
                [3.0485873222351074, 4.0181498527526855],
            ]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestSoftmax:
    def test_softmax(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.softmax
        fn_args: list = [backend.array([[-1.0, 2.0], [3.0, 4.0]]), 0]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [
                [0.01798621006309986, 0.11920291930437088],
                [0.9820137619972229, 0.8807970285415649],
            ]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestLog:
    def test_log(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.log
        fn_args: list = [backend.array([[2.0, 1e-5], [1.0, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [[0.6931471824645996, -11.512925148010254], [0.0, 1.3862943649291992]]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestIsNaN:
    def test_is_nan(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.isnan
        fn_args: list = [backend.array([[2.0, backend.nan], [backend.nan, 4.0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[False, True], [True, False]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            8,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestSqueeze:
    def test_squeeze(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.squeeze
        fn_args: list = [backend.array([[[[[2.0, 1.0], [3.0, 4.0]]]]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[2.0, 1.0], [3.0, 4.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_squeeze_edge(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.squeeze
        fn_args: list = [backend.array([[[[[[[[2.0]]]]]]]])]
        fn_kwargs: dict = {}
        ref_output = backend.array(2.0)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestReshape:
    def test_reshape(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.reshape
        fn_args: list = [backend.array([[[[[2.0, 1.0], [3.0, 4.0]]]]]), (4, 1)]
        fn_kwargs: dict = {}
        ref_output = backend.array([[2.0], [1.0], [3.0], [4.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_reshape_edge(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.reshape
        fn_args: list = [backend.array([[[[[[[[2.0]]]]]]]]), (1, 1)]
        fn_kwargs: dict = {}
        ref_output = backend.array([[2.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


bdp_without_gpu = backends_with_device_precision.copy()
names_without_gpu = names.copy()
for idx, item in enumerate(backends_with_device_precision):
    if item[0] == ml.TorchBackend and "cpu" not in item[1]:
        bdp_without_gpu.remove(item)
        names_without_gpu.pop(idx)


@pytest.mark.parametrize(
    "backendcls, device, precision", bdp_without_gpu, ids=names_without_gpu
)
class TestSort:
    def test_sort(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.sort
        fn_args: list = [backend.array([[[[[1.0, 2.0], [3.0, 4.0]]]]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[[[[1.0, 2.0], [3.0, 4.0]]]]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestExpandDims:
    def test_expand_dims(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.expand_dims
        fn_args: list = [backend.array([2.0, 3.0]), 1]
        fn_kwargs: dict = {}
        ref_output = backend.array([[2.0], [3.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestStack:
    def test_stack_dim0(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.stack
        fn_args: list = [[backend.array([2.0, 3.0]), backend.array([4.0, 5.0])], 0]
        fn_kwargs: dict = {}
        ref_output = backend.array([[2.0, 3.0], [4.0, 5.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_stack_dim1(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.stack
        fn_args: list = [[backend.array([2.0, 3.0]), backend.array([4.0, 5.0])], 1]
        fn_kwargs: dict = {}
        ref_output = backend.array([[2.0, 4.0], [3.0, 5.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestCat:
    def test_dim0(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.cat
        fn_args: list = [[backend.array([[2.0, 3.0]]), backend.array([[4.0, 5.0]])], 0]
        fn_kwargs: dict = {}
        ref_output = backend.array([[2.0, 3.0], [4.0, 5.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_dim1(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.cat
        fn_args: list = [[backend.array([[2.0, 3.0]]), backend.array([[4.0, 5.0]])], 1]
        fn_kwargs: dict = {}
        ref_output = backend.array([[2.0, 3.0, 4.0, 5.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestPad:
    def test_tuple_of_tuple(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.pad
        fn_args: list = [backend.array([[2.0, 3.0], [4.0, 5.0]]), ((0, 0), (1, 1))]
        fn_kwargs: dict = {}
        ref_output = backend.array([[0.0, 2.0, 3.0, 0.0], [0.0, 4.0, 5.0, 0.0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_tuple_of_tuple_3_dim(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.pad
        fn_args: list = [
            backend.array([[[2.0, 3.0], [4.0, 5.0]], [[2.0, 3.0], [4.0, 5.0]]]),
            ((0, 0), (1, 1), (2, 2)),
        ]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_tuple_int(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.pad
        fn_args: list = [
            backend.array([[[2.0, 3.0], [4.0, 5.0]], [[2.0, 3.0], [4.0, 5.0]]]),
            (1, 2),
        ]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 3.0, 0.0, 0.0],
                    [0.0, 4.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 3.0, 0.0, 0.0],
                    [0.0, 4.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_int(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.pad
        fn_args: list = [backend.array([[2.0, 3.0], [4.0, 5.0]]), 1]
        fn_kwargs: dict = {}
        ref_output = backend.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 0.0],
                [0.0, 4.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestAll:
    def test_all_false(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.all
        fn_args: list = [backend.array([True, False, False, True])]
        fn_kwargs: dict = {}
        ref_output = backend.array(False)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            8,
            tolerances[precision],
            tolerances[precision],
        )

    def test_all_true(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.all
        fn_args: list = [backend.array([True, True, 1.0, True])]
        fn_kwargs: dict = {}
        ref_output = backend.array(True)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            8,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestAny:
    def test_any_false(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.any
        fn_args: list = [backend.array([False, 0.0, False, 0.0])]
        fn_kwargs: dict = {}
        ref_output = backend.array(False)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            8,
            tolerances[precision],
            tolerances[precision],
        )

    def test_any_true(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.any
        fn_args: list = [backend.array([False, False, 0.0, True])]
        fn_kwargs: dict = {}
        ref_output = backend.array(True)
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            8,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestAtLeast1D:
    def test_zero_dim(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.atleast_1d
        fn_args: list = [backend.array(0)]
        fn_kwargs: dict = {}
        ref_output = backend.array([0])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_two_dim(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.atleast_1d
        fn_args: list = [backend.array([[0]])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestAtLeast2D:
    def test_zero_dim(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.atleast_2d
        fn_args: list = [backend.array(0)]
        fn_kwargs: dict = {}
        ref_output = backend.array([[0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )

    def test_one_dim(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.atleast_2d
        fn_args: list = [backend.array([0])]
        fn_kwargs: dict = {}
        ref_output = backend.array([[0]])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestWhere:
    def test_where(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.where
        input = backend.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        fn_args: list = [input < 5, input, 10 * input]
        fn_kwargs: dict = {}
        ref_output = backend.array([0, 1, 2, 3, 4, 50, 60, 70, 80, 90])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestTopK:
    def test_topk(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.topk
        input = backend.array([0, 1, 2, 3, 4, 5])
        fn_args: list = [input, 3]
        fn_kwargs: dict = {}
        ref_output = backend.array([5, 4, 3])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )


@pytest.mark.parametrize(
    "backendcls, device, precision", backends_with_device_precision, ids=names
)
class TestLinspace:
    def test_linpsace(self, backendcls, device, precision):
        backend = backendcls(device=device, precision=precision)
        fn = backend.linspace
        fn_args: list = [0, 20, 3]
        fn_kwargs: dict = {}
        ref_output = backend.array([0.0, 10.0, 20.0])
        assert_backend_results_equal(
            backend,
            fn,
            fn_args,
            fn_kwargs,
            ref_output,
            device,
            precision,
            tolerances[precision],
            tolerances[precision],
        )
