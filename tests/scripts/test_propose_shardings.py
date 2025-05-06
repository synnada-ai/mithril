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

from mithril import TorchBackend, compile
from mithril.models import Add

from .test_parallel import create_parallel_backend


def test_single_dim_mesh_divisible():
    """Test case with (4,) mesh and dimensions divisible by 4"""
    model = Add()
    model.set_shapes(left=[1, 512, 1, 4], right=[1, 1, 512, 4])
    backend = create_parallel_backend(device_mesh=(4,))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": (1, 4, 1, 1), "right": (1, 1, 4, 1)}


def test_single_dim_mesh_indivisible():
    """Test case with (4,) mesh and dimensions not divisible by 4"""
    model = Add()
    model.set_shapes(left=[1, 77, 3, 9], right=[1, 77, 3, 9])
    backend = create_parallel_backend(device_mesh=(4,))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": None, "right": None}


def test_two_dim_mesh_2_2():
    """Test case with (2, 2) mesh and dimensions divisible by both 2s"""
    model = Add()
    model.set_shapes(left=[1, 512, 4, 4], right=[1, 512, 4, 4])
    backend = create_parallel_backend(device_mesh=(2, 2))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": (1, 2, 2, 1), "right": (1, 2, 2, 1)}


def test_two_dim_mesh_3_2():
    """Test case with (3, 2) mesh and dimensions divisible by 3 and 2"""
    model = Add()
    model.set_shapes(left=[1, 9, 4, 4], right=[1, 9, 4, 4])
    backend = create_parallel_backend(device_mesh=(3, 2))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": (1, 3, 2, 1), "right": (1, 3, 2, 1)}


def test_two_dim_mesh_2_3():
    """Test case with (2, 3) mesh and dimensions divisible by 2 and 3"""
    model = Add()
    model.set_shapes(left=[1, 9, 4, 4], right=[1, 9, 4, 4])
    backend = create_parallel_backend(device_mesh=(2, 3))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    # The result will be same as (3, 2) case since we're looking for matching dimensions
    assert shardings == {"left": (1, 3, 2, 1), "right": (1, 3, 2, 1)}


def test_mixed_shape_compatibility():
    """Test case with mixed shapes - one divisible, one not"""
    model = Add()
    model.set_shapes(left=[1, 512, 4, 4], right=[1, 1, 1])
    backend = create_parallel_backend(device_mesh=(4,))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": (1, 4, 1, 1), "right": None}


def test_with_known_shardings():
    """Test case with pre-defined known shardings"""
    model = Add()
    model.set_shapes(left=[1, 512, 4, 4], right=[1, 512, 4, 4])
    backend = create_parallel_backend(device_mesh=(4,))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)

    # Define custom sharding for "left"
    given_shards = {"left": (1, 1, 1, 4)}

    shardings = pm.propose_shardings(given_shards=given_shards)  # type: ignore

    # Known sharding should be preserved and automized
    # sharding should be calculated for "right"
    assert shardings == {"left": (1, 1, 1, 4), "right": (1, 4, 1, 1)}


def test_different_input_shapes():
    """Test with different input shapes for left and right"""
    model = Add()
    model.set_shapes(left=[1, 8, 16, 4])
    backend = create_parallel_backend(device_mesh=(4, 2))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": (1, 4, 2, 1), "right": None}


def test_one_dimensional_inputs():
    """Test with one-dimensional inputs"""
    model = Add()
    model.set_shapes(left=[16], right=[16])
    backend = create_parallel_backend(device_mesh=(4,))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": (4,), "right": (4,)}


def test_large_mesh_dimension():
    """Test with mesh dimension larger than any shape dimension"""
    model = Add()
    model.set_shapes(left=[1, 2, 3, 4], right=[1, 2, 3, 4])
    backend = create_parallel_backend(device_mesh=(8,))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": None, "right": None}


def test_large_mesh_dimension_ones_in_mesh():
    """Test with mesh dimension larger than any shape dimension"""
    model = Add()
    model.set_shapes(left=[1, 2, 3, 4], right=[1, 2, 3, 4])
    backend = create_parallel_backend(device_mesh=(1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": (1, 1, 3, 1), "right": (1, 1, 3, 1)}


def test_large_mesh_dimension_shp_with_none():
    """Test with mesh dimension larger than any shape dimension"""
    model = Add()
    model.set_shapes(left=[None, None, 3, None], right=[None])
    backend = create_parallel_backend(device_mesh=(1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": (1, 1, 3, 1), "right": None}


def test_large_mesh_dimension_shp_with_ellipsis():
    """Test with mesh dimension larger than any shape dimension"""
    model = Add()
    model.set_shapes(left=[1, None, ("V2", ...), 1, 3], right=[("V1", ...)])
    backend = create_parallel_backend(device_mesh=(1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings()
    assert shardings == {"left": None, "right": None}


def test_large_mesh_dimension_shp_with_given_shards():
    """Test with mesh dimension larger than any shape dimension"""
    model = Add()
    model.set_shapes(left=[None, None, 3, None], right=[None])
    backend = create_parallel_backend(device_mesh=(1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1))
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    shardings = pm.propose_shardings({"left": (3, 1, 1, 1)})
    assert shardings == {"left": (3, 1, 1, 1), "right": None}


def test_large_mesh_dimension_shp_with_type_error():
    """Test with mesh dimension larger than any shape dimension"""
    model = Add()
    model.set_shapes(left=[None, None, 3, None], right=[None])
    backend = TorchBackend()
    pm = compile(model, backend, jit=False, data_keys={"left", "right"}, inference=True)
    with pytest.raises(TypeError) as error_info:
        # This should raise TypeError because the mesh is not compatible with the shape
        _ = pm.propose_shardings({"left": (3, 1, 1, 1)})
    assert str(error_info.value) == "Sharding is only supported for parallel backends!"
