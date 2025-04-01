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

import mithril as ml

def prepare_inputs():
    c_backend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()

    left = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    right = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    left_c = c_backend.array(left)
    right_c = c_backend.array(right)
    
    left_ggml = ggml_backend.array(left)
    right_ggml = ggml_backend.array(right)
    
    return left, right, left_c, right_c, left_ggml, right_ggml    

def test_scalar_radd():
    c_backend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()
    
    left, _, left_c, _, left_ggml, _  = prepare_inputs()
    
    scalar = 10.0
    result_c = scalar + left_c
    result_ggml = scalar + left_ggml
    expected = scalar + left
    assert np.allclose(c_backend.to_numpy(result_c), expected)
    assert np.allclose(ggml_backend.to_numpy(result_ggml), expected)
    
def test_scalar_add():
    c_backend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()
    
    left, _, left_c, _, left_ggml, _  = prepare_inputs()
    
    scalar = 2.0
    result_c = left_c + scalar
    result_ggml = left_ggml + scalar
    expected = left + scalar
    assert np.allclose(c_backend.to_numpy(result_c), expected)
    assert np.allclose(ggml_backend.to_numpy(result_ggml), expected)
    
def test_pyarray_add():
    c_backend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()
    
    left, right, left_c, right_c, left_ggml, right_ggml = prepare_inputs()
    
    result_c = left_c + right_c
    result_ggml = left_ggml + right_ggml
    expected = left + right
    assert np.allclose(c_backend.to_numpy(result_c), expected)
    assert np.allclose(ggml_backend.to_numpy(result_ggml), expected)


def test_scalar_rsubtract():
    c_backend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()

    left, _, left_c, _, left_ggml, _  = prepare_inputs()
    
    scalar = 10.0
    result_c = scalar - left_c
    result_ggml = scalar - left_ggml
    expected = scalar - left
    assert np.allclose(c_backend.to_numpy(result_c), expected)
    assert np.allclose(ggml_backend.to_numpy(result_ggml), expected)
    
def test_scalar_subtract():
    c_backend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()
    
    left, _, left_c, _, left_ggml, _  = prepare_inputs()
    
    scalar = 2.0
    result_c = left_c - scalar
    result_ggml = left_ggml - scalar
    expected = left - scalar
    assert np.allclose(c_backend.to_numpy(result_c), expected)
    assert np.allclose(ggml_backend.to_numpy(result_ggml), expected)
    
def test_pyarray_subtract():
    c_backend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()
    
    left, right, left_c, right_c, left_ggml, right_ggml = prepare_inputs()
    
    result_c = left_c - right_c
    result_ggml = left_ggml - right_ggml
    expected = left - right
    assert np.allclose(c_backend.to_numpy(result_c), expected)
    assert np.allclose(ggml_backend.to_numpy(result_ggml), expected)
    
    
def test_scalar_multiply():
    c_backend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()
    
    left, _, left_c, _, left_ggml, _  = prepare_inputs()
    
    scalar = 2.0
    result_c = left_c * scalar
    result_ggml = left_ggml * scalar
    expected = left * scalar
    assert np.allclose(c_backend.to_numpy(result_c), expected)
    assert np.allclose(ggml_backend.to_numpy(result_ggml), expected)
    
def test_pyarray_multiply():
    c_backend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()
    
    left, right, left_c, right_c, left_ggml, right_ggml = prepare_inputs()
    
    result_c = left_c * right_c
    result_ggml = left_ggml * right_ggml
    expected = left * right
    assert np.allclose(c_backend.to_numpy(result_c), expected)
    assert np.allclose(ggml_backend.to_numpy(result_ggml), expected)