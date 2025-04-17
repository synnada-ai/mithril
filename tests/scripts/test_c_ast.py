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

from mithril.framework.codegen.c_style_codegen import c_ast


def test_constant_int():
    constant_variable = c_ast.ConstantVariable("int", "my_int", c_ast.Constant(1))
    assert (
        c_ast.CStyleCodeGenerator().visit(constant_variable) == "const int my_int = 1;"
    )


def test_constant_float():
    constant_variable = c_ast.ConstantVariable("float", "my_float", c_ast.Constant(1.0))
    assert (
        c_ast.CStyleCodeGenerator().visit(constant_variable)
        == "const float my_float = 1.0;"
    )


def test_constant_int_array_tuple():
    constant_variable = c_ast.ConstantVariable(
        c_ast.Pointer("int"),
        "my_int",
        c_ast.InitializerList(
            (c_ast.Constant(1), c_ast.Constant(2), c_ast.Constant(3))
        ),
    )
    assert (
        c_ast.CStyleCodeGenerator().visit(constant_variable)
        == "const int * my_int = {1, 2, 3};"
    )


def test_constant_float_array_tuple():
    constant_variable = c_ast.ConstantVariable(
        c_ast.Pointer("float"),
        "my_float",
        c_ast.InitializerList(
            (c_ast.Constant(1.0), c_ast.Constant(2.0), c_ast.Constant(3.0))
        ),
    )
    assert (
        c_ast.CStyleCodeGenerator().visit(constant_variable)
        == "const float * my_float = {1.0, 2.0, 3.0};"
    )
