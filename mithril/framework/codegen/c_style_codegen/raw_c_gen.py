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

from typing import override

from . import c_ast, utils
from .c_gen import CGen


class RawCGen(CGen):
    dynamic_links = ["-lmithrilc"]

    @override
    def determine_struct_keys(self) -> utils.StructKeys:
        struct_keys = super().determine_struct_keys()
        struct_keys.eval_grad_input_keys = sorted(
            {
                key + "_grad"
                for key in self.pm.flat_graph.all_keys
                if self._has_grad(key)
            }
            | self.pm.flat_graph.all_keys
        )
        struct_keys.eval_output_keys = sorted(struct_keys.eval_input_keys)

        return struct_keys

    def create_key_ref(
        self, key: str, context: str, load: bool = True
    ) -> c_ast.Variable | c_ast.Expr:
        if key in self.struct_keys.eval_input_keys:
            return c_ast.Variable(f"inputs->{key}")

        else:
            return super().create_key_ref(key, context, load)

    @override
    def assign_primitive_output(
        self, target: str, source: c_ast.Expr, context: str
    ) -> c_ast.Assign:
        return c_ast.MakeStmt(source)  # type: ignore
