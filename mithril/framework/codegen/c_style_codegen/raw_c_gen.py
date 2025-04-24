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

from collections.abc import Callable, Sequence
from typing import override

from ...logical.operator import Operator
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

    def pre_process_op(
        self,
        op: Operator,
        inputs: Sequence[str | int | float | bool | None],
        context: str,
        pre_processor: Callable[
            [Operator, Sequence[str | int | float | bool | None], str],
            tuple[
                Operator, Sequence[str | int | float | bool | None], list[c_ast.Stmt]
            ],
        ]
        | None = None,
    ) -> tuple[Operator, Sequence[str | int | float | bool | None], list[c_ast.Stmt]]:
        pre_op_stmts: list[c_ast.Stmt] = []

        op, inputs, lines = self.handle_static_scalar(op, inputs)
        pre_op_stmts.extend(lines)

        op, inputs, _pre_op_stmts = super().pre_process_op(
            op, inputs, context, pre_processor
        )
        pre_op_stmts.extend(_pre_op_stmts)
        return op, inputs, pre_op_stmts

    def handle_static_scalar(
        self, op: Operator, inputs: Sequence[str | int | float | bool | None]
    ) -> tuple[Operator, Sequence[str | int | float | bool | None], list[c_ast.Stmt]]:
        # Handles scalar static inputs
        #
        # 1) Tuple:
        #   (1,2,3) -> `c_tuple my_tuple= {.size = 3, .data = (int[]) {1, 2, 3} };`
        #   Create a new key for the tuple and add it to the processed inputs
        #   Add the length of the tuple to the processed inputs
        # 2) Bool:
        #   True -> `op(..., true, ...);`
        #   False -> `op(..., false, ...);`
        #   Add the bool to the processed inputs

        pre_op_stmts: list[c_ast.Stmt] = []
        processed_inputs: list[str | int | float | bool | None] = []

        for key in inputs:
            if not (isinstance(key, str) and self.is_static_scalar(key)):
                # If the input is not a static scalar, we need to add it to
                # the processed inputs
                processed_inputs.append(key)
                continue

            value = self.pm.flat_graph.cached_data[key]
            match value:
                case tuple():
                    # (1,2,3) -> `c_tuple my_tuple= {.size = 3, .data = (int[])
                    # {1, 2, 3} };`
                    tuple_name = self.pm.flat_graph.get_next_unique_key(
                        op.formula_key + "_tuple"
                    )
                    tuple_stmt = self.tuple_generator(tuple_name, value)
                    pre_op_stmts.append(tuple_stmt)
                    processed_inputs.append(tuple_name)

                case bool() | None:
                    # True -> `op(..., true, ...);`
                    processed_inputs.append(value)

                case _:
                    processed_inputs.append(key)
                # TODO: Open this error back up
                # case _:
                #     raise ValueError(f"Unsupported static scalar type: {type(value)}")

        return op, processed_inputs, pre_op_stmts

    # TODO: Support tuple of float/bool
    def tuple_generator(self, tuple_name: str, scalar: tuple[int, ...]) -> c_ast.Stmt:
        # In Raw C we are handling tuples as creating a constant array
        # (1,2,3)-> `c_tuple my_tuple= {.size = 3, .data = (int[]) {1, 2, 3} };`

        compound_literal = c_ast.CompoundLiteral(
            type="int",
            initializer=c_ast.InitializerList(
                values=tuple(c_ast.Constant(value) for value in scalar)
            ),
        )

        # Get address of the compound literal
        initializer_dict = c_ast.InitializerDict(
            keys=("size", "data"),
            values=(c_ast.Constant(len(scalar)), compound_literal),
        )

        cast_stmt = c_ast.Cast(
            target_type="c_tuple",
            value=initializer_dict,
        )
        addressof_stmt = c_ast.AddressOf(cast_stmt)

        tuple_var = c_ast.ConstantVariable(
            c_ast.Pointer("c_tuple"), tuple_name, addressof_stmt
        )
        return tuple_var
