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

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass
class AST(ABC):
    @abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError("to_str is not implemented")


@dataclass
class Expr(AST):
    pass


@dataclass
class Stmt(AST):
    pass


@dataclass
class MakeStmt(Stmt):
    expr: Expr

    def to_str(self) -> str:
        return self.expr.to_str() + ";"


@dataclass
class Call(Expr):
    name: str
    args: Sequence[str | Expr]

    def to_str(self) -> str:
        args_str = ", ".join(
            [arg.to_str() if isinstance(arg, Expr) else arg for arg in self.args]
        )
        return f"{self.name}({args_str})"


@dataclass
class Constant(Expr):
    value: int | float | str

    def to_str(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return self.to_str()


@dataclass
class Variable(Expr):
    name: str

    def to_str(self) -> str:
        return self.name


@dataclass
class Assign(Stmt):
    target: Variable
    source: Expr | Stmt

    def to_str(self) -> str:
        result_str = f"{self.target.to_str()} = {self.source.to_str()}"
        if not isinstance(self.source, Stmt):
            result_str += ";"
        return result_str


@dataclass
class Parameter:
    type: str
    name: str

    def to_str(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class FunctionDef(Stmt):
    return_type: str
    name: str
    params: list[Parameter]
    body: Sequence[Stmt | Expr]

    def to_str(self) -> str:
        params_str = (
            ("\n\t" + ",\n\t".join([param.to_str() for param in self.params]) + "\n")
            if len(self.params) > 0
            else ""
        )
        body_str = "\n    ".join([stmt.to_str() for stmt in self.body])
        return f"\n{self.return_type} {self.name}({params_str})\n{{\n    {body_str}\n}}"


@dataclass
class Return(Stmt):
    value: Expr

    def to_str(self) -> str:
        return f"return {self.value.to_str()};"


@dataclass
class Include(AST):
    header: str
    system: bool = False  # True for system headers, False for user-defined headers

    def to_str(self) -> str:
        if self.system:
            return f"#include <{self.header}>"
        else:
            return f'#include "{self.header}"'


@dataclass
class Comment(Stmt):
    text: str
    multiline: bool = False  # True for /* */ comments, False for // comments

    def to_str(self) -> str:
        if self.multiline:
            # Format multiline comments with proper line breaks
            lines = self.text.split("\n")
            if len(lines) == 1:
                return f"/* {self.text} */"
            formatted_lines = [f" * {line}" for line in lines]
            return "/*\n" + "\n".join(formatted_lines) + "\n */"
        else:
            return f"// {self.text}"


@dataclass
class StructField:
    type: str
    name: str

    def to_str(self) -> str:
        return f"    {self.type} {self.name};"


@dataclass
class StructDef(Stmt):
    name: str
    fields: list[StructField]

    def to_str(self) -> str:
        fields_str = "\n".join(field.to_str() for field in self.fields)
        return f"\nstruct {self.name} {{\n{fields_str}\n}};\n"


@dataclass
class FILE(AST):
    includes: list[Include]
    globals: list[Stmt]
    declarations: list[
        FunctionDef
    ]  # Union[FunctionDef, VariableDecl]]  # Add other top-level declarations as needed

    def to_str(self) -> str:
        includes_str = "\n".join(include.to_str() for include in self.includes)
        globals_str = "\n".join(stmt.to_str() for stmt in self.globals)
        declarations_str = "\n\n".join(decl.to_str() for decl in self.declarations)
        return f"{includes_str}\n\n{globals_str}\n\n{declarations_str}"


@dataclass
class StructInit(Stmt):
    struct_name: str
    field_values: Mapping[str, Expr | str]
    static: bool = False

    def to_str(self) -> str:
        field_inits = [
            f".{field} = {value.to_str() if isinstance(value, Expr) else value}"
            for field, value in self.field_values.items()
        ]
        fields_str = ", ".join(field_inits)

        stmt = f"struct {self.struct_name} = {{ {fields_str} }};"
        if self.static:
            stmt = f"static {stmt}"

        return stmt


@dataclass
class StaticVariable(Stmt):
    type: str
    name: str
    initial_value: Expr | None = None

    def to_str(self) -> str:
        if self.initial_value is None:
            return f"static {self.type} {self.name};"
        return f"static {self.type} {self.name} = {self.initial_value.to_str()};"


@dataclass
class If(Stmt):
    condition: Expr
    body: list[Stmt]
    else_body: list[Stmt] | None = None

    def to_str(self) -> str:
        body_str = "\n    ".join([stmt.to_str() for stmt in self.body])
        if self.else_body is None:
            return f"if ({self.condition.to_str()}) {{\n    {body_str}\n}}"
        else:
            else_str = "\n    ".join([stmt.to_str() for stmt in self.else_body])
            return (
                f"if ({self.condition.to_str()}) {{\n    {body_str}\n}} else "
                f"{{\n    {else_str}\n}}"
            )
