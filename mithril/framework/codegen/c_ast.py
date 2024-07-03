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
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class AST(ABC):
    @abstractmethod
    def to_str(self):
        raise NotImplementedError("to_str is not implemented")


@dataclass
class Expr(AST):
    pass


@dataclass
class Stmt(AST):
    pass


@dataclass
class Call(Expr):
    name: str
    args: list[str] | list[Expr]

    def to_str(self):
        args_str = ", ".join(
            [arg.to_str() if isinstance(arg, Expr) else arg for arg in self.args]
        )
        # args_str = ", ".join(self.args)
        return f"{self.name}({args_str})"


@dataclass
class Constant(Expr):
    value: int | float

    def to_str(self):
        return str(self.value)

    def __str__(self) -> str:
        return self.to_str()


@dataclass
class Parameter:
    type: str
    name: str

    def to_str(self):
        return f"{self.type} {self.name}"


@dataclass
class FunctionDef(Stmt):
    return_type: str
    name: str
    params: list[Parameter]
    body: Sequence[Stmt | Expr]

    def to_str(self):
        params_str = (
            "\n\t" + ",\n\t".join([param.to_str() for param in self.params]) + "\n"
        )
        body_str = "\n    ".join([stmt.to_str() + ";" for stmt in self.body])
        return f"{self.return_type} {self.name}({params_str})\n{{\n    {body_str}\n}}"


@dataclass
class Return(Stmt):
    value: Expr

    def to_str(self):
        return f"return {self.value.to_str()};"


@dataclass
class Include(AST):
    header: str
    system: bool = False  # True for system headers, False for user-defined headers

    def to_str(self):
        if self.system:
            return f"#include <{self.header}>"
        else:
            return f'#include "{self.header}"'


@dataclass
class FILE(AST):
    includes: list[Include]
    globals: list[Stmt]
    declarations: list[
        FunctionDef
    ]  # Union[FunctionDef, VariableDecl]]  # Add other top-level declarations as needed

    def to_str(self):
        includes_str = "\n".join(include.to_str() for include in self.includes)
        globals_str = "\n".join(stmt.to_str() for stmt in self.globals)
        declarations_str = "\n\n".join(decl.to_str() for decl in self.declarations)
        return f"{includes_str}\n\n{globals_str}\n\n{declarations_str}"
