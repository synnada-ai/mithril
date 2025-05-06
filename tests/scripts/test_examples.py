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

import re
from pathlib import Path

from mithril.backends.with_autograd.torch_backend.parallel import TorchParallel


class TemporaryFileCleanup:
    def __init__(self, directory="."):
        self.directory = Path(directory)
        self.files_before = set()

    def __enter__(self):
        # Take a snapshot of all files before entering the context
        self.files_before = set(self.directory.rglob("*"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Take a snapshot of files after the execution
        files_after = set(self.directory.rglob("*"))
        # Identify new files created during the context
        new_files = files_after - self.files_before
        # Delete all new files
        for file_path in new_files:
            if file_path.is_file():
                file_path.unlink()  # delete the file


def test_readme_code_blocks():
    with open("README.md") as file:
        content = file.read()

    # Find Python code blocks in the README (e.g., ```python ... ```)
    code_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

    with TemporaryFileCleanup():
        if TorchParallel._instance is not None:
            TorchParallel._instance.clean_up()
        for i, code_block in enumerate(code_blocks):
            try:
                # Use exec() to run the code block, wrapped in a function scope
                exec_globals: dict = {}
                exec(code_block, exec_globals)

                # You can add optional checks if the code blocks are supposed to define
                # variables For example, assert that expected variables or outputs are
                # defined
                assert exec_globals is not None  # Make sure exec ran successfully

                print(f"Code block {i + 1} executed successfully.")

            except Exception as e:
                # Raise an assertion error for pytest to catch
                raise AssertionError(
                    f"Code block {i + 1} failed with error: {e}"
                ) from e
        if TorchParallel._instance is not None:
            TorchParallel._instance.clean_up()
