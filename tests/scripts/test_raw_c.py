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

import shutil
import subprocess
import tempfile
from pathlib import Path


def test_c_bindings():
    test_dir = Path(__file__).parent.absolute()
    c_file = test_dir / "c_tests.c"

    # Try both .dylib and .so extensions
    so_base = test_dir.parent.parent / "mithril/cores/c/raw_c/libmithrilc"
    so_file = None

    # Check which library file exists
    for ext in [".dylib", ".so"]:
        if (so_base.parent / (so_base.name + ext)).exists():
            so_file = so_base.parent / (so_base.name + ext)
            break

    if so_file is None:
        raise FileNotFoundError(
            f"Could not find libmithrilc.dylib or libmithrilc.so in {so_base.parent}"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        _tmp_dir = Path(tmp_dir)
        # Copy library to temp dir (keeping its original extension)
        tmp_lib = _tmp_dir / so_file.name
        shutil.copy(so_file, tmp_lib)

        executable = _tmp_dir / "c_tests"
        compile_cmd = [
            "cc",
            str(c_file),
            f"-L{_tmp_dir}",
            "-lmithrilc",
            f"-Wl,-rpath,{_tmp_dir}",
            "-o",
            str(executable),
        ]

        # macOS-specific flags
        import platform

        if platform.system() == "Darwin":
            compile_cmd.extend(["-Wl,-undefined,dynamic_lookup"])

        try:
            subprocess.check_call(compile_cmd, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("Compilation failed with command:", " ".join(compile_cmd))
            print("Error output:", e.stderr)
            raise

        result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            cwd=_tmp_dir,
        )
        if result.returncode != 0:
            print("Test executable failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0
