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
    so_file = test_dir.parent.parent / "mithril/cores/c/raw_c/libmithrilc.so"

    with tempfile.TemporaryDirectory() as tmp_dir:
        _tmp_dir = Path(tmp_dir)
        # Copy .so to temp dir
        tmp_so = _tmp_dir / "libmithrilc.so"
        shutil.copy(so_file, tmp_so)

        executable = _tmp_dir / "c_tests"
        compile_cmd = [
            "cc",
            str(c_file),
            str(tmp_so),
            f"-Wl,-rpath,{_tmp_dir}",  # Look in temp dir
            "-o",
            str(executable),
        ]
        subprocess.check_call(compile_cmd)

        result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            cwd=_tmp_dir,
        )
        assert result.returncode == 0
