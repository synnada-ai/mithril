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

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

# def test_c_bindings():
#     test_dir = Path(__file__).parent.absolute()
#     c_file = test_dir / "c_tests.c"
#     so_file = test_dir.parent.parent / "mithril/cores/c/raw_c/libmithrilc.so"

#     with tempfile.TemporaryDirectory() as tmp_dir:
#         _tmp_dir = Path(tmp_dir)
#         # Copy .so to temp dir
#         tmp_so = _tmp_dir / "libmithrilc.so"
#         shutil.copy(so_file, tmp_so)

#         executable = _tmp_dir / "c_tests"
#         compile_cmd = [
#             "cc",
#             str(c_file),
#             str(tmp_so),
#             f"-Wl,-rpath,{_tmp_dir}",  # Look in temp dir
#             "-o",
#             str(executable),
#         ]
#         subprocess.check_call(compile_cmd)

#         result = subprocess.run(
#             [str(executable)],
#             capture_output=True,
#             text=True,
#             cwd=_tmp_dir,
#         )
#         assert result.returncode == 0


def test_c_bindings():
    test_dir = Path(__file__).parent.absolute()
    c_file = test_dir / "c_tests.c"

    # Library detection
    so_base = test_dir.parent.parent / "mithril/cores/c/raw_c/libmithrilc"
    so_file = None
    for ext in [".dylib", ".so"]:
        candidate = so_base.parent / (so_base.name + ext)
        if candidate.exists():
            so_file = candidate
            break

    if so_file is None:
        raise FileNotFoundError(
            f"Could not find libmithrilc.dylib or libmithrilc.so in {so_base.parent}"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        _tmp_dir = Path(tmp_dir)
        tmp_lib = _tmp_dir / so_file.name
        shutil.copy(so_file, tmp_lib)

        # Debug info
        print("\n--- DEBUG INFO ---")
        print("Library path:", tmp_lib)
        print("Library exists:", tmp_lib.exists())
        print("File size:", tmp_lib.stat().st_size, "bytes")

        try:
            print("\nLibrary info:")
            subprocess.run(["file", str(tmp_lib)], check=True)
            if platform.system() == "Darwin":
                subprocess.run(["otool", "-L", str(tmp_lib)], check=True)
        except Exception as e:
            print("Couldn't inspect library:", e)

        executable = _tmp_dir / "c_tests"
        compile_cmd = [
            "cc",
            str(c_file),
            str(tmp_lib),  # Directly reference the library file
            f"-Wl,-rpath,{_tmp_dir}",
            "-o",
            str(executable),
        ]
        if platform.system() == "Darwin":
            compile_cmd.extend(["-Wl,-undefined,dynamic_lookup"])

        print("\nCompilation command:", " ".join(compile_cmd))

        try:
            # Let the output show in CI logs
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("COMPILATION FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                result.check_returncode()  # This will raise the error
        except subprocess.CalledProcessError as e:
            print("\nFULL ERROR OUTPUT:")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise

        # Run the tests
        test_result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            cwd=_tmp_dir,
        )
        if test_result.returncode != 0:
            print("TEST EXECUTION FAILED:")
            print("STDOUT:", test_result.stdout)
            print("STDERR:", test_result.stderr)
        assert test_result.returncode == 0
