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


import os
import sys

# Take the exclude argument with argparse
exclude = ""
for arg in sys.argv:
    if arg.startswith("--exclude="):
        exclude = arg.split("=")[1]
        break

license_py = """# Copyright 2022 Synnada, Inc.
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
"""

license_ch = """// Copyright 2022 Synnada, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
"""
current_directory = os.getcwd()


# Walk through the directory recursively
for root, _, files in os.walk(current_directory):
    if os.path.basename(root) == "tmp":
        continue
    for filename in files:
        if filename.endswith((".py", ".c", ".h")):  # Check for .py .h and .c files
            file_path = os.path.join(root, filename)

            if exclude != "" and exclude in file_path:
                continue

            # Check if it's a file
            if os.path.isfile(file_path):
                with open(file_path, encoding="utf-8", errors="ignore") as file:
                    lines = file.readlines()
                    if len(lines) < 13:
                        raise Exception(f"No license found in {file_path}")

                    file_license = "".join(lines[:13])

                license = license_py if filename.endswith(".py") else license_ch

                assert (
                    license == file_license
                ), f"Please update the license in {file_path}"
