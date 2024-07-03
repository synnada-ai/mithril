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
import sys

from generate_changelog import run as generate_changelog
from generate_changelog import validate_version


def get_setup_version():
    """Read the version from setup.py."""
    with open("../setup.py") as f:
        setup_content = f.read()
    # Extract version using regex
    version_match = re.search(r'version="([0-9]+\.[0-9]+\.[0-9]+)"', setup_content)
    if not version_match:
        raise ValueError("Version not found in setup.py")
    return version_match.group(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    version = args[0]
    token = args[1]

    # Validate version format
    validate_version(version)

    # Check if the provided version matches setup.py
    setup_version = get_setup_version()
    if version != setup_version:
        raise ValueError(
            f"Provided version '{version}' does not match setup.py version "
            f"'{setup_version}'"
        )

    generate_changelog(version, token)
