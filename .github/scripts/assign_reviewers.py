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

import subprocess
import sys

import yaml  # type: ignore

# Load labels from GitHub Action input
labels = sys.argv[2].split(",")
number = int(sys.argv[1])

# Load the label-user mapping from the YAML file
with open(".github/label_to_devs.yml") as file:
    label_user_mapping = yaml.safe_load(file)

# Find reviewers based on labels
for label in labels:
    reviewers = set()
    label = label.strip()
    if label in label_user_mapping:
        reviewers.update(label_user_mapping[label])
    if reviewers:
        prefix_arg = f"gh pr edit {number} --add-assignee "
        for reviewer in reviewers:
            arg = prefix_arg + str(reviewer)
            subprocess.run(arg, shell=True)
