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
number = sys.argv[1]
label = sys.argv[2]
body = sys.argv[3]

# Load the label-user mapping from the YAML file
with open(".github/label_to_devs.yml") as file:
    label_user_mapping = yaml.safe_load(file)


if related_users := label_user_mapping.get(label):
    related_users = set(related_users)
    lines = body.split("\n")

    last_line = lines[-1]
    # check if the issue already has a CC
    if last_line[:4] == "CC: ":
        lines.pop()
        users = last_line[4:].split(", ")  # find all users in the CC
        users = [user[1:] for user in users]  # remove the @
        all_users = set(users) | related_users  # combine the old and new users
    else:
        # add three lines for formatting
        lines.append("")
        lines.append("")
        lines.append("")
        all_users = related_users
    all_users.discard("")  # remove empty strings
    new_line = "CC: " + "".join(f"@{user}, " for user in all_users)[:-2]
    lines[-1] = new_line
    new_body = "".join(f"{line}\n" for line in lines)[:-1]
    new_body.replace("`", "\\`")  # escape backticks
    command = f"gh issue edit {number} --body '{new_body}'"
    subprocess.run(command, shell=True)
