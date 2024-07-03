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


# Github Changelog Generator
# This script generates a changelog for a GitHub repository based
# on the pull requests between two tags.
# It categorizes the pull requests based on GitHub
# labels and the Conventional Commits specification.
# For now, it only suports the following categories:

# - Breaking changes
# - Feature updates
# - Fixed bugs

# to utilize this script fun the following command:

# pip install PyGithub
# python3 generate_changelog.py <token> <version>

import io
import re
import sys
from contextlib import redirect_stdout

from github import Github  # type: ignore


def print_pulls(repo_name, title, pulls):
    if len(pulls) > 0:
        print(f"**{title}:**")
        print()
        for pull, commit in pulls:
            url = f"https://github.com/{repo_name}/pull/{pull.number}"
            print(f"- {pull.title} [#{pull.number}]({url}) ({commit.author.login})")
        print()


def generate_changelog(repo, repo_name, tag1, tag2):
    # get a list of commits between two tags
    comparison = repo.compare(tag1, tag2)

    # get the pull requests for these commits
    unique_pulls = []
    all_pulls = []
    for commit in comparison.commits:
        pulls = commit.get_pulls()
        for pull in pulls:
            # there can be multiple commits per PR if squash merge is not being used and
            # in this case we should get all the author names, but for now just pick one
            if pull.number not in unique_pulls:
                unique_pulls.append(pull.number)
                all_pulls.append((pull, commit))

    # split the pulls into categories
    breaking = []
    bugs = []
    docs = []
    style = []
    perf = []
    tests = []
    features = []
    other = []

    # categorize the pull requests based on GitHub labels
    print("Categorizing pull requests", file=sys.stderr)
    for pull, commit in all_pulls:
        # see if PR title uses Conventional Commits
        cc_type = ""
        cc_breaking = ""
        parts = re.findall(r"^([a-z]+)(\([a-z]+\))?(!)?:", pull.title)
        if len(parts) == 1:
            parts_tuple = parts[0]
            cc_type = parts_tuple[0]  # fix, feat, docs, chore
            cc_breaking = parts_tuple[2] == "!"

        labels = [label.name for label in pull.labels]
        if "api change" in labels or cc_breaking:
            breaking.append((pull, commit))
        elif "bug" in labels or cc_type == "fix":
            bugs.append((pull, commit))
        elif "feature" in labels or cc_type == "feat":
            features.append((pull, commit))
        elif "documentation" in labels or cc_type == "docs":
            docs.append((pull, commit))
        elif "style" in labels or cc_type == "style":
            style.append((pull, commit))
        elif "performance" in labels or cc_type == "perf":
            perf.append((pull, commit))
        elif "tests" in labels or cc_type == "test":
            tests.append((pull, commit))
        else:
            other.append((pull, commit))

    # produce the changelog content
    print("Generating changelog content", file=sys.stderr)

    print("""<!--
    Copyright 2022 Synnada, Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    -->\n""")

    print(f"# Mithril {tag2} Changelog\n")

    commit_count = str(comparison.total_commits)
    unique_contributors = set()
    for commit in comparison.commits:
        author = commit.author.login
        unique_contributors.add(author)

    # get number of contributors
    contributor_count = len(unique_contributors)

    print(
        f"This release has {commit_count} commits from {contributor_count} "
        "contributors."
    )

    print_pulls(repo_name, "Breaking changes", breaking)
    print_pulls(repo_name, "Feature updates", features)
    print_pulls(repo_name, "Documentation updates", docs)
    print_pulls(repo_name, "Style updates", style)
    print_pulls(repo_name, "Performance updates", perf)
    print_pulls(repo_name, "Test updates", tests)
    print_pulls(repo_name, "Fixed bugs", bugs)
    print_pulls(repo_name, "Other", other)


def validate_version(version: str):
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        raise ValueError("Invalid version number, must be in the format x.y.z")


def run(version: str, token: str):
    validate_version(version)
    project = "synnada-ai/mithril"

    g = Github(token)
    repo = g.get_repo(project)
    all_releases = list(repo.get_releases())

    new_release = "main"
    if len(all_releases) == 0:
        prev_release = "main"
        print(f"Generating changelog of initial release for {project}")
    else:
        prev_release = all_releases[0].tag_name
        print(
            f"Generating changelog for {project} from {prev_release} to "
            f"{new_release}"
        )

    f = io.StringIO()
    with redirect_stdout(f):
        generate_changelog(repo, project, prev_release, new_release)
    s = f.getvalue()

    with open(f"changelog/{version}.md", "w") as changelog_file:
        changelog_file.write(s)

    # Generated changelog
    print(s)


if __name__ == "__main__":
    """Process command line arguments."""
    args = sys.argv[1:]
    if not args:
        raise ValueError("Missing arguments, Usage: generate_changelog.py <token> ")

    version = args[0]
    token = args[1]
    run(version, token)
