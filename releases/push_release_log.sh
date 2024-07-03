#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Usage: ./update_changelog.sh <version> <token>
# Example: ./update_changelog.sh 0.2.5 your_github_token

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <version> <token>"
    exit 1
fi

VERSION=$1
TOKEN=$2

# Check if the current branch name is "prepare_<version>"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
EXPECTED_BRANCH="prepare_${VERSION}"

if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
    echo "Error: Current branch is '$CURRENT_BRANCH', but expected branch is '$EXPECTED_BRANCH'."
    echo "Please switch to the correct branch or rename your branch."
    exit 1
fi

# Run the Python script to generate the changelog
python3 prepare_release.py "$VERSION" "$TOKEN"

# Stage and commit the new changelog file
git add "changelog/${VERSION}.md"
git commit -m "Add changelog for version $VERSION"

# Push the changes to the GitHub repository
git push origin "$CURRENT_BRANCH"