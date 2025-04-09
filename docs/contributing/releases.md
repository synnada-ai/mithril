# Release Process

This document outlines the process for creating and publishing releases of Mithril.

## Version Numbering

Mithril follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Functionality added in a backward-compatible manner
- **PATCH**: Backward-compatible bug fixes

## Release Preparation Checklist

### 1. Code Preparation

1. Ensure all desired features and bug fixes have been merged into the main branch
2. Verify all tests pass across all supported platforms and backends
3. Check code coverage and address any significant gaps
4. Run linting and type checking to ensure code quality
5. Review and update API documentation

### 2. Documentation Updates

1. Update the changelog with all notable changes since the last release (see [Changelog Updates](#changelog-updates) below)
2. Update version numbers in all relevant files:
   - `mithril/__init__.py`
   - `setup.py`
   - Documentation references
3. Ensure all examples work with the latest code
4. Update any compatibility tables or notes

### 3. Create Release Branch

Create a release branch from the main branch:

```bash
git checkout -b release/vX.Y.Z main
```

## Changelog Updates

The changelog should be organized into the following categories:

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Features that will be removed in upcoming releases
- **Removed**: Features removed in this release
- **Fixed**: Bug fixes
- **Security**: Security-related changes

Example:

```markdown
## [1.2.0] - 2025-04-08

### Added
- Support for the MLX backend on Apple Silicon
- New pooling operators: GlobalMaxPool, GlobalAvgPool

### Changed
- Improved memory efficiency in JAX backend
- Enhanced shape inference for dynamic dimensions

### Fixed
- Fixed gradient computation for custom loss functions
- Corrected shape handling in Conv2D when using padding
```

## Release Process

### 1. Final Testing on Release Branch

1. Run the complete test suite one final time
2. Build the package and test installation from the built distribution
3. Test documentation build

### 2. Release Approval

1. Create a pull request from the release branch to main
2. Request review from core maintainers
3. Address any final issues raised during review

### 3. Finalize Release

1. Merge the release PR into main
2. Create a tag for the new version:

```bash
git checkout main
git pull
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

### 4. Build and Publish Package

1. Build source and wheel distributions:

```bash
python -m build
```

2. Upload to PyPI:

```bash
python -m twine upload dist/*
```

### 5. Create GitHub Release

1. Go to the GitHub repository and navigate to Releases
2. Create a new release using the tag created earlier
3. Copy the relevant section from the changelog into the release description
4. Attach the built distributions to the release

### 6. Post-Release Activities

1. Announce the release on appropriate channels
2. Update documentation website with the latest release
3. Create a new development branch for the next version if needed

## Hotfix Releases

For urgent bug fixes that need to be released before the next planned release:

1. Create a hotfix branch from the latest release tag:

```bash
git checkout -b hotfix/vX.Y.(Z+1) vX.Y.Z
```

2. Apply the fixes directly to this branch
3. Follow the same validation and release process as above
4. After releasing the hotfix, merge the fixes back to the main branch as well

## Release Schedule

- **Minor releases**: Every 2-3 months
- **Patch releases**: As needed for bug fixes
- **Major releases**: When significant API changes are required, typically announced at least 3 months in advance

## Long-Term Support (LTS)

- Selected versions may be designated as Long-Term Support releases
- LTS releases receive bug fixes and security updates for an extended period (typically 1 year)
- The LTS policy will be explicitly documented for each designated release