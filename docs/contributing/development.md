# Development Guide

This guide is intended for developers who want to contribute to Mithril. It covers the setup process, development workflow, and best practices for contributing to the project.

## Setting Up Development Environment

### Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.8 or higher
- Git
- A C/C++ compiler (for some backends)

### Clone the Repository

```bash
git clone https://github.com/example/mithril.git
cd mithril
```

### Create a Virtual Environment

We recommend using a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Install the development dependencies:

```bash
pip install -r requirements/dev.txt
```

Install the package in development mode:

```bash
pip install -e .
```

## Development Workflow

### Branch Strategy

Mithril follows a feature branch workflow:

1. Create a new branch for each feature or bugfix
2. Make your changes in the branch
3. Submit a pull request to merge into the main branch

```bash
# Create a new branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Add your feature"

# Push to GitHub
git push -u origin feature/your-feature-name
```

### Commit Guidelines

Commit messages should be clear and descriptive. We follow a conventional commit format:

```
<type>: <description>

[optional body]

[optional footer]
```

Types include:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code changes that neither fix bugs nor add features
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Updates to build process, dependencies, etc.

Examples:
```
feat: add support for MLX backend
fix: correct shape inference in conv2d
docs: update installation instructions
```

### Pull Requests

When submitting a pull request:

1. Update the documentation, especially if you're adding new features
2. Add or update tests to cover your changes
3. Make sure all tests pass
4. Keep your PR focused on a single concern for easier review
5. Write a clear PR description explaining the purpose and implementation details

## Running Tests

Mithril uses pytest for testing. To run all tests:

```bash
pytest tests/
```

To run a specific test file or function:

```bash
pytest tests/scripts/test_file.py::test_function
```

### Testing Guidelines

- Write tests for all new features and bug fixes
- Maintain test coverage for existing code
- Use parametrized tests for testing variations of the same functionality
- Add directed tests for specific edge cases

## Code Quality Checks

### Type Checking

Mithril uses mypy for static type checking. Run type checking with:

```bash
mypy mithril/
```

### Linting

We use ruff for linting. Check your code with:

```bash
ruff check .
```

## Documentation

All new features should be documented. Mithril uses Markdown for documentation.

- Add docstrings to all public functions, classes, and methods
- Update the appropriate Markdown files in the `docs/` directory
- For significant changes, add examples showing how to use the feature

## Building Documentation

Mithril uses MkDocs for documentation. To build and view the documentation locally:

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

## Backend Development

### Adding a New Backend

To add a new backend to Mithril:

1. Create a new directory in `mithril/backends/with_autograd/` or `mithril/backends/with_manualgrad/` depending on whether the backend supports automatic differentiation
2. Implement the `Backend` interface defined in `mithril/backends/backend.py`
3. Add necessary operator implementations
4. Add tests for the new backend
5. Update documentation

See existing backends for examples of implementation.

### Testing Backends

Mithril has a comprehensive test suite that verifies backend behavior. Some key tests for backends include:

- `tests/scripts/test_backend_fns.py`: Tests basic functions of the backend
- `tests/scripts/test_differentiablity.py`: Tests gradient computation for autograd backends
- `tests/scripts/test_randomized_models_all_backends.py`: Tests random model execution across backends

## Framework Development

### Core Components

The main components of the Mithril framework are:

- **Logical Framework**: The logical model definition layer (`mithril/framework/logical/`)
- **Physical Framework**: The compiled model representation (`mithril/framework/physical/`)
- **Backends**: Backend-specific implementations (`mithril/backends/`)
- **Code Generation**: Tools for generating code from models (`mithril/framework/codegen/`)

When developing, be mindful of the separation between these components.

### Adding New Operators

To add a new operator to Mithril:

1. Define the operator in `mithril/framework/logical/operators.py` or a specialized module
2. Implement shape and type inference for the operator
3. Implement the operator in all backends or provide a reasonable error message for unsupported backends
4. Add tests for the operator
5. Update documentation

## CI/CD Pipeline

Mithril uses GitHub Actions for continuous integration and deployment. The CI pipeline runs on every pull request and includes:

- Building the package
- Running tests
- Type checking
- Linting
- Building documentation

Ensure that your changes pass all CI checks before merging.

## Release Process

Mithril follows semantic versioning (MAJOR.MINOR.PATCH). The release process involves:

1. Creating a release branch
2. Updating version numbers
3. Generating a changelog
4. Creating a GitHub release
5. Publishing to PyPI

See the [Release Process](releases.md) document for detailed procedures.

## Getting Help

If you need help with the development process:

- Check the existing documentation
- Look at similar existing code in the codebase
- Open an issue for general questions
- Reach out to the maintainers directly for specific concerns

## Resources

- [GitHub Repository](https://github.com/example/mithril)
- [Documentation](https://example.com/mithril)
- [Issue Tracker](https://github.com/example/mithril/issues)