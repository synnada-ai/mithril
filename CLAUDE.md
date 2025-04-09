# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Install dev dependencies: `pip install -r requirements/dev.txt`
- Install package: `pip install -e .`
- Run all tests: `pytest tests/`
- Run single test: `pytest tests/scripts/test_file.py::test_function`
- Lint code: `ruff check .`
- Type check: `mypy mithril/`

## Code Style
- Line length: 88 characters
- Indentation: 4 spaces
- Imports: stdlib → third-party → project, alphabetized within groups
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: UPPER_SNAKE_CASE
- Private methods/vars: prefixed with underscore
- Type annotations required (strict mypy checking)
- Error handling: specific exceptions with descriptive messages
- Docstrings: NumPy style format
- Models use operator overloading (+=, |=) for composition
- Backend-specific code isolated in separate modules
- Tests must be thorough, including randomized model tests