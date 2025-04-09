# Code Style Guide

This document outlines the coding style and conventions used in the Mithril project. Following these guidelines ensures consistency across the codebase and makes collaboration easier.

## Python Style Guidelines

Mithril follows PEP 8 with some specific adaptations.

### Formatting

- **Line length**: Maximum 88 characters per line (compatible with Black formatter)
- **Indentation**: 4 spaces (no tabs)
- **Whitespace**:
  - No trailing whitespace
  - Surround binary operators with a single space on each side
  - No space around the equals sign in keyword arguments or default parameter values
- **Blank lines**:
  - Two blank lines before top-level classes and functions
  - One blank line before method definitions inside a class
  - Use blank lines judiciously within functions to indicate logical sections

### Naming Conventions

- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Prefix with a single underscore (`_private_method`)
- **"Dunder" methods**: Python special methods like `__init__` should be at the top of the class

### Import Style

Organize imports into the following groups, separated by a blank line:

1. Standard library imports
2. Related third-party imports
3. Local application/library-specific imports

Within each group, imports should be sorted alphabetically.

```python
# Standard library
import math
import os
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

# Third-party
import jax
import numpy as np
import torch

# Mithril imports
from mithril.backends import JaxBackend
from mithril.models import Model
```

### Type Annotations

- All public functions and methods must include type annotations
- Use the typing module for complex types
- Use optional types (`Optional[T]`) rather than default `None` values without type specification
- For collection types, specify the contained type (`List[int]`, `Dict[str, float]`)
- Use type comments for complex or multi-line expressions where annotations would hurt readability

```python
def process_data(
    inputs: Dict[str, np.ndarray], 
    batch_size: int, 
    names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Process input data in batches."""
    # Function implementation
```

## Docstrings

Mithril uses NumPy style docstrings.

### Function Docstrings

```python
def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate the accuracy of predictions compared to targets.
    
    Parameters
    ----------
    predictions : np.ndarray
        The predicted values, shape (n_samples,)
    targets : np.ndarray
        The ground truth values, shape (n_samples,)
        
    Returns
    -------
    float
        The accuracy score from 0.0 to 1.0
        
    Notes
    -----
    Accuracy is calculated as the number of correct predictions 
    divided by the total number of predictions.
    
    Examples
    --------
    >>> calculate_accuracy(np.array([0, 1, 1]), np.array([0, 1, 0]))
    0.6667
    """
```

### Class Docstrings

```python
class LinearRegression(Model):
    """
    Linear regression model implementation.
    
    This class implements a basic linear regression model with 
    optional L1/L2 regularization.
    
    Parameters
    ----------
    input_dim : int
        The input dimension
    output_dim : int
        The output dimension
    regularization : str, optional
        Type of regularization, either 'l1', 'l2', or None
    
    Attributes
    ----------
    weights : np.ndarray
        The model weights, shape (input_dim, output_dim)
    bias : np.ndarray
        The model bias, shape (output_dim,)
    """
```

## Code Organization

### Model Definition

- Use operator overloading (`|=`, `+=`) for model composition as intended
- Keep models modular, favoring composition over large monolithic classes
- Document the expected shapes of inputs and outputs

### Error Handling

- Use specific exception types rather than generic exceptions
- Include clear error messages explaining what went wrong
- For user-facing functions, validate inputs and provide helpful error messages

```python
def compile_model(model, backend_name):
    """Compile a model for the specified backend."""
    if not isinstance(model, Model):
        raise TypeError(f"Expected a Model instance, got {type(model).__name__}")
    
    if backend_name not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available backends: {', '.join(AVAILABLE_BACKENDS)}"
        )
```

### Comments

- Comments should explain "why" not "what" (the code shows what it does)
- Use comments for complex algorithms or non-obvious design decisions
- Keep comments up-to-date with the code (outdated comments are worse than no comments)

## Backend-Specific Code

- Keep backend-specific code isolated in the appropriate modules
- Use abstract interfaces to define common behavior across backends
- Document backend-specific limitations or behaviors

## Testing Style

- Write clear test names that describe what's being tested
- Use descriptive variable names in tests
- Create helper functions for common test setups
- Use parametrized tests for testing variations of the same functionality

```python
@pytest.mark.parametrize(
    "input_shape,kernel_size,expected_output_shape",
    [
        ((1, 3, 32, 32), 3, (1, 3, 30, 30)),
        ((1, 1, 5, 5), 2, (1, 1, 4, 4)),
        ((2, 3, 10, 10), 1, (2, 3, 10, 10)),
    ]
)
def test_conv2d_output_shape(input_shape, kernel_size, expected_output_shape):
    """Test that Conv2D produces correctly shaped outputs."""
    # Test implementation
```

## Continuous Integration

All code must pass the following checks in the CI pipeline:

- All tests passing
- Type checking with mypy
- Linting with ruff
- Code coverage above the project's minimum threshold

## Tools

Mithril recommends these tools for maintaining code quality:

- **Type checking**: mypy
- **Linting**: ruff
- **Formatting**: black (or ruff format)
- **Testing**: pytest

## Recommended Editor Settings

### VSCode

```json
{
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.mypyPath": "mypy",
    "editor.rulers": [88],
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "editor.detectIndentation": false,
    "files.trimTrailingWhitespace": true
}
```

### PyCharm

- Set line length to 88
- Enable PEP 8 checks
- Configure mypy integration