# Testing Guide

This document outlines the testing approach for the Mithril project. Following these guidelines ensures that the codebase remains robust and reliable.

## Testing Philosophy

Mithril's testing philosophy is based on:

1. **Comprehensive Coverage**: Every feature should have tests
2. **Multiple Levels**: Unit, integration, and model tests
3. **Randomized Testing**: Complex behaviors tested with randomized inputs
4. **Cross-Backend Validation**: Features tested across all supported backends
5. **Regression Prevention**: Tests for fixed bugs to prevent regressions

## Test Organization

Tests are organized in the `tests/` directory:

- `tests/scripts/`: Main test scripts
- `tests/json_files/`: JSON test specifications
- `tests/__init__.py`: Test utilities and shared fixtures

## Test Categories

### Unit Tests

Unit tests verify individual functions and classes in isolation. They should:

- Focus on a single unit of functionality
- Minimize dependencies on other components
- Use mocks where appropriate for isolation
- Have descriptive names indicating what they test

Example:

```python
def test_relu_applies_correct_function():
    """Test that the ReLU function zeros out negative values."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    result = relu_fn(x)
    np.testing.assert_array_equal(result, expected)
```

### Integration Tests

Integration tests verify that multiple components work together correctly:

- Test interactions between components
- Verify end-to-end functionality
- Focus on interfaces and data flow

Example:

```python
def test_model_compilation_and_inference():
    """Test that a model can be compiled and run inference correctly."""
    model = create_test_model()
    backend = JaxBackend()
    compiled_model = ml.compile(model, backend)
    
    inputs = {"input": np.random.randn(1, 10)}
    outputs = compiled_model.evaluate(inputs)
    
    assert "output" in outputs
    assert outputs["output"].shape == (1, 5)
```

### Directed Tests

Directed tests focus on specific edge cases or behaviors:

- Test corner cases and boundary conditions
- Verify handling of unusual inputs
- Test error conditions and exceptions

Example:

```python
@pytest.mark.parametrize(
    "shape1,shape2,error_expected",
    [
        ((2, 3), (3, 4), False),      # Valid matrix multiplication
        ((2, 3), (2, 4), True),       # Invalid shapes
        ((2, 0), (0, 3), False),      # Zero dimension
    ]
)
def test_matmul_shape_validation(shape1, shape2, error_expected):
    """Test matrix multiplication with various shapes."""
    a = np.ones(shape1)
    b = np.ones(shape2)
    
    if error_expected:
        with pytest.raises(ValueError):
            matmul_fn(a, b)
    else:
        result = matmul_fn(a, b)
        expected_shape = (shape1[0], shape2[1])
        assert result.shape == expected_shape
```

### Randomized Tests

Randomized tests use randomly generated inputs to explore a wide range of behaviors:

- Generate random inputs (with controlled seeds for reproducibility)
- Test across a range of shapes, sizes, and values
- Verify properties that should hold regardless of input

Example:

```python
@pytest.mark.parametrize("seed", range(10))
def test_random_model_equivalence(seed):
    """Test that random models produce the same output across backends."""
    np.random.seed(seed)
    model = generate_random_model(n_layers=3, max_dim=10)
    
    # Test across backends
    torch_backend = TorchBackend()
    jax_backend = JaxBackend()
    
    torch_model = ml.compile(model, torch_backend)
    jax_model = ml.compile(model, jax_backend)
    
    inputs = {"input": np.random.randn(2, 8)}
    
    torch_outputs = torch_model.evaluate(inputs)
    jax_outputs = jax_model.evaluate(inputs)
    
    # Compare outputs (allowing for small numerical differences)
    np.testing.assert_allclose(
        torch_outputs["output"], 
        jax_outputs["output"],
        rtol=1e-5, atol=1e-5
    )
```

### Backend Tests

Backend tests ensure that each backend works correctly and consistently:

- Test each backend's implementation of primitives
- Verify backend-specific features
- Compare results across backends for consistency

### Performance Tests

Performance tests track execution speed and memory usage:

- Benchmark critical operations
- Measure compilation time
- Monitor memory consumption
- Compare performance across backends

## Running Tests

### Running All Tests

To run all tests:

```bash
pytest tests/
```

### Running Specific Tests

To run a specific test file:

```bash
pytest tests/scripts/test_file.py
```

To run a specific test function:

```bash
pytest tests/scripts/test_file.py::test_function
```

To run tests matching a pattern:

```bash
pytest -k "matmul or conv2d"
```

### Test Options

Common pytest options:

- `-v`: Verbose output
- `-s`: Show print statements in tests
- `--pdb`: Drop into debugger on test failure
- `--junitxml=report.xml`: Generate XML test report
- `--cov=mithril`: Generate coverage report

## Writing Good Tests

### Test Naming

Name tests clearly with the pattern `test_<what_is_being_tested>_<expected_behavior>`:

```python
def test_conv2d_padding_valid_reduces_dimensions():
    # Test implementation
```

### Fixtures

Use pytest fixtures for common setup:

```python
@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    model = Model()
    model |= Linear(dimension=32)(input="input", output="hidden")
    model += Relu()(input="hidden", output="hidden_act")
    model += Linear(dimension=10)(input="hidden_act", output="output")
    return model

def test_model_compilation(sample_model):
    """Test that the model compiles successfully."""
    # Test using the fixture
```

### Assertions

Use appropriate assertions:

- `assert` for simple checks
- `pytest.raises` for exception testing
- `np.testing.assert_allclose` for numerical comparisons
- `np.testing.assert_array_equal` for exact array comparisons

### Parameterization

Use pytest's parameterize feature for testing variations:

```python
@pytest.mark.parametrize(
    "input_shape,kernel_size,stride,padding,expected_output_shape",
    [
        ((1, 3, 32, 32), 3, 1, "same", (1, 3, 32, 32)),
        ((1, 3, 32, 32), 3, 1, "valid", (1, 3, 30, 30)),
        ((1, 3, 32, 32), 3, 2, "same", (1, 3, 16, 16)),
    ]
)
def test_conv2d_output_shapes(input_shape, kernel_size, stride, padding, expected_output_shape):
    """Test Conv2D output shapes with various configurations."""
    # Test implementation
```

## Test-Driven Development

Consider using Test-Driven Development (TDD) when appropriate:

1. Write a failing test that defines the expected behavior
2. Implement the minimum code to make the test pass
3. Refactor the code while keeping tests passing

## Test Coverage

Aim for high test coverage, but focus on meaningful coverage:

- Use `pytest-cov` to measure coverage
- Aim for at least 80% coverage for core components
- Prioritize testing complex logic and edge cases
- Don't write tests just to increase coverage numbers

To check coverage:

```bash
pytest --cov=mithril tests/
```

## Test Data

For tests requiring data:

- Use synthetic data where possible
- Keep test data small and focused
- Include the data generation in the test when feasible
- For larger datasets, consider storing in a test_data directory

## Continuous Integration

Tests run automatically in the CI pipeline:

- All tests must pass before merging PRs
- Coverage changes are tracked
- Performance regressions are monitored

## Debugging Failed Tests

When tests fail:

1. Use `-v` flag to get more verbose output
2. Use `-s` flag to see print statements
3. Use `--pdb` flag to drop into debugger on failure
4. Check logs for error messages
5. Try running the test in isolation

## Adding New Tests

When adding a new feature, add appropriate tests:

- Unit tests for the core functionality
- Integration tests for interactions with other components
- Edge case tests for boundary conditions
- Backend-specific tests if needed

When fixing a bug:

1. Write a test that reproduces the bug
2. Fix the bug
3. Verify that the test now passes