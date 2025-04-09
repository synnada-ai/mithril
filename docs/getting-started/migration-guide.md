# Migration Guide

This guide helps users migrate from previous versions of Mithril to the latest version.

## Migrating from 0.0.x to 0.1.x

### Key Changes

- The `Model` class has been renamed to `LogicalModel`
- The backend selection API has been updated
- New compilation optimizations are available

### API Changes

#### Model Class Rename

```python
# Old code
from mithril import Model

# New code
from mithril import LogicalModel
```

#### Backend Selection

```python
# Old code
backend = get_backend('jax')

# New code
from mithril.backends.with_autograd.jax_backend import JaxBackend
backend = JaxBackend()
```

### Configuration Changes

The configuration file format has been updated. Old configuration files are not compatible with 0.1.x.

### Deprecated Features

The following features have been deprecated and will be removed in a future version:

- `legacy_mode` parameter in the `compile` function
- `use_old_optimizer` option

## Compatibility Matrix

| Feature | 0.0.x | 0.1.x |
|---------|-------|-------|
| Logical Models | ✓ | ✓ |
| Physical Models | ✗ | ✓ |
| JAX Backend | ✓ | ✓ |
| PyTorch Backend | ✓ | ✓ |
| NumPy Backend | ✓ | ✓ |
| MLX Backend | ✗ | ✓ |
| GGML Backend | ✗ | ✓ |
| C Backend | ✗ | ✓ |
| Advanced Compilation | ✗ | ✓ |
| Parallelization | ✗ | ✓ |