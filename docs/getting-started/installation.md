# Installation

Mithril is available on PyPI and can be installed using pip. This page covers various installation options depending on your needs.

## Basic Installation

To install the latest version of Mithril:

```bash
pip install mithril --upgrade
```

This will install Mithril with the minimum required dependencies.

## Installing with Specific Backends

Mithril supports multiple backends, including JAX, PyTorch, NumPy, and MLX. You can install Mithril with the required dependencies for your preferred backend:

### PyTorch Backend

```bash
pip install mithril[torch] --upgrade
```

### JAX Backend

```bash
pip install mithril[jax] --upgrade
```

### MLX Backend

```bash
pip install mithril[mlx] --upgrade
```

### All Backends

To install Mithril with all supported backends:

```bash
pip install mithril[all] --upgrade
```

## Development Installation

For development purposes, you can install Mithril directly from the source:

```bash
git clone https://github.com/example/mithril.git
cd mithril
pip install -e ".[dev]"
```

The `[dev]` option installs additional development dependencies, including testing and documentation tools.

## Prerequisites

Mithril requires Python 3.8 or later. Depending on the backend you choose, you might need to install additional system libraries:

- **JAX**: For GPU support, you'll need CUDA and cuDNN. See [JAX installation guide](https://github.com/google/jax#installation) for details.
- **PyTorch**: For GPU support, see [PyTorch installation guide](https://pytorch.org/get-started/locally/).
- **MLX**: For Apple Silicon support, see [MLX installation guide](https://ml-explore.github.io/mlx/build/html/install.html).

## Verifying Installation

You can verify your installation by running:

```python
import mithril as ml
print(ml.__version__)
```

## Troubleshooting

If you encounter any issues during installation, please check the following:

1. Ensure you have the latest version of pip: `pip install --upgrade pip`
2. For backend-specific issues, refer to the respective backend's installation guide
3. For Mithril-specific issues, please check our [GitHub issues](https://github.com/example/mithril/issues) or create a new issue