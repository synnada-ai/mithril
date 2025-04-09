# Changelog

This page documents all notable changes to Mithril.

## 0.1.1

*Released: 2025-02-15*

### Added
- MLX backend support for Apple Silicon
- Improved shape inference system
- New examples for CLIP and T5 models
- Support for gradient accumulation
- Additional model primitives

### Fixed
- Issue with tensor broadcasting in NumPy backend
- Memory leak in PyTorch backend when using custom ops
- Type inference for complex nested models
- Serialization of models with custom terminals

### Changed
- Refactored backend API for better extensibility
- Improved error messages for shape mismatches
- Enhanced documentation and examples
- Performance optimizations for JAX backend

## 0.1.0

*Released: 2025-01-10*

Initial release of Mithril with the following features:

### Added
- Core model composition API
- Logical and physical model abstraction
- JAX, PyTorch, and NumPy backends
- Basic parallelization support
- Shape and type inference
- Model serialization
- Gradient computation
- Basic examples and documentation