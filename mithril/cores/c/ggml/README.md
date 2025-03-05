# Mithril GGML Bindings

This directory contains bindings for the GGML tensor library for use in Mithril.

## Building the Binaries

### Option 1: Use Pre-built GGML Libraries

If you already have GGML libraries (`libggml-base.so`, `libggml-base.dylib`, etc.) available, you can just build the bindings:

```bash
# Make the script executable if needed
chmod +x compile.sh

# Build the bindings
./compile.sh
```

This will generate the `libmithrilggml.[so/dylib/dll]` file that contains the Mithril custom operations for GGML.

### Option 2: Build GGML from Source

If you don't have GGML libraries available, you can download and build them from source:

```bash
# Make the script executable if needed
chmod +x build_ggml.sh

# Build GGML and the bindings
./build_ggml.sh
```

This script will:
1. Clone the GGML repository (if not already present)
2. Build GGML with CMake
3. Copy the resulting libraries to this directory
4. Build the Mithril GGML bindings

## Using the Bindings in Python

The GGML bindings can be imported and used in Python:

```python
from mithril.cores.c.ggml import ggml_struct
# Additional imports may be needed depending on your usage
```

## Custom Operations

This binding provides the following custom operations:

- `add`: Adds two GGML tensors
- `multiplication`: Multiplies two GGML tensors

Each operation also has a corresponding gradient function for use in backpropagation:

- `add_grad`: Gradient function for addition
- `multiplication_grad`: Gradient function for multiplication

## Platform Support

The build scripts handle cross-platform compilation and will generate the appropriate library type for your system:

- Linux: `.so` extension
- macOS: `.dylib` extension
- Windows: `.dll` extension 