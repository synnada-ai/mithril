#!/bin/bash
# Script to download and build GGML from source if needed

set -e  # Exit on any error

# Detect OS and set appropriate library extension
UNAME=$(uname)
if [ "$UNAME" = "Darwin" ]; then
    LIB_EXT="dylib"
elif [ "$UNAME" = "Windows" ] || [ "$UNAME" = "MINGW"* ] || [ "$UNAME" = "MSYS"* ]; then
    LIB_EXT="dll"
else
    LIB_EXT="so"
fi

# Set directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/ggml/build"
GGML_REPO="https://github.com/ggerganov/ggml.git"
GGML_DIR="${SCRIPT_DIR}/ggml"

# Check if we need to download GGML
if [ ! -d "${GGML_DIR}" ]; then
    echo "GGML not found. Cloning from repository..."
    git clone --depth 1 "${GGML_REPO}" "${GGML_DIR}"
else
    echo "GGML directory exists. Using existing files."
fi

# Create build directory if it doesn't exist
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure CMake build
echo "Configuring GGML build..."
cmake .. -DBUILD_SHARED_LIBS=ON

# Build GGML
echo "Building GGML..."
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

# Copy the libraries to our directory
echo "Copying GGML libraries to ${SCRIPT_DIR}..."
find . -name "libggml*.${LIB_EXT}*" -exec cp {} "${SCRIPT_DIR}/" \;

# Build our own bindings
cd "${SCRIPT_DIR}"
echo "Building Mithril GGML bindings..."
./compile.sh

echo "Build completed successfully!"
echo "The following libraries are available:"
ls -l "${SCRIPT_DIR}"/*.${LIB_EXT}* 