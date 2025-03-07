#!/bin/bash
# Build GGML binaries with proper names for different platforms

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

# Basic compiler settings
CC=${CC:-cc}
CFLAGS="-O3 -fPIC"

echo "Building GGML binaries for platform: $UNAME (extension: .$LIB_EXT)"

# Compile the main library
echo "Compiling libmithrilggml.$LIB_EXT..."
${CC} ${CFLAGS} ops.c -L. -lggml-base -shared -o "libmithrilggml.$LIB_EXT"

# Make the library executable if needed
chmod +x "libmithrilggml.$LIB_EXT"

echo "Done! Created libmithrilggml.$LIB_EXT" 