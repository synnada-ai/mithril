from ....common import CGenConfig

CODEGEN_CONFIG = CGenConfig()

# File configs
CODEGEN_CONFIG.HEADER_NAME = "ggml_backend.h"


# Array configs
CODEGEN_CONFIG.ARRAY_NAME = "struct ggml_tensor"

# Function configs
CODEGEN_CONFIG.USE_OUTPUT_AS_INPUT = False
CODEGEN_CONFIG.RETURN_OUTPUT = True

