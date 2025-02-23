from dataclasses import dataclass


@dataclass
class CGenConfig:
    # Import configs
    HEADER_NAME: str = ""

    # Array configs
    ARRAY_NAME: str = ""

    # Function call configs
    USE_OUTPUT_AS_INPUT: bool = False
    RETURN_OUTPUT: bool = False
    