from typing import Tuple

from .ops import cumprod

class ShapeError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__("ShapeError: " + self.message)

# --------------------------------
# Validation functions
# --------------------------------

def validate_shape(shape: Tuple[int, ...], size: int) -> None:
    if cumprod(list(shape)) != size:
        raise ShapeError(f"checked shape {shape} does not match size {size}")

