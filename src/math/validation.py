from .ops import prod

class ShapeError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__("ShapeError: " + self.message)

# --------------------------------
# Validation functions
# --------------------------------

def validate_shape(shape: tuple[int, ...], size: int) -> None:
    if prod(list(shape)) != size:
        raise ShapeError(f"checked shape {shape} does not match size {size}")

def is_broadcastable(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> bool:
    return prod(list(shape1)) == prod(list(shape2))

