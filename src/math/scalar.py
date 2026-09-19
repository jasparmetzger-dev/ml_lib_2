from typing import Any
from decimal import Decimal

def is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, Decimal)) or issubclass(type(value), (bool, int, float, Decimal))
