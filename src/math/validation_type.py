from typing import Any
from decimal import Decimal

def is_supported_numeric(value: Any) -> bool:
    return isinstance(value, (bool, int, float, Decimal)) or issubclass(type(value), (bool, int, float, Decimal))
