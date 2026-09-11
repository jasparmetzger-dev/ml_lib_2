import pytest
import numpy as np

from decimal import Decimal
from typing import Any
from src.math.scalar import is_scalar

class NewFloat(float): pass

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (int(1), True),
        (float(1.0), True),
        (Decimal("1.0"), True),
        (NewFloat(1.0), True),
        (np.float64(1.0), True),

        (str(1), False),
        (list([1]), False),
        (tuple((1, 1)), False)
    ]
)
def test_is_scalar(value: Any, expected: Any):
    assert is_scalar(value) == expected, f"Type {type(value)} is not expected {expected}"
