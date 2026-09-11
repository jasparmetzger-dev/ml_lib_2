from decimal import Decimal
from typing import Any

import pytest

from src.math.ops import prod


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], 1),
        ([1], 1),
        ([2, 3, 4], 24),
        ([2, -3, 4], -24),
        ([1.5, 2, 0.5], 1.5),
        ([Decimal("1.5"), Decimal("2"), Decimal("0.5")], Decimal("1.5")),
        ([True, False, True], 0),
    ],
)
def test_prod_handles_edge_cases(data: list[Any], expected: Any):
    assert prod(data) == pytest.approx(expected)


def test_cumprod_raising_error():
    with pytest.raises(TypeError) as excinfo:
        prod(["two", 3])

    assert excinfo.value.__notes__
    assert "Could not multiply elements of type" in excinfo.value.__notes__[0]
    assert "<class 'str'>" in excinfo.value.__notes__[0]
