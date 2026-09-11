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
    with pytest.raises((TypeError, ValueError)):
        prod(["two", 3])
    with pytest.raises((TypeError, ValueError)):
        prod([2, [[3, 4], 5]])
    with pytest.raises((TypeError, ValueError)):
        prod(["two", "three"])

if __name__ == "__main__":
    print(prod([1, [2, 3]]))
