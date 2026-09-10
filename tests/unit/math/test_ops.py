from decimal import Decimal
from typing import Any

import pytest

from src.math.ops import prod, cumsum


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], 0),
        ([0], 0),
        ([1, 2, 3, 4], 10),
        ([1, -2, 3, -4, 5], 3),
        ([1.5, 2.5, -1.0, 0.0], 3.0),
        ([Decimal("1.5"), Decimal("2.5"), Decimal("-1.0")], Decimal("3.0")),
        ([True, False, True], 2),
    ],
)
def test_cumsum_handles_edge_cases(data: list[Any], expected:Any):
    assert cumsum(data) == pytest.approx(expected)


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
def test_cumprod_handles_edge_cases(data: list[Any], expected: Any):
    assert prod(data) == pytest.approx(expected)


def test_cumsum_raises_typeerror_with_note_for_invalid_input_types():
    with pytest.raises(TypeError) as excinfo:
        cumsum(["two", 3])

    assert excinfo.value.__notes__
    assert "Could not add elements of type" in excinfo.value.__notes__[0]
    assert "<class 'str'>" in excinfo.value.__notes__[0]


def test_cumprod_raises_typeerror_with_note_for_invalid_input_types():
    with pytest.raises(TypeError) as excinfo:
        prod(["two", 3])

    assert excinfo.value.__notes__
    assert "Could not multiply elements of type" in excinfo.value.__notes__[0]
    assert "<class 'str'>" in excinfo.value.__notes__[0]
