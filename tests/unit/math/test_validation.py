import pytest

from src.math.validation_shape import ShapeError, validate_shape
from src.math.validation_type import is_supported_numeric


def test_shape_error_stores_message_and_formats_string():
    err = ShapeError("bad shape")

    assert err.message == "bad shape"
    assert str(err) == "ShapeError: bad shape"

def test_shape_error_type():
    err = ShapeError("bad shape")
    assert isinstance(err, ShapeError)
    assert issubclass(type(err), Exception)


def test_validate_shape_accepts_matching_product():
    assert validate_shape((2, 3), 6) is None
    assert validate_shape((2, 2, 3), 12) is None

def test_validate_shape_raises_for_mismatched_size():
    with pytest.raises(
        ShapeError, match=r"ShapeError: checked shape \(2, 3\) does not match size 5"
    ):
        validate_shape((2, 3), 5)

def test_is_supported_numeric():
    assert is_supported_numeric("a") == False
    assert is_supported_numeric(()) == False
    assert is_supported_numeric({}) == False
    assert is_supported_numeric([]) == False

    assert is_supported_numeric(1) == True
    assert is_supported_numeric(1.0) == True
    assert is_supported_numeric(True) == True

    class SpecialFloat(float):
        def __init__(self) -> None: #type: ignore
            super().__init__()

    assert is_supported_numeric(SpecialFloat()) == True
