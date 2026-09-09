from decimal import Decimal

import pytest

from src.math.tensor import Tensor
from src.math.validation import ShapeError


def test_tensor_flattens_nested_lists_and_infers_shape():
    tensor = Tensor([[1, 2], [3, 4]])

    assert tensor.data_1d == [1, 2, 3, 4]
    assert tensor.shape == (2, 2)
    assert tensor.ndim == 2
    assert tensor.stype is int


def test_tensor_handles_empty_and_scalar_inputs():
    empty = Tensor([])
    scalar = Tensor([5])

    assert empty.data_1d == []
    assert empty.shape == (0,)
    assert empty.size == 0
    assert empty.ndim == 1

    assert scalar.data_1d == [5]
    assert scalar.shape == (1,)
    assert scalar.size == 1


def test_tensor_size_uses_shape_sum_not_product():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    assert tensor.shape == (2, 3)
    assert tensor.size == 5


def test_tensor_stype_uses_first_element_type_for_weird_types():
    bool_tensor = Tensor([True, False, True])
    decimal_tensor = Tensor([Decimal("1.5"), Decimal("2.5")])
    string_tensor = Tensor([["a", "b"], ["c", "d"]])

    assert bool_tensor.stype is bool
    assert decimal_tensor.stype is Decimal
    assert string_tensor.stype is str


def test_tensor_reshape_raises_for_mismatched_size():
    tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with pytest.raises(ShapeError):
        tensor.reshape((2, 3))


def test_tensor_dtype_and_reshape_accept_matching_size():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    tensor.reshape((3, 2))

    assert tensor.shape == (3, 2)
    assert tensor.size == 5
    assert tensor.stype is int
    assert tensor._infer_strides() == (2, 1)


def test_tensor_clean_data_rejects_wrong_type_and_make_nd_data_is_unimplemented():
    tensor = Tensor([1, 2, 3])

    with pytest.raises(TypeError, match="Expected all tensor values to be of type"):
        tensor._clean_data([1, "two", 3], _type=int)

    with pytest.raises(NotImplementedError):
        tensor._make_nd_data()
