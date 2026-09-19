from decimal import Decimal
from typing import Any

import pytest

from src.math.tensor import Tensor
from src.math.validation import ShapeError

class TestTensorInitialization:
    def test_tensor_flattens_nested_lists_and_infers_shape(self):
        tensor = Tensor([[1, 2], [3, 4]])

        assert tensor.data_1d == [1, 2, 3, 4]
        assert tensor.shape == (2, 2)
        assert tensor.ndim == 2
        assert tensor.stype is int

    def test_tensor_infers_nd_shape(self):
        tensor = Tensor([1, 2, 3, 4], (2, 2))
        assert tensor.shape == (2, 2)
        assert tensor.ndim == 2
        assert tensor.stype is int

    def test_tensor_handles_empty_and_scalar_inputs(self):
        empty = Tensor([])
        scalar = Tensor([5])

        assert empty.data_1d == []
        assert empty.shape == (0,)
        assert empty.size == 0
        assert empty.ndim == 1

        assert scalar.data_1d == [5]
        assert scalar.shape == (1,)
        assert scalar.size == 1

class TestTensorStridesShapes:

    @pytest.mark.parametrize(
            ("tensor", "ele_amount"),
            [
                (Tensor([1, 2, 3, 4, 5, 6], (2, 3)), 6),
                (Tensor([[1, 2, 3], [4, 5, 6]]), 6),
                (Tensor([]), 0)
            ]
    )
    def test_tensor_len_and_size(self, tensor: Tensor, ele_amount: int):
        assert len(tensor) == tensor.size and ele_amount == tensor.size

    def test_tensor_calc_offfset(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        assert tensor._calc_offset((0, 0)) == 0 # type: ignore
        assert tensor._calc_offset((0, 2)) == 2 # type: ignore
        assert tensor._calc_offset((1, 0)) == 3 # type: ignore
        assert tensor._calc_offset((1, 2)) == 5 # type: ignore

        with pytest.raises(IndexError):
            tensor._calc_offset((0, 0, 0)) # type: ignore

class TestTensorDunderNonMath:
    def test_tensor_equality(self):
        assert Tensor([[1, 2, 3], [4, 5, 6]]) == Tensor([[1, 2, 3], [4, 5, 6]])
        assert Tensor([[1, 2, 3], [4, 5, 6]]) == Tensor([1, 2, 3, 4, 5, 6], (2, 3))

    def test_tensor_geitem_setitem_int(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        assert tensor.shape == (2, 3)
        tensor[2] = 9
        assert tensor[2] == 9

    def test_tensor_geitem_setitem_tuple(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        assert tensor[(1, 0)] == 4
        assert tensor[(0, 2)] == 3
        tensor[(1, 0)] = 5
        assert tensor[(1, 0)] == 5
        with pytest.raises(IndexError):
            tensor[(0, 0, 0)] = 1
        with pytest.raises(IndexError):
                tensor[(2, 0)] = 1
        with pytest.raises(IndexError):
            value = tensor[(2, 0)] # type: ignore
        with pytest.raises(IndexError):
            value = tensor[(0, 0, 0)] # type: ignore

    @pytest.mark.parametrize(
        ("tensor", "val", "is_contained"),
        [
            (Tensor([[1, 2, 3], [4, 5, 6]]), 2, True),
            (Tensor([[1, 2, 3], [4, 5, 6]]), 2.0, True),
            (Tensor([[1, 2, 3], [4, 5, 6]]), Decimal(2), True),
            (Tensor([[1, 2, 3], [4, 5, 6]]), "two", False),
            (Tensor([[1, 0, 3], [4, 5, 6]]), 2, False),
        ]
    )
    def test_tensor_contains(self, tensor: Tensor, val: Any, is_contained: bool):
        assert (val in tensor) == is_contained

class TestTensorBasicOperations:
    def test_tensor_tensor_add_subtract(self):
        tensor1, tensor2 = Tensor([[1, 2, 3], [4, 5, 6]]), Tensor([[1, 1, 1], [1, 1, 1]])
        assert tensor1 + tensor2 == Tensor([[2, 3, 4], [5, 6, 7]])
        assert tensor1 - tensor2 == Tensor([[0, 1, 2], [3, 4, 5]])

        tensor1 += tensor2
        assert tensor1 == Tensor([[2, 3, 4], [5, 6, 7]])
        tensor1 -= tensor2
        tensor1 -= tensor2
        assert tensor1 == Tensor([[0, 1, 2], [3, 4, 5]])

        assert Tensor.add(tensor1, tensor2) == Tensor([[1, 2, 3], [4, 5, 6]])
        assert Tensor.subtract(tensor1, tensor2) == Tensor([[-1, 0, 1], [2, 3, 4]])

        tensor_int, tensor_float = Tensor([1, 2]), Tensor([2.0, 2.5])
        res = tensor_int + tensor_float
        assert res.stype == float
        for val in res._data: # type: ignore
            assert type(val) == res.stype

    def test_tensor_scalar_mult_div(self):
        tensor = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert tensor * 2 == Tensor([2, 4, 6, 8, 10, 12], (2, 3))
        assert 2 * tensor == Tensor([2, 4, 6, 8, 10, 12], (2, 3))
        tensor *= 2
        assert tensor == Tensor([2, 4, 6, 8, 10, 12], (2, 3))

        assert tensor / 2 == Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        tensor /= 2
        assert tensor == Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert pytest.approx(2 / tensor) == Tensor([2.0, 1.0, 2 / 3, 0.5, 2 / 5, 2 / 6], (2, 3))

class TestTensorMatrixMultiply:

    @pytest.mark.parametrize(
        "a_data, a_shape, b_data, b_shape, expected_data, expected_shape",
        [
            # 2D x 2D Standard
            ([1, 2, 3, 4], (2, 2), [5, 6, 7, 8], (2, 2), [19, 22, 43, 50], (2, 2)),
            # 2D x 2D Rectangular (2x3) x (3x2) -> (2x2)
            ([1, 2, 3, 4, 5, 6], (2, 3), [7, 8, 9, 10, 11, 12], (3, 2), [58, 64, 139, 154], (2, 2)),
            # 1D x 2D Vector-Matrix (3,) x (3x2) -> (2,)
            ([1, 2, 3], (3,), [7, 8, 9, 10, 11, 12], (3, 2), [58, 64], (2,)),
            # 2D x 1D Matrix-Vector (2x3) x (3,) -> (2,)
            ([1, 2, 3, 4, 5, 6], (2, 3), [1, 2, 3], (3,), [14, 32], (2,)),
            # 1D x 1D Dot Product (3,) x (3,) -> ()
            ([1, 2, 3], (3,), [4, 5, 6], (3,), [32], ()),
            # Identity Matrix Multiplication
            ([1, 0, 0, 1], (2, 2), [3, 4, 5, 6], (2, 2), [3, 4, 5, 6], (2, 2)),
            # Zero Matrix
            ([0, 0, 0, 0], (2, 2), [1, 2, 3, 4], (2, 2), [0, 0, 0, 0], (2, 2)),
        ]
    )
    def test_matmul_shapes_and_values(self, a_data, a_shape, b_data, b_shape, expected_data, expected_shape):
        a = Tensor(a_data, a_shape)
        b = Tensor(b_data, b_shape)
        res = Tensor.matrix_multiply(a, b)

        assert res.shape == expected_shape
        assert res._data == pytest.approx(expected_data)

    def test_floating_point_precision(self):
        a = Tensor([0.1, 0.2, 0.3, 0.4], (2, 2))
        b = Tensor([0.5, 0.6, 0.7, 0.8], (2, 2))
        res = Tensor.matrix_multiply(a, b)

        # Exact mathematical result: [[0.19, 0.22], [0.43, 0.50]]
        assert res._data == pytest.approx([0.19, 0.22, 0.43, 0.50], abs=1e-12)

class TestTensorTyping:
    def test_tensor_stype_uses_first_element_type_for_weird_types(self):
        bool_tensor = Tensor([True, False, True])
        decimal_tensor = Tensor([Decimal("1.5"), Decimal("2.5")])
        string_tensor = Tensor([["a", "b"], ["c", "d"]])

        assert bool_tensor.stype is bool
        assert decimal_tensor.stype is Decimal
        assert string_tensor.stype is str

    def test_tensor_clean_data_rejects_wrong_type(self):
        tensor = Tensor([1, 2, 3])

        with pytest.raises(TypeError, match="Expected all tensor values to be of type"):
            tensor._clean_data([1, "two", 3], _type=int) # type: ignore

class TestTensorShaping:
    def test_tensor_reshape_raises_for_mismatched_size(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        with pytest.raises(ShapeError):
            tensor.reshape((2, 3))


    def test_tensor_dtype_and_reshape_accept_matching_size(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])

        tensor.reshape((3, 2))

        assert tensor.shape == (3, 2)
        assert tensor.size == 6
        assert tensor.stype is int
        assert tensor._infer_strides() == (2, 1) # type: ignore



def test_tensor_outer():
    tensor1, tensor2 = Tensor([1, 2, 3, 4, 5]), Tensor([2, 2, 2, 2, 2])
    res = Tensor.outer(tensor1, tensor2)
    assert res == Tensor([5*[2], 5*[4], 5*[6], 5*[8], 5*[10]])

    with pytest.raises((TypeError, AttributeError)):
        Tensor.outer(1, [5, 6])
