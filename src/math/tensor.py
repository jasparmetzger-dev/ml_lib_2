from typing import Any, Optional

from .ops import prod
from .validation import ShapeError, is_broadcastable
from .scalar import is_scalar


class Tensor:
    def __init__(self, data: list[Any], shape: Optional[tuple[int, ...]] = None, stype: Optional[type] = None):
        self._data: list[Any] = self._make_1d(
            self._clean_data(data, _type=stype)
        )
        self._shape: tuple[int, ...] = self._infer_shape(data)
        self._strides: tuple[int, ...] = self._infer_strides()

    # -------------------------------------------------
    # DUNDER METHODS
    # -------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        """
        Equality does not support broadcasting
        Only returns True with Tensors with the same shape and data
        """

        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot compare type Tensor and {type(other)}")

        if self._shape != other._shape:
            return False

        for i, j in zip(self._data, other.data_1d):
            if i != j: return False

        return True

    def __repr__(self) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index > self.size - 1:
            raise IndexError("Index out of range")
        return self._data[index]

    def __setitem__(self, index: int, value: Any) -> None:
        if index < 0 or index > self.size - 1:
            raise IndexError("Index out of range")

        try:
            value = self.stype(value)
        except (TypeError, ValueError) as e:
            e.add_note(f"could not broadcast value of type {type(value)} to the stype {self.stype}")
            raise e

        self._data[index] = value

    def __contains__(self, value: Any) -> bool:
        try: value = self.stype(value)
        except (TypeError, ValueError): return False

        for ele in self._data:
            if ele == value:
                return True
        return False

    # -------------------------------------------------
    # ARITHMETIC DUNDER METHODS
    # -------------------------------------------------
    def __add__(self, other: Any) -> "Tensor":
        return Tensor.add(self, other)
    def __iadd__(self, other: Any) -> "Tensor":
        result = Tensor.add(self, other)
        self._data = result._data
        self._shape = result._shape
        self._strides = result._strides
        return self

    def __sub__(self, other: Any) -> "Tensor":
        return Tensor.subtract(self, other)
    def __isub__(self, other: Any) -> "Tensor":
        result = Tensor.subtract(self, other)
        self._data = result._data
        self._shape = result._shape
        self._strides = result._strides
        return self

    def __mul__(self, other: Any) -> "Tensor":
        return Tensor.scalar_multiply(self, other)
    def __rmul__(self, other: Any) -> "Tensor":
        return Tensor.scalar_multiply(self, other)
    def __imul__(self, other: Any) -> "Tensor":
        result = Tensor.scalar_multiply(self, other)
        self._data = result._data
        self._shape = result._shape
        self._strides = result._strides
        return self

    def __truediv__(self, other: Any) -> "Tensor":
        return Tensor.scalar_true_divide(self, other, is_tensor_left=True)
    def __rtruediv__(self, other: Any) -> "Tensor":
        return Tensor.scalar_true_divide(self, other, is_tensor_left=False)
    def __itruediv__(self, other: Any) -> "Tensor":
        result = Tensor.scalar_true_divide(self, other, is_tensor_left=True)
        self._data = result._data
        self._shape = result._shape
        self._strides = result._strides
        return self

    def __pow__(self, other: Any) -> "Tensor":
        return Tensor.scalar_exponentiate(self, other, is_tensor_base=True)
    def __rpow__(self, other: Any) -> "Tensor":
        return Tensor.scalar_exponentiate(self, other, is_tensor_base=False)

    def __neg__(self) -> "Tensor":
        return self * -1
    def __abs__(self) -> "Tensor":
        self._data = [abs(val) for val in self._data]
        return self

    # -------------------------------------------------
    # PROPERTIES
    # -------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return sum(list(self._shape))

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def data_nd(self) -> list[Any]:
        return self._make_nd_data(self._shape)

    @property
    def data_1d(self) -> list[Any]:
        return self._data

    @property
    def T(self) -> "Tensor":
        self._shape = self._shape[::-1]
        self._strides = self._infer_strides()
        return Tensor([])

    @property
    def stype(self) -> type:
        return type(self._data[0]) if self._data else type(None) #type: ignore


    # -------------------------------------------------
    # PUBLIC
    # -------------------------------------------------

    def reshape(self, newShape: tuple[int, ...]) -> None:
        if sum(list(newShape)) != self.size:
            raise ShapeError(f"checked shape {newShape} does not match size {self.size}")

        self._shape = newShape
        self._strides = self._infer_strides()

    def copy(self) -> "Tensor":
        return Tensor(self._data, self._shape)

    def sum(self) -> Any:
        return self.stype(sum(self._data))

    def prod(self) -> Any:
        return self.stype(prod(self._data))

    # -------------------------------------------------
    # STATIC
    # -------------------------------------------------

    @staticmethod
    def broadcast(shape: tuple[int, ...], tensor: "Tensor") -> "Tensor":
        tensor.reshape(shape)
        return tensor

    # -------------------------------------------------
    # PRIVATE
    # -------------------------------------------------


    def _make_1d(self, data: list[Any]) -> list[Any]:
        arr: list[Any] = []

        def _get_elements_recursively(val: Any) -> None:
            if isinstance(val, list):
                for ele in val: # type: ignore
                    _get_elements_recursively(ele)
            else:
                arr.append(val)

        _get_elements_recursively(data)
        return arr

    def _infer_shape(self, data: Optional[list[Any]] = None) -> tuple[int, ...]:
        if data is None: x = self._data
        else: x = data

        if not isinstance(x, list): # type: ignore
            return ()
        if len(x) == 0:
            return (0,)
        if not all(isinstance(item, list) for item in x):
            return (len(x),)

        first_shape = self._infer_shape(x[0])
        if any(self._infer_shape(item) != first_shape for item in x[1:]):
            raise TypeError("Tensor dimensions are not consistent")
        return (len(x),) + first_shape

    def _infer_strides(self, shape: Optional[tuple[int, ...]] = None) -> tuple[int, ...]:
        if shape is None:
            shape = self._shape

        if not shape:
            return ()

        strides: list[int] = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    def _make_nd_data(self, shape: Optional[tuple[int, ...]] = None) -> list[Any]:
        if shape is None:
            shape = self._shape
        raise NotImplementedError("Tensor._make_nd_data() is not implemented")

    def _clean_data(self, data: list[Any], _type: Optional[type] = None) -> list[Any]:
        if data is None: # type: ignore
            return []
        if not isinstance(data, list): # type: ignore
            data = [data]
        if _type is not None:
            def _flatten_values(val: Any) -> list[Any]:
                if isinstance(val, list):
                    out: list[Any] = []
                    for item in val: # type: ignore
                        out.extend(_flatten_values(item))
                    return out
                return [val]

            flat = _flatten_values(data)
            if any(not isinstance(item, _type) for item in flat):
                raise TypeError(f"Expected all tensor values to be of type {_type}")
        return data

    # -------------------------------------------------
    # UTIL DUNDER MATH
    # -------------------------------------------------
    @staticmethod
    def add(a: "Tensor", b: "Tensor") -> "Tensor":
        # check shapes and types
        if a.size != b.size or not is_broadcastable(a._shape, b._shape):
            raise ShapeError(f"Tensors of shape {a._shape} and {b._shape} are not broadcastable.")
        try: a.stype(b._data[0])
        except (TypeError, ValueError):
            raise TypeError(f"Elements of the Tensors with types {a.stype}, {b.stype} can not be casted into each other")

        if a._shape != b._shape:
            b = Tensor.broadcast(a._shape, b)

        new_data = [ele_a + ele_b for ele_a, ele_b in zip(a._data, b._data)]
        return Tensor(new_data, a._shape)

    @staticmethod
    def subtract(a: "Tensor", b: "Tensor") -> "Tensor":
        # check shapes and types
        if a.size != b.size or not is_broadcastable(a._shape, b._shape):
             raise ShapeError(f"Tensors of shape {a._shape} and {b._shape} are not broadcastable.")
        try: a.stype(b._data[0])
        except (TypeError, ValueError):
            raise TypeError(f"Elements of the Tensors with types {a.stype}, {b.stype} can not be casted into each other")

        if a._shape != b._shape:
            b = Tensor.broadcast(a._shape, b)

        new_data = [ele_a - ele_b for ele_a, ele_b in zip(a._data, b._data)]
        return Tensor(new_data, a._shape)

    @staticmethod
    def scalar_multiply(tensor: "Tensor", scalar: Any) -> "Tensor":
        if not is_scalar(scalar):
            raise TypeError("can only scalar_multiply() Tensor with a scalar.")

        new_data = [ele * scalar for ele in tensor._data]
        return Tensor(new_data, tensor._shape)

    @staticmethod
    def scalar_true_divide(tensor: "Tensor", scalar: Any, is_tensor_left: bool) -> "Tensor":
        if not is_scalar(scalar):
            raise TypeError("can only scalar_multiply() Tensor with a scalar.")

        if is_tensor_left:
            new_data = [ele / scalar for ele in tensor._data]
        else:
            new_data = [scalar / ele for ele in tensor._data]
        return Tensor(new_data, tensor._shape)


    @staticmethod
    def scalar_exponentiate(tensor: "Tensor", scalar: Any, is_tensor_base: bool) -> "Tensor":
        if not is_scalar(scalar):
            raise TypeError("can only scalar_multiply() Tensor with a scalar.")

        if is_tensor_base:
            new_data = [ele ** scalar for ele in tensor._data]
        else:
            new_data = [scalar ** ele for ele in tensor._data]
        return Tensor(new_data, tensor._shape)
