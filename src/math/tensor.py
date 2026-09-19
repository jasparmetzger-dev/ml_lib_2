from typing import Any, Optional

from .ops import prod
from .validation import ShapeError, is_broadcastable
from .scalar import is_scalar

# -------------------------------------------------
# TODO: test outer and @
# TODO: __iter__
# TODO: __repr__
# TODO: __getitem__ / __setitem__ for ranges
# TODO is_broadcastable(), broadcast()
# -------------------------------------------------

class Tensor:
    def __init__(self, data: list[Any], shape: Optional[tuple[int, ...]] = None, stype: Optional[type] = None):
        """
        initializes a Tensor. Two ways of initializing:
        1. Nestes lists of data
        2. 1D list of data with a precalculated shape
        """

        # 1. method
        if shape is None:
            self._data: list[Any] = self._make_1d_data(
                self._clean_data(data, _type=stype)
            )
            self._shape: tuple[int, ...]   = self._infer_shape(data)
            self._strides: tuple[int, ...] = self._infer_strides()
        # 2. method
        else:
            if prod(list(shape)) != len(data):
                raise ValueError(f"shape {shape} and data of length {len(data)} are not equal.")

            self._data: list[Any]          = self._clean_data(data, _type=stype)
            self._shape: tuple[int, ...]   = shape
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
        return f"values {self._data} with shape {self._shape}"

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, indices: tuple[int, ...] | int) -> Any:
        if type(indices) == int:
            if indices < 0 or indices > self.size - 1:
                raise IndexError("Index out of range")
            return self._data[indices]

        self.__check_item_input(indices) # type: ignore
        return self._data[self._calc_offset(indices)] # type: ignore

    def __setitem__(self, indices: tuple[int, ...] | int, value: Any) -> None:
        try:
            value = self.stype(value)
        except (TypeError, ValueError) as e:
            e.add_note(f"could not broadcast value of type {type(value)} to the stype {self.stype}")
            raise e

        if type(indices) == int:
            if indices < 0 or indices > self.size - 1:
                raise IndexError("Index out of range")
            self._data[indices] = value
            return None

        self.__check_item_input(indices) # type: ignore
        self._data[self._calc_offset(indices)] = value # type: ignore

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


    def __matmul__(self, other: "Tensor") -> "Tensor":
        return Tensor.matrix_multiply(self, other)
    def __imatmul__(self, other: "Tensor") -> "Tensor":
        result = Tensor.matrix_multiply(other, self)
        self._data = result._data
        self._shape = result._shape
        self._strides = result._strides
        return self

    # -------------------------------------------------
    # PROPERTIES
    # -------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return prod(list(self._shape))

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
    def stype(self) -> type:
        return type(self._data[0]) if self._data else type(None) #type: ignore


    # -------------------------------------------------
    # PUBLIC
    # -------------------------------------------------

    def reshape(self, newShape: tuple[int, ...]) -> None:
        if prod(list(newShape)) != self.size:
            raise ShapeError(f"checked shape {newShape} does not match size {self.size}")

        self._shape = newShape
        self._strides = self._infer_strides()

    def copy(self) -> "Tensor":
        return Tensor(self._data, self._shape)

    def T(self) -> "Tensor":
        self._shape = self._shape[::-1]
        self._strides = self._infer_strides()
        return self

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

    @staticmethod
    def outer(tensor1: "Tensor", tensor2: "Tensor") -> "Tensor":
        if tensor1.ndim != 1 or tensor2.ndim != 1:
            raise ShapeError("Tensor.outer() is only defined for ndim == 1")

        data = []
        for val in tensor1._data:
            data.append((tensor2 * val).data_1d) # type: ignore
        return Tensor(data) # type: ignore

    # -------------------------------------------------
    # PRIVATE
    # -------------------------------------------------

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

    def _calc_offset(self, indices: tuple[int, ...]) -> int:
        if len(list(indices)) != self.ndim:
            raise IndexError(f"Have {len(indices)} indices, should have {self.ndim} indices.")

        offset = 0
        for j in range(self.ndim):
            offset += indices[j] * self._strides[j]
        return offset

    def _make_1d_data(self, data: list[Any]) -> list[Any]:
        arr: list[Any] = []

        def _get_elements_recursively(val: Any) -> None:
            if isinstance(val, list):
                for ele in val: # type: ignore
                    _get_elements_recursively(ele)
            else:
                arr.append(val)

        _get_elements_recursively(data)
        return arr

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

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        Supports:
        - 2D x 2D: (m, n) x (n, p) -> (m, p)
        - 1D x 2D: (n,)   x (n, p) -> (p,)
        - 2D x 1D: (m, n) x (n,)   -> (m,)
        - 1D x 1D: (n,)   x (n,)   -> ()  (scalar tensor)
        """
        if a.ndim not in (1, 2) or b.ndim not in (1, 2):
            raise ShapeError(f"Tensors must be 1D or 2D, got a.ndim={a.ndim}, b.ndim={b.ndim}")

        # Promote 1D inputs temporarily for unified 2D matrix multiplication logic
        a_is_1d = (a.ndim == 1)
        b_is_1d = (b.ndim == 1)

        a_shape = (1, a.shape[0]) if a_is_1d else a.shape
        a_strides = (0, a._strides[0]) if a_is_1d else a._strides

        b_shape = (b.shape[0], 1) if b_is_1d else b.shape
        b_strides = (b._strides[0], 0) if b_is_1d else b._strides

        m, n_a = a_shape
        n_b, p = b_shape

        if n_a != n_b:
            raise ShapeError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")

        out_data: list[Any] = []
        for i in range(m):
            for j in range(p):
                val = 0
                for k in range(n_a):
                    idx_a = i * a_strides[0] + k * a_strides[1]
                    idx_b = k * b_strides[0] + j * b_strides[1]
                    val += a._data[idx_a] * b._data[idx_b]
                out_data.append(val)

        # Determine target output shape
        if a_is_1d and b_is_1d:
            out_shape = ()
        elif a_is_1d:
            out_shape = (p,)
        elif b_is_1d:
            out_shape = (m,)
        else:
            out_shape = (m, p)

        return Tensor(out_data, out_shape)

    # -------------------------------------------------
    # __HELPERS
    # -------------------------------------------------

    def __check_item_input(self, indices: tuple[int, ...]):
        if len(list(indices)) != self.ndim:
            raise IndexError(f"Too many indices ({len(list(indices))}) for tensor with {self.ndim} dimensions.")

        for dim, dim_idx in enumerate(indices): # type: ignore
            if dim_idx < 0 or dim_idx > self.shape[dim] - 1:
                raise IndexError(f"Index in dimension {dim} out of range.")
