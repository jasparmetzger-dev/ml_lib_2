from typing import Any, Optional

from .ops import cumsum
from .validation import ShapeError


class Tensor:
    def __init__(self, data: list[Any], stype: Optional[type] = None):
        self._data: list[Any] = self._make_1d(
            self._clean_data(data, _type=stype)
        )
        self._shape: tuple[int, ...] = self._infer_shape(data)
        self._strides: tuple[int, ...] = self._infer_strides()

    # -------------------------------------------------
    # PROPERTIES
    # -------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return cumsum(list(self._shape))

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
        if cumsum(list(newShape)) != self.size:
            raise ShapeError(f"checked shape {newShape} does not match size {self.size}")

        self._shape = newShape
        self._strides = self._infer_strides()

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

# ---------------------------------
# UTILS
# ---------------------------------
