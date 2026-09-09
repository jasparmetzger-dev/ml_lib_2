from typing import Any, Optional

from .ops import cumsum
from .validation import validate_shape

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
    def T(self) -> 'Tensor':
        self._shape = self._shape[::-1]
        self._strides = self._infer_strides()
        return Tensor([])

    @property
    def stype(self) -> type:
        return type(self._data[0]) # type: ignore

    # -------------------------------------------------
    # PUBLIC
    # -------------------------------------------------

    def reshape(self, newShape: tuple[int, ...]) -> None:
        validate_shape(newShape, self.size)

        self._shape = newShape
        self._strides = self._infer_strides()

    # -------------------------------------------------
    # PRIVATE
    # -------------------------------------------------


    def _make_1d(self, data: list[Any]) -> list[Any]:

        def _get_elements_recursively(val: Any, arr: list[Any]) -> Any:
            if isinstance(val, list):
                try:
                    for ele in val: # type: ignore
                        _get_elements_recursively(ele, arr)
                except TypeError as e:
                    e.add_note("BUG IN 'Tensor._make_1d._get_elements_recursively'")
                    raise TypeError
            else: arr.append(val)


        arr: list[Any] = []
        _get_elements_recursively(data, arr)
        return arr


    def _infer_shape(self, data: Optional[list[Any]] = None) -> tuple[int, ...]:
        if data is None: x = self._data
        else: x = data

        res: list[int] = []

        while True:
            res.append(len(x))
            try:
                x = x[0]
            except (IndexError, TypeError):
                return tuple(res)

    def _infer_strides(self, shape: Optional[tuple[int, ...]] = None) -> tuple[int, ...]:
        if shape is None: shape = self._shape

        strides: list[int] = [1] * len(self._shape)
        for i in range(len(self._shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * self._shape[i + 1]
        return tuple(strides)

    def _make_nd_data(self, shape: Optional[tuple[int, ...]] = None) -> list[Any]:
        if shape is None:
            shape = self._shape
        print("WARNING: Tensor._make_nd_data() IS NOT IMPLEMENTED!")
        raise NotImplementedError

    def _clean_data(self, data: list[Any], _type: Optional[type] = None) -> list[Any]:
        """
        Ensures type is the same (and of type '_type').
        Ensures every encapsulation is of type 'list'.
        Ensures all arrays in all dimensions have the same number of elements.
        Computationally expensive!
        """

        print("WARNING: Tensor._clean_data() IS NOT IMPLEMENTED!")
        return data

# ---------------------------------
# UTILS
# ---------------------------------
