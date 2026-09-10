from typing import Any

class nan:
    def __init__(self):
        pass

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, nan)
    def __repr__(self) -> str:
        return "nan"

    def __lt__(self, other: Any) -> bool:
        return False
    def __le__(self, other: Any) -> bool:
        return False
    def __gt__(self, other: Any) -> bool:
        return False
    def __ge__(self, other: Any) -> bool:
        return False

    def __add__(self, other: Any) -> "nan":
        return nan()
    def __radd__(self, other: Any) -> "nan":
        return nan()
    def __iadd__(self, other: Any) -> "nan":
        return nan()
    def __sub__(self, other: Any) -> "nan":
        return nan()
    def __rsub__(self, other: Any) -> "nan":
        return nan()
    def __isub__(self, other: Any) -> "nan":
        return nan()
    def __pow__(self, other: Any) -> "nan":
        return nan()
    def __rpow__(self, other: Any) -> "nan":
        return nan()
    def __ipow__(self, other: Any) -> "nan":
        return nan()
    def __truediv__(self, other: Any) -> "nan":
        return nan()
    def __rtruediv__(self, other: Any) -> "nan":
        return nan()
    def __itrudiv__(self, other: Any) -> "nan":
        return nan()
    def __neg__(self, other: Any) -> "nan":
        return nan()
    def __abs__(self, other: Any) -> "nan":
        return nan()
