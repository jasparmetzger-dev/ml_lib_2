from typing import Any

from .validation_type import is_supported_numeric


def cumsum(data: list[Any]) -> Any:
    if not data:
        return 0

    first_type = type(data[0]) #type: ignore
    if not is_supported_numeric(data[0]):
        err = TypeError(f"Could not add elements of type {first_type}")
        err.add_note(f"Could not add elements of type {first_type}")
        raise err

    try:
        res = 0
        for val in data:
            res += val
        return res
    except TypeError as e:
        item_type = type(data[0]) if data else None #type: ignore
        e.add_note(f"Could not add elements of type {item_type}")
        raise e


def cumprod(data: list[Any]) -> Any:
    if not data:
        return 1

    first_type = type(data[0]) #type: ignore
    if not is_supported_numeric(data[0]):
        err = TypeError(f"Could not multiply elements of type {first_type}")
        err.add_note(f"Could not multiply elements of type {first_type}")
        raise err

    try:
        res = 1
        for val in data:
            res *= val
        return res
    except TypeError as e:
        item_type = type(data[0]) if data else None #type: ignore
        e.add_note(f"Could not multiply elements of type {item_type}")
        raise e
