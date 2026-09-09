from typing import Any

def cumsum(data: list[Any]) -> Any:
    try:
        res = 0
        for val in data:
            res += val
        return res
    except TypeError as e:
        item_type = type(data[0]) if data else None # type: ignore

        e.add_note(f"Could not add elements of type {item_type}")
        raise e
    
def cumprod(data: list[Any]) -> Any:
    try:
        res = 1
        for val in data:
            res *= val
        return res
    except TypeError as e:
        item_type = type(data[0]) if data else None # type: ignore

        e.add_note(f"Could not multiply elements of type {item_type}")
        raise e
