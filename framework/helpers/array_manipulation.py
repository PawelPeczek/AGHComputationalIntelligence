from typing import TypeVar, List

T = TypeVar("T")


def initialize_vector(value: T, size: int) -> List[T]:
    return [value] * size
