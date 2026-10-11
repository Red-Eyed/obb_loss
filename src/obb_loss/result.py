"""Small checked outcomes for supported caller recovery."""

from typing import Generic, NamedTuple, TypeAlias, TypeVar

T = TypeVar("T")
E = TypeVar("E")


class Ok(NamedTuple, Generic[T]):
    """Successful outcome; narrow before accessing its borrowed payload."""

    value: T


class Err(NamedTuple, Generic[E]):
    """Recoverable rejection with a precise diagnostic payload."""

    error: E


Result: TypeAlias = Ok[T] | Err[E]
