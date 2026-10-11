"""Small checked outcomes for supported caller recovery."""

from typing import Generic, NamedTuple, TypeAlias, TypeVar

T = TypeVar("T")
E = TypeVar("E")


class Ok(NamedTuple, Generic[T]):
    """Represent a successful checked operation.

    Attributes:
        value: Typed success payload; narrow the Result with match before accessing
            it. Payload ownership follows the operation contract.
    """

    value: T


class Err(NamedTuple, Generic[E]):
    """Represent a recoverable rejected operation.

    Attributes:
        error: Typed failure payload explaining why the operation could not return a
            success value.
    """

    error: E


Result: TypeAlias = Ok[T] | Err[E]
