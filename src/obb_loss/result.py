"""Small typed outcomes and tensor metadata errors."""

from enum import Enum
from typing import Generic, NamedTuple, TypeAlias, TypeVar

T = TypeVar("T")
E = TypeVar("E")


class Ok(NamedTuple, Generic[T]):
    """Return a successful value.

    Attributes:
        value: Loss tensor with its autograd connection preserved.
    """

    value: T


class Err(NamedTuple, Generic[E]):
    """Return a recoverable metadata mismatch.

    Attributes:
        error: Typed reason the requested loss cannot be evaluated.
    """

    error: E


Result: TypeAlias = Ok[T] | Err[E]


class ErrorKind(Enum):
    """Distinguish shape, floating dtype and device mismatches."""

    SHAPE = "shape"
    DTYPE = "dtype"
    DEVICE = "device"


class InputError(NamedTuple):
    """Identify an invalid tensor contract.

    Attributes:
        kind: Metadata category rejected before loss evaluation.
        field: Input field or pair relationship that failed.
        detail: Required metadata contract in readable form.
    """

    kind: ErrorKind
    field: str
    detail: str
