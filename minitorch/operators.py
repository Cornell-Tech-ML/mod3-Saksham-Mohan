"""Collection of the core mathematical operators used throughout the code base."""

from typing import Callable, Iterable, TypeVar

import math

# Generic Type for map, reduce and zipWith
T = TypeVar("T")


def mul(x: float, y: float) -> float:
    """Multiply `x` by `y`"""
    return x * y


def id(x: float) -> float:
    """Return `x` unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add `x` and `y`"""
    return x + y


def neg(x: float) -> float:
    """Return the negative value of `x`"""
    return -x


def lt(x: float, y: float) -> float:
    """Returns 1.0 if x is less than y, else returns 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Returns 1.0 if x is equal to y, else returns 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum between x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Returns true if the absolute difference between `x` and `y` is less than a small tolerance (1e-2)."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Compute the sigmoid of `x`, returning a value between 0 and 1.

    The sigmoid function is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))

    This implementation avoids overflow for large negative `x` by using the equivalent:
        sigmoid(x) = exp(x) / (1 + exp(x)) when x < 0.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
    float: Sigmoid of `x`.

    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return `x` if it is positive, else return `0`. This is the ReLU activation function."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Return the natural logarithm of `x`. Raises a ValueError if `x` is less than or equal to 0."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Return the exponential of `x`, i.e., e^x."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Return the gradient of the log function with respect to `x`, using the chain rule with upstream gradient `y`."""
    return y / x


def inv(x: float) -> float:
    """Return the multiplicative inverse (1/x) of `x`. Raises a ValueError if `x` is 0."""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Return the gradient of the inverse function with respect to `x`, using the chain rule with upstream gradient `y`."""
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Return the gradient of the ReLU function with respect to `x`. If `x` is positive, return `y`, else return 0."""
    return y if x > 0 else 0.0


def map(fn: Callable[[T], T]) -> Callable[[Iterable[T]], Iterable[T]]:
    """Apply a function `fn` to each element of an iterable.

    Args:
    ----
    fn (Callable[[T], T]): Function to apply to each element.

    Returns:
    -------
    Callable[[Iterable[T]], Iterable[T]]: A function that takes an iterable and
    returns an iterable with `fn` applied to each element.

    """

    def apply(ls: Iterable[T]) -> Iterable[T]:
        acc = []
        for elem in ls:
            acc.append(fn(elem))
        return acc

    return apply


def zipWith(
    fn: Callable[[T, T], T],
) -> Callable[[Iterable[T], Iterable[T]], Iterable[T]]:
    """Apply a function `fn` to corresponding elements of two iterables.

    Args:
    ----
    fn (Callable[[T, T], T]): Function to apply to pairs of elements.

    Returns:
    -------
    Callable[[Iterable[T], Iterable[T]], Iterable[T]]: A function that takes two
    iterables and returns an iterable of results.

    """

    def apply(ls_one: Iterable[T], ls_two: Iterable[T]) -> Iterable[T]:
        combined_acc = []
        for elem_one, elem_two in zip(ls_one, ls_two):
            combined_acc.append(fn(elem_one, elem_two))
        return combined_acc

    return apply


def reduce(fn: Callable[[T, T], T], init: T) -> Callable[[Iterable[T]], T]:
    """Reduce an iterable to a single value by applying a binary function `fn` iteratively.

    Args:
    ----
        fn (Callable[[T, T], T]): Reduction function.
        init (T): Initial value for reduction.

    Returns:
    -------
        Callable[[Iterable[T]], T]: A function that reduces the iterable to a single value.

    """

    def apply(ls: Iterable[T]) -> T:
        acc = init
        for elem in ls:
            acc = fn(acc, elem)

        return acc

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Return a list where each element is the negation of the corresponding element in `ls`.

    Args:
    ----
        ls (Iterable[float]): Input list of floats.

    Returns:
    -------
        Iterable[float]: List of negated floats.

    """
    return list(map(neg)(ls))


def addLists(ls_one: Iterable[float], ls_two: Iterable[float]) -> Iterable[float]:
    """Return a list where each element is the sum of corresponding elements in `ls_one` and `ls_two`.

    Args:
    ----
        ls_one (Iterable[float]): First input list.
        ls_two (Iterable[float]): Second input list.

    Returns:
    -------
        Iterable[float]: List of summed elements.

    """
    return list(zipWith(add)(ls_one, ls_two))


def sum(ls: Iterable[float]) -> float:
    """Return the sum of all elements in the list `ls`.

    Args:
    ----
        ls (Iterable[float]): Input list of floats.

    Returns:
    -------
        float: Sum of the elements.

    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Return the product of all elements in the list `ls`.

    Args:
    ----
        ls (Iterable[float]): Input list of floats.

    Returns:
    -------
        float: Product of the elements.

    """
    return reduce(mul, 1.0)(ls)
