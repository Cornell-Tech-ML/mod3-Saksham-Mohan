from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the provided values, managing context and history.

        Args:
        ----
            vals (ScalarLike): Scalars or values convertible to scalars.

        Returns:
        -------
            Scalar: A new scalar resulting from applying the function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(float(v))

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for Log."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for Log."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for Inv."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for Inv."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x times y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for Mul."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for  Mul"""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for Neg."""
        ctx.save_for_backward(a)
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for Neg."""
        return -1 * d_output


class Sigmoid(ScalarFunction):
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

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for Sigmoid."""
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for Sigmoid."""
        (result,) = ctx.saved_values
        return operators.mul(
            d_output, operators.mul(result, operators.add(1.0, operators.neg(result)))
        )


class Relu(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for Relu."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for Relu."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for Exp."""
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for Exp."""
        (result,) = ctx.saved_values
        return operators.mul(d_output, result)


class Lt(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for Lt."""
        ctx.save_for_backward(a)
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for Lt."""
        return 0.0, 0.0


class Gt(ScalarFunction):
    """Greater than function $f(x, y) = x > y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for Gt."""
        ctx.save_for_backward(a)
        if not operators.lt(a, b) and not operators.eq(a, b):
            return 1.0
        else:
            return 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for Gt."""
        return 0.0, 0.0


class Eq(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for Eq."""
        ctx.save_for_backward(a)
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for Eq."""
        return 0.0, 0.0
