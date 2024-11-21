"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


def prod(iterable: Sequence[int]) -> int:
    """Calculate the product of all elements in an iterable.

    This function multiplies all integers in the given iterable and returns the result.

    Args:
    ----
        iterable: A sequence of integers whose product needs to be computed.

    Returns:
    -------
        The product of all integers in the iterable.

    """
    result = 1
    for x in iterable:
        result *= x
    return result


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply element-wise negation to the Tensor."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute gradients for the negation operation."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the element-wise inverse (1/x) of the Tensor."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute gradients for the inverse operation."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform element-wise addition of two Tensors."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for the addition operation."""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform element-wise multiplication of two Tensors."""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for element-wise multiplication."""
        t1, t2 = ctx.saved_tensors
        grad_t1 = t1.f.mul_zip(grad_output, t2)
        grad_t2 = t2.f.mul_zip(grad_output, t1)
        return grad_t1, grad_t2


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply the sigmoid function element-wise."""
        sigmoid = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sigmoid)
        return sigmoid

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute gradients for the sigmoid function."""
        (sigmoid,) = ctx.saved_tensors
        one_minus_sigmoid = sigmoid.f.add_zip(
            minitorch.Tensor.make([1.0], (1,), backend=sigmoid.backend),
            sigmoid.f.neg_map(sigmoid),
        )
        sigmoid_times_one_minus_sigmoid = sigmoid.f.mul_zip(sigmoid, one_minus_sigmoid)
        return sigmoid.f.mul_zip(sigmoid_times_one_minus_sigmoid, grad_output)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply the ReLU function element-wise."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute gradients for the ReLU function."""
        (t1,) = ctx.saved_tensors
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply the natural logarithm function element-wise."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute gradients for the logarithm function."""
        (t1,) = ctx.saved_tensors
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply the exponential function element-wise."""
        output = t1.f.exp_map(t1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute gradients for the exponential function."""
        (t1,) = ctx.saved_tensors
        grad_input = grad_output.f.mul_zip(t1, grad_output)
        return grad_input


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute the sum along a specified dimension."""
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute gradients for the sum operation."""
        (a_shape, dim) = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compare elements of two Tensors for less-than."""
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for the less-than comparison (no gradients)."""
        t1_shape, t2_shape = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compare elements of two Tensors for equality."""
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for the equality comparison (no gradients)."""
        t1_shape, t2_shape = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class GT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compare elements of two Tensors for greater-than."""
        return LT.apply(t2, t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for the greater-than comparison (no gradients)."""
        return zeros(grad_output.shape), zeros(grad_output.shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Check if elements of two Tensors are close within a tolerance."""
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of a Tensor."""
        ctx.save_for_backward(order)

        int_order = [int(i) for i in order._tensor._storage]

        return minitorch.Tensor(t1._tensor.permute(*int_order), backend=t1.backend)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute gradients for the permutation operation."""
        (order,) = ctx.saved_values

        rev_order = np.array([0 for _ in range(len(order._tensor._storage))])

        for i, j in enumerate(order._tensor._storage):
            rev_order[int(j)] = i

        return (
            minitorch.Tensor(
                v=grad_output._tensor.permute(*rev_order), backend=grad_output.backend
            ),
            0.0,
        )


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshape the Tensor without changing its data."""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute numerical gradients using the central difference method.

    This function estimates the gradient of a function `f` with respect to a
    specific argument using the central difference formula.

    Args:
    ----
        f: The function to differentiate.
        vals: The values at which to compute the gradient.
        arg: The index of the argument to differentiate with respect to.
        epsilon: A small perturbation value for numerical differentiation.
        ind: The index at which to perturb the argument.

    Returns:
    -------
        The estimated gradient at the specified index.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
