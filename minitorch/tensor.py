"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

from .tensor_functions import (
    Add,
    All,
    Copy,
    EQ,
    Exp,
    Inv,
    IsClose,
    GT,
    Log,
    LT,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


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


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Set whether the Tensor should require gradients.

        This method sets up the Tensor to track operations for automatic differentiation
        if the input `x` is `True`.

        Args:
        ----
        x: A boolean indicating whether this Tensor requires gradients.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if the Tensor requires gradients.

        Returns
        -------
        True if the Tensor is set to track gradients, otherwise False.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a new Tensor filled with zeros.

        This method creates a new Tensor with the specified shape or with the same
        shape as the current Tensor if no shape is provided, and fills it with zeros.

        Args:
        ----
        shape: An optional shape for the new Tensor. If not provided, the shape
               of the current Tensor is used.

        Returns:
        -------
        A new Tensor filled with zeros.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the Tensor is a constant (not part of a computation graph).

        Returns
        -------
        True if the Tensor is constant (does not have any history), otherwise False.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent Tensors in the computation graph.

        Returns
        -------
        An iterable containing the parent Tensors in the computation graph.

        Raises
        ------
        AssertionError: If the Tensor does not have a history.

        """
        assert self.history is not None
        return self.history.inputs

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return int(prod(self._tensor.shape))

    @property
    def dims(self) -> int:
        """Return the number of dimensions (axes) in the tensor."""
        return len(self._tensor.shape)

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute gradients for each input variable.

        This method computes the gradients of the loss with respect to each
        input variable using the chain rule, based on the current Tensor's history.

        Args:
        ----
        d_output: The gradient of the output with respect to the loss.

        Returns:
        -------
        An iterable of tuples containing each input variable and its corresponding gradient.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Perform backpropagation on the Tensor.

        This method computes the gradients for all the Tensors in the computation graph
        leading to the current Tensor.

        Args:
        ----
        grad_output: An optional Tensor representing the gradient of the output.
                     If not provided, it is assumed to be a scalar with a gradient of 1.0.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __radd__(self, b: TensorLike) -> Tensor:
        return Add.apply(self._ensure_tensor(b), self)

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, Neg.apply(self._ensure_tensor(b)))

    def __rsub__(self, b: TensorLike) -> Tensor:
        return Add.apply(Neg.apply(self._ensure_tensor(b)), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return GT.apply(self, self._ensure_tensor(b))

    def __rmul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), self)

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Check if all elements along a specified dimension are true.

        This method checks whether all elements in the Tensor are non-zero (or true),
        optionally along a specified dimension.

        Args:
        ----
        dim: An optional integer specifying the dimension along which to perform the check.
             If not provided, the check is performed on the entire Tensor.

        Returns:
        -------
        A Tensor containing the result of the `all` operation.

        """
        if dim is None:
            return All.apply(self)
        else:
            return All.apply(self, Tensor.make([dim], (1,), backend=self.backend))

    # def is_close(self, b: TensorLike, tol: float = 1e-5) -> Tensor:

    #     return IsClose.apply(self, self._ensure_tensor(b), self._ensure_tensor(tol))
    def is_close(self, y: Tensor) -> Tensor:
        """Check if elements of two Tensors are close within a tolerance.

        This method compares each element of the current Tensor with the corresponding
        element in the Tensor `y` to check if they are close, within a specified tolerance.

        Args:
        ----
            y: A Tensor to compare with the current Tensor.

        Returns:
        -------
            A Tensor containing the result of the comparison.

        """
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Apply the sigmoid function element-wise.

        This method applies the sigmoid activation function to each element in the Tensor.

        Returns
        -------
        A new Tensor with the sigmoid function applied to each element.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Apply the ReLU (Rectified Linear Unit) function element-wise.

        This method applies the ReLU activation function to each element in the Tensor.

        Returns
        -------
        A new Tensor with the ReLU function applied to each element.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Apply the natural logarithm function element-wise.

        This method applies the natural logarithm (log base e) function to each element in the Tensor.

        Returns
        -------
        A new Tensor with the natural logarithm function applied to each element.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Apply the exponential function element-wise.

        This method applies the exponential function (e^x) to each element in the Tensor.

        Returns
        -------
        A new Tensor with the exponential function applied to each element.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Compute the sum of all elements along a specified dimension.

        This method computes the sum of all elements in the Tensor, optionally
        along a specified dimension.

        Args:
        ----
        dim: An optional integer specifying the dimension along which to sum.
             If not provided, the sum is computed over all elements.

        Returns:
        -------
        A Tensor containing the sum of elements.

        """
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean of all elements along a specified dimension.

        This method computes the mean of all elements in the Tensor, optionally
        along a specified dimension.

        Args:
        ----
        dim: An optional integer specifying the dimension along which to compute the mean.
             If not provided, the mean is computed over all elements.

        Returns:
        -------
        A Tensor containing the mean of elements.

        """
        total_sum = self.sum(dim)
        count = prod(self.shape) if dim is None else self.shape[dim]
        return total_sum / count

    def permute(self, *order: int) -> Tensor:
        """Permute the dimensions of the Tensor.

        This method rearranges the dimensions of the Tensor according to the specified order.

        Args:
        ----
        order: A sequence of integers specifying the new order of dimensions.

        Returns:
        -------
        A new Tensor with permuted dimensions.

        """
        order_tensor = Tensor.make(list(order), (len(order),), backend=self.backend)
        return Permute.apply(self, order_tensor)

    def view(self, *shape: int) -> Tensor:
        """Reshape the Tensor without changing its underlying data.

        This method reshapes the Tensor to a new shape without changing the underlying data.

        Args:
        ----
        shape: A sequence of integers specifying the new shape.

        Returns:
        -------
        A new Tensor with the specified shape.

        """
        return View.apply(
            self, Tensor.make(list(shape), (len(shape),), backend=self.backend)
        )

    def zero_grad_(self) -> None:
        """Reset the gradients of the Tensor.

        This method sets the gradient of the Tensor to `None`, effectively zeroing
        the accumulated gradients.
        """
        self.grad = None

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape
