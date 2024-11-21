from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_new_plus = list(vals)
    vals_new_minus = list(vals)

    vals_new_plus[arg] += epsilon
    vals_new_minus[arg] -= epsilon

    return (f(*vals_new_plus) - f(*vals_new_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for this variable.

        Args:
        ----
            x: The value to be accumulated to the derivative.

        Returns:
        -------
            None

        """
        ...

    @property
    def unique_id(self) -> int:
        """Get the unique identifier for this variable.

        Returns
        -------
            int: The unique identifier for this variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Determine if this Scalar is a leaf node in the computation graph.

        A leaf node is a Scalar that was created by the user and not by an operation
        from another Scalar. This means it has no associated function that generated it.

        Returns
        -------
            bool: True if the Scalar is a leaf node (i.e., it has no `last_fn` in its history),
                  otherwise False.

        """
        ...

    def is_constant(self) -> bool:
        """Check if the Scalar is a constant value with no history of operations.

        A constant Scalar is one that was directly instantiated with a value and
        does not have any associated computation graph or history of operations.

        Returns
        -------
            bool: True if the Scalar is constant (i.e., it has no history), otherwise False.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Retrieve the parent Variables that were used to compute this Scalar.

        The parent Variables are the inputs that were provided to the last function
        that produced this Scalar. This is used to trace back through the computation
        graph during backpropagation.

        Returns
        -------
            Iterable[Variable]: An iterable of parent Variables (Scalars) that are the
                            inputs to the operation that generated this Scalar.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to propagate the derivative through the computation graph.

        This method calculates the local derivatives of the Scalar with respect to its
        inputs using the chain rule, and pairs each input with its corresponding local
        derivative to propagate the gradient backward through the graph.

        Args:
        ----
            d_output (Any): The upstream gradient or derivative of the current Scalar.

        Returns:
        -------
        Iterable[Tuple[Variable, Any]]: An iterable of tuples, each containing an input
                                        Variable and its corresponding local derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    topo_order = []

    def visit(v: Variable) -> None:
        if v.unique_id not in visited:
            visited.add(v.unique_id)
            if not v.is_constant():
                for parent in v.parents:
                    visit(parent)
                topo_order.append(v)

    visit(variable)
    return topo_order[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
    variable: The right-most variable.
    deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
    None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    topo_order = list(topological_sort(variable))
    derivatives = {variable.unique_id: deriv}

    for var in topo_order:
        current_derivative = derivatives.get(var.unique_id, 0)

        if var.is_leaf():
            var.accumulate_derivative(current_derivative)
            continue

        for parent, parent_deriv in var.chain_rule(current_derivative):
            if parent.unique_id in derivatives:
                derivatives[parent.unique_id] += parent_deriv
            else:
                derivatives[parent.unique_id] = parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve the saved tensors from the context.

        This property provides access to the tensors that were saved during the forward
        pass of a computation, which are stored in `saved_values`. These saved tensors
        are crucial for computing the backward pass, as they contain the necessary values
        for gradient calculations.

        Returns
        -------
        Tuple[Any, ...]: A tuple of saved values (tensors) that were stored during
                         the forward pass of the operation.

        """
        return self.saved_values
