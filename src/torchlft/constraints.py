"""
Constraints
"""
from abc import ABC, abstractmethod
from collections import UserList
from collections.abc import Iterable
from math import pi as π

import torch
import torch.linalg as LA

from torchlft.typing import Constraint, Tensor, BoolTensor

__all__ = [
    "boolean",
    "real",
    "positive",
    "nonnegative",
    "periodic",
    "UnitNorm",
    "SumToValue",
]


class ConstraintNotSatisfied(Exception):
    pass


class DomainError(Exception):
    pass


class _MultiConstraint(UserList):
    def __init__(self, constraints: Constraint) -> None:
        super().__init__(constraints)

        if not all([isinstance(c, Constraint) for c in constraints]):
            raise ValueError("Invalid argument - not a valid Constraint")
        self.constraints = self.data

    def check(self, value: Tensor) -> BoolTensor:
        results = {
            constraint: constraint.check(value)
            for constraint in self.constraints
        }
        if all(results):
            return True
        failed = [
            constraint
            for constraint, result in results.items()
            if result is False
        ]
        # TODO: raise or return False for broken constraints??
        # raise ConstraintNotSatisfied(

    # NOTE: in Python 3.11 we can replace with -> typing.Self
    def __add__(self, other: Constraint) -> Constraint:
        if not isinstance(other, Iterable):
            other = [other]
        return type(self)([*self, *other])

    def __iadd__(self, other: Constraint) -> None:
        if not isinstance(other, Iterable):
            other = [other]
        super().__iadd__(other)


def _constraint_add(self, other: Constraint) -> Constraint:
    return _MultiConstraint([self, other])


class _Boolean:
    def check(self, t: Tensor):
        return ((t == 0) | (t == 1)).all()


class _Real:
    def check(self, t: Tensor):
        return t.isfinite().all()


class GreaterThan:
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, t):
        return torch.all(self.lower_bound < t)


class GreaterThanEqualTo:
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, t):
        return torch.all(self.lower_bound <= t)


class LessThan:
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, t):
        return torch.all(t < self.upper_bound)


class LessThanEqualTo:
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, t):
        return torch.all(t <= self.upper_bound)


class OpenInterval:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, t):
        return torch.all((self.lower_bound <= t) & (t <= self.upper_bound))


class HalfOpenInterval:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, t):
        return torch.all((self.lower_bound <= t) & (t < self.upper_bound))


class UnitNorm:
    def __init__(self, dim: int, atol: float = 1e-5):
        self.dim = dim
        self.atol = atol

    def __eq__(self, other):
        if isinstance(other, UnitNorm):
            return (other.dim == self.dim) and (other.atol == self.atol)
        return False

    def check(self, t):
        return torch.allclose(
            LA.vector_norm(t, dim=self.dim) - 1,
            torch.zeros(1, dtype=t.dtype, device=t.device),
            atol=self.atol,
            rtol=0,
        )


class UnitBall:
    def __init__(self, dim: int, atol: float = 1e-5):
        self.dim = dim

    def __eq__(self, other):
        if isinstance(other, UnitNorm):
            return other.dim == self.dim
        return False

    def check(self, t):
        return torch.all(
            LA.vector_norm(t, dim=self.dim) <= 1,
        )


class SumToValue:
    def __init__(self, value: float, dim: int, rtol: float = 1e-5):
        self.value = value
        self.dim = dim
        self.rtol = rtol

    def __eq__(self, other):
        if isinstance(other, SumToValue):
            return (
                (other.value == self.value)
                and (other.dim == self.dim)
                and (other.rtol == self.rtol)
            )
        return False

    def check(self, t):
        return torch.allclose(
            t.sum(dim=self.dim) - self.value,
            torch.zeros(1, dtype=t.dtype, device=t.device),
            rtol=self.rtol,
        )


# TODO: stacking, spherical constraints on angles, positive definite covariance etc


boolean = _Boolean()
real = _Real()
positive = GreaterThan(0)
nonnegative = GreaterThanEqualTo(0)
periodic = HalfOpenInterval(0, 2 * π)
