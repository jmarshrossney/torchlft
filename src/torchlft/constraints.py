"""
Constraints
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import UserList
from collections.abc import Iterable
from math import pi as π
from typing import TYPE_CHECKING

import torch
import torch.linalg as LA

if TYPE_CHECKING:
    from torchlft.typing import *

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


class _Constraint(ABC):
    @abstractmethod
    def check(self, value: Tensor) -> bool:
        ...


class _MultiConstraint(UserList, _Constraint):
    def __init__(self, constraints: _Constraint) -> None:
        super().__init__(constraints)

        if not all([isinstance(c, _Constraint) for c in constraints]):
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

    def __add__(self, other: _Constraint) -> "_MultiConstraint":
        if not isinstance(other, Iterable):
            other = [other]
        return type(self)([*self, *other])

    def __iadd__(self, other: _Constraint) -> None:
        if not isinstance(other, Iterable):
            other = [other]
        super().__iadd__(other)


def _constraint_add(self, other: "_Constraint") -> _MultiConstraint:
    return _MultiConstraint([self, other])


_Constraint.__add__ = _constraint_add


class _Boolean(_Constraint):
    def check(self, t: Tensor):
        return ((t == 0) | (t == 1)).all()


class _Real(_Constraint):
    def check(self, t: Tensor):
        return t.isfinite().all()


class GreaterThan(_Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, t):
        return torch.all(self.lower_bound < t)


class GreaterThanEqualTo(_Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, t):
        return torch.all(self.lower_bound <= t)


class LessThan(_Constraint):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, t):
        return torch.all(t < self.upper_bound)


class LessThanEqualTo(_Constraint):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, t):
        return torch.all(t <= self.upper_bound)


class OpenInterval(_Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, t):
        return torch.all((self.lower_bound <= t) & (t <= self.upper_bound))


class HalfOpenInterval(_Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, t):
        return torch.all((self.lower_bound <= t) & (t < self.upper_bound))


class UnitNorm(_Constraint):
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


class UnitBall(_Constraint):
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


class SumToValue(_Constraint):
    def __init__(self, value: float, dim: int, rtol: float = 1e-5):
        self.value = value
        self.dim = dim
        self.rtol = rtol

    def __eq__(self, other):
        if isinstance(other, SumToOne):
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