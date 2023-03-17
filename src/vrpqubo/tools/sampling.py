"""
SM Harwood
16 March 2023

Simple utilities to help with sampling random variables
defined by algebraic expressions of simpler scipy distributions
"""
from numbers import Real
from typing import Union, Tuple
import numpy as np
from scipy.stats import rv_discrete, rv_continuous
# I feel like this will probably break...
from scipy.stats._distn_infrastructure import rv_discrete_frozen, rv_continuous_frozen

class SimpleSampler:
    """
    Base class to help with sampling algebraic expressions of independent random variables
    """
    def rvs(self, size: int):
        """Sample from underlying samplers/distributions"""
        raise NotImplementedError

    def __neg__(self):
        return NegatedSampler(self)

    def __add__(self, other):
        if isinstance(other, Real):
            other = ConstantSampler(other)
        return SumSampler((self, other))

    def __radd__(self, other):
        if isinstance(other, Real):
            other = ConstantSampler(other)
        return SumSampler((other, self))

    def __sub__(self, other):
        if isinstance(other, Real):
            other = ConstantSampler(-other)
            return SumSampler((self, other))
        return SumSampler((self, -other))

    def __rsub__(self, other):
        if isinstance(other, Real):
            other = ConstantSampler(other)
        return SumSampler((other, -self))

    def __mul__(self, other):
        if isinstance(other, Real):
            other = ConstantSampler(other)
        return ProductSampler((self, other))

    def __rmul__(self, other):
        if isinstance(other, Real):
            other = ConstantSampler(other)
        return ProductSampler((other, self))

    def __truediv__(self, other):
        return RatioSampler(self, other)

    def __rtruediv__(self, other):
        return RatioSampler(other, self)

# Types and Typing:
RV = Union[rv_discrete, rv_continuous, rv_discrete_frozen, rv_continuous_frozen]
Sampleable = Union[SimpleSampler, RV]
# unnecessary in python 3.10 and up
RVT = (rv_discrete, rv_continuous, rv_discrete_frozen, rv_continuous_frozen)
Sampleable_Type = (SimpleSampler, *RVT)

class WrapperSampler(SimpleSampler):
    """ Class to wrap a RV so that we can take advantage of operator algebra """
    def __init__(self, underlying: RV) -> None:
        self.underlying = underlying

    def rvs(self, size: int=1):
        return self.underlying.rvs(size)

    def mean(self) -> float:
        """Get mean of underlying distribution"""
        return self.underlying.mean()

class ConstantSampler(SimpleSampler):
    """ Class to sample a constant RV """
    def __init__(self, constant: Real) -> None:
        self.constant = constant

    def rvs(self, size: int=1):
        return self.constant*np.ones(size)

class NegatedSampler(SimpleSampler):
    """ Class to sample negated RV """
    def __init__(self, positive: Sampleable) -> None:
        self.positive = positive

    def rvs(self, size: int=1):
        return -self.positive.rvs(size)

class SumSampler(SimpleSampler):
    """ Class to sample sum of independent RVs """
    def __init__(self, summands: Tuple[Sampleable]) -> None:
        self.summands = summands

    def rvs(self, size: int=1):
        vals = [summand.rvs(size) for summand in self.summands]
        return np.sum(vals, axis=0)

class ProductSampler(SimpleSampler):
    """ Class to sample product of independent RVs """
    def __init__(self, multiplicands: Tuple[Sampleable]) -> None:
        self.multiplicands = multiplicands

    def rvs(self, size: int=1):
        vals = [multiplicand.rvs(size) for multiplicand in self.multiplicands]
        return np.prod(vals, axis=0)

class RatioSampler(SimpleSampler):
    """ Class to sample the ratio/quotient of independent RVs """
    def __init__(self,
            numerator: Union[Sampleable, Real],
            denominator: Sampleable
        ) -> None:
        if isinstance(numerator, Real):
            numerator = ConstantSampler(numerator)
        self.numerator = numerator
        self.denominator = denominator

    def rvs(self, size: int=1):
        numer = self.numerator.rvs(size)
        denom = self.denominator.rvs(size)
        return numer/denom
