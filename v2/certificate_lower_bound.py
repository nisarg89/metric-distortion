#!/usr/bin/env python3
"""Exact rational certificate for the 267/125 RSL lower bound.

For each of two rational separators, the checker reconstructs the three
functions in the manuscript, verifies their weighted combination directly for
the finite prefix, and certifies the entire infinite tail by an exact monotonic
upper bound.  It uses no floating-point arithmetic or numerical search.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F


TARGET_EXCESS_RATIO = F(71, 125)
DISTORTION_LOWER_BOUND = F(267, 125)


@dataclass(frozen=True)
class Separator:
    name: str
    weights: tuple[F, F, F]
    states: tuple[tuple[F, F], tuple[F, F]]
    cutoff: int
    tail_margin: F


SEPARATORS = (
    Separator(
        "paper separator",
        (F(45697, 250000), F(24337, 500000), F(384269, 500000)),
        ((F(1, 100), F(37, 50)), (F(1, 50), F(3, 4))),
        17,
        F(1, 15000),
    ),
    Separator(
        "independent separator",
        (
            F(184772969, 1000000000),
            F(766862450, 1000000000),
            F(48364581, 1000000000),
        ),
        ((F(1, 60), F(3, 4)), (F(1, 20), F(47, 60))),
        40,
        F(0),
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def boundary_function(degree: int) -> F:
    """Return b_d=(d+1)c^d-1 from the manuscript."""

    return (degree + 1) * TARGET_EXCESS_RATIO**degree - 1


def internal_function(degree: int, a: F, p: F) -> F:
    """Return i_d(a,p) from the manuscript."""

    return p * (
        (TARGET_EXCESS_RATIO - a) * (1 - p**degree)
        - a ** (degree + 1)
        - 1
        + (1 + a - p) ** (degree + 1)
        + p
    )


def combined_function(degree: int, separator: Separator) -> F:
    weights = separator.weights
    states = separator.states
    return (
        weights[0] * boundary_function(degree)
        + weights[1] * internal_function(degree, *states[0])
        + weights[2] * internal_function(degree, *states[1])
    )


def constant_term(separator: Separator) -> F:
    return -separator.weights[0] + sum(
        weight * p * (TARGET_EXCESS_RATIO - a + p - 1)
        for weight, (a, p) in zip(separator.weights[1:], separator.states)
    )


def expanded_function(degree: int, separator: Separator) -> F:
    """Return the exact expansion used for the infinite-tail argument."""

    weights = separator.weights
    states = separator.states
    return (
        constant_term(separator)
        + weights[0] * (degree + 1) * TARGET_EXCESS_RATIO**degree
        + sum(
            weight * p * (1 + a - p) ** (degree + 1)
            for weight, (a, p) in zip(weights[1:], states)
        )
        - sum(
            weight
            * (
                (TARGET_EXCESS_RATIO - a) * p ** (degree + 1)
                + p * a ** (degree + 1)
            )
            for weight, (a, p) in zip(weights[1:], states)
        )
    )


def tail_upper_bound(degree: int, separator: Separator) -> F:
    """Discard the nonpositive terms in the exact expansion of F_d."""

    common_base = 1 + separator.states[0][0] - separator.states[0][1]
    return (
        constant_term(separator)
        + separator.weights[0]
        * (degree + 1)
        * TARGET_EXCESS_RATIO**degree
        + sum(separator.weights[index] * separator.states[index - 1][1]
              for index in (1, 2))
        * common_base ** (degree + 1)
    )


def verify_separator(separator: Separator) -> None:
    weights = separator.weights
    states = separator.states
    cutoff = separator.cutoff

    require(all(weight > 0 for weight in weights), "weights must be positive")
    require(sum(weights) == 1, "weights must sum to one")
    require(cutoff >= 1, "tail cutoff must be positive")
    require(
        all(
            F(0) <= a < p < 1 and a < TARGET_EXCESS_RATIO
            for a, p in states
        ),
        "internal-state parameters are outside the required domain",
    )

    bases = tuple(1 + a - p for a, p in states)
    require(bases[0] == bases[1], "tail bases must agree")
    require(F(0) < bases[0] < 1, "common tail base must lie in (0,1)")
    require(constant_term(separator) < 0, "tail constant must be negative")

    direct_values = []
    for degree in range(1, cutoff + 1):
        direct = combined_function(degree, separator)
        require(
            direct == expanded_function(degree, separator),
            f"tail expansion fails at degree {degree}",
        )
        require(direct < 0, f"finite inequality fails at degree {degree}")
        direct_values.append(direct)

    # The omitted terms in expanded_function are nonpositive.  The first
    # retained exponential has successive ratio
    # c(d+2)/(d+1), which decreases with d; the second has ratio bases[0].
    # Hence this exact check proves that the tail upper bound decreases for
    # every d at least cutoff, rather than testing a finite sample of the tail.
    require(
        TARGET_EXCESS_RATIO * F(cutoff + 2, cutoff + 1) < 1,
        "first positive tail term is not decreasing",
    )
    tail_at_cutoff = tail_upper_bound(cutoff, separator)
    require(
        combined_function(cutoff, separator) <= tail_at_cutoff,
        "tail expression is not an upper bound",
    )
    require(
        tail_at_cutoff < -separator.tail_margin,
        "tail upper bound does not have the claimed negative margin",
    )

    print(
        f"{separator.name} verified:",
        f"degrees 1 through {cutoff},",
        "infinite tail certified,",
        f"largest finite value {max(direct_values)}",
    )


def main() -> None:
    require(
        1 + 2 * TARGET_EXCESS_RATIO == DISTORTION_LOWER_BOUND,
        "excess ratio and distortion bound are inconsistent",
    )
    for separator in SEPARATORS:
        verify_separator(separator)
    print(
        "Exact lower-bound certificate verified: "
        "inf_D dist(RSL(D)) >= 267/125."
    )


if __name__ == "__main__":
    main()
