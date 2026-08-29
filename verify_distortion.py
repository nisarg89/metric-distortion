#!/usr/bin/env python3
"""Exact verification of a rational candidate in Appendix B.

With no argument, this verifies the candidate printed in the paper.  An
optional JSON file may supply the same paper-facing data: the draw
distribution, rule mixture, target polynomial, dual-family parameters, and
verification domain.  Numerical search output is never trusted as a proof;
all polynomials are reconstructed here and checked using exact arithmetic.
"""

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction as Q
from math import comb
from pathlib import Path


# Sparse bivariate power polynomials.  The variables are (x,y) in the
# dual-feasibility checks and (q,x) in the outer-disjunction check.
ZERO, ONE = {}, {(0, 0): Q(1)}
FIRST, SECOND = {(1, 0): Q(1)}, {(0, 1): Q(1)}
X, Y = FIRST, SECOND
Q_OUT, X_OUT = FIRST, SECOND
R = {(0, 0): Q(1), (1, 0): Q(-1)}  # r=1-x


def add(*polynomials):
    result = {}
    for polynomial in polynomials:
        for exponent, coefficient in polynomial.items():
            result[exponent] = result.get(exponent, Q(0)) + coefficient
    return {exponent: coefficient for exponent, coefficient in result.items()
            if coefficient}


def scale(polynomial, scalar):
    return {exponent: scalar * coefficient
            for exponent, coefficient in polynomial.items()
            if scalar * coefficient}


def multiply(*polynomials):
    result = ONE
    for polynomial in polynomials:
        product = {}
        for (i, j), left in result.items():
            for (k, ell), right in polynomial.items():
                exponent = (i + k, j + ell)
                product[exponent] = product.get(exponent, Q(0)) + left * right
        result = {exponent: coefficient for exponent, coefficient in product.items()
                  if coefficient}
    return result


def power(polynomial, degree):
    result, base = ONE, polynomial
    while degree:
        if degree & 1:
            result = multiply(result, base)
        base = multiply(base, base)
        degree //= 2
    return result


def derivative(polynomial, coordinate):
    result = {}
    for exponent, coefficient in polynomial.items():
        if exponent[coordinate]:
            target = list(exponent)
            target[coordinate] -= 1
            result[tuple(target)] = coefficient * exponent[coordinate]
    return result


def substitute_second(polynomial, value):
    result = {}
    for (i, j), coefficient in polynomial.items():
        result[(i, 0)] = result.get((i, 0), Q(0)) + coefficient * value**j
    return {exponent: coefficient for exponent, coefficient in result.items()
            if coefficient}


def divide_variable(polynomial, coordinate):
    assert all(exponent[coordinate] for exponent in polynomial)
    result = {}
    for exponent, coefficient in polynomial.items():
        target = list(exponent)
        target[coordinate] -= 1
        result[tuple(target)] = coefficient
    return result


def divide_linear_second(polynomial, root):
    """Divide exactly by SECOND-root, separately in each FIRST column."""
    first_degree = max(i for i, _ in polynomial)
    second_degree = max(j for _, j in polynomial)
    result = {}
    for i in range(first_degree + 1):
        coefficients = [polynomial.get((i, j), Q(0))
                        for j in range(second_degree + 1)]
        quotient = [Q(0)] * second_degree
        quotient[-1] = coefficients[-1]
        for j in range(second_degree - 1, 0, -1):
            quotient[j - 1] = coefficients[j] + root * quotient[j]
        assert coefficients[0] + root * quotient[0] == 0
        for j, coefficient in enumerate(quotient):
            if coefficient:
                result[(i, j)] = coefficient
    return result


def divide_linear_first(polynomial, root):
    transposed = {(j, i): coefficient
                  for (i, j), coefficient in polynomial.items()}
    quotient = divide_linear_second(transposed, root)
    return {(j, i): coefficient for (i, j), coefficient in quotient.items()}


def divide_one_minus(polynomial, coordinate):
    """Divide exactly by 1-FIRST or 1-SECOND."""
    other, groups = 1 - coordinate, {}
    for exponent, coefficient in polynomial.items():
        groups.setdefault(exponent[other], {})[exponent[coordinate]] = coefficient
    result = {}
    for fixed, coefficients in groups.items():
        previous = Q(0)
        for degree in range(max(coefficients)):
            quotient = coefficients.get(degree, Q(0)) + previous
            exponent = [0, 0]
            exponent[coordinate], exponent[other] = degree, fixed
            if quotient:
                result[tuple(exponent)] = quotient
            previous = quotient
        assert coefficients.get(max(coefficients), Q(0)) + previous == 0
    return result


def as_outer_x(polynomial):
    """Regard a polynomial in x as a polynomial in the outer variables (q,x)."""
    assert all(j == 0 for _, j in polynomial)
    return {(0, i): coefficient for (i, _), coefficient in polynomial.items()}


def evaluate(polynomial, first, second):
    return sum(coefficient * first**i * second**j
               for (i, j), coefficient in polynomial.items())


DEFAULT_INPUT = {
    "schema": "metric-distortion-candidate-v1",
    "draw_distribution": {
        "1": "423239/1000000",
        "12": "471105/1000000",
        "13": "105656/1000000",
    },
    "rule_mixture": {
        "alpha_integrated_veto": "7941/50000",
        "beta_maximal_lottery": "0",
        "gamma_random_size_stable_lottery": "42059/50000",
        "C_cut": "6641/10000",
    },
    "target_polynomial_T": {
        "1": "33205/42059",
        "2": "-7941/42059",
    },
    "dual_tuple_parameters": {
        "active_y": ["80529/100000", "1"],
        "tangency_t": [f"{numerator}/100" for numerator in range(15, 61)],
    },
    "verification_domain": {
        "q": ["0", "1"],
        "x": ["0", "1"],
        "family_2_x_min": "1/5",
        "x_1_q": ["135516253/182000000", "1"],
    },
}


def rational(value):
    """Parse an integer or a rational string, rejecting inexact JSON floats."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"expected an integer or rational string, got {value!r}")
    return Q(value)


@dataclass(frozen=True)
class VerificationInput:
    draw_distribution: dict
    alpha: Q
    beta: Q
    gamma: Q
    C_cut: Q
    target_coefficients: dict
    active_y: tuple
    tangency_t: tuple
    q_domain: tuple
    x_domain: tuple
    family_2_x_min: Q
    x_1_q_domain: tuple

    @classmethod
    def from_mapping(cls, data):
        if data.get("schema") != "metric-distortion-candidate-v1":
            raise ValueError("unsupported or missing candidate schema")
        mixture = data["rule_mixture"]
        dual = data["dual_tuple_parameters"]
        domain = data["verification_domain"]
        candidate = cls(
            {int(degree): rational(weight)
             for degree, weight in data["draw_distribution"].items()},
            rational(mixture["alpha_integrated_veto"]),
            rational(mixture["beta_maximal_lottery"]),
            rational(mixture["gamma_random_size_stable_lottery"]),
            rational(mixture["C_cut"]),
            {int(degree): rational(coefficient) for degree, coefficient
             in data["target_polynomial_T"].items()},
            tuple(map(rational, dual["active_y"])),
            tuple(map(rational, dual["tangency_t"])),
            tuple(map(rational, domain["q"])),
            tuple(map(rational, domain["x"])),
            rational(domain["family_2_x_min"]),
            tuple(map(rational, domain["x_1_q"])),
        )
        candidate.validate()
        return candidate

    def validate(self):
        if (not self.draw_distribution or 1 not in self.draw_distribution
                or any(degree < 1 or weight <= 0
                       for degree, weight in self.draw_distribution.items())
                or sum(self.draw_distribution.values()) != 1):
            raise ValueError(
                "draw_distribution must be a probability law on positive "
                "degrees containing degree 1")
        if min(self.alpha, self.beta, self.gamma, self.C_cut) < 0:
            raise ValueError("rule weights and C_cut must be nonnegative")
        if self.alpha + self.beta + self.gamma != 1:
            raise ValueError("the three rule weights must sum to one")
        if self.beta != 0 or self.gamma == 0:
            raise ValueError("this verifier implements the beta=0 target used in Appendix B")
        expected_target = {1: self.C_cut / self.gamma,
                           2: -self.alpha / self.gamma}
        if self.target_coefficients != expected_target:
            raise ValueError("target_polynomial_T is inconsistent with the rule mixture")
        if (len(self.active_y) != 2
                or any(not Q(0) <= value <= Q(1) for value in self.active_y)):
            raise ValueError("active_y must contain the two values for dual families 1 and 2")
        if (not self.tangency_t
                or tuple(sorted(set(self.tangency_t))) != self.tangency_t
                or any(not Q(0) < value < Q(1) for value in self.tangency_t)):
            raise ValueError("tangency_t must be a nonempty increasing list in (0,1)")
        pp = sum(weight / (degree + 1)
                 for degree, weight in self.draw_distribution.items())
        if (self.q_domain != (Q(0), Q(1))
                or self.x_domain != (Q(0), Q(1))
                or not Q(0) <= self.family_2_x_min <= Q(1)
                or self.x_1_q_domain != (1 - pp, Q(1))):
            raise ValueError("verification_domain must cover the Appendix B domain exactly")


def load_input(path=None):
    data = DEFAULT_INPUT if path is None else json.loads(Path(path).read_text())
    return VerificationInput.from_mapping(data)


# Appendix B.3: install the rational input before constructing polynomials.
def configure(candidate):
    global a, p, alpha, beta, gamma, C_cut, T
    global ACTIVE_Y, TANGENCY_T, Q_DOMAIN, X_DOMAIN, FAMILY_2_X_MIN
    global X_1_Q_DOMAIN, RX, FX, FR, ONE_MINUS_FX, L_X, H_OUT_0
    a = candidate.draw_distribution
    p = sum(a[degree] / (degree + 1) for degree in a)
    alpha, beta, gamma, C_cut = (
        candidate.alpha, candidate.beta, candidate.gamma, candidate.C_cut)
    T = {(degree, 0): coefficient for degree, coefficient
         in candidate.target_coefficients.items() if coefficient}
    ACTIVE_Y, TANGENCY_T = candidate.active_y, candidate.tangency_t
    Q_DOMAIN, X_DOMAIN = candidate.q_domain, candidate.x_domain
    FAMILY_2_X_MIN = candidate.family_2_x_min
    X_1_Q_DOMAIN = candidate.x_1_q_domain

    RX = R_x()
    FX, FR = f(X), f(R)
    ONE_MINUS_FX = add(ONE, scale(FX, -1))
    L_X = add({(0, 0): p}, scale(Psi(X), -1), scale(Psi(R), -1),
              scale(multiply(X, FR), -1))
    H_OUT_0 = add(T, scale(X_OUT, -(1 - p)))


def f(argument):
    return add(*(scale(power(argument, degree), weight)
                 for degree, weight in a.items()))


def f_prime(argument):
    return add(*(scale(power(argument, degree - 1), degree * weight)
                 for degree, weight in a.items()))


def Psi(argument):
    return add(*(scale(power(argument, degree + 1), weight / (degree + 1))
                 for degree, weight in a.items()))


# Appendix B.4, equations (B.1)--(B.3).
def F_x(b_1):
    return add(*(scale(multiply(power(X, k - 1), power(R, degree - k),
                                power(b_1, k)),
                       weight * Q(comb(degree, k), degree - k + 1))
                 for degree, weight in a.items()
                 for k in range(1, degree + 1)))


def G_x(b_0, y):
    return add(*(scale(multiply(
        power(X, k - 1), power(R, degree - k),
        add(multiply(power(b_0, k), power(y, degree - k + 1)),
            ONE, scale(power(y, degree - k + 1), -1))),
        weight * Q(comb(degree, k), degree - k + 1))
        for degree, weight in a.items() for k in range(1, degree + 1)))


def R_x():
    return add(*(scale(power(R, j), weight / (degree + 1))
                 for degree, weight in a.items() for j in range(degree)))


def W_x(y):
    return f(add(X, multiply(R, y)))


@dataclass(frozen=True)
class DualFamily:
    index: int
    x_min: Q
    H_feas: tuple
    H_out: dict
    outer_multiplier: dict  # positive factor multiplying T(q)-U(q,x;z_j)


def dual_family_1_or_2(index, y_j):
    """Construct the H^feas and H^out polynomials induced by z_j, j=1,2."""
    G0, A_x = G_x(ZERO, Y), G_x(ONE, ZERO)
    G0_y, W_y = derivative(G0, 1), derivative(W_x(Y), 1)
    Wj, Wj_y = substitute_second(W_x(Y), y_j), substitute_second(W_y, y_j)
    G0j, G0j_y = substitute_second(G0, y_j), substitute_second(G0_y, y_j)

    Delta_j = add(multiply(add(FX, scale(Wj, -1)), G0j_y),
                  scale(multiply(add(A_x, scale(G0j, -1)), Wj_y), -1))
    outer_multiplier_x = scale(divide_variable(Delta_j, 0), -1)  # -Delta_j/x
    H1 = divide_linear_first(divide_linear_first(outer_multiplier_x, 1), 1)
    H2, H3 = scale(G0j_y, -1), Wj_y

    # Cleared numerators of I_{j,0}, I_{j,1}, and the derivative at b_1=0.
    I0 = add(multiply(G0j_y, Wj), scale(multiply(Wj_y, G0j), -1),
             scale(multiply(G0j_y, W_x(Y)), -1), multiply(Wj_y, G0))
    f_r_over_r = add(*(scale(power(R, degree - 1), weight)
                       for degree, weight in a.items()))
    f_ry_over_r = add(*(scale(multiply(power(R, degree - 1), power(Y, degree)),
                                weight) for degree, weight in a.items()))
    x_multiplier = multiply(X, outer_multiplier_x)
    I1 = add(I0, multiply(Wj_y, f_ry_over_r), scale(x_multiplier, -1))
    H4 = add(scale(multiply(G0j_y, X), -a[1]), multiply(Wj_y, f_r_over_r),
             scale(x_multiplier, -1))
    H5 = divide_linear_second(divide_linear_second(I0, y_j), y_j)
    H6 = divide_variable(I1, 1)

    # The same clearing factor gives H^out_j from U(q,x;z_j).
    objective_at_q_zero = add(scale(G0j_y, -p), multiply(Wj_y, RX),
                              multiply(G0j_y, Wj),
                              scale(multiply(Wj_y, G0j), -1))
    u_numerator = add(multiply(G0j_y, Wj),
                      scale(multiply(Wj_y, G0j), -1))
    multiplier = as_outer_x(outer_multiplier_x)
    H_out = add(multiply(multiplier, T),
                scale(as_outer_x(objective_at_q_zero), -1),
                multiply(Q_OUT, as_outer_x(u_numerator)))
    return DualFamily(index, X_DOMAIN[0] if index == 1 else FAMILY_2_X_MIN,
                      (H1, H2, H3, H4, H5, H6), H_out, multiplier)


def dual_family_3():
    """Construct the three H^feas polynomials and H^out induced by z_3."""
    H1 = divide_one_minus(divide_variable(L_X, 0), 0)
    ry, transition = multiply(R, Y), add(X, multiply(R, Y))
    numerator = add({(0, 0): p}, scale(Psi(R), -1), Psi(ry),
                    scale(Psi(transition), -1))
    I30 = add(multiply(numerator, ONE_MINUS_FX),
              scale(multiply(L_X, add(ONE, scale(f(transition), -1))), -1))
    I31 = add(I30, multiply(X, ONE_MINUS_FX, add(f(ry), scale(FR, -1))))
    H2 = divide_one_minus(I30, 1)
    H3 = divide_one_minus(divide_variable(I31, 1), 1)

    multiplier_x = multiply(FR, ONE_MINUS_FX)
    objective_constant = add(scale(L_X, -(1 - p)),
                             multiply(X, R, RX, ONE_MINUS_FX))
    multiplier = as_outer_x(multiplier_x)
    H_out = add(multiply(multiplier, T),
                scale(as_outer_x(objective_constant), -1),
                scale(multiply(Q_OUT, as_outer_x(L_X)), -1))
    return DualFamily(3, X_DOMAIN[0], (H1, H2, H3), H_out, multiplier)


def dual_family_at_tangency(index, tau_j):
    """Construct the two H^feas polynomials and H^out induced by z_j."""
    x_tau = scale(X, tau_j)
    Z_j = add(multiply(L_X, f_prime(x_tau)),
              multiply(ONE_MINUS_FX,
                       add(f(add(R, x_tau)), scale(f(x_tau), -1))))
    H1 = divide_one_minus(Z_j, 0)
    H2 = divide_one_minus(divide_variable(
        add(Z_j, scale(multiply(ONE_MINUS_FX, FR), -1)), 0), 0)

    L_over_x = divide_variable(L_X, 0)
    u_numerator = add(Z_j, scale(multiply(ONE_MINUS_FX, FR), -1),
                      scale(L_over_x, -1))
    v_numerator = add(scale(Z_j, tau_j),
                      scale(multiply(L_over_x, f(x_tau)), -1),
                      scale(multiply(R, ONE_MINUS_FX,
                                     F_x(scale(ONE, tau_j))), -1))
    fixed_numerator = add(scale(L_over_x, p),
                          multiply(R, ONE_MINUS_FX, RX))
    dual_numerator = multiply(X_OUT, add(
        as_outer_x(fixed_numerator),
        multiply(add(ONE, scale(Q_OUT, -1)), as_outer_x(u_numerator)),
        multiply(Q_OUT, as_outer_x(v_numerator))))
    H_out = divide_one_minus(
        add(multiply(T, as_outer_x(Z_j)), scale(dual_numerator, -1)), 1)
    return DualFamily(index, X_DOMAIN[0], (H1, H2), H_out, as_outer_x(H1))


def construct_dual_families():
    families = [dual_family_1_or_2(index, y_j)
                for index, y_j in enumerate(ACTIVE_Y, start=1)]
    families.append(dual_family_3())
    families.extend(dual_family_at_tangency(index, tau_j)
                    for index, tau_j in enumerate(TANGENCY_T, start=4))
    return families


# Appendix B.5: exact tensor-product Bernstein coefficients on a rational box.
def bernstein_1d(coefficients, left, right):
    degree, width = len(coefficients) - 1, right - left
    shifted = [Q(0)] * (degree + 1)
    for k, coefficient in enumerate(coefficients):
        for j in range(k + 1):
            shifted[j] += (coefficient * comb(k, j) * left**(k - j)
                           * width**j)
    return [sum(shifted[k] * Q(comb(i, k), comb(degree, k))
                for k in range(i + 1)) for i in range(degree + 1)]


def bernstein_minimum(polynomial, first_left, first_right,
                      second_left, second_right):
    first_degree = max((i for i, _ in polynomial), default=0)
    second_degree = max((j for _, j in polynomial), default=0)
    matrix = [[polynomial.get((i, j), Q(0))
               for j in range(second_degree + 1)]
              for i in range(first_degree + 1)]
    converted = [[Q(0)] * (second_degree + 1)
                 for _ in range(first_degree + 1)]
    for j in range(second_degree + 1):
        column = bernstein_1d([matrix[i][j] for i in range(first_degree + 1)],
                              first_left, first_right)
        for i, coefficient in enumerate(column):
            converted[i][j] = coefficient
    return min(coefficient for row in converted
               for coefficient in bernstein_1d(row, second_left, second_right))


def certify_polynomial(polynomial, box, strict=False):
    """Subdivide at rational midpoints until Bernstein coefficients certify a sign."""
    pending = [(box, 0)]
    while pending:
        current, depth = pending.pop()
        minimum = bernstein_minimum(polynomial, *current)
        if minimum > 0 if strict else minimum >= 0:
            continue
        assert depth < 64, current
        a0, a1, b0, b1 = current
        if a1 - a0 >= b1 - b0:
            midpoint = (a0 + a1) / 2
            children = ((a0, midpoint, b0, b1), (midpoint, a1, b0, b1))
        else:
            midpoint = (b0 + b1) / 2
            children = ((a0, a1, b0, midpoint), (a0, a1, midpoint, b1))
        pending.extend((child, depth + 1) for child in children)


def verify_dual_feasibility(families):
    """Verify the H^feas polynomials in the order and on the domains of B.4."""
    for family in families[:2]:
        for index, polynomial in enumerate(family.H_feas, start=1):
            domain = ((family.x_min, X_DOMAIN[1], Q(0), Q(1)) if index >= 5
                      else (family.x_min, X_DOMAIN[1], Q(0), Q(0)))
            certify_polynomial(polynomial, domain, strict=(index == 1))

    family = families[2]
    certify_polynomial(family.H_feas[0], (*X_DOMAIN, Q(0), Q(0)))
    for polynomial in family.H_feas[1:]:
        certify_polynomial(polynomial, (*X_DOMAIN, Q(0), Q(1)))

    for family in families[3:]:
        certify_polynomial(family.H_feas[0], (*X_DOMAIN, Q(0), Q(0)),
                           strict=True)
        certify_polynomial(family.H_feas[1], (*X_DOMAIN, Q(0), Q(0)))


@dataclass(frozen=True)
class OuterCandidate:
    index: int
    H_out: dict
    multiplier: dict


def verify_outer_disjunction(initial_boxes, candidates_for_box):
    """Verify the finite disjunction from B.5 by rational midpoint subdivision."""
    pending = [(box, 0) for box in initial_boxes]
    while pending:
        box, depth = pending.pop()
        q0, q1, x0, x1 = box
        q_midpoint, x_midpoint = (q0 + q1) / 2, (x0 + x1) / 2
        candidates = candidates_for_box(box)

        # A negative midpoint value rules out nonnegativity on this box.
        candidates = [candidate for candidate in candidates
                      if evaluate(candidate.H_out, q_midpoint, x_midpoint) >= 0]
        # This ordering changes only runtime: the quotient is T(q)-U(q,x;z_j).
        candidates.sort(
            key=lambda candidate: evaluate(candidate.H_out, q_midpoint, x_midpoint)
            / evaluate(candidate.multiplier, q_midpoint, x_midpoint),
            reverse=True)
        if any(bernstein_minimum(candidate.H_out, *box) >= 0
               for candidate in candidates):
            continue

        assert depth < 64, box
        if q1 - q0 >= x1 - x0:
            midpoint = (q0 + q1) / 2
            children = ((q0, midpoint, x0, x1), (midpoint, q1, x0, x1))
        else:
            midpoint = (x0 + x1) / 2
            children = ((q0, q1, x0, midpoint), (q0, q1, midpoint, x1))
        pending.extend((child, depth + 1) for child in children)


def verify_candidate_coverage(families):
    """Verify exactly the interior and x=1 disjunctions displayed in B.5."""
    c3 = OuterCandidate(0, H_OUT_0, ONE)
    dual_candidates = [OuterCandidate(family.index, family.H_out,
                                      family.outer_multiplier)
                       for family in families]

    def interior_candidates(box):
        x0 = box[2]
        return [c3] + [candidate for candidate in dual_candidates
                       if candidate.index != 2 or x0 >= FAMILY_2_X_MIN]

    # Splitting first at the z_2 threshold makes its eligibility exact.
    verify_outer_disjunction(
        [(Q_DOMAIN[0], Q_DOMAIN[1], X_DOMAIN[0], FAMILY_2_X_MIN),
         (Q_DOMAIN[0], Q_DOMAIN[1], FAMILY_2_X_MIN, X_DOMAIN[1])],
        interior_candidates)

    boundary_candidates = [c3] + [candidate for candidate in dual_candidates
                                  if candidate.index >= 4]
    verify_outer_disjunction(
        [(*X_1_Q_DOMAIN, X_DOMAIN[1], X_DOMAIN[1])],
        lambda _box: boundary_candidates)


def main(input_path=None):
    configure(load_input(input_path))
    families = construct_dual_families()
    assert [family.index for family in families] == list(range(1, len(families) + 1))
    verify_dual_feasibility(families)
    verify_candidate_coverage(families)
    print("All polynomial nonnegativity checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", help="candidate JSON emitted by the search")
    main(parser.parse_args().candidate)
