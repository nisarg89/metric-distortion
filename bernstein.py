#!/usr/bin/env python3
"""Exact multivariate polynomial and Bernstein-certificate utilities.

The module is deliberately independent of metric distortion.  A verification
problem consists of rational boxes and finite alternatives of polynomial sign
conditions.  At every point of a box, all polynomials in at least one
alternative must have the requested sign.  Adaptive verification proves this
by subdivision and the sufficient Bernstein-coefficient test.

Static certificates store one exact Bernstein control net at the root of each
problem and a fixed binary subdivision tree.  Checking a certificate expands
every root control net back into the power basis, follows every recorded split,
and checks the selected alternative at every leaf.  Thus certificate checking
does not search for a subdivision.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb
from typing import Any, Mapping, Sequence


SCHEMA = "metric-distortion-polynomial-problem-v1"
CERTIFICATE_SCHEMA = "metric-distortion-bernstein-certificate-v1"

Exponent = tuple[int, ...]
Polynomial = dict[Exponent, Fraction]
Box = tuple[tuple[Fraction, Fraction], ...]


def rational(value: Any) -> Fraction:
    """Parse an exact JSON rational and reject booleans and floating point."""

    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"expected an integer or rational string, got {value!r}")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid rational value {value!r}") from error


def fraction_text(value: Fraction | int) -> str:
    value = Fraction(value)
    return (str(value.numerator) if value.denominator == 1
            else f"{value.numerator}/{value.denominator}")


def clean(polynomial: Mapping[Exponent, Fraction | int]) -> Polynomial:
    return {
        tuple(exponent): Fraction(coefficient)
        for exponent, coefficient in polynomial.items()
        if coefficient
    }


def zero() -> Polynomial:
    return {}


def constant(dimension: int, value: Fraction | int) -> Polynomial:
    value = Fraction(value)
    return {} if not value else {(0,) * dimension: value}


def variable(dimension: int, axis: int) -> Polynomial:
    if not 0 <= axis < dimension:
        raise ValueError("variable axis lies outside the polynomial dimension")
    exponent = [0] * dimension
    exponent[axis] = 1
    return {tuple(exponent): Fraction(1)}


def dimension(polynomial: Polynomial, default: int | None = None) -> int:
    dimensions = {len(exponent) for exponent in polynomial}
    if len(dimensions) > 1:
        raise ValueError("polynomial contains exponents of different dimensions")
    if dimensions:
        answer = dimensions.pop()
        if default is not None and answer != default:
            raise ValueError("polynomial dimension does not match its variables")
        return answer
    if default is None:
        raise ValueError("the zero polynomial needs an explicit dimension")
    return default


def add(*polynomials: Polynomial) -> Polynomial:
    answer: dict[Exponent, Fraction] = defaultdict(Fraction)
    expected: int | None = None
    for polynomial in polynomials:
        if polynomial:
            current = dimension(polynomial)
            if expected is None:
                expected = current
            elif current != expected:
                raise ValueError("cannot add polynomials of different dimensions")
        for exponent, coefficient in polynomial.items():
            answer[exponent] += coefficient
    return clean(answer)


def scale(polynomial: Polynomial, scalar: Fraction | int) -> Polynomial:
    scalar = Fraction(scalar)
    return {} if not scalar else {
        exponent: scalar * coefficient
        for exponent, coefficient in polynomial.items()
        if coefficient
    }


def multiply(first: Polynomial, second: Polynomial,
             *, dimension_hint: int | None = None) -> Polynomial:
    if not first or not second:
        return {}
    first_dimension = dimension(first, dimension_hint)
    second_dimension = dimension(second, dimension_hint)
    if first_dimension != second_dimension:
        raise ValueError("cannot multiply polynomials of different dimensions")
    answer: dict[Exponent, Fraction] = defaultdict(Fraction)
    for exponent, coefficient in first.items():
        for other_exponent, other_coefficient in second.items():
            target = tuple(left + right for left, right in
                           zip(exponent, other_exponent))
            answer[target] += coefficient * other_coefficient
    return clean(answer)


def product_polynomials(*polynomials: Polynomial,
                        dimension_hint: int) -> Polynomial:
    answer = constant(dimension_hint, 1)
    for polynomial in polynomials:
        answer = multiply(answer, polynomial, dimension_hint=dimension_hint)
    return answer


def power(polynomial: Polynomial, exponent: int,
          *, dimension_hint: int) -> Polynomial:
    if exponent < 0:
        raise ValueError("polynomial powers must be nonnegative")
    answer = constant(dimension_hint, 1)
    base = polynomial
    remaining = exponent
    while remaining:
        if remaining & 1:
            answer = multiply(answer, base, dimension_hint=dimension_hint)
        base = multiply(base, base, dimension_hint=dimension_hint)
        remaining //= 2
    return answer


def compose_univariate(coefficients: Sequence[Fraction | int],
                       argument: Polynomial, *, dimension_hint: int) -> Polynomial:
    answer = zero()
    for coefficient in reversed(coefficients):
        answer = add(
            multiply(answer, argument, dimension_hint=dimension_hint),
            constant(dimension_hint, Fraction(coefficient)),
        )
    return answer


def derivative(polynomial: Polynomial, axis: int) -> Polynomial:
    if polynomial and not 0 <= axis < dimension(polynomial):
        raise ValueError("derivative axis lies outside the polynomial dimension")
    answer: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        if exponent[axis]:
            target = list(exponent)
            target[axis] -= 1
            answer[tuple(target)] = coefficient * exponent[axis]
    return clean(answer)


def evaluate(polynomial: Polynomial,
             point: Sequence[Fraction | int]) -> Fraction:
    values = tuple(Fraction(value) for value in point)
    if polynomial:
        dimension(polynomial, len(values))
    return sum(
        coefficient * _monomial(values, exponent)
        for exponent, coefficient in polynomial.items()
    )


def _monomial(values: Sequence[Fraction], exponent: Sequence[int]) -> Fraction:
    answer = Fraction(1)
    for value, degree in zip(values, exponent):
        answer *= value ** degree
    return answer


def divide_variable(polynomial: Polynomial, axis: int) -> Polynomial:
    answer: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        if exponent[axis] < 1:
            raise ValueError("polynomial is not divisible by the selected variable")
        target = list(exponent)
        target[axis] -= 1
        answer[tuple(target)] = coefficient
    return clean(answer)


def _univariate_divide_one_minus(coefficients: Sequence[Fraction]) -> list[Fraction]:
    coefficients = list(coefficients)
    while len(coefficients) > 1 and not coefficients[-1]:
        coefficients.pop()
    if len(coefficients) < 2:
        raise ValueError("constant polynomial is not divisible by 1-x")
    quotient = [Fraction(0)] * (len(coefficients) - 1)
    quotient[0] = coefficients[0]
    for index in range(1, len(quotient)):
        quotient[index] = coefficients[index] + quotient[index - 1]
    if coefficients[-1] != -quotient[-1]:
        raise ValueError("polynomial is not divisible by 1-x")
    while len(quotient) > 1 and not quotient[-1]:
        quotient.pop()
    return quotient


def divide_one_minus_variable(polynomial: Polynomial, axis: int) -> Polynomial:
    if not polynomial:
        return {}
    polynomial_dimension = dimension(polynomial)
    if not 0 <= axis < polynomial_dimension:
        raise ValueError("division axis lies outside the polynomial dimension")
    other_axes = tuple(index for index in range(polynomial_dimension)
                       if index != axis)
    groups: dict[tuple[int, ...], dict[int, Fraction]] = defaultdict(dict)
    for exponent, coefficient in polynomial.items():
        key = tuple(exponent[index] for index in other_axes)
        groups[key][exponent[axis]] = coefficient
    answer: Polynomial = {}
    for other_exponents, terms in groups.items():
        degree = max(terms)
        coefficients = [terms.get(index, Fraction(0))
                        for index in range(degree + 1)]
        quotient = _univariate_divide_one_minus(coefficients)
        for index, coefficient in enumerate(quotient):
            if coefficient:
                exponent = [0] * polynomial_dimension
                exponent[axis] = index
                for other_axis, other_exponent in zip(other_axes, other_exponents):
                    exponent[other_axis] = other_exponent
                answer[tuple(exponent)] = coefficient
    return clean(answer)


def polynomial_record(polynomial: Polynomial, variables: Sequence[str],
                      *, relation: str = "nonnegative") -> dict[str, Any]:
    variable_tuple = tuple(variables)
    if polynomial:
        dimension(polynomial, len(variable_tuple))
    if relation not in {"nonnegative", "positive"}:
        raise ValueError("relation must be 'nonnegative' or 'positive'")
    return {
        "variables": list(variable_tuple),
        "relation": relation,
        "terms": [
            [list(exponent), fraction_text(coefficient)]
            for exponent, coefficient in sorted(polynomial.items())
        ],
    }


def polynomial_from_record(record: Mapping[str, Any],
                           expected_variables: Sequence[str] | None = None,
                           ) -> tuple[tuple[str, ...], str, Polynomial]:
    if not isinstance(record, Mapping):
        raise TypeError("each polynomial record must be a JSON object")
    variables_raw = record.get("variables")
    if (not isinstance(variables_raw, list)
            or any(not isinstance(name, str) or not name for name in variables_raw)
            or len(set(variables_raw)) != len(variables_raw)):
        raise ValueError("polynomial variables must be distinct nonempty strings")
    variables = tuple(variables_raw)
    if expected_variables is not None and variables != tuple(expected_variables):
        raise ValueError("polynomial variables do not match the verification problem")
    relation = record.get("relation")
    if relation not in {"nonnegative", "positive"}:
        raise ValueError("polynomial relation must be 'nonnegative' or 'positive'")
    terms = record.get("terms")
    if not isinstance(terms, list):
        raise ValueError("polynomial terms must be a list")
    answer: Polynomial = {}
    previous: Exponent | None = None
    for term in terms:
        if not isinstance(term, list) or len(term) != 2:
            raise ValueError("each polynomial term must be [exponents, coefficient]")
        exponent_raw, coefficient_raw = term
        if (not isinstance(exponent_raw, list)
                or len(exponent_raw) != len(variables)
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 0 for value in exponent_raw)):
            raise ValueError("polynomial exponents must be nonnegative integers")
        exponent = tuple(exponent_raw)
        if previous is not None and exponent <= previous:
            raise ValueError("polynomial terms must be sorted with no duplicates")
        previous = exponent
        coefficient = rational(coefficient_raw)
        if not coefficient:
            raise ValueError("zero coefficients must be omitted")
        answer[exponent] = coefficient
    return variables, relation, answer


@dataclass(frozen=True)
class ParsedPolynomial:
    name: str
    variables: tuple[str, ...]
    relation: str
    coefficients: Polynomial


@dataclass(frozen=True)
class VerificationProblem:
    name: str
    variables: tuple[str, ...]
    domain: Box
    alternatives: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class PolynomialProblem:
    raw: Mapping[str, Any]
    metadata: Mapping[str, Any]
    polynomials: Mapping[str, ParsedPolynomial]
    problems: tuple[VerificationProblem, ...]


def parse_problem(data: Mapping[str, Any]) -> PolynomialProblem:
    if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
        raise ValueError(f"candidate schema must be {SCHEMA!r}")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON object")
    polynomial_records = data.get("polynomials")
    if not isinstance(polynomial_records, Mapping) or not polynomial_records:
        raise ValueError("polynomials must be a nonempty JSON object")
    parsed_polynomials: dict[str, ParsedPolynomial] = {}
    for name, record in polynomial_records.items():
        if not isinstance(name, str) or not name:
            raise ValueError("polynomial names must be nonempty strings")
        variables, relation, coefficients = polynomial_from_record(record)
        parsed_polynomials[name] = ParsedPolynomial(
            name, variables, relation, coefficients
        )

    problem_records = data.get("problems")
    if not isinstance(problem_records, list) or not problem_records:
        raise ValueError("problems must be a nonempty list")
    parsed_problems: list[VerificationProblem] = []
    problem_names: set[str] = set()
    referenced: set[str] = set()
    for record in problem_records:
        if not isinstance(record, Mapping):
            raise ValueError("each verification problem must be a JSON object")
        name = record.get("name")
        if not isinstance(name, str) or not name or name in problem_names:
            raise ValueError("verification problem names must be distinct strings")
        problem_names.add(name)
        variables_raw = record.get("variables")
        if (not isinstance(variables_raw, list)
                or any(not isinstance(variable_name, str) or not variable_name
                       for variable_name in variables_raw)
                or len(set(variables_raw)) != len(variables_raw)):
            raise ValueError("problem variables must be distinct nonempty strings")
        variables = tuple(variables_raw)
        domain_raw = record.get("domain")
        if not isinstance(domain_raw, list) or len(domain_raw) != len(variables):
            raise ValueError("problem domain must give one interval per variable")
        domain: list[tuple[Fraction, Fraction]] = []
        for interval in domain_raw:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("each domain interval must have two endpoints")
            left, right = map(rational, interval)
            if left > right:
                raise ValueError("domain interval has its endpoints reversed")
            domain.append((left, right))
        alternatives_raw = record.get("alternatives")
        if not isinstance(alternatives_raw, list) or not alternatives_raw:
            raise ValueError("each problem must contain at least one alternative")
        alternatives: list[tuple[str, ...]] = []
        for alternative_raw in alternatives_raw:
            if (not isinstance(alternative_raw, list) or not alternative_raw
                    or any(not isinstance(item, str) for item in alternative_raw)):
                raise ValueError("each alternative must be a nonempty polynomial list")
            alternative = tuple(alternative_raw)
            if len(set(alternative)) != len(alternative):
                raise ValueError("an alternative repeats a polynomial")
            for polynomial_name in alternative:
                if polynomial_name not in parsed_polynomials:
                    raise ValueError(f"unknown polynomial {polynomial_name!r}")
                polynomial = parsed_polynomials[polynomial_name]
                if polynomial.variables != variables:
                    raise ValueError(
                        f"polynomial {polynomial_name!r} has the wrong variables"
                    )
                referenced.add(polynomial_name)
            alternatives.append(alternative)
        parsed_problems.append(VerificationProblem(
            name, variables, tuple(domain), tuple(alternatives)
        ))
    unused = set(parsed_polynomials) - referenced
    if unused:
        raise ValueError(f"unreferenced polynomials: {sorted(unused)}")
    return PolynomialProblem(
        raw=data,
        metadata=metadata,
        polynomials=parsed_polynomials,
        problems=tuple(parsed_problems),
    )


def _affine_to_unit(polynomial: Polynomial, box: Box) -> Polynomial:
    polynomial_dimension = len(box)
    if polynomial:
        dimension(polynomial, polynomial_dimension)
    answer: dict[Exponent, Fraction] = defaultdict(Fraction)
    for exponent, coefficient in polynomial.items():
        ranges = [range(degree + 1) for degree in exponent]
        for target in product(*ranges):
            value = coefficient
            for degree, target_degree, (left, right) in zip(
                    exponent, target, box):
                value *= (comb(degree, target_degree)
                          * left ** (degree - target_degree)
                          * (right - left) ** target_degree)
            answer[target] += value
    return clean(answer)


def _degrees(polynomial: Polynomial, polynomial_dimension: int) -> tuple[int, ...]:
    return tuple(
        max((exponent[axis] for exponent in polynomial), default=0)
        for axis in range(polynomial_dimension)
    )


def _group_by_other_axes(polynomial: Polynomial, axis: int,
                         polynomial_dimension: int,
                         ) -> dict[tuple[int, ...], dict[int, Fraction]]:
    other_axes = tuple(index for index in range(polynomial_dimension)
                       if index != axis)
    groups: dict[tuple[int, ...], dict[int, Fraction]] = defaultdict(dict)
    for exponent, coefficient in polynomial.items():
        key = tuple(exponent[index] for index in other_axes)
        groups[key][exponent[axis]] = coefficient
    return groups


def power_to_bernstein_on_box(polynomial: Polynomial, box: Box,
                              ) -> tuple[tuple[int, ...], Polynomial]:
    """Return the exact tensor-product Bernstein control net on ``box``."""

    polynomial_dimension = len(box)
    transformed = _affine_to_unit(polynomial, box)
    degrees = _degrees(transformed, polynomial_dimension)
    current = transformed
    for axis, degree in enumerate(degrees):
        groups = _group_by_other_axes(current, axis, polynomial_dimension)
        other_axes = tuple(index for index in range(polynomial_dimension)
                           if index != axis)
        converted: Polynomial = {}
        for other_exponents, terms in groups.items():
            for index in range(degree + 1):
                coefficient = sum(
                    terms.get(power_index, Fraction(0))
                    * Fraction(comb(index, power_index),
                               comb(degree, power_index))
                    for power_index in range(index + 1)
                )
                if coefficient:
                    exponent = [0] * polynomial_dimension
                    exponent[axis] = index
                    for other_axis, other_exponent in zip(
                            other_axes, other_exponents):
                        exponent[other_axis] = other_exponent
                    converted[tuple(exponent)] = coefficient
        current = converted
    return degrees, clean(current)


def bernstein_to_power(control: Polynomial, degrees: Sequence[int]) -> Polynomial:
    """Expand an exact tensor-product Bernstein control net on the unit box."""

    degree_tuple = tuple(degrees)
    polynomial_dimension = len(degree_tuple)
    if control:
        dimension(control, polynomial_dimension)
    current = control
    for axis, degree in enumerate(degree_tuple):
        groups = _group_by_other_axes(current, axis, polynomial_dimension)
        other_axes = tuple(index for index in range(polynomial_dimension)
                           if index != axis)
        converted: Polynomial = {}
        for other_exponents, terms in groups.items():
            for power_index in range(degree + 1):
                coefficient = comb(degree, power_index) * sum(
                    terms.get(bernstein_index, Fraction(0))
                    * comb(power_index, bernstein_index)
                    * (-1) ** (power_index - bernstein_index)
                    for bernstein_index in range(power_index + 1)
                )
                if coefficient:
                    exponent = [0] * polynomial_dimension
                    exponent[axis] = power_index
                    for other_axis, other_exponent in zip(
                            other_axes, other_exponents):
                        exponent[other_axis] = other_exponent
                    converted[tuple(exponent)] = coefficient
        current = converted
    return clean(current)


def _control_size(degrees: Sequence[int]) -> int:
    answer = 1
    for degree in degrees:
        answer *= degree + 1
    return answer


def control_minimum(control: Polynomial,
                    degrees: Sequence[int]) -> Fraction:
    if len(control) < _control_size(degrees):
        return min(Fraction(0), min(control.values(), default=Fraction(0)))
    return min(control.values(), default=Fraction(0))


def control_satisfies(control: Polynomial, degrees: Sequence[int],
                      relation: str) -> bool:
    minimum = control_minimum(control, degrees)
    if relation == "nonnegative":
        return minimum >= 0
    if relation == "positive":
        return len(control) == _control_size(degrees) and minimum > 0
    raise ValueError(f"unknown polynomial relation {relation!r}")


def split_control(control: Polynomial, degrees: Sequence[int], axis: int,
                  ) -> tuple[Polynomial, Polynomial]:
    """Split a tensor-product control net at the selected axis midpoint."""

    degree_tuple = tuple(degrees)
    polynomial_dimension = len(degree_tuple)
    if not 0 <= axis < polynomial_dimension:
        raise ValueError("split axis lies outside the control-net dimension")
    other_axes = tuple(index for index in range(polynomial_dimension)
                       if index != axis)
    left: Polynomial = {}
    right: Polynomial = {}
    other_ranges = [range(degree_tuple[index] + 1) for index in other_axes]
    for other_indices in product(*other_ranges):
        line: list[Fraction] = []
        for index in range(degree_tuple[axis] + 1):
            exponent = [0] * polynomial_dimension
            exponent[axis] = index
            for other_axis, other_index in zip(other_axes, other_indices):
                exponent[other_axis] = other_index
            line.append(control.get(tuple(exponent), Fraction(0)))
        levels = [line]
        while len(levels[-1]) > 1:
            levels.append([
                (levels[-1][index] + levels[-1][index + 1]) / 2
                for index in range(len(levels[-1]) - 1)
            ])
        degree = degree_tuple[axis]
        left_line = [levels[index][0] for index in range(degree + 1)]
        right_line = [levels[degree - index][index]
                      for index in range(degree + 1)]
        for index in range(degree + 1):
            exponent = [0] * polynomial_dimension
            exponent[axis] = index
            for other_axis, other_index in zip(other_axes, other_indices):
                exponent[other_axis] = other_index
            target = tuple(exponent)
            if left_line[index]:
                left[target] = left_line[index]
            if right_line[index]:
                right[target] = right_line[index]
    return clean(left), clean(right)


def _control_record(degrees: Sequence[int], control: Polynomial) -> dict[str, Any]:
    return {
        "degrees": list(degrees),
        "coefficients": [
            [list(exponent), fraction_text(coefficient)]
            for exponent, coefficient in sorted(control.items())
        ],
    }


def _control_from_record(record: Mapping[str, Any], dimension_value: int,
                         ) -> tuple[tuple[int, ...], Polynomial]:
    if not isinstance(record, Mapping):
        raise ValueError("control-net record must be a JSON object")
    degrees_raw = record.get("degrees")
    if (not isinstance(degrees_raw, list)
            or len(degrees_raw) != dimension_value
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in degrees_raw)):
        raise ValueError("control-net degrees are invalid")
    degrees = tuple(degrees_raw)
    coefficients_raw = record.get("coefficients")
    if not isinstance(coefficients_raw, list):
        raise ValueError("control-net coefficients must be a list")
    control: Polynomial = {}
    previous: Exponent | None = None
    for entry in coefficients_raw:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("each control coefficient must have index and value")
        exponent_raw, coefficient_raw = entry
        if (not isinstance(exponent_raw, list)
                or len(exponent_raw) != dimension_value
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 0 or value > degree
                       for value, degree in zip(exponent_raw, degrees))):
            raise ValueError("control-net index lies outside its degree range")
        exponent = tuple(exponent_raw)
        if previous is not None and exponent <= previous:
            raise ValueError("control coefficients must be sorted without duplicates")
        previous = exponent
        coefficient = rational(coefficient_raw)
        if not coefficient:
            raise ValueError("zero control coefficients must be omitted")
        control[exponent] = coefficient
    return degrees, control


@dataclass
class CertificateCounts:
    nodes: int = 0
    leaves: int = 0
    maximum_depth: int = 0


def _accepted_alternative(problem: VerificationProblem,
                          parsed: PolynomialProblem,
                          controls: Mapping[str, tuple[tuple[int, ...], Polynomial]],
                          ) -> int | None:
    for index, alternative in enumerate(problem.alternatives):
        if all(control_satisfies(
            controls[name][1], controls[name][0],
            parsed.polynomials[name].relation,
        ) for name in alternative):
            return index
    return None


def _split_all_controls(
    controls: Mapping[str, tuple[tuple[int, ...], Polynomial]], axis: int,
) -> tuple[
    dict[str, tuple[tuple[int, ...], Polynomial]],
    dict[str, tuple[tuple[int, ...], Polynomial]],
]:
    left: dict[str, tuple[tuple[int, ...], Polynomial]] = {}
    right: dict[str, tuple[tuple[int, ...], Polynomial]] = {}
    for name, (degrees, control) in controls.items():
        left_control, right_control = split_control(control, degrees, axis)
        left[name] = (degrees, left_control)
        right[name] = (degrees, right_control)
    return left, right


def _subdivide(
    problem: VerificationProblem,
    parsed: PolynomialProblem,
    controls: Mapping[str, tuple[tuple[int, ...], Polynomial]],
    box: Box,
    depth: int,
    maximum_depth: int,
    counts: CertificateCounts,
) -> dict[str, Any]:
    counts.nodes += 1
    counts.maximum_depth = max(counts.maximum_depth, depth)
    alternative = _accepted_alternative(problem, parsed, controls)
    if alternative is not None:
        counts.leaves += 1
        return {"alternative": alternative}
    if depth >= maximum_depth:
        minima = {
            name: fraction_text(control_minimum(control, degrees))
            for name, (degrees, control) in controls.items()
        }
        raise AssertionError(
            f"maximum subdivision depth reached in {problem.name!r}: {minima}"
        )
    widths = tuple(right - left for left, right in box)
    if not widths or max(widths) <= 0:
        raise AssertionError(
            f"no alternative is certified on the point domain {problem.name!r}"
        )
    axis = max(range(len(widths)), key=lambda index: widths[index])
    left_controls, right_controls = _split_all_controls(controls, axis)
    midpoint = sum(box[axis], Fraction(0)) / 2
    left_box = list(box)
    right_box = list(box)
    left_box[axis] = (box[axis][0], midpoint)
    right_box[axis] = (midpoint, box[axis][1])
    return {
        "split": axis,
        "children": [
            _subdivide(problem, parsed, left_controls, tuple(left_box),
                       depth + 1, maximum_depth, counts),
            _subdivide(problem, parsed, right_controls, tuple(right_box),
                       depth + 1, maximum_depth, counts),
        ],
    }


def find_certificate(data: Mapping[str, Any], maximum_depth: int = 64,
                     ) -> tuple[dict[str, Any], dict[str, CertificateCounts]]:
    """Find an exact Bernstein subdivision certificate for every problem."""

    if maximum_depth < 0:
        raise ValueError("maximum_depth must be nonnegative")
    parsed = parse_problem(data)
    certificate_problems: list[dict[str, Any]] = []
    all_counts: dict[str, CertificateCounts] = {}
    for problem in parsed.problems:
        names = sorted({name for alternative in problem.alternatives
                        for name in alternative})
        controls = {
            name: power_to_bernstein_on_box(
                parsed.polynomials[name].coefficients, problem.domain
            )
            for name in names
        }
        counts = CertificateCounts()
        tree = _subdivide(
            problem, parsed, controls, problem.domain, 0, maximum_depth, counts
        )
        certificate_problems.append({
            "name": problem.name,
            "root_controls": {
                name: _control_record(*controls[name]) for name in names
            },
            "tree": tree,
        })
        all_counts[problem.name] = counts
    return {
        "schema": CERTIFICATE_SCHEMA,
        "problems": certificate_problems,
    }, all_counts


def _verify_tree(
    node: Mapping[str, Any],
    problem: VerificationProblem,
    parsed: PolynomialProblem,
    controls: Mapping[str, tuple[tuple[int, ...], Polynomial]],
    box: Box,
    depth: int,
    counts: CertificateCounts,
) -> None:
    if not isinstance(node, Mapping):
        raise ValueError("every certificate-tree node must be a JSON object")
    counts.nodes += 1
    counts.maximum_depth = max(counts.maximum_depth, depth)
    if set(node) == {"alternative"}:
        alternative = node["alternative"]
        if (isinstance(alternative, bool) or not isinstance(alternative, int)
                or not 0 <= alternative < len(problem.alternatives)):
            raise ValueError("certificate leaf selects an invalid alternative")
        for name in problem.alternatives[alternative]:
            degrees, control = controls[name]
            relation = parsed.polynomials[name].relation
            if not control_satisfies(control, degrees, relation):
                raise AssertionError(
                    f"leaf of {problem.name!r} does not certify {name!r}"
                )
        counts.leaves += 1
        return
    if set(node) != {"split", "children"}:
        raise ValueError("certificate node must be exactly a leaf or a split")
    axis = node["split"]
    children = node["children"]
    if (isinstance(axis, bool) or not isinstance(axis, int)
            or not 0 <= axis < len(problem.variables)):
        raise ValueError("certificate split has an invalid axis")
    if not isinstance(children, list) or len(children) != 2:
        raise ValueError("certificate split must have exactly two children")
    if box[axis][0] == box[axis][1]:
        raise ValueError("certificate attempts to split a zero-width interval")
    left_controls, right_controls = _split_all_controls(controls, axis)
    midpoint = sum(box[axis], Fraction(0)) / 2
    left_box = list(box)
    right_box = list(box)
    left_box[axis] = (box[axis][0], midpoint)
    right_box[axis] = (midpoint, box[axis][1])
    _verify_tree(children[0], problem, parsed, left_controls, tuple(left_box),
                 depth + 1, counts)
    _verify_tree(children[1], problem, parsed, right_controls, tuple(right_box),
                 depth + 1, counts)


def verify_certificate(data: Mapping[str, Any],
                       certificate: Mapping[str, Any],
                       ) -> dict[str, CertificateCounts]:
    """Verify a fixed certificate, including coverage and polynomial identity."""

    parsed = parse_problem(data)
    if (not isinstance(certificate, Mapping)
            or certificate.get("schema") != CERTIFICATE_SCHEMA):
        raise ValueError(f"certificate schema must be {CERTIFICATE_SCHEMA!r}")
    records = certificate.get("problems")
    if not isinstance(records, list) or len(records) != len(parsed.problems):
        raise ValueError("certificate has the wrong number of problems")
    all_counts: dict[str, CertificateCounts] = {}
    for problem, record in zip(parsed.problems, records):
        if not isinstance(record, Mapping) or record.get("name") != problem.name:
            raise ValueError("certificate problems are missing or out of order")
        root_records = record.get("root_controls")
        if not isinstance(root_records, Mapping):
            raise ValueError("certificate problem omits its root control nets")
        names = sorted({name for alternative in problem.alternatives
                        for name in alternative})
        if set(root_records) != set(names):
            raise ValueError("certificate root controls do not match the alternatives")
        controls: dict[str, tuple[tuple[int, ...], Polynomial]] = {}
        for name in names:
            degrees, control = _control_from_record(
                root_records[name], len(problem.variables)
            )
            transformed = _affine_to_unit(
                parsed.polynomials[name].coefficients, problem.domain
            )
            expanded = bernstein_to_power(control, degrees)
            if expanded != transformed:
                raise AssertionError(
                    f"Bernstein coefficients do not represent polynomial {name!r}"
                )
            controls[name] = (degrees, control)
        counts = CertificateCounts()
        _verify_tree(
            record.get("tree"), problem, parsed, controls, problem.domain, 0, counts
        )
        all_counts[problem.name] = counts
    return all_counts


def summarize_counts(counts: Mapping[str, CertificateCounts]) -> str:
    problem_count = len(counts)
    leaves = sum(record.leaves for record in counts.values())
    maximum_depth = max((record.maximum_depth for record in counts.values()),
                        default=0)
    return (f"{problem_count} polynomial problems, {leaves} accepted boxes, "
            f"maximum depth {maximum_depth}")
