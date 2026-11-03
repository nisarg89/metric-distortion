#!/usr/bin/env python3
r"""Numerically search for a sparse integer draw distribution.

This program implements the discovery process for the direct-potential bound.
It enumerates integer supports, numerically optimizes the probability weights
against a finite-grid longest-path problem, refines a shortlist on a finer
grid, and rationalizes the best candidate.  It then writes the exact polynomial
verification problems associated with that rational candidate.

Every optimization and ranking performed here is numerical and heuristic.  In
particular, the resulting objective is evidence from below for the optimum of
the sufficient potential framework, not an upper bound on distortion.  Only a
successful run of 'verify_distortion.py' supplies an exact continuum proof.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import optimize

from bernstein import (
    SCHEMA,
    add,
    compose_univariate,
    constant,
    divide_one_minus_variable,
    divide_variable,
    fraction_text,
    multiply,
    polynomial_record,
    scale,
    variable,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "candidate.json"


# ---------------------------------------------------------------------------
# Numerical fixed-distribution oracle.


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    answer = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if answer.ndim != 1 or not len(answer) or not np.all(np.isfinite(answer)):
        raise ValueError("weights must be a nonempty finite vector")
    total = float(answer.sum())
    if total <= 0:
        answer[:] = 1.0
        total = float(len(answer))
    answer /= total
    answer[-1] += 1.0 - float(answer.sum())
    return answer


def grid_endpoint(degrees: Sequence[int], weights: Sequence[float],
                  state_count: int) -> tuple[float, float]:
    """Return the sampled longest-path endpoint and boundary maximizer."""

    degree_array = np.asarray(degrees, dtype=np.float64)
    weight_array = normalize_weights(weights)
    if len(degree_array) != len(weight_array):
        raise ValueError("degrees and weights must have equal length")
    if state_count < 2:
        raise ValueError("state_count must be at least 2")
    n = int(state_count)
    grid = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    activation = np.power(grid[:, None], degree_array[None, :]) @ weight_array
    primitive = (
        np.power(grid[:, None], (degree_array + 1.0)[None, :])
        @ (weight_array / (degree_array + 1.0))
    )
    mean = float(np.dot(weight_array, 1.0 / (degree_array + 1.0)))
    qbar = np.empty(n + 1, dtype=np.float64)
    qbar[0] = mean
    qbar[1:n] = (
        -np.expm1(np.log(grid[1:n, None]) * degree_array[None, :])
        @ (weight_array / (degree_array + 1.0))
    )
    qbar[n] = 0.0

    longest = np.full(n + 1, -np.inf, dtype=np.float64)
    longest[0] = 0.0
    for target in range(1, n + 1):
        best = float(longest[target - 1])
        for p_index in range(1, min(target, n - 1) + 1):
            sources = np.arange(p_index + 1, dtype=np.int64)
            shifted = sources + target - p_index
            mixed = (
                primitive[target] - primitive[shifted]
                - primitive[p_index] + primitive[sources]
            )
            edges = (
                grid[sources] - grid[p_index] + mixed / qbar[p_index]
            )
            best = max(best, float(np.max(longest[sources] + edges)))
        longest[target] = best

    boundary = grid + (mean - activation) / (1.0 - mean)
    totals = longest + boundary
    boundary_index = int(np.argmax(totals))
    endpoint = max(float(longest[n]), float(totals[boundary_index]))
    return endpoint, float(grid[boundary_index])


def continuous_boundary_maximizer(degrees: Sequence[int],
                                  weights: Sequence[float]) -> float:
    degree_array = np.asarray(degrees, dtype=np.float64)
    weight_array = normalize_weights(weights)
    mean = float(np.dot(weight_array, 1.0 / (degree_array + 1.0)))

    def negative_boundary(value: float) -> float:
        activation = float(np.dot(weight_array, np.power(value, degree_array)))
        return -(value + (mean - activation) / (1.0 - mean))

    answer = optimize.minimize_scalar(
        negative_boundary, bounds=(0.0, 1.0), method="bounded",
        options={"xatol": 1e-14},
    )
    return float(answer.x)


def continuous_boundary_endpoint(degrees: Sequence[int],
                                 weights: Sequence[float],
                                 ) -> tuple[float, float]:
    degree_array = np.asarray(degrees, dtype=np.float64)
    weight_array = normalize_weights(weights)
    maximizer = continuous_boundary_maximizer(degrees, weights)
    mean = float(np.dot(weight_array, 1.0 / (degree_array + 1.0)))
    activation = float(np.dot(
        weight_array, np.power(maximizer, degree_array)
    ))
    endpoint = maximizer + (mean - activation) / (1.0 - mean)
    return maximizer, endpoint


def candidate_starts(size: int, count: int, seed: int,
                     extra: Sequence[float] | None = None) -> list[np.ndarray]:
    starts: list[np.ndarray] = []

    def insert(weights: Sequence[float]) -> None:
        normalized = normalize_weights(weights)
        key = tuple(np.round(normalized, 14))
        if all(tuple(np.round(old, 14)) != key for old in starts):
            starts.append(normalized)

    insert([1.0 / size] * size)
    for index in range(size):
        insert([float(index == coordinate) for coordinate in range(size)])
    if extra is not None:
        insert(extra)
    rng = np.random.default_rng(seed)
    while len(starts) < count:
        insert(rng.dirichlet(np.ones(size)))
    return starts


def optimize_support(support: Sequence[int], state_count: int, seed: int,
                     initial_samples: int, local_starts: int,
                     maximum_iterations: int,
                     extra_start: Sequence[float] | None = None,
                     differential_evolution: bool = False,
                     differential_iterations: int = 60,
                     differential_population: int = 12,
                     ) -> dict[str, Any]:
    support = tuple(int(value) for value in support)
    if tuple(sorted(set(support))) != support or support[0] < 1:
        raise ValueError("support must be strictly increasing and positive")
    if len(support) == 1:
        endpoint, boundary = grid_endpoint(support, [1.0], state_count)
        return {
            "support": list(support), "weights": [1.0],
            "endpoint": endpoint, "boundary": boundary,
        }

    cache: dict[tuple[float, ...], float] = {}

    def objective(raw_weights: Sequence[float]) -> float:
        weights = normalize_weights(raw_weights)
        key = tuple(np.round(weights, 14))
        if key not in cache:
            cache[key] = grid_endpoint(support, weights, state_count)[0]
        return cache[key]

    starts = candidate_starts(
        len(support), max(initial_samples, local_starts), seed, extra_start
    )
    if differential_evolution:
        def from_logits(logits: Sequence[float]) -> np.ndarray:
            extended = np.append(np.asarray(logits, dtype=np.float64), 0.0)
            extended -= float(np.max(extended))
            return normalize_weights(np.exp(extended))

        global_answer = optimize.differential_evolution(
            lambda logits: objective(from_logits(logits)),
            bounds=[(-14.0, 14.0)] * (len(support) - 1),
            seed=seed, maxiter=differential_iterations,
            popsize=differential_population, polish=False, workers=1,
            updating="immediate",
            tol=1e-7,
        )
        starts.append(from_logits(global_answer.x))

    ranked = sorted(starts, key=lambda weights: (
        objective(weights), tuple(float(value) for value in weights)
    ))
    constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
    }
    for start in ranked[:local_starts]:
        answer = optimize.minimize(
            objective, start, method="SLSQP",
            bounds=[(0.0, 1.0)] * len(support),
            constraints=(constraint,),
            options={
                "maxiter": maximum_iterations,
                "ftol": 1e-11,
                "disp": False,
            },
        )
        objective(answer.x)
    best_key, best_endpoint = min(
        cache.items(), key=lambda item: (item[1], item[0])
    )
    weights = np.asarray(best_key, dtype=np.float64)
    endpoint, boundary = grid_endpoint(support, weights, state_count)
    return {
        "support": list(support),
        "weights": [float(value) for value in weights],
        "endpoint": float(max(endpoint, best_endpoint)),
        "boundary": boundary,
    }


def canonical_supports(maximum_degree: int,
                       maximum_support_size: int) -> list[tuple[int, ...]]:
    if maximum_degree < 1:
        raise ValueError("maximum_degree must be positive")
    if not 1 <= maximum_support_size <= maximum_degree:
        raise ValueError("maximum_support_size is outside the degree range")
    return [
        support
        for size in range(1, maximum_support_size + 1)
        for support in combinations(range(1, maximum_degree + 1), size)
    ]


def _coarse_job(arguments: tuple[Any, ...]) -> dict[str, Any]:
    return optimize_support(*arguments)


def report_progress(label: str, record: dict[str, Any], completed: int,
                    total: int, best_endpoint: float,
                    progress_every: int) -> float:
    endpoint = float(record["endpoint"])
    if endpoint < best_endpoint:
        print(
            f"{label}: new best after {completed}/{total}: "
            f"support={record['support']}, endpoint={endpoint:.12g}",
            flush=True,
        )
        best_endpoint = endpoint
    if completed == total or (progress_every and completed % progress_every == 0):
        print(f"{label}: completed {completed}/{total}", flush=True)
    return best_endpoint


def numerical_search(args: argparse.Namespace) -> dict[str, Any]:
    supports = canonical_supports(args.max_degree, args.max_support_size)
    jobs = [
        (
            support, args.grid_size, args.seed + index,
            args.initial_samples, args.local_starts, args.maxiter,
        )
        for index, support in enumerate(supports)
    ]
    if args.workers == 1:
        coarse_iterator = map(_coarse_job, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        coarse_iterator = executor.map(_coarse_job, jobs)
    coarse = []
    best_endpoint = math.inf
    try:
        for completed, record in enumerate(coarse_iterator, start=1):
            coarse.append(record)
            best_endpoint = report_progress(
                "coarse screen", record, completed, len(jobs), best_endpoint,
                args.progress_every,
            )
    finally:
        if executor is not None:
            executor.shutdown()
    coarse.sort(key=lambda record: (
        record["endpoint"], len(record["support"]), record["support"]
    ))
    shortlist = coarse[:min(args.shortlist, len(coarse))]

    refined: list[dict[str, Any]] = []
    best_endpoint = math.inf
    for rank, record in enumerate(shortlist):
        result = optimize_support(
            record["support"], args.refine_grid_size,
            args.seed + 100000 + rank,
            args.refine_initial_samples,
            args.refine_local_starts,
            args.refine_maxiter,
            extra_start=record["weights"],
        )
        refined.append(result)
        best_endpoint = report_progress(
            "shortlist refinement", result, rank + 1, len(shortlist),
            best_endpoint, args.progress_every,
        )
    refined.sort(key=lambda record: (
        record["endpoint"], len(record["support"]), record["support"]
    ))

    if args.global_refinement:
        global_candidates = refined[:min(args.global_shortlist, len(refined))]
        globally_refined = []
        best_endpoint = math.inf
        for rank, record in enumerate(global_candidates):
            result = optimize_support(
                record["support"], args.refine_grid_size,
                args.seed + 200000 + rank,
                args.global_initial_samples,
                args.global_local_starts,
                args.global_maxiter,
                extra_start=record["weights"],
                differential_evolution=True,
                differential_iterations=args.differential_iterations,
                differential_population=args.differential_population,
            )
            globally_refined.append(result)
            best_endpoint = report_progress(
                "global refinement", result, rank + 1,
                len(global_candidates), best_endpoint, args.progress_every,
            )
        finalists = globally_refined
    else:
        finalists = refined[:min(args.global_shortlist, len(refined))]

    final_records = []
    best_endpoint = math.inf
    for rank, record in enumerate(finalists):
        grid_value, _ = grid_endpoint(
            record["support"], record["weights"], args.final_grid_size
        )
        boundary_point, boundary_value = continuous_boundary_endpoint(
            record["support"], record["weights"]
        )
        result = dict(record)
        result["endpoint"] = max(float(record["endpoint"]), grid_value,
                                 boundary_value)
        result["final_grid_endpoint"] = grid_value
        result["continuous_boundary_endpoint"] = boundary_value
        result["continuous_boundary_maximizer"] = boundary_point
        final_records.append(result)
        best_endpoint = report_progress(
            "final reevaluation", result, rank + 1, len(finalists),
            best_endpoint, args.progress_every,
        )
    final_records.sort(key=lambda record: (
        record["endpoint"], len(record["support"]), record["support"]
    ))
    best = final_records[0]
    best["searched_support_count"] = len(supports)
    best["coarse_grid_size"] = args.grid_size
    best["refine_grid_size"] = args.refine_grid_size
    best["final_grid_size"] = args.final_grid_size
    best["refined_support_count"] = len(shortlist)
    best["globally_refined_support_count"] = (
        min(args.global_shortlist, len(refined)) if args.global_refinement else 0
    )
    return best


# ---------------------------------------------------------------------------
# Exact polynomial encoding for a rational threshold potential.


def uni_clean(coefficients: Sequence[F]) -> list[F]:
    answer = list(coefficients)
    while len(answer) > 1 and not answer[-1]:
        answer.pop()
    return answer


def uni_add(first: Sequence[F], second: Sequence[F]) -> list[F]:
    answer = [F(0)] * max(len(first), len(second))
    for index, coefficient in enumerate(first):
        answer[index] += coefficient
    for index, coefficient in enumerate(second):
        answer[index] += coefficient
    return uni_clean(answer)


def uni_scale(coefficients: Sequence[F], scalar: F) -> list[F]:
    return uni_clean([scalar * coefficient for coefficient in coefficients])


def uni_evaluate(coefficients: Sequence[F], value: F) -> F:
    answer = F(0)
    for coefficient in reversed(coefficients):
        answer = answer * value + coefficient
    return answer


def uni_derivative(coefficients: Sequence[F]) -> list[F]:
    return ([F(index) * coefficients[index]
             for index in range(1, len(coefficients))] or [F(0)])


def uni_divide_x(coefficients: Sequence[F]) -> list[F]:
    if coefficients[0]:
        raise ValueError("univariate polynomial is not divisible by x")
    return uni_clean(coefficients[1:])


def uni_divide_one_minus(coefficients: Sequence[F]) -> list[F]:
    coefficients = uni_clean(coefficients)
    if len(coefficients) < 2:
        raise ValueError("polynomial is not divisible by 1-x")
    quotient = [F(0)] * (len(coefficients) - 1)
    quotient[0] = coefficients[0]
    for index in range(1, len(quotient)):
        quotient[index] = coefficients[index] + quotient[index - 1]
    if coefficients[-1] != -quotient[-1]:
        raise ValueError("polynomial is not divisible by 1-x")
    return uni_clean(quotient)


def univariate_polynomial(coefficients: Sequence[F]) -> dict[tuple[int], F]:
    return {
        (degree,): coefficient
        for degree, coefficient in enumerate(coefficients)
        if coefficient
    }


def activation(degrees: Sequence[int], weights: Sequence[F]) -> list[F]:
    answer = [F(0)] * (max(degrees) + 1)
    for degree, weight in zip(degrees, weights):
        answer[degree] += weight
    return uni_clean(answer)


def primitive(coefficients: Sequence[F]) -> list[F]:
    return [F(0)] + [
        coefficient / (index + 1)
        for index, coefficient in enumerate(coefficients)
    ]


def potential(coefficients: Sequence[F], mean: F, endpoint: F) -> list[F]:
    delta = 1 - mean
    return uni_add(
        [endpoint - mean / delta, F(-1)],
        uni_scale(coefficients, 1 / delta),
    )


def build_cells(activation_coefficients: Sequence[F],
                primitive_coefficients: Sequence[F],
                potential_coefficients: Sequence[F],
                mean: F, threshold: F,
                ) -> tuple[dict[tuple[int, ...], F], ...]:
    dimension = 3
    one = constant(dimension, 1)
    x = variable(dimension, 0)
    y = variable(dimension, 1)
    z = variable(dimension, 2)
    t = threshold
    one_minus_t = 1 - t
    qfun = uni_add(
        [mean], uni_scale(uni_divide_x(primitive_coefficients), F(-1))
    )
    rfun = uni_divide_one_minus(uni_add(
        [F(0), mean], uni_scale(primitive_coefficients, F(-1))
    ))

    # Cell 001: a=t*x*y, p=t*y, A=t+(1-t)*z.
    p = scale(y, t)
    a = scale(multiply(x, y), t)
    u = add(p, scale(a, -1))
    A = add(constant(dimension, t), scale(z, one_minus_t))
    shifted = add(a, A, scale(p, -1))
    mixed = add(
        compose_univariate(primitive_coefficients, A, dimension_hint=dimension),
        scale(compose_univariate(
            primitive_coefficients, shifted, dimension_hint=dimension
        ), -1),
        scale(compose_univariate(
            primitive_coefficients, p, dimension_hint=dimension
        ), -1),
        compose_univariate(primitive_coefficients, a, dimension_hint=dimension),
    )
    lower001 = add(
        multiply(
            add(compose_univariate(
                potential_coefficients, A, dimension_hint=dimension
            ), u),
            compose_univariate(qfun, p, dimension_hint=dimension),
        ),
        scale(mixed, -1),
    )

    # Cell 011: a=t*x, p=t+(1-t)*y, A=p+(1-p)*z.
    a = scale(x, t)
    p = add(constant(dimension, t), scale(y, one_minus_t))
    one_minus_p = scale(add(one, scale(y, -1)), one_minus_t)
    v = multiply(one_minus_p, z)
    A = add(p, v)
    u = add(p, scale(a, -1))
    shifted = add(a, v)
    mixed = add(
        compose_univariate(primitive_coefficients, A, dimension_hint=dimension),
        scale(compose_univariate(
            primitive_coefficients, shifted, dimension_hint=dimension
        ), -1),
        scale(compose_univariate(
            primitive_coefficients, p, dimension_hint=dimension
        ), -1),
        compose_univariate(primitive_coefficients, a, dimension_hint=dimension),
    )
    divided_mixed = scale(
        divide_one_minus_variable(mixed, 1), 1 / one_minus_t
    )
    lower011 = add(
        multiply(
            add(compose_univariate(
                potential_coefficients, A, dimension_hint=dimension
            ), u),
            compose_univariate(rfun, p, dimension_hint=dimension),
        ),
        scale(multiply(p, divided_mixed), -1),
    )

    # Cell 111: a=t+(1-t)*x, u=(1-a)*y,
    # v=(1-a)*(1-y)*z.
    a = add(constant(dimension, t), scale(x, one_minus_t))
    remaining = scale(add(one, scale(x, -1)), one_minus_t)
    u = multiply(remaining, y)
    p = add(a, u)
    v = multiply(multiply(remaining, add(one, scale(y, -1))), z)
    A = add(p, v)
    shifted = add(a, v)
    potential_difference = add(
        compose_univariate(
            potential_coefficients, A, dimension_hint=dimension
        ),
        scale(compose_univariate(
            potential_coefficients, a, dimension_hint=dimension
        ), -1),
    )
    delta_bar = scale(
        divide_one_minus_variable(potential_difference, 0), 1 / one_minus_t
    )
    mixed = add(
        compose_univariate(primitive_coefficients, A, dimension_hint=dimension),
        scale(compose_univariate(
            primitive_coefficients, shifted, dimension_hint=dimension
        ), -1),
        scale(compose_univariate(
            primitive_coefficients, p, dimension_hint=dimension
        ), -1),
        compose_univariate(primitive_coefficients, a, dimension_hint=dimension),
    )
    normalized_mixed = divide_variable(mixed, 1)
    normalized_mixed = divide_variable(normalized_mixed, 2)
    normalized_mixed = divide_one_minus_variable(normalized_mixed, 1)
    normalized_mixed = divide_one_minus_variable(normalized_mixed, 0)
    normalized_mixed = divide_one_minus_variable(normalized_mixed, 0)
    normalized_mixed = scale(
        normalized_mixed, 1 / (one_minus_t * one_minus_t)
    )
    lower111 = add(
        multiply(
            add(delta_bar, y),
            compose_univariate(rfun, p, dimension_hint=dimension),
        ),
        scale(multiply(multiply(multiply(p, y), z), normalized_mixed), -1),
    )
    return lower001, lower011, lower111


def build_threshold_problem(
    degrees: Sequence[int], weights: Sequence[F | int | str],
    threshold: F | int | str, endpoint: F | int | str,
    *, title: str, result_version: str,
) -> dict[str, Any]:
    degree_tuple = tuple(int(degree) for degree in degrees)
    weight_tuple = tuple(F(weight) for weight in weights)
    threshold = F(threshold)
    endpoint = F(endpoint)
    if (not degree_tuple or tuple(sorted(set(degree_tuple))) != degree_tuple
            or degree_tuple[0] < 1):
        raise ValueError("degrees must be strictly increasing positive integers")
    if (len(weight_tuple) != len(degree_tuple)
            or any(weight <= 0 for weight in weight_tuple)
            or sum(weight_tuple) != 1):
        raise ValueError("weights must be positive and sum to one")
    if not 0 < threshold < 1 or endpoint <= 0:
        raise ValueError("threshold and endpoint are outside their ranges")

    activation_coefficients = activation(degree_tuple, weight_tuple)
    primitive_coefficients = primitive(activation_coefficients)
    mean = uni_evaluate(primitive_coefficients, F(1))
    potential_coefficients = potential(
        activation_coefficients, mean, endpoint
    )
    cells = build_cells(
        activation_coefficients, primitive_coefficients,
        potential_coefficients, mean, threshold,
    )
    claimed_distortion = 1 + 2 * endpoint

    polynomials: dict[str, dict[str, Any]] = {}

    def insert(name: str, polynomial: dict[tuple[int, ...], F],
               variables: Sequence[str], relation: str = "nonnegative") -> None:
        polynomials[name] = polynomial_record(
            polynomial, variables, relation=relation
        )

    zero_variables: list[str] = []
    mass_error = sum(weight_tuple) - 1
    bound_error = claimed_distortion - (1 + 2 * endpoint)
    insert("probability_mass_identity", constant(0, mass_error), zero_variables)
    insert("probability_mass_identity_reverse",
           constant(0, -mass_error), zero_variables)
    insert("distortion_identity", constant(0, bound_error), zero_variables)
    insert("distortion_identity_reverse",
           constant(0, -bound_error), zero_variables)
    for index, weight in enumerate(weight_tuple):
        insert(f"probability_{index}_positive", constant(0, weight),
               zero_variables, "positive")
    insert(
        "activation_below_mean",
        constant(0, mean - uni_evaluate(
            activation_coefficients, threshold
        )),
        zero_variables, "positive",
    )
    activation_derivative = uni_derivative(activation_coefficients)
    insert(
        "threshold_derivative",
        constant(0, uni_evaluate(activation_derivative, threshold)
                 - (1 - mean)),
        zero_variables, "positive",
    )
    insert(
        "boundary_polynomial",
        univariate_polynomial(potential_coefficients), ["r"],
    )
    insert(
        "potential_derivative",
        univariate_polynomial(uni_derivative(potential_coefficients)), ["u"],
    )
    for name, polynomial in zip(("001", "011", "111"), cells):
        insert(f"cell_{name}", polynomial, ["x", "y", "z"])

    parameter_polynomials = [
        "probability_mass_identity", "probability_mass_identity_reverse",
        "distortion_identity", "distortion_identity_reverse",
        *[f"probability_{index}_positive"
          for index in range(len(weight_tuple))],
    ]
    problems = [
        {
            "name": "exact_parameter_checks",
            "variables": [],
            "domain": [],
            "alternatives": [parameter_polynomials],
        },
        {
            "name": "threshold_checks",
            "variables": [],
            "domain": [],
            "alternatives": [[
                "activation_below_mean", "threshold_derivative"
            ]],
        },
        {
            "name": "boundary_inequality",
            "variables": ["r"],
            "domain": [["0", fraction_text(threshold)]],
            "alternatives": [["boundary_polynomial"]],
        },
        {
            "name": "potential_monotonicity",
            "variables": ["u"],
            "domain": [[fraction_text(threshold), "1"]],
            "alternatives": [["potential_derivative"]],
        },
        *[
            {
                "name": f"internal_cell_{name}",
                "variables": ["x", "y", "z"],
                "domain": [["0", "1"], ["0", "1"], ["0", "1"]],
                "alternatives": [[f"cell_{name}"]],
            }
            for name in ("001", "011", "111")
        ],
    ]
    return {
        "schema": SCHEMA,
        "metadata": {
            "title": title,
            "result_version": result_version,
            "verification_statement": (
                "Exact threshold-potential conditions for a single "
                "random-size stable lottery."
            ),
            "draw_distribution": [
                {
                    "degree": degree,
                    "probability": fraction_text(weight),
                }
                for degree, weight in zip(degree_tuple, weight_tuple)
            ],
            "mean": fraction_text(mean),
            "threshold": fraction_text(threshold),
            "potential_endpoint": fraction_text(endpoint),
            "claimed_distortion_bound": fraction_text(claimed_distortion),
        },
        "polynomials": polynomials,
        "problems": problems,
    }


def ceil_fraction(value: float, denominator: int, margin_units: int = 0) -> F:
    return F(math.ceil(float(value) * denominator) + margin_units, denominator)


def candidate_from_search(record: dict[str, Any], args: argparse.Namespace,
                          ) -> dict[str, Any]:
    raw_support = tuple(record["support"])
    raw_weights = normalize_weights(record["weights"])
    scaled = raw_weights * args.weight_denominator
    numerators = np.floor(scaled).astype(np.int64)
    missing = int(args.weight_denominator - int(numerators.sum()))
    order = sorted(
        range(len(numerators)),
        key=lambda index: (-(scaled[index] - numerators[index]), index),
    )
    for index in order[:missing]:
        numerators[index] += 1
    retained = [index for index, numerator in enumerate(numerators)
                if numerator > 0]
    support = tuple(raw_support[index] for index in retained)
    weights = tuple(
        F(int(numerators[index]), args.weight_denominator)
        for index in retained
    )
    boundary, boundary_endpoint = continuous_boundary_endpoint(
        support, [float(weight) for weight in weights]
    )
    threshold = ceil_fraction(boundary, args.threshold_denominator)
    numerical_endpoint = max(float(record["endpoint"]), boundary_endpoint)
    endpoint = ceil_fraction(
        numerical_endpoint, args.endpoint_denominator,
        margin_units=args.endpoint_margin_units,
    )
    problem = build_threshold_problem(
        support, weights, threshold, endpoint,
        title="Numerically discovered integer draw distribution",
        result_version="search-output",
    )
    problem["metadata"]["search"] = {
        "status": "numerical heuristic, not a proof",
        "searched_support_count": str(record["searched_support_count"]),
        "coarse_grid_size": str(record["coarse_grid_size"]),
        "refine_grid_size": str(record["refine_grid_size"]),
        "final_grid_size": str(record["final_grid_size"]),
        "refined_support_count": str(record["refined_support_count"]),
        "globally_refined_support_count": str(
            record["globally_refined_support_count"]
        ),
        "numerical_grid_endpoint": format(record["endpoint"], ".17g"),
        "numerical_boundary_endpoint": format(boundary_endpoint, ".17g"),
        "endpoint_rounding_margin_units": str(args.endpoint_margin_units),
        "global_optimality_certified": "false",
    }
    return problem


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-degree", type=int, default=20)
    parser.add_argument("--max-support-size", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--refine-grid-size", type=int, default=80)
    parser.add_argument("--final-grid-size", type=int, default=320)
    parser.add_argument("--shortlist", type=int, default=100)
    parser.add_argument("--initial-samples", type=int, default=6)
    parser.add_argument("--local-starts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--refine-initial-samples", type=int, default=24)
    parser.add_argument("--refine-local-starts", type=int, default=5)
    parser.add_argument("--refine-maxiter", type=int, default=160)
    parser.add_argument("--global-shortlist", type=int, default=20)
    parser.add_argument("--global-initial-samples", type=int, default=24)
    parser.add_argument("--global-local-starts", type=int, default=6)
    parser.add_argument("--global-maxiter", type=int, default=240)
    parser.add_argument("--differential-iterations", type=int, default=60)
    parser.add_argument("--differential-population", type=int, default=12)
    parser.add_argument(
        "--global-refinement", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--weight-denominator", type=int, default=10**12)
    parser.add_argument("--threshold-denominator", type=int, default=10**6)
    parser.add_argument("--endpoint-denominator", type=int, default=10**11)
    parser.add_argument("--endpoint-margin-units", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if min(args.shortlist, args.initial_samples, args.local_starts,
           args.maxiter, args.refine_initial_samples,
           args.refine_local_starts, args.refine_maxiter,
           args.global_shortlist, args.global_initial_samples,
           args.global_local_starts, args.global_maxiter,
           args.differential_iterations, args.differential_population,
           args.workers, args.weight_denominator,
           args.threshold_denominator, args.endpoint_denominator) < 1:
        parser.error("counts, iteration limits, workers, and denominators must be positive")
    if args.endpoint_margin_units < 0:
        parser.error("--endpoint-margin-units must be nonnegative")
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative")
    if args.refine_grid_size < args.grid_size:
        parser.error("--refine-grid-size must be at least --grid-size")
    if args.final_grid_size < args.refine_grid_size:
        parser.error("--final-grid-size must be at least --refine-grid-size")

    record = numerical_search(args)
    problem = candidate_from_search(record, args)
    write_json(args.output, problem)
    distribution = problem["metadata"]["draw_distribution"]
    print("Numerical search only; no global optimum or distortion bound is proved.")
    print("Best candidate found:", distribution)
    print("Numerical potential endpoint:", record["endpoint"])
    print(f"Wrote exact polynomial problems to {args.output.resolve()}")
    print(
        "Verify exactly with: python3 verify_distortion.py "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
