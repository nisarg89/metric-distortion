#!/usr/bin/env python3
r"""Numerical search and rationalization from Appendix B.5--B.6.

This is candidate generation, not a proof.  The default run refines the
selected support {1,12,13}, rationalizes the result as in the paper, and
writes ``candidate.json`` beside this file.  Verify that file exactly with

    python3 verify_distortion.py candidate.json

The ``dual`` and ``grid`` modes expose the discretized dual, constraint
generation, the estimates \hat L_D and \hat g_D, and the outer rule-mixture
LP in the notation of the paper.  They remain floating-point heuristics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb, exp, log
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, linprog, minimize_scalar


# ---------------------------------------------------------------------------
# Appendix B.5: the discretized primal/dual at fixed (D,q,x).


def normalize(D):
    total = sum(D.values())
    if total <= 0:
        raise ValueError("D must have positive mass")
    return {int(d): float(a_d) / total for d, a_d in D.items() if a_d > 0}


def p_D(D):
    return sum(a_d / (d + 1) for d, a_d in D.items())


def f_D(D, t):
    t = np.asarray(t)
    return sum(a_d * np.power(t, d) for d, a_d in D.items())


def f_prime_D(D, t):
    t = np.asarray(t)
    return sum(d * a_d * np.power(t, d - 1) for d, a_d in D.items())


def Psi_D(D, t):
    t = np.asarray(t)
    return sum(a_d * np.power(t, d + 1) / (d + 1)
               for d, a_d in D.items())


def W_x(D, x, y):
    return f_D(D, x + (1 - x) * np.asarray(y))


def w_x(D, x, b_1):
    return f_D(D, x * np.asarray(b_1))


def F_x(D, x, b_1):
    b_1 = np.asarray(b_1)
    r, answer = 1 - x, np.zeros_like(b_1, dtype=float)
    for d, a_d in D.items():
        for k in range(1, d + 1):
            answer += (a_d * comb(d, k) * x ** (k - 1) * r ** (d - k)
                       * np.power(b_1, k) / (d - k + 1))
    return answer


def G_x(D, x, b_0, y):
    b_0, y = np.broadcast_arrays(np.asarray(b_0), np.asarray(y))
    r, answer = 1 - x, np.zeros_like(b_0, dtype=float)
    for d, a_d in D.items():
        for k in range(1, d + 1):
            y_power = np.power(y, d - k + 1)
            answer += (a_d * comb(d, k) * x ** (k - 1) * r ** (d - k)
                       * (np.power(b_0, k) * y_power + 1 - y_power)
                       / (d - k + 1))
    return answer


def R_x(D, x):
    r = 1 - x
    return sum(a_d * sum(r ** j for j in range(d)) / (d + 1)
               for d, a_d in D.items())


@dataclass(frozen=True)
class NumericalDual:
    L_hat: float
    lambda_: float
    mu: float
    u: float
    v: float
    delta_0: float = float("nan")
    delta_1: float = float("nan")
    active_E: tuple | None = None
    active_C_1: float | None = None
    iterations: int = 0


def solve_discretized_dual(D, q, x, E, C_1):
    """Solve the ordinary finite LP dual displayed in Appendix B.5."""
    D = normalize(D)
    E, C_1 = np.asarray(E, dtype=float), np.asarray(C_1, dtype=float)
    b_0, y = E[:, 0], E[:, 1]
    rows_E = np.column_stack((
        -W_x(D, x, y), -G_x(D, x, b_0, y),
        -np.ones(len(E)), np.zeros(len(E)),
    ))
    rows_C_1 = np.column_stack((
        -w_x(D, x, C_1), -F_x(D, x, C_1),
        np.zeros(len(C_1)), -np.ones(len(C_1)),
    ))
    answer = linprog(
        (p_D(D), R_x(D, x), 1 - q, q),
        A_ub=np.row_stack((rows_E, rows_C_1)),
        b_ub=-np.concatenate((b_0, C_1)),
        bounds=((0, None), (0, None), (None, None), (None, None)),
        method="highs",
    )
    if answer.status == 3:
        return NumericalDual(-float("inf"), *(float("nan"),) * 4)
    if not answer.success:
        raise RuntimeError(f"dual failed at (q,x)=({q},{x}): {answer.message}")
    return NumericalDual(float(answer.fun), *map(float, answer.x))


def largest_violations(D, x, dual, seed):
    """Numerically compute delta_0 and delta_1 from Appendix B.6."""
    lambda_, mu, u, v = dual.lambda_, dual.mu, dual.u, dual.v

    def violation_E(point):
        b_0, y = point
        return float(b_0 - u - lambda_ * W_x(D, x, y)
                     - mu * G_x(D, x, b_0, y))

    def violation_C_1(point):
        b_1 = float(point[0])
        return float(b_1 - v - lambda_ * w_x(D, x, b_1)
                     - mu * F_x(D, x, b_1))

    answer_E = differential_evolution(
        lambda point: -violation_E(point), ((0, 1), (0, 1)),
        seed=seed, popsize=12, maxiter=250, tol=1e-11,
        polish=True, workers=1,
    )
    answer_C_1 = differential_evolution(
        lambda point: -violation_C_1(point), ((0, 1),),
        seed=seed + 1, popsize=12, maxiter=250, tol=1e-11,
        polish=True, workers=1,
    )
    return (-float(answer_E.fun), tuple(map(float, answer_E.x))), (
        -float(answer_C_1.fun), float(answer_C_1.x[0]))


def constraint_generated_dual(D, q, x, points=9, tolerance=1e-9,
                              max_iterations=50, seed=1):
    """Repeat finite-dual solution and numerical separation."""
    D = normalize(D)
    mesh = np.linspace(0, 1, points)
    E, C_1 = list(product(mesh, mesh)), list(mesh)
    for iteration in range(max_iterations + 1):
        dual = solve_discretized_dual(D, q, x, E, C_1)
        if not np.isfinite(dual.L_hat):
            return dual
        separated_E, separated_C_1 = largest_violations(
            D, x, dual, seed + 2 * iteration)
        delta_0, active_E = separated_E
        delta_1, active_C_1 = separated_C_1
        if max(delta_0, delta_1) <= tolerance or iteration == max_iterations:
            return NumericalDual(
                dual.L_hat, dual.lambda_, dual.mu, dual.u, dual.v,
                delta_0, delta_1, active_E, active_C_1, iteration,
            )
        if delta_0 > tolerance:
            E.append(active_E)
        if delta_1 > tolerance:
            C_1.append(active_C_1)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Appendix B.6: finite (q,x) sets and the outer rule-mixture LP.


def rho_D(D):
    """Numerically solve f_D(rho_D)=p_D."""
    lower, upper, pp = 0.0, 1.0, p_D(D)
    for _ in range(90):
        middle = (lower + upper) / 2
        if f_D(D, middle) < pp:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def Lambda_D(D):
    """Numerical q-downarrow-0 value from the observed one-active-y pattern."""
    pp, a_1 = p_D(D), D.get(1, 0.0)

    def ratio(y):
        value = float(f_D(D, y))
        denominator = a_1 * value + pp * (float(f_prime_D(D, y)) - a_1)
        return -1e6 if value <= pp or denominator <= 0 else 2 * (value - pp) / denominator

    answer = minimize_scalar(
        lambda y: -ratio(y), bounds=(0.6, 1.0), method="bounded",
        options={"xatol": 1e-14},
    )
    return -float(answer.fun), float(answer.x)


def estimate_g_D(D, q, x_values, state_points, separate=False, seed=1):
    r"""Return \hat g_D(q)=max_x min{x\hat L_D(q,x),(1-p)x}."""
    mesh = np.linspace(0, 1, state_points)
    E, C_1 = list(product(mesh, mesh)), mesh
    best = (-float("inf"), None, None)
    for index, x in enumerate(x_values):
        dual = (constraint_generated_dual(
            D, q, float(x), points=state_points, seed=seed + index)
                if separate else solve_discretized_dual(D, q, float(x), E, C_1))
        value = min(float(x) * dual.L_hat, (1 - p_D(D)) * float(x))
        if value > best[0]:
            best = (value, float(x), dual)
    return best


def solve_rule_mixture(D, q_values, g_hat_values):
    """Solve the finite outer LP for alpha,beta,gamma,C_cut."""
    Lambda, _ = Lambda_D(D)
    rows = [(q * q, min(q, 0.5), g_hat, -q)
            for q, g_hat in zip(q_values, g_hat_values)]
    rows.extend(((0, 1, Lambda, -1), (1, 0.5, rho_D(D), -1)))
    answer = linprog(
        (0, 0, 0, 1), A_ub=np.asarray(rows), b_ub=np.zeros(len(rows)),
        A_eq=((1, 1, 1, 0),), b_eq=(1,),
        bounds=((0, None),) * 4, method="highs",
    )
    if not answer.success:
        raise RuntimeError(answer.message)
    return tuple(map(float, answer.x))


# ---------------------------------------------------------------------------
# Selected-support refinement, active dual patterns, and rationalization.


def logit(value):
    return log(value / (1 - value))


def sigmoid(value):
    return 1 / (1 + exp(-value))


def draw_distribution_from_log_ratios(degrees, parameters):
    """Normalize two unconstrained log mass ratios on a three-degree support."""
    degree_one_mass = sigmoid(float(parameters[0]))
    remaining_split = sigmoid(float(parameters[1]))
    masses = (degree_one_mass,
              (1 - degree_one_mass) * remaining_split,
              (1 - degree_one_mass) * (1 - remaining_split))
    return dict(zip(degrees, masses))


def active_pattern_losses(degrees, parameters, details=False):
    """Final beta=0 refinement using the three observed active regimes."""
    D = draw_distribution_from_log_ratios(degrees, parameters)
    alpha = float(parameters[2])
    gamma, pp = 1 - alpha, p_D(D)
    Lambda, active_y = Lambda_D(D)
    rho = rho_D(D)

    def endpoint(x):
        r = 1 - x
        support_budget = (pp - float(Psi_D(D, r)) / r) / x
        atom_cost = (pp - float(Psi_D(D, x)) - float(Psi_D(D, r))) / (x * r)
        mass = support_budget / atom_cost
        q = 1 - pp - mass * (1 - float(f_D(D, x)))
        remaining = 1 - q - mass
        if min(q, mass, remaining) <= 0:
            return -1e6, None
        comparison_loss = x * mass / q
        return alpha * q + gamma * comparison_loss, (
            q, x, comparison_loss, mass, remaining)

    endpoint_answer = minimize_scalar(
        lambda x: -endpoint(x)[0], bounds=(0.48, 0.72), method="bounded",
        options={"xatol": 1e-14},
    )
    endpoint_loss, endpoint_data = endpoint(float(endpoint_answer.x))
    losses = (gamma * Lambda, alpha + gamma * rho, endpoint_loss)
    value = max(losses)
    if not details:
        return value
    return value, {
        "D": D, "alpha": alpha, "beta": 0.0, "gamma": gamma,
        "C_cut": value, "Lambda_D": Lambda, "active_y": active_y,
        "rho_D": rho, "endpoint": endpoint_data, "active_losses": losses,
    }


def search_selected_support(degrees, seed, maxiter):
    if len(degrees) != 3 or degrees[0] != 1:
        raise ValueError("the final refinement expects three degrees beginning with 1")
    bounds = ((logit(0.39), logit(0.46)),
              (logit(0.55), logit(1 - 1e-10)),
              (0.13, 0.19))
    answer = differential_evolution(
        lambda parameters: active_pattern_losses(degrees, parameters),
        bounds, seed=seed, popsize=24, maxiter=maxiter, tol=2e-11,
        polish=True, workers=1,
    )
    value, details = active_pattern_losses(degrees, answer.x, details=True)
    if not answer.success:
        raise RuntimeError(answer.message)
    return value, details


def fraction_text(value):
    return (str(value.numerator) if value.denominator == 1
            else f"{value.numerator}/{value.denominator}")


def rationalize(details):
    """Apply the rational choices made in Appendix B.6."""
    rational_D = {
        1: Fraction(423239, 10**6),
        12: Fraction(471105, 10**6),
        13: Fraction(105656, 10**6),
    }
    if set(details["D"]) != set(rational_D):
        raise ValueError("exact rationalization is recorded only for support {1,12,13}")
    if (max(abs(details["D"][d] - float(rational_D[d])) for d in rational_D) > 2e-4
            or abs(details["C_cut"] - 0.6641) > 2e-4):
        raise RuntimeError(
            "numerical search did not return close enough to the paper's rational candidate")

    alpha = Fraction(round(details["alpha"] * 50000), 50000)
    gamma = 1 - alpha
    C_cut = Fraction(round(details["C_cut"] * 10000), 10000)
    active_y = Fraction(round(details["active_y"] * 100000), 100000)
    pp = sum(a_d / (d + 1) for d, a_d in rational_D.items())
    return {
        "schema": "metric-distortion-candidate-v1",
        "draw_distribution": {
            "1": "423239/1000000",
            "12": "471105/1000000",
            "13": "105656/1000000",
        },
        "rule_mixture": {
            "alpha_integrated_veto": fraction_text(alpha),
            "beta_maximal_lottery": "0",
            "gamma_random_size_stable_lottery": fraction_text(gamma),
            "C_cut": fraction_text(C_cut),
        },
        "target_polynomial_T": {
            "1": fraction_text(C_cut / gamma),
            "2": fraction_text(-alpha / gamma),
        },
        "dual_tuple_parameters": {
            "active_y": [fraction_text(active_y), "1"],
            "tangency_t": [f"{numerator}/100" for numerator in range(15, 61)],
        },
        "verification_domain": {
            "q": ["0", "1"], "x": ["0", "1"],
            "family_2_x_min": "1/5",
            "x_1_q": [fraction_text(1 - pp), "1"],
        },
    }


def candidate_mode(args):
    degrees = tuple(sorted(set(map(int, args.degrees.split(",")))))
    value, details = search_selected_support(degrees, args.seed, args.maxiter)
    candidate = rationalize(details)
    output = Path(args.output)
    output.write_text(json.dumps(candidate, indent=2) + "\n")
    print("Numerical search only (not a proof):")
    print("D =", ", ".join(f"a_{d}={details['D'][d]:.10f}" for d in degrees))
    print("(alpha,beta,gamma) =",
          f"({details['alpha']:.10f},0,{details['gamma']:.10f})")
    print(f"C_cut = {value:.10f}; active y = {details['active_y']:.10f}")
    print(f"Wrote rational candidate to {output}")
    verifier = Path(__file__).resolve().with_name("verify_distortion.py")
    print(f"Verify with: python3 {verifier} {output.resolve()}")


def dual_mode(args):
    D = {1: 423239, 12: 471105, 13: 105656}
    answer = constraint_generated_dual(
        D, args.q, args.x, points=args.states,
        tolerance=args.tolerance, seed=args.seed,
    )
    if not np.isfinite(answer.L_hat):
        print(f"The discretized primal is infeasible at (q,x)=({args.q:g},{args.x:g}).")
        return
    print(f"L_hat_D({args.q:g},{args.x:g}) = {answer.L_hat:.12g}")
    print("(lambda,mu,u,v) =",
          tuple(round(value, 12) for value in
                (answer.lambda_, answer.mu, answer.u, answer.v)))
    print(f"delta_0 = {answer.delta_0:.3g} at {answer.active_E}")
    print(f"delta_1 = {answer.delta_1:.3g} at {answer.active_C_1}")
    print(f"constraint-generation iterations = {answer.iterations}")
    if max(answer.delta_0, answer.delta_1) > args.tolerance:
        print("Stopped before the requested numerical separation tolerance.")


def grid_mode(args):
    D = normalize({1: 423239, 12: 471105, 13: 105656})
    q_values = np.linspace(args.q_min, 1, args.q_count)
    x_values = np.linspace(args.x_min, 1, args.x_count)
    estimates = []
    for index, q in enumerate(q_values):
        g_hat, x, _ = estimate_g_D(
            D, float(q), x_values, args.states, args.separate,
            args.seed + index * args.x_count,
        )
        if x is None:
            raise RuntimeError(f"no feasible searched x at q={q:g}; lower --x-min")
        estimates.append(g_hat)
        print(f"q={q:.6g}  g_hat_D={g_hat:.10g}  active x={x:.6g}")
    alpha, beta, gamma, C_cut = solve_rule_mixture(D, q_values, estimates)
    print("(alpha,beta,gamma,C_cut) =",
          tuple(round(value, 10) for value in (alpha, beta, gamma, C_cut)))
    print("Numerical grid output is a search heuristic, not a certificate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("candidate", "dual", "grid"),
                        default="candidate")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--degrees", default="1,12,13")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--output", default=str(Path(__file__).with_name("candidate.json")))
    parser.add_argument("--q", type=float, default=0.4)
    parser.add_argument("--x", type=float, default=0.6)
    parser.add_argument("--states", type=int, default=9)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--q-min", type=float, default=0.05)
    parser.add_argument("--x-min", type=float, default=0.05)
    parser.add_argument("--q-count", type=int, default=9)
    parser.add_argument("--x-count", type=int, default=21)
    parser.add_argument("--separate", action="store_true")
    args = parser.parse_args()
    {"candidate": candidate_mode, "dual": dual_mode, "grid": grid_mode}[args.mode](args)


if __name__ == "__main__":
    main()
