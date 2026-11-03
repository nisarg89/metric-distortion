#!/usr/bin/env python3
"""Unit tests for the proof-agnostic Bernstein certificate machinery."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as F
import unittest

from bernstein import (
    SCHEMA,
    _affine_to_unit,
    bernstein_to_power,
    find_certificate,
    parse_problem,
    polynomial_record,
    power_to_bernstein_on_box,
    split_control,
    verify_certificate,
)


def disjunction_problem() -> dict:
    return {
        "schema": SCHEMA,
        "metadata": {"title": "toy disjunction"},
        "polynomials": {
            "x": polynomial_record({(1,): F(1)}, ["x"]),
            "minus_x": polynomial_record({(1,): F(-1)}, ["x"]),
        },
        "problems": [{
            "name": "absolute-value cover",
            "variables": ["x"],
            "domain": [["-1", "1"]],
            "alternatives": [["x"], ["minus_x"]],
        }],
    }


class BernsteinConversionTests(unittest.TestCase):
    def test_exact_round_trip_on_nonunit_box(self) -> None:
        polynomial = {
            (0, 0): F(2),
            (1, 0): F(3, 5),
            (2, 0): F(-7, 11),
            (1, 2): F(13, 17),
        }
        box = ((F(-2), F(3)), (F(1, 3), F(7, 5)))
        degrees, control = power_to_bernstein_on_box(polynomial, box)
        self.assertEqual(
            bernstein_to_power(control, degrees),
            _affine_to_unit(polynomial, box),
        )

    def test_midpoint_split_matches_direct_conversion(self) -> None:
        polynomial = {
            (0, 0): F(1, 7),
            (1, 0): F(-3, 4),
            (2, 1): F(5, 9),
            (0, 2): F(11, 13),
        }
        box = ((F(-1), F(3)), (F(2), F(5)))
        degrees, control = power_to_bernstein_on_box(polynomial, box)
        left, right = split_control(control, degrees, 0)
        midpoint = F(1)
        self.assertEqual(
            (degrees, left),
            power_to_bernstein_on_box(
                polynomial, ((F(-1), midpoint), box[1])
            ),
        )
        self.assertEqual(
            (degrees, right),
            power_to_bernstein_on_box(
                polynomial, ((midpoint, F(3)), box[1])
            ),
        )


class CertificateTests(unittest.TestCase):
    def test_disjunction_requires_and_records_subdivision(self) -> None:
        problem = disjunction_problem()
        certificate, counts = find_certificate(problem)
        self.assertEqual(counts["absolute-value cover"].leaves, 2)
        self.assertEqual(counts["absolute-value cover"].maximum_depth, 1)
        self.assertEqual(
            verify_certificate(problem, certificate)[
                "absolute-value cover"
            ].leaves,
            2,
        )

    def test_tampered_coefficient_is_rejected(self) -> None:
        problem = disjunction_problem()
        certificate, _ = find_certificate(problem)
        tampered = deepcopy(certificate)
        coefficients = tampered["problems"][0]["root_controls"]["x"][
            "coefficients"
        ]
        coefficients[0][1] = "-2"
        with self.assertRaisesRegex(
            AssertionError, "do not represent polynomial"
        ):
            verify_certificate(problem, tampered)

    def test_tampered_leaf_choice_is_rejected(self) -> None:
        problem = disjunction_problem()
        certificate, _ = find_certificate(problem)
        tampered = deepcopy(certificate)
        tampered["problems"][0]["tree"]["children"][0] = {
            "alternative": 0
        }
        with self.assertRaisesRegex(AssertionError, "does not certify"):
            verify_certificate(problem, tampered)

    def test_strict_relation_rejects_zero(self) -> None:
        problem = {
            "schema": SCHEMA,
            "polynomials": {
                "zero": polynomial_record({}, [], relation="positive"),
            },
            "problems": [{
                "name": "strict point check",
                "variables": [],
                "domain": [],
                "alternatives": [["zero"]],
            }],
        }
        with self.assertRaisesRegex(AssertionError, "point domain"):
            find_certificate(problem)

    def test_inexact_json_coefficient_is_rejected(self) -> None:
        problem = disjunction_problem()
        problem["polynomials"]["x"]["terms"][0][1] = 1.0
        with self.assertRaisesRegex(TypeError, "rational string"):
            parse_problem(problem)


if __name__ == "__main__":
    unittest.main()
