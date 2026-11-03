"""Regression tests for certificate rejection and rigorous interval bounds."""
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from certificate_lower_bound import verify
from intervals import Failure, certify_box

ROOT = Path(__file__).resolve().parents[1]
ZERO = (F(0), F(0))


class IntervalTests(unittest.TestCase):
    def test_boundary_zero_is_proved(self):
        result = certify_box(lambda z: -z[0], ((F(0), F(1)), ZERO, ZERO, ZERO))
        self.assertGreater(result['leaves'], 0)

    def test_positive_constant_is_rejected(self):
        with self.assertRaises(Failure):
            certify_box(lambda z: z[0]*0+1, (ZERO, ZERO, ZERO, ZERO))

    def test_narrow_positive_region_is_rejected(self):
        # The midpoint and endpoints are negative. A finite sample is insufficient.
        def residual(z):
            x = z[0]-F(1, 3)
            return F(1, 10000)-x*x
        with self.assertRaises(Failure):
            certify_box(residual, ((F(0), F(1)), ZERO, ZERO, ZERO))

    def test_fourth_coordinate_can_require_subdivision(self):
        def residual(z):
            x = z[3]-F(1, 2)
            return -x*x-F(1, 100)
        result = certify_box(residual, (ZERO, ZERO, ZERO, (F(0), F(1))))
        self.assertGreater(result['nodes'], 1)

    def test_box_limit_does_not_accept_unproved_claim(self):
        def residual(z):
            x = z[0]-F(1, 2)
            return -x*x-F(1, 100)
        with self.assertRaises(Failure):
            certify_box(residual, ((F(0), F(1)), ZERO, ZERO, ZERO), limit=1)


class LowerBoundTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT/'data/lower-bound.json').read_text())

    def test_v4_integer_inequality(self):
        self.assertEqual(verify(self.data)['status'], 'verified')

    def test_changed_finite_gap_is_rejected(self):
        self.data['finite_gaps'][0] = '0'
        with self.assertRaises(AssertionError):
            verify(self.data)

    def test_changed_tail_start_is_rejected(self):
        self.data['tail_start'] = 100
        with self.assertRaises(AssertionError):
            verify(self.data)

    def test_wrong_target_bound_is_rejected(self):
        self.data['distortion_lower_bound'] = '213609/100000'
        with self.assertRaises(AssertionError):
            verify(self.data)


class InputRejectionTests(unittest.TestCase):
    def run_audit(self, directory, optimized=False):
        command = [sys.executable, '-B']
        if optimized:
            command.append('-O')
        command += [str(ROOT/'verify_distortion.py'), '--audit-only', '--data-dir', str(directory)]
        return subprocess.run(command, capture_output=True, text=True)

    def test_optimized_python_is_rejected(self):
        result = self.run_audit(ROOT/'data', optimized=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('without -O', result.stderr)

    def test_float_coefficient_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)/'data'
            shutil.copytree(ROOT/'data', data)
            for name in ['finite-family-witness.json', 'rank-cut-witness.json']:
                path = data/name
                witness = json.loads(path.read_text())
                witness['mass_cells'][0]['lambda_numerator'][0] = 0.0
                path.write_text(json.dumps(witness))
            result = self.run_audit(data)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Non-rational JSON input', result.stderr)

    def test_shared_but_wrong_correction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)/'data'
            shutil.copytree(ROOT/'data', data)
            for name in ['finite-family-witness.json', 'rank-cut-witness.json']:
                path = data/name
                witness = json.loads(path.read_text())
                witness['mass_cells'][0]['lambda_numerator'][0] = '1'
                path.write_text(json.dumps(witness))
            self.assertNotEqual(self.run_audit(data).returncode, 0)


if __name__ == '__main__':
    unittest.main()
