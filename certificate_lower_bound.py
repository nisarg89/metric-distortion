#!/usr/bin/env python3
"""Check rational parameters and integer-degree inequalities for the lower bound.

This verifies the finite-degree gaps and the bound on the infinite tail of
positive integer degrees for distortion strictly greater than 267/125.
"""
from fractions import Fraction as F
import argparse
import json
from pathlib import Path

def verify(data):
    if not __debug__:
        raise RuntimeError('Run without -O: the certificate checks require assertions.')
    assert data['schema'] == 'metric-distortion-integer-lower-v1'
    c, r, p, a, b = (F(data[key]) for key in ['c', 'r', 'p', 'a', 'b'])
    assert c == F(71, 125) and r == F(1, 39) and p == F(3, 4)
    assert a == p*r and b == a+1-p and 0 < a < b < c < p < 1
    assert 1+2*c == F(data['distortion_lower_bound']) == F(267, 125)
    for degree, expected in enumerate(data['finite_gaps'], 1):
        difference = (c-a)*p**degree + a**(degree+1) - b**(degree+1) - (c-b)*(degree+1)*c**degree
        assert difference == F(expected) > 0
    assert len(data['finite_gaps']) == 2 and data['tail_start'] == 3
    degree = data['tail_start']
    stronger = (c-a)*p**degree-b**(degree+1)-(c-b)*(degree+1)*c**degree
    assert stronger > F(data['tail_margin']) == F(1, 200)
    # After division by p**d, both right-hand terms decrease for every d>=3.
    assert 0 < b/p < 1
    assert F(degree+2, degree+1)*c/p == 5*c/(4*p) < 1
    z = p/(p+c-p*r)
    assert 0 < z <= 1 and p*(r+(1-z)/z) == c
    # In the first case p=1 and r=c, so the calibration gives z=1.
    assert data['candidate_counts'] == [c.denominator+1, r.denominator+1] == [126, 40]
    return {'status': 'verified', 'lower_bound': '267/125 = 2.136',
            'finite_degrees_checked': [1, 2, 3], 'all_degrees_at_least_3': 'strictly decreasing normalized tail'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent/'data/lower-bound.json')
    args = parser.parse_args()
    def reject(value):
        raise ValueError('Non-rational JSON input: ' + value)
    data = json.loads(args.data.read_text(), parse_float=reject, parse_constant=reject)
    print(json.dumps(verify(data), indent=2))
    print('Integer lower-bound arithmetic verified: distortion > 267/125 = 2.136.')

if __name__ == '__main__':
    main()
