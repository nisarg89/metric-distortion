#!/usr/bin/env python3
"""Verify the 2.13609 certificate for metric-distortion-v4.tex.

The default (also --full) recomputes every continuous polynomial bound and all
algebraic identities. --audit-only checks inputs and the one-variable robustness
bound; it does not certify the multivariate inequalities. No optimizer is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CODE = Path(__file__).resolve().parent

def main():
    if not __debug__:
        raise RuntimeError('Run without -O: the certificate checks require assertions.')
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='recompute the complete certificate (default)')
    mode.add_argument('--audit-only', action='store_true', help='audit inputs only; not a complete certificate')
    parser.add_argument('--data-dir', type=Path, default=CODE / 'data')
    parser.add_argument('--output-dir', type=Path, help='keep generated reports here (otherwise use a temporary directory)')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='metric-distortion-check-') as temporary:
        output = (args.output_dir or Path(temporary)).resolve()
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, METRIC_DISTORTION_DATA_DIR=str(args.data_dir.resolve()),
                   METRIC_DISTORTION_OUTPUT_DIR=str(output), PYTHONDONTWRITEBYTECODE='1')
        scripts = ['audit_finite_witness.py', 'verify_robustness_bound.py'] if args.audit_only else [
            'finite_family_verifier.py', 'finish_finite_certificate.py',
            'audit_finite_witness.py', 'verify_rank_cut.py',
            'finish_rank_cut_certificate.py', 'verify_robustness_bound.py']
        # Check the two frozen tables agree before any expensive calculation.
        def read(name):
            def reject(value):
                raise ValueError('Non-rational JSON input: ' + value)
            return json.loads((args.data_dir / name).read_text(), parse_float=reject, parse_constant=reject)
        ff = read('finite-family-witness.json')
        rr = read('rank-cut-witness.json')
        for field in ['g_degrees', 'g_weights', 'mass_cells', 'mu', 'C', 'eta', 'b', 'root_intervals', 'endpoints']:
            assert ff[field] == rr[field], ('inconsistent witness tables', field)
        for script in scripts:
            print('Checking ' + script, flush=True)
            subprocess.run([sys.executable, '-B', str(CODE / script)], env=env, cwd=CODE, check=True)
        if args.audit_only:
            print('Input and robustness audits passed. Multivariate bounds were NOT checked.')
            return
        finite = json.loads((output / 'finite-family-verification.json').read_text())
        rank = json.loads((output / 'rank-cut-verification.json').read_text())
        assert finite['status'] == rank['status'] == 'verified'
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        summary = {'status': 'verified', 'manuscript': 'metric-distortion-v4.tex',
                   'distortion': '213609/100000', 'decimal_bound': '2.13609',
                   'finite_polynomial_boxes': finite['direct_poly_boxes'],
                   'additional_rank_cut_boxes': rank['added_polynomial_boxes'],
                   'inputs': {p.name: sha(p) for p in sorted(args.data_dir.glob('*.json')) if p.name != 'verification-summary.json'},
                   'code': {p.name: sha(p) for p in sorted(CODE.glob('*.py'))}}
        (output / 'verification-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('Complete v4 certificate verified: distortion <= 213609/100000 = 2.13609.')

if __name__ == '__main__':
    main()
