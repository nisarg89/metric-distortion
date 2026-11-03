# Exact verification for metric distortion 2.13609

This repository contains checks for the numerical and polynomial claims underlying the paper's distortion bounds for random-size stable lotteries.

## What is verified

- **Upper bound 2.13609:** the fixed coefficient identities, interval coverage, endpoint values, measure nonnegativity and total masses, and polynomial inequalities over their complete continuous domains. The draw distribution has probabilities $19/73$, $35/146$, and $1/2$ on $1$, $2$, and $32$, respectively.
- **Approximation guarantee:** the auxiliary polynomial bound $(1-p)\lambda(p)+\tau_0(p)\leq16$.
- **Lower bound 2.136:** the rational parameter identities and integer-degree inequalities, including the bound covering the infinite tail of degrees.

## Run the checks

Use Python 3.10 or later. From the repository root:

```console
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B certificate_distortion.py
python -B certificate_lower_bound.py
python -B -m unittest discover -s tests -v
```

The upper-bound command performs the complete verification and can take several minutes. The lower-bound command uses only Python's standard library. Run without Python's `-O` option, which disables assertions.

`python -B verify_distortion.py --full` is equivalent to the upper-bound command. Use `--output-dir verification-results` to retain generated reports; otherwise they are temporary. The optional `--audit-only` mode checks the inputs and the approximation bound, but does not check the multivariate inequalities.

## Code and data

Verification programs are in the repository root; fixed rational inputs are in [`data/`](data/).

| Data file | Contents |
| --- | --- |
| [`parameters.json`](data/parameters.json) | Draw probabilities and rational parameters. |
| [`finite-family-witness.json`](data/finite-family-witness.json) | Auxiliary coefficient table. |
| [`rank-cut-witness.json`](data/rank-cut-witness.json) | Coefficient table and measures for the upper-bound checks. |
| [`lower-bound.json`](data/lower-bound.json) | Rational parameters and claimed gaps for the lower-bound checks. |

[`verification-summary.json`](data/verification-summary.json) records input and code hashes, runtime versions, and box counts from a complete successful verification. The commands recompute the checks independently of this saved summary.

## Archived artifacts

Earlier artifacts and their original data are preserved in [`v1/`](v1/) and [`v2/`](v2/). From the repository root:

```console
python3 -B v1/certificate_distortion_v1.py
python3 -B v2/certificate_distortion.py
python3 -B v2/certificate_lower_bound.py
```
