# Exact verification for metric distortion 2.13609

This repository contains the computational checks for `metric-distortion-v4.tex`. The upper-bound artifact verifies the polynomial inequalities used to prove distortion at most $213609/100000=2.13609$ for a random-size stable lottery. Its draw distribution is supported on $\{1,2,32\}$, with probabilities $19/73$, $35/146$, and $1/2$, respectively. The lower-bound artifact checks the arithmetic in the v4 proof that every fixed integer draw distribution admits a profile on which every stable lottery has distortion strictly greater than $267/125=2.136$.

The repository root is the current v4 artifact. Earlier published artifacts are preserved in [`v1/`](v1/) and [`v2/`](v2/), with their original data and historical bounds.

## Run the current checks

Use Python 3.10 or later. From the repository root:

```console
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B certificate_distortion.py
python -B certificate_lower_bound.py
python -B -m unittest discover -s tests -v
```

The upper-bound command rechecks every continuous polynomial domain, including boundary faces, and the exact identities composing the rank-cut inequality. It uses rational inputs, exact symbolic polynomial arithmetic, and 160-bit outward Arb enclosures through `python-flint`. It can take several minutes. It performs no numerical optimization or root search. The lower-bound command uses only Python's standard library and also certifies the infinite tail of integer degrees.

The equivalent explicit command for the upper bound is:

```console
python -B verify_distortion.py --full
```

Reports are written to a temporary directory and removed after the check. To keep them, pass `--output-dir verification-results`. For a quick input and one-variable robustness audit, use `--audit-only`; that mode does **not** verify the multivariate inequalities and is not a complete upper-bound certificate. Run without Python's `-O` option, which disables assertions.

[`data/verification-summary.json`](data/verification-summary.json) records the input and code hashes, runtime versions, and box counts from a complete successful run. The verification commands recompute the bounds; they do not rely on this saved summary.

## Code and data

The fixed mathematical inputs are JSON data in [`data/`](data/). Verification code is in Python files outside that directory. The checker does not execute code embedded in data or documentation, and a complete pass reconstructs both witness tables and checks that they agree exactly with the fixed inputs. Saved success messages are not used to accept an inequality.

| File | Purpose |
| --- | --- |
| [`certificate_distortion.py`](certificate_distortion.py) | Entry point for the complete v4 upper-bound check. |
| [`verify_distortion.py`](verify_distortion.py) | Runs the continuous-domain checks, input audits, exact identities, and robustness check. |
| [`intervals.py`](intervals.py) | Proof-independent outward interval arithmetic, automatic differentiation, and box subdivision. |
| [`distortion_polynomials.py`](distortion_polynomials.py) | The integer draw distribution's polynomial score and primitive. |
| [`finite_family_verifier.py`](finite_family_verifier.py), [`finish_finite_certificate.py`](finish_finite_certificate.py) | Auxiliary scalar polynomial bounds, their endpoint cases, and coefficient identities. |
| [`verify_rank_cut.py`](verify_rank_cut.py), [`finish_rank_cut_certificate.py`](finish_rank_cut_certificate.py) | The additional crossing bounds, nonnegative measures of common mass, and exact branch identities used by the rank-cut criterion. |
| [`audit_finite_witness.py`](audit_finite_witness.py) | Exact reconstruction of correction coefficients and coverage of the rational mass and ranking intervals. |
| [`verify_robustness_bound.py`](verify_robustness_bound.py) | Exact rational Bernstein checks of $(1-p)\lambda(p)+\tau_0(p)\leq16$, used for approximate stability. |
| [`certificate_lower_bound.py`](certificate_lower_bound.py) | The finite-degree inequalities and decreasing-tail argument from the v4 integer lower-bound proof. |
| [`data/parameters.json`](data/parameters.json) | Exact draw probabilities, normalization, interpolation parameters, and ranking intervals. |
| [`data/finite-family-witness.json`](data/finite-family-witness.json) | The auxiliary rational correction table. |
| [`data/rank-cut-witness.json`](data/rank-cut-witness.json) | The fixed coefficient table and measures specified in the v4 certificate appendix. |
| [`data/lower-bound.json`](data/lower-bound.json) | Rational parameters and claimed gaps for the v4 lower-bound arithmetic. |

## How the checks match v4

The rank-cut criterion in v4 has common measure mass $C_0=113609/200000$, so its distortion bound is $1+2C_0=2.13609$. The coefficient table uses $b=2840189/5000000$ and stability normalization $\mu=1085/4818$.

On each rational mass interval, `lambda_numerator` and `charge_numerators` give polynomials in ascending powers of $p$. Dividing by $p(1-p)$ gives $\lambda(p)$ and $\tau_j(p)$ for $0<p<1$. At an interior breakpoint, use the interval beginning there. At $p=0,1$, use the separately specified endpoint values. The table also specifies the nine allowed ranking intervals, cutoffs, atoms, and densities. The input audit checks the coefficient definitions and interval coverage. The complete check verifies the auxiliary polynomial bounds, the additional rank-cut bounds, nonnegativity and total mass of the measures, both mass endpoints, and the exact identities covering the local inequality's branches.

A box is accepted only when an outward enclosure proves the required sign. The checker may use the interval mean-value theorem, restrict a maximum to a face when a partial derivative has a certified sign, or subdivide a box. Floating point is used only to choose subdivision coordinates and report diagnostics; all acceptance decisions use rigorous enclosures. Algebraic factor cancellations and boundary extensions are checked by exact polynomial identities.

The filename `verify_v3.py` cited in the v4 manuscript is retained as a compatibility entry point: `python -B verify_v3.py --full` runs this same current check. Files cited there without a directory prefix are in `data/` when they are JSON inputs and in the repository root when they are Python code.

The manuscript proves the passage from these scalar inequalities to arbitrary profiles, including the rearrangement of marked mass, the geometric bound on ranking cuts, and the averaging argument using stability. The computation checks the scalar inequalities and identities needed by that proof. The robustness bound supports distortion $2.13609+\varepsilon$ for the manuscript's approximation algorithm; it does not claim that a finite implementation computes an exact stable lottery.

## Earlier versions

Both archives remain independently runnable, and their historical data is unchanged.

```console
python3 -B v1/certificate_distortion_v1.py
python3 -B v2/certificate_distortion.py
python3 -B v2/certificate_lower_bound.py
```

[`v1/`](v1/) contains the earlier mixed-rule certificate formerly in `old/`. [`v2/`](v2/) contains the previous top-level single-lottery certificate, generic Bernstein verifier, numerical search, tests, and documentation. Its optional search dependencies are listed in `v2/requirements.txt`. The numerical search and verification instructions for that historical result are in [`v2/README.md`](v2/README.md).
