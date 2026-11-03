# Archived version 2: distortion at most 2.1441

Run the commands below from this directory. This archive preserves the earlier result; the current 2.13609 artifact is in the repository root.

# Exact certificates for randomized metric distortion

This repository contains the search and exact-verification artifact for *Improving Randomized Metric Distortion to 2.1441*. Numerical discovery, adaptive certification, and checking a frozen certificate are separate steps. It also contains the exact rational certificate for the universal lower bound $267/125$.

The current result uses a single random-size stable lottery with

$$
\Pr[D^\star=1]=\frac{462688493532}{10^{12}},\quad
\Pr[D^\star=2]=\frac{194607598780}{10^{12}},\quad
\Pr[D^\star=9]=\frac{156952022902}{10^{12}},\quad
\Pr[D^\star=10]=\frac{185751884786}{10^{12}}.
$$

Its exact threshold-potential certificate gives distortion at most $107200742653/50000000000=2.14401485306$.

## Files

- [`certificate_distortion.py`](certificate_distortion.py) is the reader-facing, frozen certificate for the current result. It checks a fixed subdivision and performs no adaptive search.
- [`certificate_lower_bound.py`](certificate_lower_bound.py) is the reader-facing certificate for the universal lower bound $267/125$. It follows the manuscript proof directly and performs no search.
- [`candidate.json`](candidate.json) records the current exact polynomial problems.
- [`verify_distortion.py`](verify_distortion.py) is the generic adaptive verifier and certificate generator.
- [`bernstein.py`](bernstein.py) contains proof-independent exact polynomial, Bernstein-basis, subdivision, and certificate-checking routines. It contains no metric-distortion polynomials.
- [`search_distortion.py`](search_distortion.py) is the numerical search for the current integer draw distribution and the proof-specific conversion of a rationalized result into `candidate.json`.
- [`../v1/candidate_v1.json`](../v1/candidate_v1.json) and [`../v1/certificate_distortion_v1.py`](../v1/certificate_distortion_v1.py) archive the earlier certificate for the mixed rule with distortion bound $11641/5000=2.3282$.

## Check the exact certificates

Python 3.9 or later is sufficient, and no third-party package is required:

```console
$ python3 -B certificate_distortion.py
Exact certificate verified: 7 polynomial problems, 192 accepted boxes, maximum depth 48

$ python3 -B certificate_lower_bound.py
paper separator verified: degrees 1 through 17, infinite tail certified, largest finite value -5709803/2500000000000
independent separator verified: degrees 1 through 40, infinite tail certified, largest finite value -170580257/225000000000000
Exact lower-bound certificate verified: inf_D dist(RSL(D)) >= 267/125.

$ python3 -B ../v1/certificate_distortion_v1.py
Exact certificate verified: 56 polynomial problems, 1150 accepted boxes, maximum depth 31
```

The lower-bound certificate reconstructs the manuscript's three scalar functions, checks the finite rational inequalities, and certifies the infinite tail using the exact monotonicity argument from the proof. It does not use the Bernstein routines needed by the upper-bound certificates.

Each distortion-certificate file embeds the relevant polynomial problem, exact root Bernstein control nets, a fixed binary midpoint-subdivision tree, and the alternative selected at every leaf. The checker:

1. expands each recorded root control net back into the power basis and checks exact equality with the stated polynomial after the stated affine change of variables;
2. follows the recorded tree, using exact de Casteljau subdivision, so its two children cover their parent's box at every split; and
3. checks that every coefficient required by the selected alternative is nonnegative at each leaf (or positive when the input explicitly asks for strict positivity).

Thus these commands check the recorded proof rather than search for a new subdivision. Zero coefficients are stored sparsely but treated as present zeros.

## Generate a frozen certificate adaptively

To repeat the one-time adaptive calculation that produced the current certificate, run

```console
$ python3 -B verify_distortion.py candidate.json \
    --certificate certificate_distortion.py
```

The same generic verifier handles the structurally different previous proof:

```console
$ python3 -B verify_distortion.py ../v1/candidate_v1.json \
    --certificate regenerated-certificate-v1.py
```

The output file is written only after adaptive exact verification succeeds; the generated static checker is then run once. These calculations use exact `fractions.Fraction` arithmetic and can take several minutes.

## Common candidate format

Both JSON files use the same proof-agnostic schema. A verification problem specifies a rational box $B$, polynomial alternatives $\mathcal P_1,\ldots,\mathcal P_k$, and a sign relation for each polynomial. Its meaning is

$$
\forall z\in B,\qquad \bigvee_{j=1}^k\ \bigwedge_{P\in\mathcal P_j} P(z)\geq 0,
$$

with $>$ replacing $\geq$ where requested. The complete input is the conjunction of all its verification problems. This accommodates both proofs without making their mathematical data artificially identical: the current threshold-potential proof mostly has one polynomial per problem, whereas the earlier outer-cover argument uses many alternative polynomials.

All coefficients and domain endpoints are JSON integers or rational strings. Floating-point values are rejected.

## Reproduce the numerical search

The search requires NumPy and SciPy. The versions in [`requirements.txt`](requirements.txt) reproduce the development environment:

```console
$ python3 -m venv .venv
$ source .venv/bin/activate
$ python3 -m pip install -r requirements.txt
$ python3 -B search_distortion.py
```

By default, the program enumerates all 6,195 subsets of $\{1,\ldots,20\}$ of size at most four. It optimizes weights on a 32-state grid, refines the best 100 supports on an 80-state grid, applies differential evolution followed by SLSQP to the best 20 refined supports, and reevaluates those finalists on a 320-state grid together with the continuous boundary obstruction. It then rationalizes the selected law, threshold, and endpoint and writes the resulting exact polynomial problems to `candidate.json`. Use `--workers` to parallelize the coarse screen and `--help` for all controls. A full run is expensive.

A quick format smoke test is:

```console
$ python3 -B search_distortion.py \
    --max-degree 1 --max-support-size 1 \
    --grid-size 2 --refine-grid-size 2 --final-grid-size 2 \
    --shortlist 1 \
    --no-global-refinement --output /tmp/search-candidate.json
$ python3 -B verify_distortion.py /tmp/search-candidate.json --maximum-depth 8
```

The finite-grid objectives, SLSQP runs, differential evolution, shortlist decisions, and reported “best candidate” are numerical heuristics. They prove neither global optimality of the weights nor optimality of the draw distribution, and a generated candidate proves no distortion bound unless `verify_distortion.py` succeeds on it. The checked-in `candidate.json` is the rational candidate on which that exact continuum verification has succeeded.
