# Improving Randomized Metric Distortion to 2.3282

This repository contains the computational artifact accompanying the paper *Improving Randomized Metric Distortion to $2.3282$*. It separates numerical discovery from exact verification:

- [`search_distortion.py`](search_distortion.py) numerically searches for and rationalizes a candidate. Its calculations use floating-point optimization and are heuristic.
- [`candidate.json`](candidate.json) records the resulting rational candidate in a machine-readable format.
- [`verify_distortion.py`](verify_distortion.py) reconstructs the polynomial inequalities from the rational data and verifies them using exact arithmetic and Bernstein subdivision. It uses only the Python standard library.

The checked candidate uses

$$
\Pr[D^\star=1]=\frac{423239}{10^6},\qquad
\Pr[D^\star=12]=\frac{471105}{10^6},\qquad
\Pr[D^\star=13]=\frac{105656}{10^6},
$$

and mixes Integrated Veto and the random-size stable lottery with weights $7941/50000$ and $42059/50000$, respectively. The verified target is

$$
T(q)=\frac{33205}{42059}q-\frac{7941}{42059}q^2,
$$

which yields distortion $11641/5000=2.3282$.

## Exact verification

Python 3.9 or later is sufficient; no third-party packages are required. Run

```console
$ python3 verify_distortion.py
All polynomial nonnegativity checks passed.
```

This command verifies the candidate built into the script. To verify the checked-in JSON file instead, run

```console
$ python3 verify_distortion.py candidate.json
All polynomial nonnegativity checks passed.
```

Any failed check terminates the program. A successful run verifies the dual-feasibility inequalities on their stated domains and the finite candidate-cover disjunction used in Appendix B of the paper. Depending on the machine, the exact rational computation can take several minutes.

## Reproducing the numerical search

The search code was tested with Python 3.9.6, NumPy 1.26.3, and SciPy 1.13.1. A reproducible environment can be created with

```console
$ python3 -m venv .venv
$ source .venv/bin/activate
$ python3 -m pip install -r requirements.txt
```

Run the selected-support search and rationalization with

```console
$ python3 search_distortion.py
```

This overwrites `candidate.json` with the rationalized candidate and prints the exact verification command. The default seed is fixed, but intermediate floating-point values may vary slightly across platforms and library versions.

Two additional modes expose the numerical programs described in the paper:

```console
# Discretized dual and numerical constraint generation at fixed (q,x)
$ python3 search_distortion.py dual --q 0.4 --x 0.6

# Finite q- and x-grid search followed by the outer rule-mixture LP
$ python3 search_distortion.py grid

# Also run numerical separation for each fixed dual
$ python3 search_distortion.py grid --separate
```

Use `python3 search_distortion.py --help` for all grid sizes, tolerances, seeds, and output options.

## Candidate format

`candidate.json` contains the paper-facing inputs:

- the draw distribution $D$;
- the rule weights $(\alpha,\beta,\gamma)$ and $C_{\mathrm{cut}}$;
- the coefficients of $T(q)$;
- the active $y$-values and tangency $t$-values defining the dual families; and
- the verification domains.

All exact values are encoded as integers or rational strings. The verifier rejects floating-point data, a target inconsistent with the rule mixture, or a domain that does not cover the domain required in the paper.

## Scope of the computation

The numerical search explains how the rational candidate was found. It does **not** prove that the numerical dual solutions are feasible over continuous state spaces, that the sampled envelope is an upper bound, or that $D^\star$ is optimal among draw distributions. Those numerical outputs become part of the proof only after rationalization and successful execution of `verify_distortion.py`.
