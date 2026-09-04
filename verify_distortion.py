#!/usr/bin/env python3
"""Find and optionally write an exact Bernstein subdivision certificate.

The input is proof-agnostic.  It contains rational polynomials grouped into
verification problems.  Each problem asks that, at every point of a rational
box, at least one listed alternative have all of its polynomials nonnegative
(or positive when explicitly requested).

The adaptive calculation uses exact ``fractions.Fraction`` arithmetic.  When
``--certificate`` is supplied, a deterministic Python checker containing the
fixed subdivision tree and root Bernstein control nets is written only after
the complete verification succeeds.  The generated checker is then run once.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from bernstein import find_certificate, summarize_counts, verify_certificate


HERE = Path(__file__).resolve().parent
DEFAULT_CANDIDATE = HERE / "candidate.json"


def load_candidate(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(data, Mapping):
        raise ValueError("the candidate JSON root must be an object")
    return data


def _embedded_json(data: Mapping[str, Any]) -> str:
    rendered = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True)
    if "'''" in rendered:
        return repr(rendered)
    return "r'''\n" + rendered + "\n'''"


def _certificate_description(problem: Mapping[str, Any]) -> str:
    metadata = problem.get("metadata", {})
    title = str(metadata.get("title", "Exact polynomial certificate"))
    bound = metadata.get("claimed_distortion_bound")
    distribution = metadata.get("draw_distribution")
    lines = [title]
    if bound is not None:
        lines.append(f"Claimed distortion bound: {bound}.")
    if isinstance(distribution, list):
        atoms = []
        for atom in distribution:
            if isinstance(atom, Mapping):
                atoms.append(f"D={atom.get('degree')}: {atom.get('probability')}")
        if atoms:
            lines.append("Draw distribution: " + ", ".join(atoms) + ".")
    lines.append(
        "This generated checker verifies a fixed exact Bernstein certificate; "
        "it does not search for subdivisions."
    )
    return "\n\n".join(lines)


def certificate_source(problem: Mapping[str, Any],
                       certificate: Mapping[str, Any]) -> str:
    """Render a deterministic, relocatable Python certificate checker."""

    problem_literal = _embedded_json(problem)
    certificate_literal = _embedded_json(certificate)
    description_literal = repr(_certificate_description(problem))
    return f'''#!/usr/bin/env python3
{description_literal}

from __future__ import annotations

import json
from pathlib import Path
import sys


_HERE = Path(__file__).resolve()
for _PARENT in (_HERE.parent, *_HERE.parents):
    if (_PARENT / "bernstein.py").is_file():
        sys.path.insert(0, str(_PARENT))
        break
else:
    raise RuntimeError("could not locate bernstein.py")

from bernstein import summarize_counts, verify_certificate


PROBLEM = json.loads({problem_literal})
CERTIFICATE = json.loads({certificate_literal})


def main() -> None:
    counts = verify_certificate(PROBLEM, CERTIFICATE)
    print("Exact certificate verified:", summarize_counts(counts))


if __name__ == "__main__":
    main()
'''


def write_atomic(path: Path, contents: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
        path.chmod(0o755)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate", nargs="?", type=Path, default=DEFAULT_CANDIDATE,
        help="polynomial verification problem (default: candidate.json)",
    )
    parser.add_argument(
        "--certificate", type=Path,
        help=("write a static Python certificate checker inside the source "
              "directory"),
    )
    parser.add_argument(
        "--maximum-depth", type=int, default=64,
        help="maximum adaptive subdivision depth per problem (default: 64)",
    )
    args = parser.parse_args(argv)

    candidate_path = args.candidate.resolve()
    if args.certificate is not None:
        certificate_path = args.certificate.resolve()
        protected = {candidate_path, Path(__file__).resolve(),
                     (HERE / "bernstein.py").resolve()}
        if certificate_path in protected:
            parser.error("the certificate output may not overwrite an input program")
        if HERE not in certificate_path.parents:
            parser.error("the certificate output must be inside the source directory")
    else:
        certificate_path = None

    problem = load_candidate(candidate_path)
    certificate, counts = find_certificate(
        problem, maximum_depth=args.maximum_depth
    )
    # Check the fixed representation before serializing it.  This exercises
    # the reverse Bernstein expansion and the non-searching tree checker.
    checked_counts = verify_certificate(problem, certificate)
    if checked_counts != counts:
        raise AssertionError("adaptive and static certificate counts disagree")
    print(
        "Adaptive exact verification succeeded:", summarize_counts(counts),
        flush=True,
    )

    if certificate_path is not None:
        write_atomic(
            certificate_path, certificate_source(problem, certificate)
        )
        subprocess.run(
            [sys.executable, str(certificate_path)],
            cwd=certificate_path.parent,
            check=True,
        )
        print(f"Wrote static certificate to {certificate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
