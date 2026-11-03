"""Locations of immutable rational inputs and generated verification reports."""
from pathlib import Path
import json
import os

if not __debug__:
    raise RuntimeError("Run without -O: the certificate checks require assertions.")

CODE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('METRIC_DISTORTION_DATA_DIR', CODE / 'data')).resolve()
OUTPUT = Path(os.environ.get('METRIC_DISTORTION_OUTPUT_DIR', CODE / 'verification-results')).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)

def load_data(name):
    def reject_float(value):
        raise ValueError("Rational input must use integers or rational strings, not floats: " + value)
    return json.loads((DATA / name).read_text(), parse_float=reject_float,
                      parse_constant=reject_float)
