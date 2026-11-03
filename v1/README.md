# Archived version 1: distortion at most 2.3282

This directory preserves the earlier mixed-rule result with bound 11641/5000.
The current 2.13609 artifact is in the repository root. The original candidate,
frozen certificate, and generic Bernstein kernel are preserved here.

From the repository root (Python 3.9+, standard library only):

```console
python3 -B v1/certificate_distortion_v1.py
```

To regenerate its certificate, run from the repository root:

```console
python3 -B v2/verify_distortion.py v1/candidate_v1.json --certificate v2/regenerated-certificate-v1.py
```
