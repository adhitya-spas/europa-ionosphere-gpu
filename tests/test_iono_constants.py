import os
import re
import inspect
import importlib
from physics_constants import C_LIGHT, C_IONO, R_EUROPA


def test_constants_present():
    assert isinstance(C_LIGHT, float)
    assert isinstance(C_IONO, float)
    assert isinstance(R_EUROPA, float)


def test_sanity_slab():
    # tiny slab check (fast)
    from tools.iono_consistency_check import slab_test
    ok, rel_err, stec, tec_inv = slab_test(Ne_value=1e10, alt_top_m=200e3, f_low=8.5e6, f_high=9.5e6, tol_rel=0.05)
    assert ok, f"Slab consistency failed: rel_err={rel_err}"


def _is_in_triple_quotes(line_idx, lines):
    # naive triple-quote state machine
    in_triple = False
    for i, L in enumerate(lines[:line_idx+1]):
        if "\"\"\"" in L or "'''" in L:
            # toggle for each occurrence (conservative)
            in_triple = not in_triple
    return in_triple


def test_no_forbidden_literals_in_code():
    """Scan .py files for forbidden numeric literals outside comments/docstrings."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    forbidden = ["80.6", "6.371e6", "6.371E6", "6.371e+6"]
    py_files = []
    for root, _, files in os.walk(repo_root):
        for fn in files:
            if fn.endswith('.py'):
                py_files.append(os.path.join(root, fn))

    for path in py_files:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            # skip pure comments
            if stripped.startswith('#'):
                continue
            # skip lines inside naive triple-quote detection
            if _is_in_triple_quotes(i, lines):
                continue
            for lit in forbidden:
                if lit in line:
                    raise AssertionError(f"Forbidden literal '{lit}' found in {path}:{i+1}: {line.strip()}")
