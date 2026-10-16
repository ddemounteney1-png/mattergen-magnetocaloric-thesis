"""Small helpers used by prepare_dataset.py.

They live in their own module so they can be imported and tested without
running the dataset preparation itself. The code is unchanged from
prepare_dataset.py, apart from importing pymatgen inside parse_target_formula.
"""

import math
import re
from pathlib import Path


def normalise_stem(stem: str) -> str:
    """Canonical form of a CIF stem for grouping/matching (never used for renaming)."""
    s = re.sub(r'_sqs$', '', stem, flags=re.IGNORECASE)
    s = re.sub(r'[\s_]', '', s)
    return s.lower()

def parse_target_formula(filename: str):
    """
    Extract target formula from filename stem.
    Rule: take stem up to (not including) the first underscore,
    strip spaces (FIX 1), replace digit-p-digit with digit.digit, parse as Composition.
    Returns (Composition, None) on success, (None, error_str) on failure.
    """
    stem = Path(filename).stem
    raw  = stem.split('_')[0]
    raw  = raw.replace(' ', '')                          # FIX 1
    raw  = re.sub(r'(?<=\d)p(?=\d)', '.', raw)
    from pymatgen.core import Composition

    try:
        comp = Composition(raw)
        return comp, None
    except Exception as e:
        return None, str(e)

def nan_or_float(v) -> float:
    """Convert CSV cell to float, returning nan on failure or blank."""
    if v is None:
        return float('nan')
    try:
        f = float(v)
        return float('nan') if math.isnan(f) else f
    except (ValueError, TypeError):
        return float('nan')

def fmt_val(v):
    """CSV cell: empty string for NaN/None, raw value otherwise."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return v

def prop_sort_key(row_tuple):
    """Sort key: prefer 5 T field, then highest field, then highest |dS|."""
    _, _, fld, ds = row_tuple
    fld_f = nan_or_float(fld)
    ds_f  = nan_or_float(ds)
    if math.isnan(fld_f): fld_f = 0.0
    if math.isnan(ds_f):  ds_f  = 0.0
    is_5t = 1 if abs(fld_f - 5.0) < 1e-6 else 0
    return (-is_5t, -fld_f, -abs(ds_f))
