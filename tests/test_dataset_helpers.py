import math

import pytest

from dataset_helpers import fmt_val, nan_or_float, normalise_stem, parse_target_formula, prop_sort_key


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("Ni50Mn35In15_sqs", "ni50mn35in15"),
        ("Ni50Mn35In15_SQS", "ni50mn35in15"),
        ("Gd5 Si2 Ge2_icsd", "gd5si2ge2icsd"),
        ("LaFe11p6Si1p4", "lafe11p6si1p4"),
    ],
)
def test_normalise_stem(stem, expected):
    assert normalise_stem(stem) == expected


@pytest.mark.parametrize("value", [None, "", "abc", float("nan")])
def test_nan_or_float_returns_nan_for_missing_values(value):
    assert math.isnan(nan_or_float(value))


def test_nan_or_float_parses_numbers():
    assert nan_or_float("18.9") == 18.9
    assert nan_or_float(5) == 5.0


def test_fmt_val_blanks_missing_values():
    assert fmt_val(None) == ""
    assert fmt_val(float("nan")) == ""
    assert fmt_val(3.2) == 3.2


def test_prop_sort_key_prefers_5_tesla_then_field_then_entropy_change():
    rows = [
        ("a", None, "2.0", "30.0"),
        ("b", None, "5.0", "4.0"),
        ("c", None, "5.0", "-12.0"),
        ("d", None, "7.0", "1.0"),
        ("e", None, "", ""),
    ]
    ordered = [r[0] for r in sorted(rows, key=prop_sort_key)]
    assert ordered == ["c", "b", "d", "a", "e"]


def test_parse_target_formula_handles_decimal_p_and_spaces():
    pytest.importorskip("pymatgen")
    comp, err = parse_target_formula("LaFe11p6Si1p4_sqs.cif")
    assert err is None
    assert comp["Fe"] == pytest.approx(11.6) and comp["Si"] == pytest.approx(1.4)
    comp, err = parse_target_formula("Gd5 Si2 Ge2_icsd.cif")
    from pymatgen.core import Composition

    assert err is None and comp == Composition("Gd5Si2Ge2")


def test_parse_target_formula_reports_errors_instead_of_raising(monkeypatch):
    pymatgen_core = pytest.importorskip("pymatgen.core")

    def unparseable(formula):
        raise ValueError(f"cannot parse {formula}")

    # pymatgen accepts unknown symbols as dummy species, so simulate a parse failure directly
    monkeypatch.setattr(pymatgen_core, "Composition", unparseable)
    comp, err = parse_target_formula("Fe2Si_x.cif")
    assert comp is None
    assert "cannot parse Fe2Si" in err
