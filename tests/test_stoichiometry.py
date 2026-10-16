import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def quality():
    pytest.importorskip("pymatgen")
    return load_script("data_parsing/analyze_cif_quality.py")


def test_target_formula_fractions_sum_to_one(quality):
    fractions = quality.parse_target_formula("Ni50Mn35In15")
    assert fractions == pytest.approx({"Ni": 0.50, "Mn": 0.35, "In": 0.15})
    assert quality.parse_target_formula("Qq2Zz3") == {}


def test_stoich_deviation_is_largest_per_element_error_in_percent(quality):
    target = {"Ni": 0.50, "Mn": 0.35, "In": 0.15}
    actual = {"Ni": 0.50, "Mn": 0.30, "In": 0.20}
    worst, per_element = quality.stoich_deviation(actual, target)
    assert worst == pytest.approx(5.0)
    assert per_element["Ni"] == pytest.approx(0.0)


def test_missing_element_counts_as_full_error(quality):
    worst, _ = quality.stoich_deviation({"Fe": 1.0}, {"Fe": 0.9, "Co": 0.1})
    assert worst == pytest.approx(10.0)


@pytest.mark.parametrize(
    "err, missing, rank_delta, found, expected",
    [
        (0.0, set(), 0, False, "SKIP"),
        (5.0, set(), 0, True, "PASS"),
        (5.1, set(), 0, True, "WARN"),
        (10.0, set(), 0, True, "WARN"),
        (10.1, set(), 0, True, "FAIL"),
        (1.0, {"Co"}, 0, True, "WARN"),
        (1.0, set(), -1, True, "WARN"),
        (1.0, set(), -2, True, "FAIL"),
    ],
)
def test_status_thresholds(quality, err, missing, rank_delta, found, expected):
    assert quality.status_from_metrics(err, missing, set(), rank_delta, found) == expected
