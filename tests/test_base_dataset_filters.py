"""Filters and deduplication used to build the 274,602-structure base dataset."""

import numpy as np
import pytest

from conftest import load_script

pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure  # noqa: E402


@pytest.fixture(scope="module")
def mp_filter():
    return load_script("base_dataset/02_load_filter_mp.py")


@pytest.fixture(scope="module")
def merge():
    pytest.importorskip("tqdm")
    return load_script("base_dataset/04_merge_dedup.py")


def rocksalt(a=5.64):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_ordered_inorganic_structure_passes(mp_filter):
    prim, reason = mp_filter.filter_structure(rocksalt(), "test")
    assert reason is None
    assert len(prim) == 2


def test_partial_occupancy_is_rejected(mp_filter):
    s = Structure(Lattice.cubic(3.0), [{"Fe": 0.5}], [[0, 0, 0]])
    assert mp_filter.filter_structure(s, "test")[1].startswith("partial_occupancy")


def test_mixed_occupancy_is_rejected(mp_filter):
    s = Structure(Lattice.cubic(3.0), [{"Fe": 0.5, "Co": 0.5}], [[0, 0, 0]])
    assert mp_filter.filter_structure(s, "test")[1] == "mixed_occupancy"


def test_structures_over_51_atoms_are_rejected(mp_filter):
    rng = np.random.default_rng(0)
    s = Structure(Lattice.cubic(15.0), ["Fe"] * 52, rng.random((52, 3)))
    assert mp_filter.filter_structure(s, "test")[1] == "too_many_atoms:52"


def test_organic_structures_are_rejected(mp_filter):
    s = Structure(Lattice.cubic(6.0), ["C", "H", "H", "H", "H"],
                  [[0, 0, 0], [0.1, 0.1, 0.1], [0.9, 0.9, 0.1], [0.9, 0.1, 0.9], [0.1, 0.9, 0.9]])
    assert mp_filter.filter_structure(s, "test")[1] == "organic_CH"


def test_dedup_keeps_highest_priority_source_and_merges_metadata(merge):
    entries = [
        {"source": "COD", "source_id": "cod-1", "structure": rocksalt(5.64), "band_gap": 5.0},
        {"source": "MP", "source_id": "mp-22862", "structure": rocksalt(5.65), "band_gap": None},
        {"source": "Alexandria", "source_id": "alex-9", "structure": rocksalt(7.5), "band_gap": None},
    ]
    kept, n_dups = merge.dedup_formula_group(entries)
    assert n_dups == 1
    assert [e["source_id"] for e in kept] == ["mp-22862", "alex-9"]
    # metadata the MP entry lacked is filled in from the COD duplicate
    assert kept[0]["band_gap"] == 5.0


def test_merge_metadata_never_overwrites_identity_fields(merge):
    keep = {"source": "MP", "formula": "NaCl", "band_gap": None}
    drop = {"source": "COD", "formula": "ClNa", "band_gap": 4.9}
    merged = merge.merge_metadata(keep, drop)
    assert merged == {"source": "MP", "formula": "NaCl", "band_gap": 4.9}
