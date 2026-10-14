import collections

import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def export():
    pytest.importorskip("tqdm")
    return load_script("base_dataset/05_export_mattergen.py")


def _entries():
    systems = {"Cubic": 500, "Hexagonal": 200, "Triclinic": 37, "Monoclinic": 3}
    return [{"id": f"{cs}-{i}", "crystal_system": cs} for cs, n in systems.items() for i in range(n)]


def test_every_entry_lands_in_exactly_one_split(export):
    entries = _entries()
    train, val, test = export.stratified_split(list(entries))
    ids = [e["id"] for e in train + val + test]
    assert sorted(ids) == sorted(e["id"] for e in entries)


def test_split_is_80_10_10_within_each_crystal_system(export):
    train, val, test = export.stratified_split(_entries())
    count = lambda split: collections.Counter(e["crystal_system"] for e in split)  # noqa: E731
    assert count(train)["Cubic"] == 400 and count(val)["Cubic"] == 50 and count(test)["Cubic"] == 50
    assert count(train)["Hexagonal"] == 160 and count(val)["Hexagonal"] == 20
    # small groups still put at least one structure in training
    assert count(train)["Monoclinic"] >= 1


def test_split_is_reproducible_with_the_same_seed(export):
    first = export.stratified_split(_entries(), seed=42)
    second = export.stratified_split(_entries(), seed=42)
    assert [[e["id"] for e in s] for s in first] == [[e["id"] for e in s] for s in second]
