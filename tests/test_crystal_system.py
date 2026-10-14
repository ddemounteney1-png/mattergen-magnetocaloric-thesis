import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def export():
    pytest.importorskip("tqdm")
    return load_script("base_dataset/05_export_mattergen.py")


@pytest.mark.parametrize(
    "sg, expected",
    [
        (1, "Triclinic"), (2, "Triclinic"),
        (3, "Monoclinic"), (15, "Monoclinic"),
        (16, "Orthorhombic"), (74, "Orthorhombic"),
        (75, "Tetragonal"), (142, "Tetragonal"),
        (143, "Trigonal"), (167, "Trigonal"),
        (168, "Hexagonal"), (194, "Hexagonal"),
        (195, "Cubic"), (230, "Cubic"),
        (None, "Unknown"),
    ],
)
def test_crystal_system_boundaries(export, sg, expected):
    assert export.get_crystal_system(sg) == expected


def test_both_crystal_system_lookups_agree(export):
    """The base-dataset export and the CIF quality report use separate lookups."""
    pytest.importorskip("pymatgen")
    quality = load_script("data_parsing/analyze_cif_quality.py")
    for sg in range(1, 231):
        assert quality.sg_to_crystal_system(sg) == export.get_crystal_system(sg).lower()
