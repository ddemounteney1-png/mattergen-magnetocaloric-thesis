"""Re-run a few thesis figure scripts from committed results.

Skipped automatically when Git LFS files have not been downloaded.
"""

import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT


def _lfs_pointer(path):
    with open(path, "rb") as fh:
        return fh.read(40).startswith(b"version https://git-lfs")


@pytest.mark.parametrize(
    "script, output, needs",
    [
        ("fig17_sun_comparison.py", "fig17_sun_comparison.png", None),
        ("fig22_element_freq_by_ds.py", "fig22_element_freq_by_ds.png", None),
        ("fig11_convergence_comparison.py", "fig11_convergence_comparison.png", "results/loss_curves/pretrain_losses.csv"),
    ],
)
def test_figure_script_runs(script, output, needs, tmp_figures):
    pytest.importorskip("matplotlib")
    if needs and _lfs_pointer(REPO_ROOT / needs):
        pytest.skip("Git LFS data not downloaded")
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "code" / "analysis" / script)],
        cwd=tmp_figures.parent, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert (tmp_figures / output).stat().st_size > 10_000
