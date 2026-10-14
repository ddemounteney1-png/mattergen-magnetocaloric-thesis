import importlib

import project_paths


def test_committed_folders_are_inside_the_repository():
    root = project_paths.REPO_ROOT
    for folder in (project_paths.DATA, project_paths.METRICS, project_paths.LOSS_CURVES, project_paths.DOCS):
        assert folder.is_dir()
        assert root in folder.parents


def test_working_folders_follow_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_WORK_DIR", str(tmp_path))
    monkeypatch.delenv("BASE_DATASET_DIR", raising=False)
    monkeypatch.setenv("MATTERGEN_DIR", str(tmp_path / "mg"))
    paths = importlib.reload(project_paths)
    try:
        assert paths.WORK == tmp_path
        assert paths.BASE_DATASET_WORK == tmp_path / "base_dataset"
        assert paths.MATTERGEN_DIR == tmp_path / "mg"
    finally:
        monkeypatch.undo()
        importlib.reload(project_paths)
