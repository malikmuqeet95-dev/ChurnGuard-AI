from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_workflow_exists():
    assert WORKFLOW_PATH.exists()


def test_ci_workflow_uses_python_312():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'python-version: "3.12"' in content


def test_ci_workflow_runs_pytest():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "python -m pytest tests/ -v" in content


def test_ci_workflow_uses_requirements_freeze():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "requirements-freeze.txt" in content


def test_ci_workflow_runs_on_main_push():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "push:" in content
    assert "main" in content