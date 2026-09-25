from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "docker.yml"


def test_docker_ci_workflow_exists():
    assert WORKFLOW_PATH.exists()


def test_docker_ci_uses_production_dockerfile():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "Dockerfile.production" in content


def test_docker_ci_builds_image():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "docker build" in content
    assert "churnguard-ai:ci" in content


def test_docker_ci_runs_container():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "docker run" in content
    assert "churnguard-ci" in content


def test_docker_ci_checks_health():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "/health" in content


def test_docker_ci_checks_readiness():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "/ready" in content


def test_docker_ci_has_cleanup():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "if: always()" in content
    assert "docker rm -f churnguard-ci" in content
